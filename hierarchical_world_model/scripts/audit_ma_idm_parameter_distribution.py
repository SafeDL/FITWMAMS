#!/usr/bin/env python3
"""Compare calibrated MA-IDM population draws with fitted highD driver MAPs.

This is a calibration-cohort consistency audit, not an independent validation:
the driver MAP estimates and empirical-Bayes population posterior use the same
observed calibration cohort. Held-out behavioral prediction metrics remain the
separate check against unseen recordings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_model_baselines.models.bayesian_ma_idm.src.model import (  # noqa: E402
    LOWER,
    UPPER,
)
from traffic_components.src.core.utils import file_sha256, save_json  # noqa: E402


POSTERIOR = ROOT / "external_model_baselines/models/bayesian_ma_idm/evidence/deployment/ma_idm_all_251.npz"
KEY_PARAMETERS = ("v0_mps", "s0_m", "T_s", "alpha_mps2", "beta_mps2")


def _population_draws(
    log_mean: np.ndarray,
    log_cov: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = rng.integers(len(log_mean), size=count)
    draws = np.empty((count, log_mean.shape[-1]), dtype=np.float64)
    for index in np.unique(selected):
        rows = np.flatnonzero(selected == index)
        draws[rows] = rng.multivariate_normal(
            log_mean[index], log_cov[index], size=len(rows), check_valid="raise"
        )
    return np.clip(np.exp(draws), lower, upper)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    ordered = np.asarray(values)[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(ordered) != 0.0) + 1, len(values)]
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        result[order[begin:end]] = 0.5 * (begin + end - 1)
    return result


def _correlation(values: np.ndarray) -> np.ndarray:
    return np.corrcoef(np.stack([_rank(values[:, i]) for i in range(values.shape[1])]))


def _wasserstein_1d(left: np.ndarray, right: np.ndarray) -> float:
    q = np.linspace(0.0, 1.0, 1001)
    return float(np.mean(np.abs(np.quantile(left, q) - np.quantile(right, q))))


def _ks_1d(left: np.ndarray, right: np.ndarray) -> float:
    points = np.sort(np.concatenate((left, right)))
    left_cdf = np.searchsorted(np.sort(left), points, side="right") / len(left)
    right_cdf = np.searchsorted(np.sort(right), points, side="right") / len(right)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def build_report(*, posterior_path: Path = POSTERIOR, draws: int = 50_000, seed: int = 8127) -> dict:
    with np.load(posterior_path, allow_pickle=False) as archive:
        names = [str(x) for x in archive["parameter_names"]]
        indexes = [names.index(name) for name in KEY_PARAMETERS]
        driver_map = np.asarray(archive["driver_map"][:, indexes], dtype=np.float64)
        population = _population_draws(
            archive["population_log_mean"][:, indexes],
            archive["population_log_cov"][:, indexes][:, :, indexes],
            lower=LOWER[indexes],
            upper=UPPER[indexes],
            count=draws,
            seed=seed,
        )
        recording_id = np.asarray(archive["recording_id"], dtype=int)
        follower_id = np.asarray(archive["follower_id"], dtype=int)
    cv_path = posterior_path.parents[1] / "population" / "summary.json"
    cv = json.loads(cv_path.read_text(encoding="utf-8"))
    ma_cv = cv["models"]["ma_idm"]["pair_weighted_horizons"]
    parameters = {}
    for column, name in enumerate(KEY_PARAMETERS):
        calibrated = driver_map[:, column]
        predictive = population[:, column]
        q_map = np.quantile(calibrated, [0.05, 0.25, 0.5, 0.75, 0.95])
        q_pop = np.quantile(predictive, [0.05, 0.25, 0.5, 0.75, 0.95])
        parameters[name] = {
            "driver_map_q05_q25_q50_q75_q95": [float(x) for x in q_map],
            "population_posterior_predictive_q05_q25_q50_q75_q95": [float(x) for x in q_pop],
            "wasserstein_1d": _wasserstein_1d(calibrated, predictive),
            "wasserstein_over_driver_map_iqr": _wasserstein_1d(calibrated, predictive)
            / max(float(q_map[3] - q_map[1]), 1.0e-8),
            "ks_statistic": _ks_1d(calibrated, predictive),
            "fraction_driver_maps_inside_population_90pct_interval": float(
                np.mean((calibrated >= q_pop[0]) & (calibrated <= q_pop[-1]))
            ),
        }
    corr_map = _correlation(driver_map)
    corr_population = _correlation(population)
    correlations = {
        "parameter_order": list(KEY_PARAMETERS),
        "driver_map_spearman": corr_map.tolist(),
        "population_posterior_predictive_spearman": corr_population.tolist(),
        "mean_absolute_pairwise_difference": float(
            np.mean(np.abs(corr_map[np.triu_indices(len(KEY_PARAMETERS), 1)]
                           - corr_population[np.triu_indices(len(KEY_PARAMETERS), 1)]))
        ),
    }
    return {
        "schema": "ma_idm_population_parameter_distribution_audit_v1",
        "calibration_method": "Empirical-Bayes correlated log-normal hierarchical MAP with joint population covariance draws",
        "source_artifact": str(posterior_path.relative_to(ROOT)),
        "source_sha256": file_sha256(posterior_path),
        "calibration_pairs": int(len(driver_map)),
        "unique_follower_ids": int(len(np.unique(follower_id))),
        "recording_pair_counts": {
            str(recording): int(np.sum(recording_id == recording))
            for recording in np.unique(recording_id)
        },
        "leave_one_recording_out_behavior_validation": {
            "source": str(cv_path.relative_to(ROOT)),
            "protocol": cv["protocol"],
            "horizons": {
                seconds: {
                    "position_rmse_m": float(values["position_rmse_m"]),
                    "speed_rmse_mps": float(values["speed_rmse_mps"]),
                    "acceleration_crps_mps2": float(values["acceleration_crps_mps2"]),
                    "position_90_coverage": float(values["position_90_coverage"]),
                }
                for seconds, values in ma_cv.items()
            },
            "interpretation": "Independent recording holdouts exist, but the 90% position intervals are under-dispersed.",
        },
        "posterior_predictive_draws": int(draws),
        "posterior_predictive_seed": int(seed),
        "decision_grid_hz": 5,
        "parameters": parameters,
        "joint_dependence": correlations,
        "interpretation": (
            "The two parameter distributions compare hierarchical population posterior-predictive draws with per-pair MAP fits from the same observed calibration cohort. "
            "This checks fit/population consistency and preserves correlated heterogeneity; it is not independent proof of parameter calibration. "
            "Separate leave-one-recording-out behavior evaluation is available but shows under-dispersed 90% position intervals; collision frequency is not a calibration target."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=8127)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/hierarchical_world_model/evaluation/ma_idm_parameter_distribution_audit.json",
    )
    args = parser.parse_args()
    if args.draws < 1:
        raise ValueError("draws must be positive")
    report = build_report(draws=args.draws, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, args.output)
    print(report)


if __name__ == "__main__":
    main()
