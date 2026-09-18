"""Scalable full-cohort empirical-Bayes MA-IDM calibration.

This keeps the paper's hierarchical driver parameters and shared SE-GP
discrepancy while making all locally eligible 25 Hz highD pairs usable on a
CPU.  GP likelihoods use the public notebook's 4 s windows with a 2 s hop.
As in that notebook, adjacent likelihood blocks overlap by 50%; this is a
source-protocol property, not an assertion that the blocks are independent.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from .data import load_ragged_pairs
from .model import LOWER, UPPER, PARAMETER_NAMES, _nearest_pd
from .reference_kernels import idm, se_kernel

# The plant remains native highD 25 Hz, but the stochastic driver is sampled
# and held at 5 Hz.  Fitting its discrepancy on every 25 Hz finite difference
# while simulating one value per 0.2 s is a different stochastic process.
# Work on the paper's decision grid instead: 20 points per four-second window.
PLANT_DT, DECISION_STRIDE = .04, 5
DT, WINDOW, HOP = PLANT_DT * DECISION_STRIDE, 20, 10
# Paper Eq. (19) contains this independent action/velocity term alongside the
# persistent GP.  It must be estimated with the GP hyperparameters rather than
# silently fixed to the author-ring simulator's illustrative .1 value.
DEFAULT_IID_ACTION_SIGMA = .10


def _residual(pair: dict[str, np.ndarray], theta: np.ndarray) -> np.ndarray:
    indices = np.arange(0, len(pair["follower_v"]) - DECISION_STRIDE, DECISION_STRIDE, dtype=int)
    velocity = pair["follower_v"]
    leader = pair["leader_v"]
    closing = velocity[indices] - leader[indices]
    return ((velocity[indices + DECISION_STRIDE] - velocity[indices]) / DT
            - idm(pair["gap"][indices], velocity[indices], closing, theta))


def _starts(length: int) -> np.ndarray:
    """Paper-notebook window starts: 20 decision points with a 10-point hop."""
    return np.arange(0, length - WINDOW + 1, HOP, dtype=np.int64)


def _blocks(values: np.ndarray) -> np.ndarray:
    starts = _starts(len(values))
    return np.stack([values[start:start + WINDOW] for start in starts])


def _iid_map(pair: dict[str, np.ndarray], initial: np.ndarray, prior_mean: np.ndarray | None = None,
             prior_precision: np.ndarray | None = None) -> np.ndarray:
    def objective(log_theta: np.ndarray) -> float:
        residual = _residual(pair, np.exp(log_theta))
        value = .5 * len(residual) * np.log(np.mean(residual ** 2) + 1.e-8)
        if prior_mean is not None and prior_precision is not None:
            delta = log_theta - prior_mean
            value += .5 * float(delta @ prior_precision @ delta)
        return float(value)
    result = minimize(objective, np.log(initial), method="L-BFGS-B", bounds=list(zip(np.log(LOWER[:5]), np.log(UPPER[:5]))), options={"maxiter": 80, "ftol": 1.e-8})
    if not np.isfinite(result.fun):
        raise RuntimeError("individual IDM MAP failed")
    return np.exp(result.x)


def _gp_factor(sigma: float, ell: float, iid_sigma: float) -> tuple[tuple[np.ndarray, bool], float]:
    times = np.arange(WINDOW, dtype=float) * DT
    covariance = se_kernel(times, times, sigma, ell)
    covariance.flat[:: WINDOW + 1] += float(iid_sigma) ** 2
    factor = cho_factor(covariance, lower=True, check_finite=False)
    return factor, 2. * float(np.sum(np.log(np.diag(factor[0]))))


def _gp_nll(blocks: np.ndarray, sigma: float, ell: float, iid_sigma: float) -> float:
    factor, logdet = _gp_factor(sigma, ell, iid_sigma)
    solved = cho_solve(factor, blocks.T, check_finite=False).T
    return float(.5 * (np.sum(blocks * solved) + len(blocks) * (logdet + WINDOW * np.log(2 * np.pi))))


def _fit_global_gp(pairs: list[dict[str, np.ndarray]], theta: np.ndarray) -> np.ndarray:
    all_blocks = np.concatenate([_blocks(_residual(pair, value)) for pair, value in zip(pairs, theta)])
    result = minimize(lambda x: _gp_nll(all_blocks, *np.exp(x)), np.log([.12, 1.5, DEFAULT_IID_ACTION_SIGMA]), method="L-BFGS-B",
                      bounds=[(np.log(.005), np.log(3.)), (np.log(.08), np.log(8.)), (np.log(.005), np.log(1.))],
                      options={"maxiter": 100, "ftol": 1.e-8})
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(f"global GP fit failed: {result.message}")
    return np.exp(result.x)


def _gp_map(pair: dict[str, np.ndarray], initial: np.ndarray, factor: tuple[np.ndarray, bool], prior_mean: np.ndarray,
            prior_precision: np.ndarray) -> np.ndarray:
    def objective(log_theta: np.ndarray) -> float:
        blocks = _blocks(_residual(pair, np.exp(log_theta)))
        solved = cho_solve(factor, blocks.T, check_finite=False).T
        delta = log_theta - prior_mean
        return .5 * float(np.sum(blocks * solved) + delta @ prior_precision @ delta)
    result = minimize(objective, np.log(initial), method="L-BFGS-B", bounds=list(zip(np.log(LOWER[:5]), np.log(UPPER[:5]))), options={"maxiter": 60, "ftol": 1.e-8})
    if not np.isfinite(result.fun):
        raise RuntimeError("GP individual IDM MAP failed")
    return np.exp(result.x)


def fit_full_population(dataset_path: str | Path, output_path: str | Path, *, model: str = "ma_idm",
                        training_recordings: list[int] | None = None, posterior_draws: int = 256,
                        seed: int = 20260915, train_fraction: float = 1.,
                        training_vehicle_classes: list[str] | None = None) -> Path:
    """Fit all eligible pairs (or an explicit recording-held-out CV training fold)."""
    _, all_pairs = load_ragged_pairs(dataset_path)
    recordings = None if training_recordings is None else {int(value) for value in training_recordings}
    classes = None if training_vehicle_classes is None else {str(value) for value in training_vehicle_classes}
    pairs = [pair for pair in all_pairs if (recordings is None or int(pair["recording_id"]) in recordings)
             and (classes is None or str(pair["vehicle_class"]) in classes)]
    if not .2 <= train_fraction <= 1.:
        raise ValueError("train_fraction must be in [.2, 1]")
    if train_fraction < 1.:
        # A contiguous prefix leaves genuinely future time origins available for
        # train-only dispersion calibration.  Keep all static pair metadata.
        pairs = [{**pair, **{key: value[:max(WINDOW + 1, int(len(value) * train_fraction))]
                            for key, value in pair.items() if isinstance(value, np.ndarray)}} for pair in pairs]
    if len(pairs) < 3:
        raise RuntimeError("full-cohort fit needs at least three drivers")
    initial = np.array([33., 2., 1.6, 1.5, 1.67])
    maps = np.asarray([_iid_map(pair, initial) for pair in pairs])
    # One empirical-Bayes shrinkage step supplies the hierarchical population prior.
    logs = np.log(maps); population_mean = np.mean(logs, axis=0); population_cov = _nearest_pd(np.cov(logs, rowvar=False))
    precision = np.linalg.inv(population_cov + np.eye(5) * .05)
    maps = np.asarray([_iid_map(pair, value, population_mean, precision) for pair, value in zip(pairs, maps)])
    if model == "ma_idm":
        sigma, ell, iid_sigma = _fit_global_gp(pairs, maps)
        factor, _ = _gp_factor(sigma, ell, iid_sigma)
        logs = np.log(maps); population_mean = np.mean(logs, axis=0); population_cov = _nearest_pd(np.cov(logs, rowvar=False))
        precision = np.linalg.inv(population_cov + np.eye(5) * .05)
        maps = np.asarray([_gp_map(pair, value, factor, population_mean, precision) for pair, value in zip(pairs, maps)])
    elif model == "b_idm":
        sigma = float(np.sqrt(np.mean(np.concatenate([_residual(pair, value) for pair, value in zip(pairs, maps)]) ** 2)))
        ell, iid_sigma = 1.5, DEFAULT_IID_ACTION_SIGMA
    else:
        raise ValueError("model must be b_idm or ma_idm")
    logs = np.log(maps); population_mean = np.mean(logs, axis=0); population_cov = _nearest_pd(np.cov(logs, rowvar=False) + np.eye(5) * .0025)
    rng = np.random.default_rng(seed); means, covariances = [], []
    iid_draws = []
    for _ in range(posterior_draws):
        sampled = logs[rng.integers(0, len(logs), len(logs))]
        mean5, covariance5 = np.mean(sampled, axis=0), _nearest_pd(np.cov(sampled, rowvar=False) + np.eye(5) * .0025)
        mean = np.r_[mean5, np.log([sigma, ell])]
        covariance = np.zeros((7, 7)); covariance[:5, :5] = covariance5; covariance[5, 5] = .0025; covariance[6, 6] = .0025
        means.append(mean); covariances.append(covariance)
        # A small log-scale hyperposterior approximation retains uncertainty
        # in the newly fitted shared iid action term without changing the
        # legacy seven-parameter population artifact schema.
        iid_draws.append(float(np.exp(rng.normal(np.log(iid_sigma), .05))))
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, parameter_names=np.asarray(PARAMETER_NAMES), population_log_mean=np.asarray(means), population_log_cov=np.asarray(covariances),
                        driver_map=np.column_stack((maps, np.full(len(maps), sigma), np.full(len(maps), ell))), driver_train_indices=np.arange(len(maps)),
                        recording_id=np.asarray([pair["recording_id"] for pair in pairs]), follower_id=np.asarray([pair["follower_id"] for pair in pairs]), model=np.asarray(model),
                        iid_action_sigma_draws=np.asarray(iid_draws),
                        fit_backend=np.asarray("full_cohort_empirical_bayes_hierarchical_map_5hz_decision"), dt_s=np.asarray(DT), calibration_window_s=np.asarray(4.))
    report = {"model": model, "fit_backend": "full_cohort_empirical_bayes_hierarchical_map_5hz_decision", "native_fps": 25,
              "driver_decision_fps": 5, "decision_dt_s": DT, "pairs": len(pairs),
              "training_recordings": sorted(recordings) if recordings is not None else "all qualifying recordings",
              "training_vehicle_classes": sorted(classes) if classes is not None else "all classes", "all_pair_observations": int(sum(len(pair["gap"]) - 1 for pair in pairs)),
              "training_fraction_per_pair": train_fraction,
              "gp_window_s": 4, "gp_window_points": WINDOW, "gp_hop_points": HOP, "gp_hop_s": HOP * DT,
              "gp_windows_are": "author-notebook 5 Hz decision-grid 4 s windows with 2 s hop; overlapping blocks are retained", "global_gp_sigma_mps2": float(sigma), "global_gp_ell_s": float(ell), "global_iid_action_sigma_mps2": float(iid_sigma),
              "parameter_prior": "empirical-Bayes correlated log-normal; one MAP shrinkage iteration",
              "innovation_distribution": "gaussian",
              "author_reference": "hierarchical MA-IDM notebook: shared l/s2_f and 4 s GP covariance windows"}
    path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
