"""Shared POT/GPD peak EVT fitting for highD scenarios."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import genpareto

from process_highD.src.evt_diagnostics import write_evt_diagnostic_plots
from process_highD.src.io_utils import load_config, resolve_data_path
from tools.evt import fit_evt_model, fit_gpd_excess
from tools.highd_exposure import extract_independent_peaks
from tools.io import write_json


EVT_FIT_DEFAULTS: dict[str, Any] = {
    "min_exceedances": 20,
    "max_tail_fraction": 0.25,
    "max_threshold_candidates": 400,
    "min_threshold_exceedance_rate": 0.10,
    "random_seed": 42,
}
logger = logging.getLogger(__name__)


def _gpd_gof_statistics(excess: np.ndarray, *, xi: float, beta: float) -> dict[str, float]:
    values = np.sort(np.asarray(excess, dtype=np.float64))
    values = values[np.isfinite(values) & (values > 0.0)]
    n = int(values.size)
    if n == 0:
        return {
            "ks": float("nan"),
            "cramer_von_mises": float("nan"),
            "anderson_darling": float("nan"),
        }
    cdf = genpareto.cdf(
        values,
        c=float(xi),
        loc=0.0,
        scale=max(float(beta), 1.0e-12),
    )
    cdf = np.clip(cdf, 1.0e-12, 1.0 - 1.0e-12)
    i = np.arange(1, n + 1, dtype=np.float64)
    ks = float(
        max(
            np.max(np.abs(cdf - (i - 1.0) / n)),
            np.max(np.abs(i / n - cdf)),
        )
    )
    cvm = float((1.0 / (12.0 * n)) + np.sum((cdf - (2.0 * i - 1.0) / (2.0 * n)) ** 2))
    ad = float(
        -n
        - np.mean(
            (2.0 * i - 1.0)
            * (np.log(cdf) + np.log(1.0 - cdf[::-1]))
        )
    )
    return {
        "ks": ks,
        "cramer_von_mises": cvm,
        "anderson_darling": ad,
    }


def _gpd_tail_gof(
    *,
    model: Any,
    values: np.ndarray,
    bootstrap_samples: int,
    random_seed: int,
) -> dict[str, Any]:
    excess = np.asarray(values, dtype=np.float64)
    excess = excess[np.isfinite(excess) & (excess > float(model.u))] - float(model.u)
    excess = excess[excess > 0.0]
    n = int(excess.size)
    observed = _gpd_gof_statistics(excess, xi=float(model.xi), beta=float(model.beta))
    if n < 5 or int(bootstrap_samples) <= 0:
        return {
            "method": "gpd_parametric_bootstrap",
            "num_exceedances": n,
            "bootstrap_samples_requested": int(bootstrap_samples),
            "bootstrap_samples_used": 0,
            "statistics": observed,
            "p_values": {
                "ks": float("nan"),
                "cramer_von_mises": float("nan"),
                "anderson_darling": float("nan"),
            },
        }

    rng = np.random.default_rng(int(random_seed))
    boot_stats: dict[str, list[float]] = {
        "ks": [],
        "cramer_von_mises": [],
        "anderson_darling": [],
    }
    for _ in range(int(bootstrap_samples)):
        sample = genpareto.rvs(
            c=float(model.xi),
            loc=0.0,
            scale=max(float(model.beta), 1.0e-12),
            size=n,
            random_state=rng,
        )
        try:
            xi_hat, beta_hat = fit_gpd_excess(sample)
        except ValueError:
            continue
        stats = _gpd_gof_statistics(sample, xi=xi_hat, beta=beta_hat)
        for key, value in stats.items():
            if np.isfinite(value):
                boot_stats[key].append(float(value))

    p_values = {}
    for key, value in observed.items():
        samples = np.asarray(boot_stats[key], dtype=np.float64)
        samples = samples[np.isfinite(samples)]
        p_values[key] = (
            float((np.sum(samples >= float(value)) + 1.0) / (samples.size + 1.0))
            if samples.size
            else float("nan")
        )
    return {
        "method": "gpd_parametric_bootstrap_refit",
        "num_exceedances": n,
        "bootstrap_samples_requested": int(bootstrap_samples),
        "bootstrap_samples_used": int(
            max(len(values) for values in boot_stats.values())
        ),
        "statistics": observed,
        "p_values": p_values,
        "interpretation": (
            "GOF tests are reported as supporting diagnostics; threshold stability, "
            "QQ/PP plots, survival overlay, and sensitivity plots remain primary."
        ),
    }


def _score_cache_path(cfg: dict[str, Any], config_path: Path, filename: str) -> Path:
    highd_events_dir = resolve_data_path(cfg["paths"]["output_dir"], config_path)
    return highd_events_dir / filename


def _load_scored_events(
    path: Path,
    *,
    required_columns: set[str],
    scenario_label: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"highD {scenario_label} score cache not found: {path}. "
            "Run process_highD/scripts/extract_highd_events.py first."
        )
    frame = pd.read_csv(path)
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise KeyError(f"{path} is missing required columns: {missing}")
    return frame


def fit_highd_peak_evt(
    *,
    config_path: Path,
    score_filename: str,
    peak_config_key: str,
    declustering_config_path: tuple[str, ...],
    required_columns: set[str],
    score_column: str,
    peak_value_key: str,
    scenario_label: str,
    summary_model_type: str,
    collision_critical_level_mode: str,
    model_type: str | None = None,
    summary_extra: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    score_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    peak_cfg = cfg[peak_config_key]
    declustering_config = cfg
    for key in declustering_config_path:
        declustering_config = declustering_config[key]
    score_csv = _score_cache_path(cfg, config_path, score_filename)
    model_path = resolve_data_path(peak_cfg["model_path"], config_path)
    summary_path = resolve_data_path(peak_cfg["summary_path"], config_path)
    figure_dir = model_path.parent / "figures"

    scores = _load_scored_events(
        score_csv,
        required_columns=required_columns,
        scenario_label=scenario_label,
    )
    if score_filter is not None:
        before = len(scores)
        scores = score_filter(scores).copy()
        if scores.empty:
            raise RuntimeError(
                f"No highD {scenario_label} score rows remain after score_filter"
            )
        removed = before - len(scores)
        if removed:
            logger.info(
                "Filtered %d highD %s score rows before EVT fitting",
                removed,
                scenario_label,
            )
    target_fps = float(cfg["sampling"]["target_fps"])
    group_keys = tuple(str(item) for item in declustering_config["group_keys"])
    run_length_seconds = float(declustering_config["run_length_seconds"])
    peaks = extract_independent_peaks(
        scores,
        run_length_seconds=run_length_seconds,
        fps=target_fps,
        group_keys=group_keys,
        score_column=score_column,
    )
    if not peaks:
        raise RuntimeError(
            f"No independent highD {scenario_label} risk peaks were extracted"
        )

    y_peaks = np.asarray([row[peak_value_key] for row in peaks], dtype=np.float64)
    model = fit_evt_model(
        y_peaks,
        min_exceedances=int(EVT_FIT_DEFAULTS["min_exceedances"]),
        max_tail_fraction=EVT_FIT_DEFAULTS["max_tail_fraction"],
        max_threshold_candidates=int(EVT_FIT_DEFAULTS["max_threshold_candidates"]),
        min_threshold_exceedance_rate=float(
            EVT_FIT_DEFAULTS["min_threshold_exceedance_rate"]
        ),
        bootstrap_samples=int(peak_cfg["bootstrap_samples"]),
        random_seed=int(EVT_FIT_DEFAULTS["random_seed"]),
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_type is None:
        model.to_json(model_path)
    else:
        model.to_json(model_path, model_type=model_type)

    collision_critical_level = float(
        peak_cfg.get("collision_critical_level", float("nan"))
    )
    figures = write_evt_diagnostic_plots(
        figure_dir,
        model=model,
        values=y_peaks,
        collision_critical_level=collision_critical_level,
        **(plot_kwargs or {}),
    )
    gof = _gpd_tail_gof(
        model=model,
        values=y_peaks,
        bootstrap_samples=int(peak_cfg.get("gof_bootstrap_samples", peak_cfg["bootstrap_samples"])),
        random_seed=int(peak_cfg.get("random_seed", EVT_FIT_DEFAULTS["random_seed"])),
    )
    tail_peaks = int(np.sum(y_peaks > float(model.u)))
    summary: dict[str, Any] = {
        "model_path": str(model_path),
        "score_csv": str(score_csv),
        "model_type": summary_model_type,
        "num_independent_peaks": int(len(peaks)),
        "num_tail_peaks": tail_peaks,
        "u": float(model.u),
        "xi": float(model.xi),
        "beta": float(model.beta),
        "exceedance_rate": float(model.exceedance_rate),
        "collision_critical_level": collision_critical_level,
        "collision_critical_level_mode": collision_critical_level_mode,
        "return_levels": model.return_levels,
        "return_level_ci": model.return_level_ci,
        "declustering_run_length_seconds": run_length_seconds,
        "declustering_group_keys": list(group_keys),
        "tail_gof": gof,
        "audit_protocol": {
            "risk_variable_source": "declustered independent event-level peak risk",
            "pot_threshold_role": "GPD fitting threshold u; not the safety-critical threshold",
            "safety_threshold_role": (
                "Use exposure summary human_calibrated_safety_threshold for x_star"
            ),
            "diagnostics": [
                "empirical histogram and CCDF/survival overlay",
                "mean residual life",
                "shape and modified-scale threshold stability",
                "GPD QQ/PP plot",
                "parametric-bootstrap KS/CvM/AD goodness-of-fit",
                "all-vehicle-km return-level threshold and threshold sensitivity",
            ],
        },
        "figures": figures,
    }
    if summary_extra:
        summary.update(summary_extra)
    write_json(summary_path, summary)
    logger.info(
        "Saved %s peak EVT model to %s | peaks=%d tail_peaks=%d u=%.6f",
        scenario_label,
        model_path,
        len(peaks),
        tail_peaks,
        model.u,
    )
    return cfg
