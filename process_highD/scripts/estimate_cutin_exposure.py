#!/usr/bin/env python3
"""Fit highD cut-in EVT peaks and estimate natural driving exposure rates."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from process_highD.src.io_utils import load_config, resolve_data_path
from process_highD.src.evt_fitting import fit_highd_peak_evt
from process_highD.src.evt_diagnostics import write_evt_diagnostic_plots
from process_highD.src.evt_mileage_threshold import (
    km_return_level_threshold,
    is_human_threshold_enabled,
    target_return_period_km,
    write_km_threshold_plots,
)
from tools.evt import gpd_conditional_survival, load_evt_model
from tools.highd_exposure import (
    KM_PER_MILE,
    extract_independent_peaks,
    peak_rate_summary,
)
from tools.io import write_csv, write_json


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "highd_default.yaml"
logger = logging.getLogger(__name__)
REQUIRED_SCORE_COLUMNS = {
    "event_id",
    "recording_id",
    "ego_id",
    "target_id",
    "start_frame",
    "end_frame",
    "anchor_frame",
    "is_cutin",
    "y_cutin",
}


def _semantic_cutin_scores(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["is_cutin"].astype(float) >= 0.5]


def _fit_evt_model(config_path: Path) -> dict[str, Any]:
    return fit_highd_peak_evt(
        config_path=config_path,
        score_filename="cutin_event_scores.csv",
        peak_config_key="cutin_evt_peak",
        declustering_config_path=("cutin_evt_peak", "declustering"),
        required_columns=REQUIRED_SCORE_COLUMNS,
        score_column="y_cutin",
        peak_value_key="y_cutin_max",
        scenario_label="cut-in",
        model_type="gpd_pot_cutin_risk",
        summary_model_type="gpd_pot_cutin_independent_peak_risk",
        collision_critical_level_mode="deferred_human_all_vehicle_km_return_level",
        summary_extra={
            "risk_variable": "y_cutin",
            "risk_scoring_window_frames": 100,
            "risk_scoring_window_source": (
                "cut-in fixed context from anchor_frame over "
                "cutin.context_horizon_steps; longitudinal risk starts at cross_frame"
            ),
        },
        score_filter=_semantic_cutin_scores,
        plot_kwargs={
            "risk_variable": "Y_cutin",
            "histogram_filename": "peak_evt_y_cutin_histogram.png",
            "histogram_key": "peak_y_cutin_histogram",
        },
    )


def _paths(cfg: dict[str, Any], config_path: Path) -> dict[str, Path]:
    events_dir = resolve_data_path(cfg["paths"]["output_dir"], config_path)
    peak_cfg = cfg["cutin_evt_peak"]
    independent_peaks = resolve_data_path(
        peak_cfg["independent_peaks_path"],
        config_path,
    )
    return {
        "exposure_csv": events_dir / "exposure_per_recording.csv",
        "score_csv": events_dir / "cutin_event_scores.csv",
        "evt_model": resolve_data_path(peak_cfg["model_path"], config_path),
        "evt_summary": resolve_data_path(peak_cfg["summary_path"], config_path),
        "independent_peaks": independent_peaks,
        "summary": independent_peaks.parent / "highd_cutin_exposure_summary.json",
    }


def _load_exposure(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Per-recording exposure CSV not found: {path}. "
            "Run process_highD/scripts/extract_highd_events.py first."
        )
    frame = pd.read_csv(path)
    required = {"all_vehicle_miles", "all_vehicle_hours"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{path} is missing required columns: {missing}")
    return frame


def _load_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"highD cut-in score cache not found: {path}. "
            "Run process_highD/scripts/extract_highd_events.py first."
        )
    frame = pd.read_csv(path)
    required = {
        *REQUIRED_SCORE_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{path} is missing required columns: {missing}")
    before = len(frame)
    frame = frame[frame["is_cutin"].astype(float) >= 0.5].copy()
    if frame.empty:
        raise RuntimeError(f"No semantic cut-in score rows remain in {path}")
    removed = before - len(frame)
    if removed:
        logger.info(
            "Filtered %d non semantic cut-in score rows before exposure estimation",
            removed,
        )
    return frame


def _return_period(rate: float) -> float:
    return float(1.0 / rate) if float(rate) > 0.0 else float("inf")


def _collision_level_from_config(
    peak_cfg: dict[str, Any],
    *,
    model: Any,
    tail_peak_rate_per_km: float,
    tail_peak_rate_per_hour: float,
) -> tuple[float, str, dict[str, Any]]:
    if not is_human_threshold_enabled(peak_cfg):
        raise ValueError(
            "cutin_evt_peak.human_safety_threshold.enabled must be true; "
            "cut-in safety-critical threshold is human-calibrated from highD "
            "all-vehicle-km exposure"
        )
    threshold = km_return_level_threshold(
        model=model,
        tail_peak_rate_per_km=tail_peak_rate_per_km,
        tail_peak_rate_per_hour=tail_peak_rate_per_hour,
        target_return_km=target_return_period_km(peak_cfg),
    )
    return (
        float(threshold["target_level"]),
        "human_all_vehicle_km_return_level",
        threshold,
    )


def _critical_rate_summary(
    *,
    all_peak_rate_per_mile: float,
    all_peak_rate_per_hour: float,
    tail_peak_rate_per_mile: float,
    tail_peak_rate_per_hour: float,
    critical_probability_per_peak: float,
    tail_conditional_probability: float,
    use_evt_tail: bool,
) -> dict[str, float | str]:
    if use_evt_tail:
        intensity_mile = float(tail_peak_rate_per_mile * tail_conditional_probability)
        intensity_hour = float(tail_peak_rate_per_hour * tail_conditional_probability)
        rate_source = "evt_tail_extrapolation"
    else:
        intensity_mile = float(all_peak_rate_per_mile * critical_probability_per_peak)
        intensity_hour = float(all_peak_rate_per_hour * critical_probability_per_peak)
        rate_source = "empirical_independent_peaks"
    return {
        "critical_level_rate_source": rate_source,
        "tail_conditional_probability_above_critical_level": float(
            tail_conditional_probability
        ),
        "critical_level_probability_per_independent_peak": float(
            critical_probability_per_peak
        ),
        "highd_safety_critical_intensity_per_mile": intensity_mile,
        "highd_safety_critical_return_period_miles": _return_period(
            intensity_mile
        ),
        "highd_safety_critical_intensity_per_km": float(
            intensity_mile / KM_PER_MILE
        ),
        "highd_safety_critical_return_period_km": (
            float(KM_PER_MILE / intensity_mile)
            if intensity_mile > 0.0
            else float("inf")
        ),
        "highd_safety_critical_intensity_per_hour": intensity_hour,
        "highd_safety_critical_return_period_hours": _return_period(
            intensity_hour
        ),
    }


def _log_cutin_metrics(
    *,
    model: Any,
    rates: dict[str, float],
    collision_level: float,
    collision_probability_per_peak: float,
    critical_summary: dict[str, float | str],
) -> None:
    tail_rate_per_km = rates["tail_peak_rate_per_mile"] / KM_PER_MILE
    logger.info(
        (
            "Human highD cut-in tail event Y>u: u=%.6g P(Y>u)=%.6g "
            "rate=%.6g/all-vehicle-mile %.6g/all-vehicle-km %.6g/hour "
            "return=%.6g miles %.6g km %.6g hours"
        ),
        float(model.u),
        float(model.exceedance_rate),
        rates["tail_peak_rate_per_mile"],
        tail_rate_per_km,
        rates["tail_peak_rate_per_hour"],
        _return_period(rates["tail_peak_rate_per_mile"]),
        _return_period(tail_rate_per_km),
        _return_period(rates["tail_peak_rate_per_hour"]),
    )
    logger.info(
        (
            "Human highD cut-in safety-critical event Y>=%.6g: "
            "P(Y>=level | Y>u)=%.6g P(Y>=level)=%.6g "
            "rate=%.6g/all-vehicle-mile %.6g/all-vehicle-km %.6g/hour "
            "return=%.6g miles %.6g km %.6g hours"
        ),
        float(collision_level),
        critical_summary["tail_conditional_probability_above_critical_level"],
        float(collision_probability_per_peak),
        critical_summary["highd_safety_critical_intensity_per_mile"],
        critical_summary["highd_safety_critical_intensity_per_km"],
        critical_summary["highd_safety_critical_intensity_per_hour"],
        critical_summary["highd_safety_critical_return_period_miles"],
        critical_summary["highd_safety_critical_return_period_km"],
        critical_summary["highd_safety_critical_return_period_hours"],
    )
    logger.info(
        "Human highD cut-in safety-critical rate source: %s",
        critical_summary["critical_level_rate_source"],
    )


def _refresh_evt_diagnostics_with_exposure_threshold(
    *,
    summary_path: Path,
    figure_dir: Path,
    model: Any,
    values: np.ndarray,
    collision_level: float,
    collision_level_mode: str,
    human_threshold: dict[str, Any],
) -> dict[str, str]:
    figures = write_evt_diagnostic_plots(
        figure_dir,
        model=model,
        values=values,
        collision_critical_level=float(collision_level),
        risk_variable="Y_cutin",
        histogram_filename="peak_evt_y_cutin_histogram.png",
        histogram_key="peak_y_cutin_histogram",
    )
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = {}
    summary.update(
        {
            "collision_critical_level": float(collision_level),
            "collision_critical_level_mode": str(collision_level_mode),
            "human_calibrated_safety_threshold": human_threshold,
            "figures": figures,
        }
    )
    write_json(summary_path, summary)
    return figures


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    cfg = _fit_evt_model(DEFAULT_CONFIG_PATH)
    paths = _paths(cfg, DEFAULT_CONFIG_PATH)
    peak_cfg = cfg["cutin_evt_peak"]
    decluster_cfg = peak_cfg["declustering"]

    model = load_evt_model(paths["evt_model"])
    scores = _load_scores(paths["score_csv"])
    exposure = _load_exposure(paths["exposure_csv"])

    all_vehicle_miles = float(exposure["all_vehicle_miles"].sum())
    all_vehicle_km = all_vehicle_miles * KM_PER_MILE
    all_vehicle_hours = float(exposure["all_vehicle_hours"].sum())
    target_fps = float(cfg["sampling"]["target_fps"])
    group_keys = tuple(str(item) for item in decluster_cfg["group_keys"])
    run_length_seconds = float(decluster_cfg["run_length_seconds"])
    all_peaks = extract_independent_peaks(
        scores,
        run_length_seconds=run_length_seconds,
        fps=target_fps,
        group_keys=group_keys,
        score_column="y_cutin",
    )
    peaks = [
        peak
        for peak in all_peaks
        if float(peak["y_cutin_max"]) > float(model.u)
    ]

    rates = peak_rate_summary(
        total_exposure_miles=all_vehicle_miles,
        total_exposure_hours=all_vehicle_hours,
        num_independent_tail_peaks=len(peaks),
    )
    all_peak_rates = peak_rate_summary(
        total_exposure_miles=all_vehicle_miles,
        total_exposure_hours=all_vehicle_hours,
        num_independent_tail_peaks=len(all_peaks),
    )
    collision_level, collision_level_mode, human_threshold = _collision_level_from_config(
        peak_cfg,
        model=model,
        tail_peak_rate_per_km=rates["tail_peak_rate_per_mile"] / KM_PER_MILE,
        tail_peak_rate_per_hour=rates["tail_peak_rate_per_hour"],
    )
    use_evt_tail = bool(collision_level > float(model.u))
    tail_conditional_probability = (
        gpd_conditional_survival(
            collision_level,
            u=float(model.u),
            xi=float(model.xi),
            beta=float(model.beta),
        )
        if use_evt_tail
        else 1.0
    )
    collision_probability_per_peak = float(model.survival(collision_level))
    critical_summary = _critical_rate_summary(
        all_peak_rate_per_mile=all_peak_rates["tail_peak_rate_per_mile"],
        all_peak_rate_per_hour=all_peak_rates["tail_peak_rate_per_hour"],
        tail_peak_rate_per_mile=rates["tail_peak_rate_per_mile"],
        tail_peak_rate_per_hour=rates["tail_peak_rate_per_hour"],
        critical_probability_per_peak=collision_probability_per_peak,
        tail_conditional_probability=tail_conditional_probability,
        use_evt_tail=use_evt_tail,
    )
    y_all_peaks = np.asarray(
        [row["y_cutin_max"] for row in all_peaks], dtype=np.float64
    )
    figures = write_km_threshold_plots(
        paths["independent_peaks"].parent / "figures",
        values=y_all_peaks,
        model=model,
        total_exposure_km=all_vehicle_km,
        tail_peak_rate_per_km=rates["tail_peak_rate_per_mile"] / KM_PER_MILE,
        target_return_km=float(human_threshold["target_return_period_km"]),
        target_level=collision_level,
        risk_variable=r"$Y_{\mathrm{cutin}}$",
        bootstrap_samples=int(peak_cfg["bootstrap_samples"]),
        distance_min_km=float(peak_cfg.get("distance_plot_min_km", 10.0)),
        distance_max_km=float(peak_cfg.get("distance_plot_max_km", 1.0e6)),
        random_seed=int(peak_cfg.get("random_seed", 42)),
        filename_prefix="cutin_peak_evt",
    )
    evt_figures = _refresh_evt_diagnostics_with_exposure_threshold(
        summary_path=paths["evt_summary"],
        figure_dir=paths["evt_model"].parent / "figures",
        model=model,
        values=y_all_peaks,
        collision_level=collision_level,
        collision_level_mode=collision_level_mode,
        human_threshold=human_threshold,
    )

    paths["independent_peaks"].parent.mkdir(parents=True, exist_ok=True)
    write_csv(paths["independent_peaks"], peaks)
    write_json(
        paths["summary"],
        {
            "evt_model_path": str(paths["evt_model"]),
            "evt_tail_threshold_u": float(model.u),
            "collision_critical_level": collision_level,
            "collision_critical_level_mode": collision_level_mode,
            "human_calibrated_safety_threshold": human_threshold,
            "exposure_denominator": "all_vehicle_km",
            "total_exposure_km": all_vehicle_km,
            "total_exposure_hours": all_vehicle_hours,
            "all_vehicle_km": all_vehicle_km,
            "all_vehicle_miles": all_vehicle_miles,
            "all_vehicle_hours": all_vehicle_hours,
            "num_tail_events_before_declustering": int(
                np.sum(
                    pd.to_numeric(scores["y_cutin"], errors="coerce")
                    > float(model.u)
                )
            ),
            "num_independent_peaks_before_tail_filter": int(len(all_peaks)),
            "num_independent_tail_peaks": int(len(peaks)),
            "independent_peak_rate_per_mile": all_peak_rates[
                "tail_peak_rate_per_mile"
            ],
            "independent_peak_rate_per_hour": all_peak_rates[
                "tail_peak_rate_per_hour"
            ],
            "tail_peak_rate_per_mile": rates["tail_peak_rate_per_mile"],
            "tail_peak_rate_per_km": rates["tail_peak_rate_per_mile"] / KM_PER_MILE,
            "tail_peak_rate_per_all_vehicle_km": rates["tail_peak_rate_per_mile"] / KM_PER_MILE,
            "tail_peak_rate_per_all_vehicle_mile": rates["tail_peak_rate_per_mile"],
            "tail_peak_rate_per_all_vehicle_hour": rates["tail_peak_rate_per_hour"],
            "tail_peak_rate_per_hour": rates["tail_peak_rate_per_hour"],
            "tail_threshold_probability_per_independent_peak": float(
                model.exceedance_rate
            ),
            "tail_threshold_return_period_miles": _return_period(
                rates["tail_peak_rate_per_mile"]
            ),
            "tail_threshold_return_period_km": _return_period(
                rates["tail_peak_rate_per_mile"] / KM_PER_MILE
            ),
            "tail_threshold_return_period_hours": _return_period(
                rates["tail_peak_rate_per_hour"]
            ),
            "critical_level_rate_source": critical_summary[
                "critical_level_rate_source"
            ],
            "safety_critical_level_tail_conditional_probability": critical_summary[
                "tail_conditional_probability_above_critical_level"
            ],
            "safety_critical_level_probability_per_independent_peak": (
                collision_probability_per_peak
            ),
            "highd_safety_critical_intensity_per_mile": critical_summary[
                "highd_safety_critical_intensity_per_mile"
            ],
            "highd_safety_critical_return_period_miles": critical_summary[
                "highd_safety_critical_return_period_miles"
            ],
            "highd_safety_critical_intensity_per_km": critical_summary[
                "highd_safety_critical_intensity_per_km"
            ],
            "highd_safety_critical_return_period_km": critical_summary[
                "highd_safety_critical_return_period_km"
            ],
            "highd_safety_critical_intensity_per_hour": critical_summary[
                "highd_safety_critical_intensity_per_hour"
            ],
            "highd_safety_critical_return_period_hours": critical_summary[
                "highd_safety_critical_return_period_hours"
            ],
            "declustering_run_length_seconds": run_length_seconds,
            "declustering_group_keys": list(group_keys),
            "declustering_representative": str(decluster_cfg["representative"]),
            "figures": figures,
            "evt_diagnostic_figures": evt_figures,
        },
    )
    _log_cutin_metrics(
        model=model,
        rates=rates,
        collision_level=collision_level,
        collision_probability_per_peak=collision_probability_per_peak,
        critical_summary=critical_summary,
    )
    logger.info(
        (
            "Wrote highD cut-in exposure summary to %s | "
            "all_vehicle_km=%.6f peaks=%d rate/km=%.6g"
        ),
        paths["summary"].parent,
        all_vehicle_km,
        len(peaks),
        rates["tail_peak_rate_per_mile"] / KM_PER_MILE,
    )


if __name__ == "__main__":
    main()
