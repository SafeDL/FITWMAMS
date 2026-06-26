#!/usr/bin/env python3
"""Fit highD following EVT model and estimate independent tail-peak exposure.

This is the single entry-point for following long-tail distribution modeling:
it fits the POT/GPD model, then reads per-recording exposure pre-computed by
extract_highd_events.py and estimates independent tail-event rates.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from process_highD.src.io_utils import resolve_data_path
from process_highD.src.evt_fitting import fit_highd_peak_evt
from process_highD.src.evt_mileage_threshold import (
    km_return_level_threshold,
    is_human_threshold_enabled,
    target_return_period_km,
    write_km_threshold_plots,
)
from tools.evt import (
    gpd_conditional_survival,
    load_evt_model,
)
from tools.highd_exposure import (
    KM_PER_MILE,
    collision_distance_summary,
    extract_independent_peaks,
    peak_rate_summary,
)
from tools.io import write_csv, write_json


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "highd_default.yaml"
logger = logging.getLogger(__name__)
REQUIRED_COLUMNS = {
    "event_id",
    "recording_id",
    "ego_id",
    "target_id",
    "start_frame",
    "end_frame",
    "anchor_frame",
    "y_long",
}


def _fit_evt_model(config_path: Path) -> dict[str, Any]:
    return fit_highd_peak_evt(
        config_path=config_path,
        score_filename="following_event_scores.csv",
        peak_config_key="following_evt_peak",
        declustering_config_path=("following_exposure", "declustering"),
        required_columns=REQUIRED_COLUMNS,
        score_column="y_long",
        peak_value_key="y_long_max",
        scenario_label="following",
        summary_model_type="gpd_pot_longitudinal_independent_peak_risk",
        collision_critical_level_mode="deferred_human_all_vehicle_km_return_level",
        summary_extra={
            "risk_variable": "y_long",
            "risk_scoring_window_frames": 125,
            "risk_scoring_window_source": (
                "following fixed context from context_anchor_frame over "
                "following.min_future_steps"
            ),
        },
    )


def _load_exposure_csv(path: Path) -> list[dict[str, Any]]:
    """Read per-recording exposure pre-computed by extract_highd_events.py."""
    if not path.exists():
        raise FileNotFoundError(
            f"Per-recording exposure CSV not found: {path}. "
            "Run process_highD/scripts/extract_highd_events.py first."
        )
    frame = pd.read_csv(path)
    return frame.to_dict("records")


def _load_scored_events(path: Path, model: Any) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"highD following score cache not found: {path}. "
            "Run process_highD/scripts/extract_highd_events.py first."
        )
    scores = pd.read_csv(path)
    required = {
        "event_id",
        "recording_id",
        "ego_id",
        "target_id",
        "start_frame",
        "end_frame",
        "anchor_frame",
        "y_long",
    }
    missing = sorted(required - set(scores.columns))
    if missing:
        raise KeyError(f"{path} is missing required columns: {missing}")
    y_long = pd.to_numeric(scores["y_long"], errors="coerce").to_numpy()
    scores["risk_score"] = np.asarray(model.score(y_long), dtype=np.float64)
    scores["evt_tail_probability"] = np.asarray(
        model.survival(y_long),
        dtype=np.float64,
    )
    return scores


def _paths(cfg: dict[str, Any], config_path: Path) -> dict[str, Path]:
    paths_cfg = cfg["paths"]
    highd_events_dir = resolve_data_path(paths_cfg["output_dir"], config_path)
    exposure_cfg = cfg["following_exposure"]
    peak_cfg = cfg["following_evt_peak"]
    return {
        "exposure_csv": highd_events_dir / "exposure_per_recording.csv",
        "score_csv": highd_events_dir / "following_event_scores.csv",
        "evt_model": resolve_data_path(peak_cfg["model_path"], config_path),
        "output_dir": resolve_data_path(exposure_cfg["output_dir"], config_path),
    }


OBSOLETE_RECORDING_PLOTS = (
    "following_exposure_miles_by_recording.png",
    "all_vehicle_exposure_miles_by_recording.png",
    "independent_tail_peaks_by_recording.png",
    "tail_peak_rate_per_mile_by_recording.png",
    "tail_peak_rate_per_all_vehicle_mile_by_recording.png",
)


def _remove_obsolete_recording_plots(figure_dir: Path) -> None:
    """Remove old per-recording bar charts that are no longer reported."""
    for name in OBSOLETE_RECORDING_PLOTS:
        path = figure_dir / name
        if path.exists():
            path.unlink()


# ── collision level ──


def _collision_level_from_config(
    peak_cfg: dict[str, Any],
    *,
    tail_peak_rate_per_km: float,
    tail_peak_rate_per_hour: float,
    model: Any,
) -> tuple[float, dict[str, Any], dict[str, Any] | None]:
    if not is_human_threshold_enabled(peak_cfg):
        raise ValueError(
            "following_evt_peak.human_safety_threshold.enabled must be true; "
            "the following safety threshold is fixed by the audited 300 km "
            "human all-vehicle return level."
        )
    target_km = target_return_period_km(peak_cfg)
    threshold = km_return_level_threshold(
        model=model,
        tail_peak_rate_per_km=tail_peak_rate_per_km,
        tail_peak_rate_per_hour=tail_peak_rate_per_hour,
        target_return_km=target_km,
    )
    return (
        float(threshold["target_level"]),
        {
            "collision_critical_level_mode": "human_all_vehicle_km_return_level",
            "collision_critical_reference_km": float(
                threshold["target_return_period_km"]
            ),
            "collision_critical_expected_tail_exceedances": threshold[
                "expected_tail_exceedances_at_target_km"
            ],
        },
        threshold,
    )


def _return_period(rate: float) -> float:
    return float(1.0 / rate) if float(rate) > 0.0 else float("inf")


def _log_human_exposure_metrics(
    *,
    model: Any,
    rates: dict[str, float],
    collision_level: float,
    collision_probability_per_peak: float,
    collision_summary: dict[str, float],
) -> None:
    tail_return_miles = _return_period(rates["tail_peak_rate_per_mile"])
    tail_rate_per_km = rates["tail_peak_rate_per_mile"] / KM_PER_MILE
    tail_return_km = _return_period(tail_rate_per_km)
    tail_return_hours = _return_period(rates["tail_peak_rate_per_hour"])
    logger.info(
        (
            "Human highD tail event Y>u: u=%.6g P(Y>u)=%.6g "
            "rate=%.6g/mile %.6g/km %.6g/hour "
            "return=%.6g miles %.6g km %.6g hours"
        ),
        float(model.u),
        float(model.exceedance_rate),
        rates["tail_peak_rate_per_mile"],
        tail_rate_per_km,
        rates["tail_peak_rate_per_hour"],
        tail_return_miles,
        tail_return_km,
        tail_return_hours,
    )
    logger.info(
        (
            "Human highD safety-critical event Y>=%.6g: "
            "P(Y>=level | Y>u)=%.6g P(Y>=level)=%.6g "
            "rate=%.6g/mile %.6g/km %.6g/hour "
            "return=%.6g miles %.6g km %.6g hours"
        ),
        float(collision_level),
        collision_summary["tail_conditional_probability_above_collision_level"],
        float(collision_probability_per_peak),
        collision_summary["highd_safety_critical_intensity_per_mile"],
        collision_summary["highd_safety_critical_intensity_per_km"],
        collision_summary["highd_safety_critical_intensity_per_hour"],
        collision_summary["highd_safety_critical_return_period_miles"],
        collision_summary["highd_safety_critical_return_period_km"],
        collision_summary["highd_safety_critical_return_period_hours"],
    )


# ── main ──


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    cfg = _fit_evt_model(DEFAULT_CONFIG_PATH)
    paths = _paths(cfg, DEFAULT_CONFIG_PATH)
    exposure_cfg = cfg["following_exposure"]
    peak_cfg = cfg["following_evt_peak"]
    decluster_cfg = exposure_cfg["declustering"]

    model = load_evt_model(paths["evt_model"])
    scores = _load_scored_events(paths["score_csv"], model)
    exposure_rows = _load_exposure_csv(paths["exposure_csv"])

    following_ego_miles = sum(row["following_ego_miles"] for row in exposure_rows)
    following_ego_hours = sum(row["following_ego_hours"] for row in exposure_rows)
    following_ego_km = following_ego_miles * KM_PER_MILE
    all_vehicle_miles = sum(row["all_vehicle_miles"] for row in exposure_rows)
    all_vehicle_hours = sum(row["all_vehicle_hours"] for row in exposure_rows)
    all_vehicle_km = all_vehicle_miles * KM_PER_MILE
    total_km = all_vehicle_km
    total_hours = all_vehicle_hours

    target_fps = float(cfg["sampling"]["target_fps"])
    group_keys = tuple(str(item) for item in decluster_cfg["group_keys"])
    run_length_seconds = float(decluster_cfg["run_length_seconds"])
    all_peaks = extract_independent_peaks(
        scores,
        run_length_seconds=run_length_seconds,
        fps=target_fps,
        group_keys=group_keys,
    )
    peaks = [
        peak
        for peak in all_peaks
        if float(peak["y_long_max"]) > float(model.u)
    ]

    rates = peak_rate_summary(
        total_exposure_miles=all_vehicle_miles,
        total_exposure_hours=total_hours,
        num_independent_tail_peaks=len(peaks),
    )
    tail_peak_rate_per_km = rates["tail_peak_rate_per_mile"] / KM_PER_MILE
    y_all_peaks = np.asarray(
        [row["y_long_max"] for row in all_peaks], dtype=np.float64
    )

    (
        collision_level,
        collision_level_meta,
        human_threshold,
    ) = _collision_level_from_config(
        peak_cfg,
        tail_peak_rate_per_km=tail_peak_rate_per_km,
        tail_peak_rate_per_hour=rates["tail_peak_rate_per_hour"],
        model=model,
    )
    tail_conditional_probability_at_collision = gpd_conditional_survival(
        collision_level,
        u=float(model.u),
        xi=float(model.xi),
        beta=float(model.beta),
    )
    collision_probability_per_peak = float(model.survival(collision_level))
    collision_summary = collision_distance_summary(
        tail_peak_rate_per_mile=rates["tail_peak_rate_per_mile"],
        tail_peak_rate_per_hour=rates["tail_peak_rate_per_hour"],
        tail_conditional_probability_above_collision_level=(
            tail_conditional_probability_at_collision
        ),
    )
    all_vehicle_collision_summary = collision_summary

    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    _remove_obsolete_recording_plots(figure_dir)

    figures = write_km_threshold_plots(
        figure_dir,
        values=y_all_peaks,
        model=model,
        total_exposure_km=total_km,
        tail_peak_rate_per_km=tail_peak_rate_per_km,
        target_return_km=(
            float(human_threshold["target_return_period_km"])
            if human_threshold is not None
            else max(
                collision_summary["highd_safety_critical_return_period_km"],
                1.0,
            )
        ),
        target_level=collision_level,
        risk_variable=r"$Y_{\mathrm{long}}$",
        bootstrap_samples=int(peak_cfg["bootstrap_samples"]),
        distance_min_km=float(peak_cfg["distance_plot_min_km"]),
        distance_max_km=float(peak_cfg["distance_plot_max_km"]),
        random_seed=int(peak_cfg["random_seed"]),
        filename_prefix="peak_evt",
    )

    summary = {
        "evt_model_path": str(paths["evt_model"]),
        "evt_tail_threshold_u": float(model.u),
        "collision_critical_level": collision_level,
        "collision_critical_level_mode": collision_level_meta[
            "collision_critical_level_mode"
        ],
        "collision_critical_reference_km": collision_level_meta[
            "collision_critical_reference_km"
        ],
        "collision_critical_expected_tail_exceedances": collision_level_meta[
            "collision_critical_expected_tail_exceedances"
        ],
        "human_calibrated_safety_threshold": human_threshold,
        "total_exposure_km": all_vehicle_km,
        "total_exposure_hours": all_vehicle_hours,
        "following_ego_km": following_ego_km,
        "following_ego_miles": following_ego_miles,
        "following_ego_hours": following_ego_hours,
        "all_vehicle_km": all_vehicle_km,
        "all_vehicle_miles": all_vehicle_miles,
        "all_vehicle_hours": all_vehicle_hours,
        "ego_mile_fraction_of_all_vehicle": (
            float(following_ego_miles / all_vehicle_miles)
            if all_vehicle_miles > 0.0
            else 0.0
        ),
        "num_tail_events_before_declustering": int(
            np.sum(
                pd.to_numeric(scores["y_long"], errors="coerce") > float(model.u)
            )
        ),
        "num_independent_peaks_before_tail_filter": int(len(all_peaks)),
        "num_independent_tail_peaks": int(len(peaks)),
        "tail_peak_rate_per_mile": rates["tail_peak_rate_per_mile"],
        "tail_peak_rate_per_hour": rates["tail_peak_rate_per_hour"],
        "tail_peak_rate_per_km": rates["tail_peak_rate_per_mile"] / KM_PER_MILE,
        "tail_peak_rate_per_all_vehicle_km": rates["tail_peak_rate_per_mile"] / KM_PER_MILE,
        "tail_peak_rate_per_all_vehicle_mile": rates["tail_peak_rate_per_mile"],
        "tail_peak_rate_per_all_vehicle_hour": rates["tail_peak_rate_per_hour"],
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
        "safety_critical_level_tail_conditional_probability": collision_summary[
            "tail_conditional_probability_above_safety_critical_level"
        ],
        "safety_critical_level_probability_per_independent_peak": (
            collision_probability_per_peak
        ),
        "highd_safety_critical_intensity_per_mile": collision_summary[
            "highd_safety_critical_intensity_per_mile"
        ],
        "highd_safety_critical_return_period_miles": collision_summary[
            "highd_safety_critical_return_period_miles"
        ],
        "highd_safety_critical_intensity_per_km": collision_summary[
            "highd_safety_critical_intensity_per_km"
        ],
        "highd_safety_critical_return_period_km": collision_summary[
            "highd_safety_critical_return_period_km"
        ],
        "highd_safety_critical_intensity_per_hour": collision_summary[
            "highd_safety_critical_intensity_per_hour"
        ],
        "highd_safety_critical_return_period_hours": collision_summary[
            "highd_safety_critical_return_period_hours"
        ],
        "safety_critical_intensity_per_all_vehicle_mile": (
            all_vehicle_collision_summary[
                "highd_safety_critical_intensity_per_mile"
            ]
        ),
        "safety_critical_return_period_all_vehicle_miles": (
            all_vehicle_collision_summary[
                "highd_safety_critical_return_period_miles"
            ]
        ),
        "safety_critical_intensity_per_all_vehicle_hour": (
            all_vehicle_collision_summary[
                "highd_safety_critical_intensity_per_hour"
            ]
        ),
        "safety_critical_return_period_all_vehicle_hours": (
            all_vehicle_collision_summary[
                "highd_safety_critical_return_period_hours"
            ]
        ),
        "declustering_run_length_seconds": run_length_seconds,
        "declustering_group_keys": list(group_keys),
        "declustering_representative": str(
            decluster_cfg["representative"]
        ),
        "exposure_denominator": "all_vehicle_km",
        "figures": figures,
    }

    write_csv(output_dir / "highd_independent_tail_peaks.csv", peaks)
    write_json(output_dir / "highd_exposure_summary.json", summary)
    _log_human_exposure_metrics(
        model=model,
        rates=rates,
        collision_level=collision_level,
        collision_probability_per_peak=collision_probability_per_peak,
        collision_summary=collision_summary,
    )
    logger.info(
        "Wrote highD exposure summary to %s | all_vehicle_km=%.6f peaks=%d rate/km=%.6g",
        output_dir,
        all_vehicle_km,
        len(peaks),
        rates["tail_peak_rate_per_mile"] / KM_PER_MILE,
    )


if __name__ == "__main__":
    main()
