"""Compact PPO result payloads for comparable ADS evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.io import resolve_path


ROOT = Path(__file__).resolve().parents[2]


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _estimate_block(summary: dict[str, Any]) -> dict[str, Any]:
    counts = dict(summary.get("simulation_counts", {}) or {})
    block: dict[str, Any] = {
        "probability": summary.get("probability"),
        "standard_error": summary.get("probability_standard_error"),
        "ci95": [
            summary.get("probability_ci95_lower"),
            summary.get("probability_ci95_upper"),
        ],
        "closed_loop_evaluations": counts.get("closed_loop_evaluations"),
    }
    reliability = dict(summary.get("reliability", {}) or {})
    if reliability:
        block["reliability"] = reliability.get("status")
    if "num_levels" in summary:
        block["num_levels"] = summary.get("num_levels")
    if "stop_reason" in summary:
        block["stop_reason"] = summary.get("stop_reason")
    return block


def _global_exposure(summary: dict[str, Any]) -> dict[str, Any] | None:
    comparison = dict(summary.get("global_risk_exposure_comparison", {}) or {})
    if not comparison:
        return None
    all_vehicle = dict(comparison.get("all_vehicle_exposure_mapping", {}) or {})
    return {
        "strict": comparison.get(
            "strict_global_exposure_interpretation",
            summary.get("strict_probability_interpretation"),
        ),
        "ads_to_highd_intensity_ratio": comparison.get(
            "ads_to_highd_intensity_ratio",
            comparison.get("ads_to_highd_intensity_ratio_per_all_vehicle_km"),
        ),
        "intensity_per_all_vehicle_km": comparison.get(
            "ads_safety_critical_intensity_per_all_vehicle_km",
            all_vehicle.get("intensity_per_km"),
        ),
        "return_period_all_vehicle_km": comparison.get(
            "ads_safety_critical_return_period_all_vehicle_km",
            all_vehicle.get("return_period_km"),
        ),
    }


def _monte_carlo_summary_path(config: dict[str, Any], config_dir: Path) -> Path | None:
    output_dir = dict(config.get("monte_carlo", {}) or {}).get("output_dir")
    if not output_dir:
        return None
    return resolve_path(str(output_dir), config_dir) / "latent_monte_carlo_summary.json"


def _idm_subset_summary_path(event_type: str) -> Path:
    dirname = "cutin" if event_type == "cut_in" else "following"
    return ROOT / "IDM_subset" / "results" / dirname / "latent_subset_summary.json"


def compact_ppo_result(
    summary: dict[str, Any],
    *,
    summary_path: Path,
    config: dict[str, Any],
    config_dir: Path,
) -> dict[str, Any]:
    """Build the small public result JSON for a PPO subset run."""

    event_type = str(summary.get("event_type", ""))
    idm_summary = _load_json_if_exists(_idm_subset_summary_path(event_type)) or {}
    mc_path = _monte_carlo_summary_path(config, config_dir)
    mc_summary = _load_json_if_exists(mc_path) if mc_path is not None else None

    payload: dict[str, Any] = {
        "ads": "PPO",
        "policy": summary.get("policy", {}),
        "event_type": event_type,
        "failure_event": summary.get("failure_event"),
        "fairness_checks": {
            "same_evt_threshold_as_idm": (
                summary.get("failure_threshold") == idm_summary.get("failure_threshold")
            ),
            "same_subset_num_samples_as_idm": (
                summary.get("num_samples") == idm_summary.get("num_samples")
            ),
            "same_p0_as_idm": summary.get("p0") == idm_summary.get("p0"),
        },
        "subset": _estimate_block(summary),
        "source_files": {
            "subset_summary": _repo_path(summary_path),
        },
    }
    exposure = _global_exposure(summary)
    if exposure is not None:
        payload["global_exposure"] = exposure
    if mc_summary is not None and mc_path is not None:
        payload["monte_carlo"] = _estimate_block(mc_summary)
        payload["source_files"]["monte_carlo_summary"] = _repo_path(mc_path)
    return payload
