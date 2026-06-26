"""Shared context helpers for rollout scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .io import load_npz


# Per-context metadata keys stored in tail context NPZ files. Dataset-level EVT
# constants that are not repeated per row should be loaded from the matching
# tail_context_summary.json instead.
CONTEXT_META_KEYS = (
    "recording_id",
    "event_id",
    "ego_id",
    "target_id",
    "anchor_frame",
    "cross_frame",
    "cutin_start_frame",
    "cutin_end_frame",
    "risk_start_index",
    "risk_start_frame",
    "context_horizon_steps",
    "source_lane",
    "target_lane",
    "source_type",
    "tail_threshold",
    "tail_score_threshold",
    "tail_selection_method",
    "tail_sampling_method",
    "collision_critical_level",
    "peak_id",
    "representative_event_id",
    "base_context_index",
    "base_event_id",
    "synthetic_context",
    "context_model_method",
    "context_feature_distance",
    "event_steps",
    "initial_gap",
    "initial_closing_speed",
    "recorded_min_gap",
    "recorded_min_ttc",
    "collision",
    "near_collision",
    "y_long",
    "y_cutin",
    "risk_score",
    "evt_tail_probability",
    "completion_gap",
    "post_cutin_min_gap",
    "post_cutin_min_ttc",
    "cutin_duration_seconds",
    "cross_lateral_offset",
    "min_abs_lateral_offset",
    "max_abs_lateral_velocity",
    "is_front_cutin",
)


def load_context_npz(path: str | Path) -> dict[str, np.ndarray]:
    return load_npz(path)


def context_from_npz(raw: dict[str, np.ndarray], idx: int) -> dict[str, Any]:
    required = ("scenario_conditions", "initial_states", "ego_length", "adv_length")
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(
            f"Context dataset is missing required arrays: {missing}"
        )
    context: dict[str, Any] = {
        "scenario_conditions": raw["scenario_conditions"][idx],
        "initial_states": raw["initial_states"][idx],
        "ego_length": float(raw["ego_length"][idx]),
        "adv_length": float(raw["adv_length"][idx]),
    }
    for key in CONTEXT_META_KEYS:
        if key in raw:
            arr = raw[key]
            value = arr.item() if getattr(arr, "ndim", 1) == 0 else arr[idx]
            context[key] = value.item() if hasattr(value, "item") else value
    if "risk_score" not in context and "criticality_score" in raw:
        value = raw["criticality_score"][idx]
        context["risk_score"] = value.item() if hasattr(value, "item") else value
    return context
