"""Leakage-safe history context features for action diffusion."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


FOLLOWING_CONTEXT_KEYS: Tuple[str, ...] = (
    "ego_vx_current",
    "lead_vx_current",
    "relative_speed_current",
    "gap_current",
    "lateral_offset_current",
    "ego_ax_current",
    "lead_ax_current",
    "gap_change_rate",
    "relative_speed_trend",
    "relative_acceleration",
    "lead_brake_indicator",
    "min_gap_in_prefix",
    "max_closing_speed_in_prefix",
)


def extract_following_context(
    history: np.ndarray,
    ego_length: float,
    lead_length: float,
    dt: float,
) -> Dict[str, float]:
    """Extract current/history-only car-following context in ego-current frame."""
    states = np.asarray(history, dtype=np.float32)
    ego = states[:, 0]
    lead = states[:, 1]
    gaps = lead[:, 0] - ego[:, 0] - 0.5 * (ego_length + lead_length)
    lateral = lead[:, 1] - ego[:, 1]
    rel_speed = ego[:, 2] - lead[:, 2]
    elapsed = max((len(states) - 1) * float(dt), 1e-6)
    return {
        "ego_vx_current": float(ego[-1, 2]),
        "lead_vx_current": float(lead[-1, 2]),
        "relative_speed_current": float(rel_speed[-1]),
        "gap_current": float(gaps[-1]),
        "lateral_offset_current": float(lateral[-1]),
        "ego_ax_current": float(ego[-1, 4]),
        "lead_ax_current": float(lead[-1, 4]),
        "gap_change_rate": float((gaps[-1] - gaps[0]) / elapsed),
        "relative_speed_trend": float((rel_speed[-1] - rel_speed[0]) / elapsed),
        "relative_acceleration": float(ego[-1, 4] - lead[-1, 4]),
        "lead_brake_indicator": float(np.min(lead[:, 4]) < -0.5),
        "min_gap_in_prefix": float(np.min(gaps)),
        "max_closing_speed_in_prefix": float(np.maximum(rel_speed, 0.0).max()),
    }


def extract_context(
    history: np.ndarray,
    ego_length: float,
    adv_length: float,
    dt: float,
) -> tuple[np.ndarray, List[str]]:
    feats = extract_following_context(history, ego_length, adv_length, dt)
    keys = list(FOLLOWING_CONTEXT_KEYS)
    return np.asarray([feats[k] for k in keys], dtype=np.float32), keys
