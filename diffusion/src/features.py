"""Anchor-frame scenario-condition features for action diffusion."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


FOLLOWING_SCENARIO_CONDITION_KEYS: Tuple[str, ...] = (
    "ego_vx_0",
    "initial_gap",
    "initial_delta_v",
    "lead_ax_0",
    "lead_speed_change",
    "lead_min_ax",
    "lead_braking_duration",
)

CUTIN_SCENARIO_CONDITION_KEYS: Tuple[str, ...] = (
    "ego_vx_0",
    "initial_gap",
    "initial_lateral_offset",
    "initial_delta_vx",
    "target_ax_0",
    "target_vy_0",
    "target_ay_0",
    "final_lateral_offset",
    "time_to_cross",
    "target_speed_change",
)

def _initial_pair(initial_states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(initial_states, dtype=np.float32)
    if states.shape != (2, 6):
        raise ValueError(
            "initial_states must have shape [2, 6] for ego and target, "
            f"got {tuple(states.shape)}"
        )
    return states[0], states[1]


def _future_window(future_states: np.ndarray) -> np.ndarray:
    future = np.asarray(future_states, dtype=np.float32)
    if future.ndim != 3 or future.shape[1:] != (2, 6):
        raise ValueError(
            "future_states must have shape [horizon, 2, 6], "
            f"got {tuple(future.shape)}"
        )
    if future.shape[0] <= 0:
        raise ValueError("future_states must contain at least one future step")
    return future


def _gap(ego: np.ndarray, adv: np.ndarray, ego_length: float, adv_length: float) -> float:
    return float(adv[0] - ego[0] - 0.5 * (float(ego_length) + float(adv_length)))


def extract_following_scenario_condition(
    initial_states: np.ndarray,
    future_states: np.ndarray,
    ego_length: float,
    lead_length: float,
    *,
    dt: float,
) -> Dict[str, float]:
    """Extract compressed car-following scenario conditions."""
    ego, lead = _initial_pair(initial_states)
    future = _future_window(future_states)
    gap = _gap(ego, lead, ego_length, lead_length)
    delta_v = float(ego[2] - lead[2])
    lead_ax = np.concatenate(
        [
            np.asarray([lead[4]], dtype=np.float32),
            future[:, 1, 4].astype(np.float32),
        ],
        axis=0,
    )
    return {
        "ego_vx_0": float(ego[2]),
        "initial_gap": gap,
        "initial_delta_v": delta_v,
        "lead_ax_0": float(lead[4]),
        "lead_speed_change": float(future[-1, 1, 2] - lead[2]),
        "lead_min_ax": float(np.min(lead_ax)),
        "lead_braking_duration": float(np.sum(lead_ax < 0.0) * float(dt)),
    }


def extract_cutin_scenario_condition(
    initial_states: np.ndarray,
    future_states: np.ndarray,
    ego_length: float,
    target_length: float,
    *,
    dt: float,
    metadata: dict[str, Any] | None = None,
) -> Dict[str, float]:
    """Extract compressed cut-in scenario conditions (10-dim)."""
    ego, target = _initial_pair(initial_states)
    future = _future_window(future_states)
    gap = _gap(ego, target, ego_length, target_length)

    # Time-to-cross: seconds from anchor to lane-crossing.
    time_to_cross: float = 0.0
    if metadata is not None:
        cross_frame = metadata.get("cross_frame")
        anchor_frame = metadata.get("anchor_frame")
        if cross_frame is not None and anchor_frame is not None:
            time_to_cross = float(int(cross_frame) - int(anchor_frame)) * float(dt)

    return {
        "ego_vx_0": float(ego[2]),
        "initial_gap": gap,
        "initial_lateral_offset": float(target[1] - ego[1]),
        "initial_delta_vx": float(ego[2] - target[2]),
        "target_ax_0": float(target[4]),
        "target_vy_0": float(target[3]),
        "target_ay_0": float(target[5]),
        "final_lateral_offset": float(future[-1, 1, 1] - future[-1, 0, 1]),
        "time_to_cross": time_to_cross,
        "target_speed_change": float(future[-1, 1, 2] - target[2]),
    }


def extract_scenario_condition(
    initial_states: np.ndarray,
    future_states: np.ndarray,
    ego_length: float,
    adv_length: float,
    *,
    event_type: str = "following",
    dt: float,
    metadata: dict[str, Any] | None = None,
) -> tuple[np.ndarray, List[str]]:
    if str(event_type) == "cut_in":
        feats = extract_cutin_scenario_condition(
            initial_states,
            future_states,
            ego_length,
            adv_length,
            dt=dt,
            metadata=metadata,
        )
        keys = list(CUTIN_SCENARIO_CONDITION_KEYS)
    else:
        feats = extract_following_scenario_condition(
            initial_states,
            future_states,
            ego_length,
            adv_length,
            dt=dt,
        )
        keys = list(FOLLOWING_SCENARIO_CONDITION_KEYS)
    return np.asarray([feats[k] for k in keys], dtype=np.float32), keys
