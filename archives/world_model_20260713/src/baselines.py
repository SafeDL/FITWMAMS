"""Rule-based background traffic baselines for world-model comparison."""
from __future__ import annotations

import numpy as np

from .schema import SLOT_NAMES


def constant_velocity_actions(current_states: np.ndarray, horizon_steps: int) -> np.ndarray:
    """Return zero acceleration actions `[K, 6, 2]`."""
    n = int(np.asarray(current_states).shape[0])
    return np.zeros((n, int(horizon_steps), len(SLOT_NAMES), 2), dtype=np.float32)


def constant_acceleration_actions(
    current_states: np.ndarray,
    horizon_steps: int,
    *,
    ax_clip: tuple[float, float] = (-6.0, 4.0),
    ay_clip: tuple[float, float] = (-2.0, 2.0),
) -> np.ndarray:
    """Repeat each slot's current acceleration over the prediction horizon."""
    current = np.asarray(current_states, dtype=np.float32)
    actions = np.zeros((len(current), int(horizon_steps), len(SLOT_NAMES), 2), dtype=np.float32)
    actions[..., 0] = np.clip(current[:, None, 1:, 4], ax_clip[0], ax_clip[1])
    actions[..., 1] = np.clip(current[:, None, 1:, 5], ay_clip[0], ay_clip[1])
    return actions


def idm_like_same_lane_actions(
    current_states: np.ndarray,
    current_valid: np.ndarray,
    horizon_steps: int,
    *,
    desired_time_gap_s: float = 0.9,
    min_gap_m: float = 2.0,
    accel_max_mps2: float = 1.5,
    brake_comfort_mps2: float = 3.0,
) -> np.ndarray:
    """A small same-lane car-following baseline for the six-slot schema.

    It mainly adjusts the same-rear vehicle based on the ego gap and otherwise
    repeats current acceleration. This is a diagnostic baseline, not a full IDM
    simulator.
    """
    actions = constant_acceleration_actions(current_states, horizon_steps)
    current = np.asarray(current_states, dtype=np.float32)
    valid = np.asarray(current_valid, dtype=bool)
    same_rear_idx = SLOT_NAMES.index("same_rear")
    rear_agent_idx = same_rear_idx + 1
    rel_x = current[:, rear_agent_idx, 0]
    rear_v = current[:, rear_agent_idx, 2]
    ego_v = current[:, 0, 2]
    closing = rear_v - ego_v
    desired_gap = min_gap_m + np.maximum(rear_v, 0.0) * desired_time_gap_s
    actual_gap = np.maximum(-rel_x, 0.1)
    brake = -brake_comfort_mps2 * np.maximum(desired_gap / actual_gap - 1.0, 0.0)
    accel = np.where(closing > 0.0, brake, accel_max_mps2 * 0.15)
    mask = valid[:, rear_agent_idx]
    actions[mask, :, same_rear_idx, 0] = np.clip(accel[mask], -8.0, 2.0)[:, None]
    actions[~mask, :, same_rear_idx, :] = 0.0
    return actions.astype(np.float32)

