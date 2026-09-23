"""Common control limits and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ControlLimits:
    acceleration_min_mps2: float = -8.0
    acceleration_max_mps2: float = 4.0
    yaw_rate_abs_max_rps: float = 0.6
    lateral_acceleration_abs_max_mps2: float = 4.0


def apply_control_limits(
    requested: np.ndarray,
    states: np.ndarray,
    valid: np.ndarray,
    limits: ControlLimits | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Apply the one method-independent physical control boundary."""
    cfg = limits or ControlLimits()
    raw = np.asarray(requested, np.float32)
    state = np.asarray(states, np.float32)
    present = np.asarray(valid, bool)
    if raw.shape != (*state.shape[:-1], 2) or present.shape != state.shape[:-1]:
        raise ValueError("requested/states/valid shapes do not align")
    if not np.isfinite(raw).all():
        raise ValueError("requested controls contain NaN or Inf")
    output = raw.copy()
    output[..., 0] = np.clip(
        output[..., 0],
        cfg.acceleration_min_mps2,
        cfg.acceleration_max_mps2,
    )
    speed = np.linalg.norm(state[..., 2:4], axis=-1)
    speed_yaw_limit = cfg.lateral_acceleration_abs_max_mps2 / np.maximum(speed, 1.0e-3)
    yaw_limit = np.minimum(cfg.yaw_rate_abs_max_rps, speed_yaw_limit)
    output[..., 1] = np.clip(output[..., 1], -yaw_limit, yaw_limit)
    output[~present] = 0.0
    correction = output - raw
    rewritten = np.any(np.abs(correction) > 1.0e-6, axis=-1) & present
    return output, {
        "rewritten": rewritten,
        "correction_l2": np.linalg.norm(correction, axis=-1),
        "requested": raw,
        "applied": output,
    }
