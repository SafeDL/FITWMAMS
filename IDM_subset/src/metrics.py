"""Shared IDM rollout collision and clearance measurements."""

from __future__ import annotations

import numpy as np
from world_model.src.core.evaluation_scope import scoped_agent_valid


def collision_and_min_gap(
    states: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure ego/background overlap and minimum same-lane front clearance."""
    values = np.asarray(states, np.float32)
    present = np.asarray(scoped_agent_valid(valid), bool)
    if values.ndim != 4 or values.shape[2:] != (7, 6):
        raise ValueError("states must have shape [batch,frames,7,6]")
    if present.shape != (values.shape[0], 7):
        raise ValueError("valid must have shape [batch,7]")
    ego = values[:, :, :1]
    background = values[:, :, 1:]
    active = present[:, None, 1:]
    longitudinal = background[..., 0] - ego[..., 0]
    lateral = np.abs(background[..., 1] - ego[..., 1])
    collision = (
        active & (np.abs(longitudinal) < 4.8) & (lateral < 1.8)
    ).any(axis=(1, 2))
    front_gap = np.where(
        active & (longitudinal > 0.0) & (lateral < 1.8),
        np.maximum(longitudinal - 4.8, 0.0),
        np.inf,
    )
    minimum = front_gap.min(axis=(1, 2))
    minimum = np.where(np.isfinite(minimum), minimum, 1_000.0)
    return collision.astype(bool), minimum.astype(np.float32)
