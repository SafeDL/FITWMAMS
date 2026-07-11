"""Ego-current coordinate frame helpers for trajectory windows."""
from __future__ import annotations

from typing import Dict

import numpy as np


def compute_ego_frame(
    ego_state_current: np.ndarray,
    world_heading_x: float = 1.0,
    world_heading_y: float = 0.0,
) -> Dict[str, float]:
    """Build the ego-current frame from the last ego state."""
    h_norm = float(np.hypot(world_heading_x, world_heading_y))
    if h_norm < 1e-6:
        rot_cos, rot_sin = 1.0, 0.0
    else:
        rot_cos = float(world_heading_x / h_norm)
        rot_sin = float(world_heading_y / h_norm)
    return {
        "origin_x": float(ego_state_current[0]),
        "origin_y": float(ego_state_current[1]),
        "rot_cos": rot_cos,
        "rot_sin": rot_sin,
    }


def world_to_ego_states(states_world: np.ndarray, frame: Dict[str, float]) -> np.ndarray:
    """Transform [T, actors, state_features] states into the ego-current frame."""
    ox = frame["origin_x"]
    oy = frame["origin_y"]
    c = frame["rot_cos"]
    s = frame["rot_sin"]
    out = states_world.copy()

    px = states_world[..., 0] - ox
    py = states_world[..., 1] - oy
    out[..., 0] = c * px + s * py
    out[..., 1] = -s * px + c * py

    vx = states_world[..., 2]
    vy = states_world[..., 3]
    out[..., 2] = c * vx + s * vy
    out[..., 3] = -s * vx + c * vy

    ax = states_world[..., 4]
    ay = states_world[..., 5]
    out[..., 4] = c * ax + s * ay
    out[..., 5] = -s * ax + c * ay
    return out
