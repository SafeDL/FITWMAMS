"""Action integration helpers for natural car-following rollouts."""
from __future__ import annotations

import numpy as np

from .types import VehicleState


def integrate_following_actions(
    initial: VehicleState,
    actions: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Integrate lead-car longitudinal acceleration.

    ``actions`` must be acceleration in m/s^2. If a model is trained with
    jerk actions, decode and integrate jerk to acceleration before calling
    this helper.

    Returns states with shape ``[H + 1, 6]`` in the same local frame as
    ``initial``.
    """
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim == 2:
        ax_seq = a[:, 0]
    else:
        ax_seq = a.reshape(-1)
    states = np.zeros((len(ax_seq) + 1, 6), dtype=np.float32)
    states[0] = initial.as_feature()
    x = float(initial.x)
    y = float(initial.y)
    vx = max(float(initial.vx), 0.0)
    vy = float(initial.vy)
    ay = float(initial.ay)
    for i, ax in enumerate(ax_seq):
        ax_f = float(ax)
        x = x + vx * dt + 0.5 * ax_f * dt * dt
        vx = max(vx + ax_f * dt, 0.0)
        states[i + 1] = np.asarray([x, y, vx, vy, ax_f, ay], dtype=np.float32)
    return states
