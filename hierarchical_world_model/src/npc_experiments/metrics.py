"""Shared physical traffic features and distribution distances."""

import numpy as np
from scipy.signal import savgol_filter

DT = 0.04
FEATURE_EDGES = {
    "speed": np.r_[-np.inf, np.linspace(0, 65, 53), np.inf],
    "acceleration": np.r_[-np.inf, np.linspace(-8, 4, 49), np.inf],
    "gap": np.r_[-np.inf, np.linspace(0, 150, 51), np.inf],
    "headway": np.r_[-np.inf, np.linspace(0, 10, 41), np.inf],
    "ttc": np.r_[-np.inf, [0, 1, 3, 5, 10, 20], np.inf],
}
JOINT_EDGES = (
    np.r_[-np.inf, np.linspace(0, 150, 26), np.inf],
    np.r_[-np.inf, np.linspace(-20, 20, 26), np.inf],
)


def hellinger(left, right):
    """Compare probability masses, retaining underflow and overflow bins."""
    left, right = np.asarray(left, float), np.asarray(right, float)
    p = left / left.sum(axis=-1, keepdims=True)
    q = right / right.sum(axis=-1, keepdims=True)
    return np.sqrt(0.5 * np.square(np.sqrt(p) - np.sqrt(q)).sum(axis=-1))


def longitudinal_acceleration(states):
    speed = np.linalg.norm(states[..., 2:4], axis=-1)
    return (states[..., 2:4] * states[..., 4:6]).sum(-1) / np.maximum(speed, 0.5)


def diagnostic_states(states, window):
    """Same retrospective position-to-motion operator for real, plan and execution."""
    values = np.asarray(states, np.float32).copy()
    for derivative, columns in ((1, slice(2, 4)), (2, slice(4, 6))):
        values[..., columns] = savgol_filter(
            values[..., :2],
            window,
            3,
            deriv=derivative,
            delta=DT,
            axis=1,
            mode="interp",
        )
    return values
