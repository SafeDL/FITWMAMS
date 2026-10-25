"""Feedforward and feedback tracking of a stored position plan."""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.signal import savgol_filter


@dataclass(frozen=True)
class TrackingConfig:
    longitudinal_frequency: float = 2.0
    lateral_frequency: float = 2.0
    derivative_window: int = 11


def tracking_reference(origin, positions, *, dt=0.04, window=11):
    """Differentiate once at reset; retain the original position targets.

    The plant translates at the old heading. Segment headings therefore give
    its discrete-time reference, rather than the tangent at the interval end.
    No logged velocity, acceleration, or future ADS action is used.
    """
    path = np.concatenate((np.asarray(origin)[:, None], np.asarray(positions)), 1)
    length = min(int(window), path.shape[1] // 2 * 2 - 1)
    if length < 5:
        velocity = np.gradient(path.astype(np.float64), dt, axis=1)
        acceleration = np.gradient(velocity, dt, axis=1)
    else:
        velocity = savgol_filter(
            path.astype(np.float64), length, 3, deriv=1, delta=dt, axis=1
        )
        acceleration = savgol_filter(
            path.astype(np.float64), length, 3, deriv=2, delta=dt, axis=1
        )
    speed = np.linalg.norm(velocity, axis=-1)
    heading = np.arctan2(velocity[..., 1], np.maximum(velocity[..., 0], 1e-4))
    yaw = (
        velocity[..., 0] * acceleration[..., 1]
        - velocity[..., 1] * acceleration[..., 0]
    ) / np.maximum(speed**2, 0.25)
    longitudinal = (velocity * acceleration).sum(-1) / np.maximum(speed, 0.5)
    heading = heading + 0.5 * dt * yaw
    return np.concatenate(
        (
            path,
            speed[..., None],
            heading[..., None],
            longitudinal[..., None],
            yaw[..., None],
        ),
        -1,
    ).astype(np.float32)


def tracking_actions(
    current,
    reference,
    config,
    *,
    min_acceleration=-8.0,
    max_acceleration=4.0,
    max_yaw_rate=0.6,
):
    """Critically damped position/speed and cross-track/heading feedback."""
    speed = torch.linalg.vector_norm(current[..., 2:4], dim=-1)
    heading = torch.atan2(current[..., 3], current[..., 2].clamp_min(1e-4))
    error = reference[..., :2] - current[..., :2]
    along = error[..., 0] * heading.cos() + error[..., 1] * heading.sin()
    cross = -error[..., 0] * heading.sin() + error[..., 1] * heading.cos()
    heading_error = torch.atan2(
        torch.sin(reference[..., 3] - heading), torch.cos(reference[..., 3] - heading)
    )
    longitudinal = config.longitudinal_frequency
    lateral = config.lateral_frequency
    acceleration = (
        reference[..., 4]
        + 2 * longitudinal * (reference[..., 2] - speed)
        + longitudinal**2 * along
    )
    yaw = (
        reference[..., 5]
        + 2 * lateral * heading_error
        + lateral**2 * cross / speed.clamp_min(0.5)
    )
    return torch.stack(
        (
            acceleration.clamp(min_acceleration, max_acceleration),
            yaw.clamp(-max_yaw_rate, max_yaw_rate),
        ),
        -1,
    )
