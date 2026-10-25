"""Observed short-history motion; no unexecuted controller actions."""

import torch


def observed_acceleration(
    current,
    previous=None,
    previous_valid=None,
    *,
    history=None,
    history_valid=None,
    dt=0.04,
):
    ax = torch.zeros_like(current[..., 2])
    ay = torch.zeros_like(ax)
    if history is None and previous is not None:
        history = torch.stack((previous, current), 1)
        history_valid = torch.stack(
            (previous_valid, torch.ones_like(previous_valid)), 1
        )
    if history is not None and history.shape[1] > 1:
        history, valid = history[:, -6:], history_valid[:, -6:]
        # Only the uninterrupted observed suffix belongs to this physical history.
        valid = valid.flip(1).long().cumprod(1).flip(1).bool()
        ax = torch.where(valid[:, -2], (current[..., 2] - history[:, -2, :, 2]) / dt, 0)
        times = torch.arange(1 - history.shape[1], 1, device=current.device) * dt
        weight = valid.to(current.dtype)
        count = weight.sum(1).clamp_min(1)
        mean_t = (times[None, :, None] * weight).sum(1) / count
        centered = times[None, :, None] - mean_t[:, None]
        ay = (centered * history[..., 3] * weight).sum(1) / (
            centered.square() * weight
        ).sum(1).clamp_min(1e-8)
    return torch.stack((ax.clamp(-8, 4), ay), -1)


def advance_motion(current, acceleration, time):
    """Broadcastable CA in x; 0.4 s lateral trend followed by constant velocity."""
    vx = current[..., 2].clamp(0, 100)
    ax, ay = acceleration.unbind(-1)
    duration = torch.where(
        ax < 0, torch.minimum(time, vx / (-ax).clamp_min(1e-8)), time
    )
    duration = torch.where(
        ax > 0, torch.minimum(duration, (100 - vx) / ax.clamp_min(1e-8)), duration
    )
    future_vx = (vx + ax * time).clamp(0, 100)
    x = current[..., 0] + vx * duration + 0.5 * ax * duration.square()
    x = x + future_vx * (time - duration)
    lateral_duration = time.clamp_max(0.4)
    y = (
        current[..., 1]
        + current[..., 3] * time
        + ay * (lateral_duration * time - 0.5 * lateral_duration.square())
    )
    vy = current[..., 3] + ay * lateral_duration
    return torch.stack((x, y), -1), torch.stack((future_vx, vy), -1)


def motion_forecast(
    current,
    previous,
    previous_valid,
    times,
    *,
    dt=0.04,
    history=None,
    history_valid=None,
):
    acceleration = observed_acceleration(
        current,
        previous,
        previous_valid,
        history=history,
        history_valid=history_valid,
        dt=dt,
    )
    position, velocity = advance_motion(
        current[:, None], acceleration[:, None], times[None, :, None]
    )
    return position, torch.atan2(velocity[..., 1], velocity[..., 0].clamp_min(1e-8))


def lateral_direction(current, acceleration):
    trend = current[..., 3] + 0.4 * acceleration[..., 1]
    return torch.where(trend < 0, -torch.ones_like(trend), torch.ones_like(trend))
