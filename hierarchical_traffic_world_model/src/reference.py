"""Soft-reference rebasing and physical feature extraction."""

from __future__ import annotations

import torch


def _heading(velocity: torch.Tensor) -> torch.Tensor:
    safe_x = torch.where(
        velocity[..., 0].abs() < 1.0e-4,
        torch.full_like(velocity[..., 0], 1.0e-4),
        velocity[..., 0],
    )
    return torch.atan2(velocity[..., 1], safe_x)


def rebase_soft_preview(
    current_background: torch.Tensor,
    reference: torch.Tensor,
    reference_base: torch.Tensor,
    *,
    dt_s: float = 0.04,
    velocity_horizon_s: float = 0.40,
    endpoint_offset_weight: float = 0.0,
) -> torch.Tensor:
    """Make the plan locally continuous while retaining a soft long-horizon anchor.

    A cubic Hermite offset matches the realized position and velocity at the
    response boundary, then decays to zero at the preview endpoint.  The
    reference is therefore neither hard execution nor a repeatedly translated
    plan that preserves accumulated rollout error.
    """
    if current_background.ndim != 3 or current_background.shape[-2:] != (6, 6):
        raise ValueError("current_background must have shape [batch,6,6]")
    if reference.ndim != 4 or reference.shape[2:] != (6, 2):
        raise ValueError("reference must have shape [batch,frames,6,2]")
    if reference_base.shape != current_background[..., :2].shape:
        raise ValueError("reference_base must have shape [batch,6,2]")
    frames = int(reference.shape[1])
    elapsed = torch.arange(
        1, frames + 1, device=reference.device, dtype=reference.dtype
    ) * float(dt_s)
    horizon_s = min(float(frames) * float(dt_s), float(velocity_horizon_s))
    progress = (elapsed / horizon_s).clamp_max(1.0)
    hermite_position = 2.0 * progress**3 - 3.0 * progress**2 + 1.0
    hermite_velocity = progress**3 - 2.0 * progress**2 + progress
    plan_velocity = (reference[:, 0] - reference_base) / float(dt_s)
    velocity_delta = current_background[..., 2:4] - plan_velocity
    position_delta = current_background[..., :2] - reference_base
    velocity_correction = (
        horizon_s
        * hermite_velocity[None, :, None, None]
        * velocity_delta[:, None]
    )
    position_weight = endpoint_offset_weight + (
        1.0 - endpoint_offset_weight
    ) * hermite_position
    position_correction = position_weight[None, :, None, None] * position_delta[:, None]
    return reference + position_correction + velocity_correction


def preview_features(
    current_background: torch.Tensor,
    rebased_preview: torch.Tensor,
    *,
    dt_s: float,
) -> torch.Tensor:
    """Summarize a modifiable preview without exposing absolute future state."""
    frames = rebased_preview.shape[1]
    indices = torch.as_tensor(
        (min(4, frames - 1), min(14, frames - 1), frames - 1),
        device=rebased_preview.device,
    )
    selected = rebased_preview.index_select(1, indices)
    displacement = selected - current_background[:, None, :, :2]
    first_velocity = (rebased_preview[:, 0] - current_background[..., :2]) / float(dt_s)
    final_velocity = (rebased_preview[:, -1] - rebased_preview[:, -2]) / float(dt_s)
    velocity_delta = final_velocity - first_velocity
    return torch.cat(
        (
            displacement.permute(0, 2, 1, 3).flatten(2),
            velocity_delta,
        ),
        dim=-1,
    )


def soft_reference_controls(
    current_background: torch.Tensor,
    rebased_preview: torch.Tensor,
    *,
    dt_s: float,
    min_acceleration: float,
    max_acceleration: float,
    max_yaw_rate: float,
) -> torch.Tensor:
    """Recover frame controls whose kinematics follow the soft positions."""
    previous_position = torch.cat(
        (current_background[:, None, :, :2], rebased_preview[:, :-1]), dim=1
    )
    velocity = (rebased_preview - previous_position) / float(dt_s)
    previous_velocity = torch.cat(
        (current_background[:, None, :, 2:4], velocity[:, :-1]), dim=1
    )
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    previous_speed = torch.linalg.vector_norm(previous_velocity, dim=-1)
    acceleration = (speed - previous_speed) / float(dt_s)
    heading = _heading(velocity)
    previous_heading = _heading(previous_velocity)
    heading_delta = torch.atan2(
        torch.sin(heading - previous_heading),
        torch.cos(heading - previous_heading),
    )
    yaw_rate = heading_delta / float(dt_s)
    return torch.stack(
        (
            acceleration.clamp(min_acceleration, max_acceleration),
            yaw_rate.clamp(-max_yaw_rate, max_yaw_rate),
        ),
        dim=-1,
    )


def response_relevance(
    current: torch.Tensor,
    valid: torch.Tensor,
    *,
    lane_width_m: float = 3.6,
) -> torch.Tensor:
    """Continuous ego-to-background interaction relevance in `[0,1]`."""
    ego = current[:, :1]
    background = current[:, 1:]
    dx = (background[..., 0] - ego[..., 0]).abs()
    dy = (background[..., 1] - ego[..., 1]).abs()
    closing = (ego[..., 2] - background[..., 2]).abs()
    longitudinal = torch.exp(-dx / 35.0)
    lane = torch.exp(-torch.square(dy / float(lane_width_m)))
    conflict = torch.exp(-closing / 12.0)
    relevance = longitudinal * lane * (0.6 + 0.4 * conflict)
    return relevance * valid[:, 1:].float()
