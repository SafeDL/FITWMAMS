"""Soft-reference rebasing and physical feature extraction."""

from __future__ import annotations

import torch


def planned_lateral_position_at_x(origin_xy, path_xy, query_x):
    """Read a forward spatial path at one or several x positions per actor."""
    path = torch.cat((origin_xy[:, None], path_xy), dim=1).permute(0, 2, 1, 3)
    path_x, path_y = path[..., 0], path[..., 1]
    query = query_x[..., None] if query_x.ndim == 2 else query_x.transpose(1, 2)
    upper = torch.searchsorted(
        path_x.contiguous(), query.contiguous(), right=True
    ).clamp(1, path_x.shape[-1] - 1)
    lower = upper - 1
    lower_x = path_x.gather(-1, lower)
    upper_x = path_x.gather(-1, upper)
    lower_y = path_y.gather(-1, lower)
    upper_y = path_y.gather(-1, upper)
    fraction = ((query - lower_x) / (upper_x - lower_x).clamp_min(1.0e-4)).clamp(0, 1)
    lateral = lower_y + fraction * (upper_y - lower_y)
    lateral = torch.where(query >= path_x[..., -1:], path_y[..., -1:], lateral)
    return lateral.squeeze(-1) if query_x.ndim == 2 else lateral.transpose(1, 2)


def unreachable_reference(
    current,
    preview,
    remaining_frames,
    *,
    dt_s=0.04,
    min_acceleration=-8.0,
    max_acceleration=4.0,
    position_tolerance_m=0.1,
):
    """Check the last available preview point against a longitudinal motion envelope."""
    steps = remaining_frames.clamp(1, preview.shape[1])
    endpoint = (
        preview[..., 0].transpose(1, 2).gather(2, (steps - 1)[..., None]).squeeze(-1)
    )
    elapsed = steps.to(current.dtype) * dt_s
    speed = current[..., 2].clamp_min(0)
    braking = -float(min_acceleration)
    stop_time = speed / braking
    brake_time = torch.minimum(elapsed, stop_time)
    minimum_distance = speed * brake_time - 0.5 * braking * brake_time.square()
    maximum_distance = speed * elapsed + 0.5 * max_acceleration * elapsed.square()
    displacement = endpoint - current[..., 0]
    return (displacement < minimum_distance - position_tolerance_m) | (
        displacement > maximum_distance + position_tolerance_m
    )


def continue_retired_reference(
    current,
    preview,
    base,
    retired,
    *,
    reference,
    reference_base,
    dt_s=0.04,
    jerk_limit_mps3=2.0,
):
    """Keep road intent but remove an obsolete longitudinal position/speed schedule.

    Continue x/vx from actual motion and relax ax with a bounded jerk. Read
    the lateral path at those x positions, not at its obsolete clock times.
    """
    elapsed = (
        torch.arange(
            1, preview.shape[1] + 1, device=current.device, dtype=current.dtype
        )
        * dt_s
    )
    acceleration = current[..., 4].clamp(-8, 4)
    relaxed = acceleration.sign()[:, None] * (
        acceleration.abs()[:, None] - jerk_limit_mps3 * elapsed[None, :, None]
    ).clamp_min(0)
    velocity = (current[:, None, :, 2] + relaxed.cumsum(1) * dt_s).clamp_min(0)
    previous_velocity = torch.cat(
        (current[:, None, :, 2].clamp_min(0), velocity[:, :-1]), 1
    )
    position = current[:, None, :, 0] + (
        0.5 * (previous_velocity + velocity) * dt_s
    ).cumsum(1)
    lateral = planned_lateral_position_at_x(reference_base, reference, position)
    base_lateral = planned_lateral_position_at_x(
        reference_base, reference, current[..., 0]
    )
    modified = torch.stack(
        (
            torch.where(retired[:, None], position, preview[..., 0]),
            torch.where(retired[:, None], lateral, preview[..., 1]),
        ),
        -1,
    )
    modified_base = torch.stack(
        (
            torch.where(retired, current[..., 0], base[..., 0]),
            torch.where(retired, base_lateral, base[..., 1]),
        ),
        -1,
    )
    return modified, modified_base


def recover_reference(
    current,
    valid,
    preview,
    base,
    remaining_frames,
    retired,
    cfg,
    *,
    reference,
    reference_base,
):
    """Retire an unreachable timetable once, then continue from realised motion."""
    retired = retired | (
        valid
        & unreachable_reference(
            current,
            preview,
            remaining_frames,
            dt_s=cfg.dt_s,
            min_acceleration=cfg.min_acceleration_mps2,
            max_acceleration=cfg.max_acceleration_mps2,
        )
    )
    preview, base = continue_retired_reference(
        current,
        preview,
        base,
        retired,
        dt_s=cfg.dt_s,
        reference=reference,
        reference_base=reference_base,
        jerk_limit_mps3=cfg.max_residual_jerk_mps3,
    )
    return preview, base, retired


def project_reference_progress(
    current_background_xy: torch.Tensor,
    reference: torch.Tensor,
    previous_start: torch.Tensor,
    *,
    nominal_index: int,
    local_window: int = 25,
) -> torch.Tensor:
    """Return monotone per-agent preview starts from realized path progress.

    Only a local forward window is searched.  A stopped or delayed vehicle can
    therefore hold its spatial progress instead of being pulled toward the
    plan's wall-clock index, while progress cannot jump backwards.
    """
    if current_background_xy.ndim != 3 or current_background_xy.shape[-2:] != (6, 2):
        raise ValueError("current_background_xy must have shape [batch,6,2]")
    if (
        reference.ndim != 4
        or reference.shape[0] != current_background_xy.shape[0]
        or reference.shape[2:] != (6, 2)
    ):
        raise ValueError("reference must have shape [batch,time,6,2]")
    if previous_start.shape != current_background_xy.shape[:2]:
        raise ValueError("previous_start must have shape [batch,6]")
    if int(nominal_index) == 0:
        return torch.zeros_like(previous_start)
    time = reference.shape[1]
    lower = (previous_start - 1).clamp_min(0)
    upper = torch.maximum(
        torch.full_like(lower, min(time, int(nominal_index) + int(local_window))),
        lower + 1,
    ).clamp_max(time)
    indices = torch.arange(time, device=reference.device)
    allowed = (indices >= lower[..., None]) & (indices < upper[..., None])
    distance = torch.linalg.vector_norm(
        reference.permute(0, 2, 1, 3) - current_background_xy[:, :, None],
        dim=-1,
    )
    nearest = distance.masked_fill(~allowed, float("inf")).argmin(dim=-1)
    return torch.maximum(previous_start, (nearest + 1).clamp_max(time - 1))


def gather_progress_preview(
    reference: torch.Tensor,
    reference_base: torch.Tensor,
    starts: torch.Tensor,
    *,
    preview_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather a fixed preview independently for each background actor."""
    batch, time, actors, _ = reference.shape
    offsets = torch.arange(preview_frames, device=reference.device)
    indices = (starts[..., None] + offsets).clamp_max(time - 1)
    expanded = reference.permute(0, 2, 1, 3)
    preview = torch.gather(
        expanded,
        2,
        indices[..., None].expand(batch, actors, preview_frames, 2),
    ).permute(0, 2, 1, 3)
    previous_index = (starts - 1).clamp_min(0)
    prior = torch.gather(
        expanded, 2, previous_index[..., None, None].expand(batch, actors, 1, 2)
    ).squeeze(2)
    base = torch.where((starts > 0)[..., None], prior, reference_base)
    return preview, base


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
    longitudinal_endpoint_offset_weight: float | None = None,
    lateral_endpoint_offset_weight: float | None = None,
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
        horizon_s * hermite_velocity[None, :, None, None] * velocity_delta[:, None]
    )
    # The legacy scalar keeps the released checkpoint behaviour.  Reaction
    # experiments may opt into a timing-invariant longitudinal reference:
    # retain accumulated x-offset while still letting y return to lane
    # geometry.  The split is explicit so callers cannot accidentally claim
    # an absolute K reference after an interactive divergence.
    longitudinal_weight = (
        endpoint_offset_weight
        if longitudinal_endpoint_offset_weight is None
        else float(longitudinal_endpoint_offset_weight)
    )
    lateral_weight = (
        endpoint_offset_weight
        if lateral_endpoint_offset_weight is None
        else float(lateral_endpoint_offset_weight)
    )
    endpoint_weights = position_delta.new_tensor((longitudinal_weight, lateral_weight))
    position_weight = (
        endpoint_weights + (1.0 - endpoint_weights) * hermite_position[:, None]
    )
    position_correction = position_weight[None, :, None] * position_delta[:, None]
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
