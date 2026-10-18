"""Shared forecasts and applicability scores for an existing NPC plan."""

import torch
from .motion import motion_forecast, observed_acceleration, lateral_direction
from .scene_geometry import (
    torch_map_lane_geometry,
    torch_nearest_lane_geometry,
    paired_rectangle_separation,
)
from .stochastic_drivers.driver_policy import physical_leader_observation


def planned_lane_change(plan, current_y, map_polylines, map_valid):
    """Mapped lane transitions distinguish maneuvers from within-lane correction."""
    lanes, present, widths = torch_map_lane_geometry(map_polylines, map_valid)
    planned, _, _, _ = torch_nearest_lane_geometry(plan[..., 1], lanes, present, widths)
    current, _, _, _ = torch_nearest_lane_geometry(current_y, lanes, present, widths)
    return (planned != planned[:, :1]).any(1) | (current != planned[:, -1])


def forecast_poses(context, forecast_owned, times):
    current = context.current
    previous = None if context.history.shape[1] < 2 else context.history[:, -2]
    previous_valid = None if previous is None else context.history_valid[:, -2]
    position, heading = motion_forecast(
        current,
        previous,
        previous_valid,
        times,
        history=context.history,
        history_valid=context.history_valid,
    )
    start, steps = context.response_index, len(times)
    plan = context.planned_path_xy[:, start : start + steps]
    origin = context.planned_current_xy
    direction = torch.diff(torch.cat((origin[:, None], plan), 1), dim=1)
    plan_heading = torch.atan2(direction[..., 1], direction[..., 0].clamp_min(1e-8))
    actual_heading = torch.atan2(current[..., 3], current[..., 2])
    initial_heading = plan_heading[:, 0]
    heading_offset = torch.atan2(
        (actual_heading[:, 1:] - initial_heading).sin(),
        (actual_heading[:, 1:] - initial_heading).cos(),
    )
    plan = plan + (current[:, 1:, :2] - origin)[:, None]
    plan_heading = plan_heading + heading_offset[:, None]
    position = torch.cat(
        (
            position[:, :, :1],
            torch.where(forecast_owned[:, None, :, None], position[:, :, 1:], plan),
        ),
        2,
    )
    heading = torch.cat(
        (
            heading[:, :, :1],
            torch.where(forecast_owned[:, None], heading[:, :, 1:], plan_heading),
        ),
        2,
    )
    braking = torch.zeros_like(context.current_valid)
    if previous is not None:
        braking = ((current[..., 2] - previous[..., 2]) / 0.04 < -0.5) & previous_valid
    lateral = current[..., 3].abs() > 0.15
    group = braking.long() + 2 * lateral.long()
    cold = torch.ones_like(lateral) if previous is None else ~previous_valid
    group = torch.where(cold, 4 + lateral.long(), group)
    changes = planned_lane_change(
        context.planned_path_xy[:, start:],
        current[:, 1:, 1],
        context.map_polylines,
        context.map_polyline_valid,
    )
    group[:, 1:] = torch.where(forecast_owned, group[:, 1:], 6 + changes.long())
    return position, heading, group, changes


def forecast_bundle(context, forecast_owned, bank, *, horizon_s=1.5, dt=0.04):
    steps = min(
        int(horizon_s / dt + 1e-6),
        context.planned_path_xy.shape[1] - context.response_index,
    )
    if steps <= 0:
        return None
    if bank.samples.shape[-2] < steps:
        raise ValueError("Trajectory bank does not cover the requested horizon")
    times = torch.arange(1, steps + 1, device=context.current.device) * dt
    position, heading, group, changes = forecast_poses(context, forecast_owned, times)
    acceleration = observed_acceleration(
        context.current,
        history=context.history,
        history_valid=context.history_valid,
        dt=dt,
    )
    sign = lateral_direction(context.current, acceleration)
    displacement = position[:, :, 1:, 1] - context.current[:, None, 1:, 1]
    largest = displacement.abs().argmax(1, keepdim=True)
    trend = displacement.gather(1, largest).squeeze(1)
    sign[:, 1:] = torch.where(
        forecast_owned, sign[:, 1:], torch.where(trend < 0, -1.0, 1.0)
    )
    errors = bank.samples[group][..., :steps, :].to(context.current).clone()
    errors[..., 1:] *= sign[..., None, None, None, None]
    return position, heading, errors, changes, times, acceleration


def conflict_scores(
    context,
    owned,
    bank,
    *,
    horizon_s=1.5,
    dt=0.04,
    return_curve=False,
    support_threshold=1 / 16,
    forecast_owned=None,
    forecast=None,
):
    """Whole-trajectory empirical conflict support; no future ADS control input."""
    current = context.current
    if forecast_owned is None:
        forecast_owned = owned
    empty = current.new_zeros((len(current), current.shape[1] - 1))
    if forecast is None:
        forecast = forecast_bundle(
            context, forecast_owned, bank, horizon_s=horizon_s, dt=dt
        )
    if forecast is None:
        return empty, torch.full_like(empty, -1), torch.full_like(empty, -1).long()
    position, heading, errors, changes, times, _ = forecast
    steps = len(times)
    own_bound = errors[:, :, 0, :, :, :2].abs().amax(2).permute(0, 2, 1, 3)
    other_bound = errors[:, :, 1, :, :, :2].abs().amax(2).permute(0, 2, 1, 3)
    radius = context.vehicle_dimensions.norm(dim=-1) * 0.5
    delta = position[..., None, :, :] - position[..., :, None, :]
    reach = (radius[:, None, :, None] + radius[:, None, None, :])[..., None]
    reach = reach + own_bound[..., :, None, :] + other_bound[..., None, :, :]
    potential = (delta.abs() <= reach).all(-1)
    dx = current[:, None, :, 0] - current[:, :, None, 0]
    changes = torch.cat((torch.zeros_like(owned[:, :1]), changes), 1)
    entering = (position[..., 1] - current[:, None, :, 1]).abs().gt(0.25) & changes[
        :, None
    ]
    directed = dx[:, None].gt(0) | entering[..., :, None]
    pending = (
        torch.cat((torch.zeros_like(owned[:, :1]), ~owned), 1) & context.current_valid
    )
    pairs = pending[:, :, None] & context.current_valid[:, None, :]
    pairs &= ~torch.eye(current.shape[1], dtype=torch.bool, device=current.device)[None]
    possible = potential & directed & pairs[:, None]
    indices = possible.any(1).nonzero()
    support = current.new_zeros((len(current), current.shape[1], current.shape[1]))
    medians = torch.full_like(support, torch.inf)
    curve = (
        current.new_zeros((len(current), steps, current.shape[1]))
        if return_curve
        else None
    )
    for chunk in indices.split(512):
        if not len(chunk):
            continue
        b, i, j = chunk.unbind(-1)
        own = errors[b, i, 0]
        other = errors[b, j, 1]
        own_position = position[b, :, i][:, None] + own[..., :2]
        other_position = position[b, :, j][:, None] + other[..., :2]
        own_angle = heading[b, :, i][:, None] + own[..., 2]
        other_angle = heading[b, :, j][:, None] + other[..., 2]
        gap = paired_rectangle_separation(
            own_position,
            own_angle,
            context.vehicle_dimensions[b, i][:, None, None],
            other_position,
            other_angle,
            context.vehicle_dimensions[b, j][:, None, None],
        )
        hits = gap.le(0) & directed[b, :, i, j][:, None]
        ever = hits.any(-1)
        q = ever.float().mean(-1)
        support[b, i, j] = q
        if return_curve:
            prefix = hits.long().cummax(-1).values.float().mean(1)
            for k in range(steps):
                flat_index = b * (steps * current.shape[1]) + k * current.shape[1] + i
                curve.flatten().scatter_reduce_(
                    0, flat_index, prefix[:, k], reduce="amax"
                )
        first = (
            torch.where(ever, times[hits.long().argmax(-1)], torch.inf).sort(-1).values
        )
        count = ever.sum(-1)
        low = ((count - 1).clamp_min(0) // 2)[:, None]
        high = (count // 2)[:, None]
        medians[b, i, j] = 0.5 * (
            first.gather(-1, low) + first.gather(-1, high)
        ).squeeze(-1)
    if return_curve:
        return curve[:, :, 1:]
    score = support.amax(-1)
    tied = support.eq(score[..., None])
    earliest = medians.masked_fill(~tied, torch.inf).amin(-1, keepdim=True)
    neighbour = (tied & medians.eq(earliest)).long().argmax(-1)
    time = medians.gather(-1, neighbour[..., None]).squeeze(-1)
    triggered = pending & score.ge(support_threshold) & score.gt(0)
    return (
        score[:, 1:],
        torch.where(triggered, time, -1)[:, 1:],
        torch.where(triggered, neighbour, -1)[:, 1:],
    )


def interaction_scores(context, forecast, previous, maneuver_started):
    """Return raw headway growth (s) and corridor release importance (1/s).

    `previous` is the preceding actual state/validity, including across snapshots.
    Both relations and all future poses are shared with the occupancy/NPC kernels.
    """
    current, valid, dims = (
        context.current,
        context.current_valid,
        context.vehicle_dimensions,
    )
    shape = current.shape[:2]
    scores = current.new_zeros((*shape, 2))
    influencers = torch.full((*shape, 2), -1, dtype=torch.long, device=current.device)
    event_times = current.new_full((*shape, 2), -1)
    if previous is None or forecast is None:
        return scores[:, 1:], influencers[:, 1:], event_times[:, 1:]
    past, past_valid = previous
    gap, speed, leader_speed, leader = physical_leader_observation(current, valid, dims)
    old_gap, old_speed, _, old_leader = physical_leader_observation(
        past, past_valid, dims
    )
    position, heading, errors, _, times, acceleration = forecast
    b = torch.arange(len(current), device=current.device)[:, None]
    j = leader.clamp_min(0)
    established = leader.ge(0) & leader.eq(old_leader) & valid & past_valid
    accelerating = acceleration[b, j, 0] > 0
    # A faster vehicle far ahead is not a short-horizon following constraint.
    # Reuse the prediction horizon and physical acceleration bound; this only
    # restricts optional acceleration responses, never conflict detection.
    horizon = times[-1]
    following_relevant = (gap <= speed * horizon) & (
        leader_speed - speed <= context.cfg.max_acceleration_mps2 * horizon
    )
    own = position.permute(0, 2, 1, 3)[:, :, None] + errors[:, :, 0, ..., :2]
    other = position[b, :, j][:, :, None] + errors[b, j, 1, ..., :2]
    angle_i = heading.permute(0, 2, 1)[:, :, None] + errors[:, :, 0, ..., 2]
    angle_j = heading[b, :, j][:, :, None] + errors[b, j, 1, ..., 2]
    delta = other - own
    along = delta[..., 0] * angle_i.cos() + delta[..., 1] * angle_i.sin()
    relative = angle_j - angle_i
    length, width = (dims * 0.5).unbind(-1)
    predicted_gap = (
        along
        - length[..., None, None]
        - length[b, j][..., None, None] * relative.cos().abs()
        - width[b, j][..., None, None] * relative.sin().abs()
    )
    growth, index = (predicted_gap - gap[..., None, None]).clamp_min(0).max(-1)
    # Midpoint of the central two samples is the ordinary median for 16 paths.
    following = torch.quantile(growth / speed.clamp_min(1)[..., None], 0.5, dim=-1)
    following = torch.where(
        established & accelerating & following_relevant, following, 0
    )
    scores[..., 0] = following
    influencers[..., 0] = torch.where(following > 0, leader, -1)
    event_times[..., 0] = torch.where(
        following > 0, torch.quantile(times[index], 0.5, dim=-1), -1
    )

    old_j = old_leader.clamp_min(0)
    # Test departure from the observer's previous corridor, not a corridor that
    # moved because the observer itself changed lanes. Disappearing tracks do not count.
    corridor = current.clone()
    corridor[..., 1] = past[..., 1]
    corridor[..., 2:4] = past[..., 2:4]
    delta = current[b, old_j, :2] - corridor[..., :2]
    angle = torch.atan2(corridor[..., 3], corridor[..., 2])
    other_angle = torch.atan2(current[b, old_j, 3], current[b, old_j, 2])
    relative = other_angle - angle
    clearance = (
        (-delta[..., 0] * angle.sin() + delta[..., 1] * angle.cos()).abs()
        - width
        - length[b, old_j] * relative.sin().abs()
        - width[b, old_j] * relative.cos().abs()
    )
    departure = (current[b, old_j, 1] - past[..., 1]).abs() > (
        past[b, old_j, 1] - past[..., 1]
    ).abs()
    released = (
        old_leader.ge(0)
        & old_leader.ne(leader)
        & valid
        & past_valid
        & valid[b, old_j]
        & clearance.gt(0)
        & departure
        & gap.gt(old_gap)
    )
    plan = context.planned_path_xy[:, context.response_index :]
    if plan.shape[1] > 1:
        n = min(5, plan.shape[1] - 1)
        initial_v = (plan[:, 0, :, 0] - context.planned_current_xy[..., 0]) / 0.04
        later_v = (plan[:, n, :, 0] - plan[:, n - 1, :, 0]) / 0.04
        responding = later_v > initial_v + 1e-3
    else:
        responding = torch.zeros_like(maneuver_started)
    released[:, 1:] &= ~(responding | maneuver_started)
    release = torch.where(released, old_speed / old_gap.clamp_min(1e-3), 0)
    scores[..., 1] = release
    influencers[..., 1] = torch.where(release > 0, old_leader, -1)
    event_times[..., 1] = torch.where(release > 0, 0.0, -1.0)
    return scores[:, 1:], influencers[:, 1:], event_times[:, 1:]
