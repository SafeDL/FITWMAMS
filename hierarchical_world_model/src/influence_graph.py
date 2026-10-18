"""Physical leader and adjacent-lane relations for NPC decisions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .scene_geometry import (
    torch_adjacent_lane_index,
    torch_map_lane_geometry,
    torch_nearest_lane_geometry,
)


@dataclass(frozen=True)
class DynamicInteractionRelations:
    """Current actor relations, recomputed once before every joint action."""

    lane_index: torch.Tensor
    leader_index: torch.Tensor
    leader_gap_m: torch.Tensor
    leader_ttc_s: torch.Tensor
    adjacent_front_index: torch.Tensor
    adjacent_rear_index: torch.Tensor
    adjacent_lane_index: torch.Tensor
    crossing_conflict: torch.Tensor


def build_dynamic_interaction_relations(
    states: torch.Tensor,
    valid: torch.Tensor,
    map_polylines: torch.Tensor,
    map_polyline_valid: torch.Tensor,
    *,
    vehicle_lengths_m: torch.Tensor | None = None,
    lane_half_width_m: float = 1.8,
    prediction_horizon_s: float = 1.5,
) -> DynamicInteractionRelations:
    """Resolve leaders and adjacent-lane neighbors from realized state only."""
    if states.ndim != 3 or states.shape[1:] != (7, 6):
        raise ValueError("states must have shape [batch,7,6]")
    if valid.shape != states.shape[:2]:
        raise ValueError("valid must have shape [batch,7]")
    lane_y, lane_present, lane_width = torch_map_lane_geometry(
        map_polylines,
        map_polyline_valid,
        fallback_width_m=2.0 * float(lane_half_width_m),
    )
    lane_index, _, current_width, _ = torch_nearest_lane_geometry(
        states[..., 1], lane_y, lane_present, lane_width
    )
    actor_half_width = 0.5 * current_width
    batch, actors = states.shape[:2]
    other = states[:, None]
    follower = states[:, :, None]
    dx = other[..., 0] - follower[..., 0]
    dy = other[..., 1] - follower[..., 1]
    future_dy = dy + (other[..., 3] - follower[..., 3]) * float(prediction_horizon_s)
    overlap = (
        torch.minimum(dy.abs(), future_dy.abs()) < actor_half_width[..., None]
    ) | (dy * future_dy <= 0.0)
    identity = torch.eye(actors, dtype=torch.bool, device=states.device)[None]
    candidate = valid[:, :, None] & valid[:, None, :] & ~identity & dx.gt(0.0) & overlap
    lengths = (
        torch.full(states.shape[:2], 4.8, dtype=states.dtype, device=states.device)
        if vehicle_lengths_m is None
        else torch.as_tensor(
            vehicle_lengths_m, dtype=states.dtype, device=states.device
        )
    )
    if lengths.shape != states.shape[:2]:
        raise ValueError("vehicle_lengths_m must have shape [batch,7]")
    pair_length = 0.5 * (lengths[:, :, None] + lengths[:, None, :])
    gap = dx - pair_length
    closing = follower[..., 2] - other[..., 2]
    projected_gap = gap - closing * float(prediction_horizon_s)
    unsafe = candidate & projected_gap.lt(
        torch.maximum(torch.full_like(projected_gap, 2.0), 0.1 * follower[..., 2])
    )
    nearest_score = torch.where(candidate, gap, torch.full_like(gap, float("inf")))
    risk_score = torch.where(unsafe, projected_gap, torch.full_like(gap, float("inf")))
    leader_index = torch.where(
        unsafe.any(dim=-1), risk_score.argmin(dim=-1), nearest_score.argmin(dim=-1)
    )
    has_leader = candidate.any(dim=-1)
    selected_gap = torch.gather(gap, 2, leader_index[..., None]).squeeze(-1)
    selected_closing = torch.gather(closing, 2, leader_index[..., None]).squeeze(-1)
    selected_ttc = torch.where(
        has_leader & selected_gap.gt(0.0) & selected_closing.gt(1.0e-4),
        selected_gap / selected_closing.clamp_min(1.0e-4),
        torch.full_like(selected_gap, float("inf")),
    )
    leader_index = torch.where(
        has_leader, leader_index, torch.full_like(leader_index, -1)
    )
    selected_gap = torch.where(
        has_leader, selected_gap.clamp_min(0.05), torch.full_like(selected_gap, 1.0e6)
    )

    # Side axis is [-y, +y].  Resolve nearest front and rear independently.
    adjacent_front = torch.full(
        (batch, actors, 2), -1, dtype=torch.long, device=states.device
    )
    adjacent_rear = torch.full_like(adjacent_front, -1)
    adjacent_lane = torch.full_like(adjacent_front, -1)
    for side_index, side in enumerate((-1.0, 1.0)):
        target, has_target = torch_adjacent_lane_index(
            lane_index, lane_y, lane_present, lane_width, side=int(side)
        )
        safe_target = target.clamp_min(0)
        adjacent_lane[..., side_index] = torch.where(
            has_target, safe_target, torch.full_like(target, -1)
        )
        in_target = lane_index[:, None, :].eq(safe_target[:, :, None])
        common = (
            valid[:, :, None]
            & valid[:, None, :]
            & ~identity
            & in_target
            & has_target[..., None]
        )
        front_score = torch.where(
            common & dx.gt(0.0), dx, torch.full_like(dx, float("inf"))
        )
        rear_score = torch.where(
            common & dx.lt(0.0), -dx, torch.full_like(dx, float("inf"))
        )
        front = front_score.argmin(dim=-1)
        rear = rear_score.argmin(dim=-1)
        adjacent_front[..., side_index] = torch.where(
            torch.isfinite(front_score.amin(dim=-1)), front, torch.full_like(front, -1)
        )
        adjacent_rear[..., side_index] = torch.where(
            torch.isfinite(rear_score.amin(dim=-1)), rear, torch.full_like(rear, -1)
        )
    crossing = (
        valid[:, :, None]
        & valid[:, None, :]
        & ~identity
        & ~(dy.abs() < actor_half_width[..., None])
        & overlap
    )
    return DynamicInteractionRelations(
        lane_index=lane_index,
        leader_index=leader_index,
        leader_gap_m=selected_gap,
        leader_ttc_s=selected_ttc,
        adjacent_front_index=adjacent_front,
        adjacent_rear_index=adjacent_rear,
        adjacent_lane_index=adjacent_lane,
        crossing_conflict=crossing,
    )
