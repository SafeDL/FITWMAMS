"""Current-state adjacent-lane features and occupancy diagnostics."""

from __future__ import annotations

import torch

from ..influence_graph import DynamicInteractionRelations
from ..scene_geometry import torch_map_lane_geometry

LANE_CHOICE_FEATURE_NAMES = (
    "own_speed_mps",
    "own_acceleration_mps2",
    "current_front_gap_m",
    "current_closing_mps",
    "candidate_front_gap_m",
    "candidate_front_speed_advantage_mps",
    "candidate_rear_gap_m",
    "candidate_rear_closing_mps",
)


def lane_choice_features(
    states: torch.Tensor,
    lengths: torch.Tensor,
    relation: DynamicInteractionRelations,
) -> torch.Tensor:
    """Eight causal features per NPC and adjacent lane, ordered lower/higher y."""
    speed = torch.linalg.vector_norm(states[..., 2:4], dim=-1)
    own = states[:, 1:]
    own_speed = speed[:, 1:]
    leader_speed = torch.gather(speed, 1, relation.leader_index[:, 1:].clamp_min(0))
    leader_speed = torch.where(
        relation.leader_index[:, 1:].ge(0), leader_speed, own_speed
    )
    front_index = relation.adjacent_front_index[:, 1:]
    rear_index = relation.adjacent_rear_index[:, 1:]
    expanded_states = states[:, None].expand(-1, 6, -1, -1)
    expanded_speed = speed[:, None].expand(-1, 6, -1)
    expanded_lengths = lengths[:, None].expand(-1, 6, -1)
    front = torch.gather(
        expanded_states, 2, front_index.clamp_min(0)[..., None].expand(-1, -1, -1, 6)
    )
    rear = torch.gather(
        expanded_states, 2, rear_index.clamp_min(0)[..., None].expand(-1, -1, -1, 6)
    )
    front_speed = torch.gather(expanded_speed, 2, front_index.clamp_min(0))
    rear_speed = torch.gather(expanded_speed, 2, rear_index.clamp_min(0))
    front_length = torch.gather(expanded_lengths, 2, front_index.clamp_min(0))
    rear_length = torch.gather(expanded_lengths, 2, rear_index.clamp_min(0))
    own_length = lengths[:, 1:, None]
    front_gap = front[..., 0] - own[..., None, 0] - 0.5 * (own_length + front_length)
    rear_gap = own[..., None, 0] - rear[..., 0] - 0.5 * (own_length + rear_length)
    front_gap = torch.where(front_index.ge(0), front_gap.clamp(0.0, 120.0), 120.0)
    rear_gap = torch.where(rear_index.ge(0), rear_gap.clamp(0.0, 120.0), 120.0)
    own_speed_pair = own_speed[..., None].expand(-1, -1, 2)
    leader_speed_pair = leader_speed[..., None].expand(-1, -1, 2)
    front_speed = torch.where(front_index.ge(0), front_speed, own_speed_pair)
    rear_speed = torch.where(rear_index.ge(0), rear_speed, own_speed_pair)
    return torch.stack(
        (
            own_speed_pair,
            own[..., 4, None].expand(-1, -1, 2),
            relation.leader_gap_m[:, 1:, None].clamp_max(120.0).expand(-1, -1, 2),
            own_speed_pair - leader_speed_pair,
            front_gap,
            front_speed - leader_speed_pair,
            rear_gap,
            rear_speed - own_speed_pair,
        ),
        dim=-1,
    )


def lane_change_clearance(
    states: torch.Tensor,
    valid: torch.Tensor,
    map_polylines: torch.Tensor,
    map_valid: torch.Tensor,
    relation: DynamicInteractionRelations,
    *,
    times_s: tuple[float, ...] = (0.0, 1.5, 3.0),
) -> torch.Tensor:
    """Target-lane occupancy at specified future maneuver times."""
    lane_y, lane_present, lane_width = torch_map_lane_geometry(map_polylines, map_valid)
    target_index = relation.adjacent_lane_index[:, 1:]
    safe_index = target_index.clamp_min(0)
    target_y = torch.gather(lane_y[:, None].expand(-1, 6, -1), 2, safe_index)
    target_width = torch.gather(lane_width[:, None].expand(-1, 6, -1), 2, safe_index)
    npc = states[:, 1:]
    other = states[:, None]
    dx = other[..., 0] - npc[:, :, None, 0]
    own = torch.arange(7, device=states.device)[None, None] == (
        torch.arange(6, device=states.device)[None, :, None] + 1
    )
    clear = target_index.ge(0) & torch.gather(
        lane_present[:, None].expand(-1, 6, -1), 2, safe_index
    )
    for time in times_s:
        future_y = other[..., 1] + other[..., 3] * time
        future_dx = dx + (other[..., 2] - npc[:, :, None, 2]) * time
        occupied = (
            valid[:, None, :, None]
            & ~own[..., None]
            & (
                (future_y[..., None] - target_y[..., None, :]).abs()
                < 0.5 * target_width[..., None, :]
            )
            & future_dx[..., None].gt(-18.0)
            & future_dx[..., None].lt(12.0)
        ).any(dim=2)
        clear &= ~occupied
    return clear
