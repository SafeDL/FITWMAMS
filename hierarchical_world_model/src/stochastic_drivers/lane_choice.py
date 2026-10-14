"""Current-state adjacent-lane features and occupancy diagnostics."""

from __future__ import annotations

import torch

from ..influence_graph import DynamicInteractionRelations

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
