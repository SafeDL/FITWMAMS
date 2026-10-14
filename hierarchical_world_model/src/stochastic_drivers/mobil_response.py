"""MOBIL lane-candidate evaluation using the episode's MA-IDM drivers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..influence_graph import DynamicInteractionRelations
from .idm_population import torch_idm


@dataclass(frozen=True)
class MOBILDecision:
    incentive_mps2: torch.Tensor
    safe: torch.Tensor
    selected: torch.Tensor
    best_side: torch.Tensor
    target_lane_index: torch.Tensor


def evaluate_mobil_candidates(
    states: torch.Tensor,
    valid: torch.Tensor,
    theta: torch.Tensor,
    relations: DynamicInteractionRelations,
    *,
    vehicle_lengths_m: torch.Tensor | None = None,
    politeness: float | torch.Tensor = 0.3,
    minimum_incentive_mps2: float | torch.Tensor = 0.3,
    safe_braking_mps2: float | torch.Tensor = 4.0,
    plan_keep_bias_mps2: float | torch.Tensor = 0.1,
) -> MOBILDecision:
    """Evaluate lower-y and higher-y candidate lanes for all six NPCs."""
    batch = states.shape[0]
    if states.shape[1:] != (7, 6) or valid.shape != states.shape[:2]:
        raise ValueError("MOBIL expects states [batch,7,6] and valid [batch,7]")
    if theta.shape != (batch, 6, 5):
        raise ValueError("theta must have shape [batch,6,5]")

    def actor_parameter(value: float | torch.Tensor, name: str) -> torch.Tensor:
        result = torch.as_tensor(value, dtype=states.dtype, device=states.device)
        if result.ndim == 0:
            return result.expand(batch, 6)
        if result.shape == (6,):
            return result[None].expand(batch, -1)
        if result.shape != (batch, 6):
            raise ValueError(f"{name} must be scalar, [6], or [batch,6]")
        return result

    politeness_value = actor_parameter(politeness, "politeness")
    incentive_threshold = actor_parameter(
        minimum_incentive_mps2, "minimum_incentive_mps2"
    )
    safe_braking = actor_parameter(safe_braking_mps2, "safe_braking_mps2")
    plan_keep_bias = actor_parameter(plan_keep_bias_mps2, "plan_keep_bias_mps2")
    lengths = (
        torch.full(states.shape[:2], 4.8, dtype=states.dtype, device=states.device)
        if vehicle_lengths_m is None
        else torch.as_tensor(vehicle_lengths_m, dtype=states.dtype, device=states.device)
    )
    actor_index = torch.arange(1, 7, device=states.device)[None].expand(batch, -1)

    def acceleration(follower_actor: torch.Tensor, leader_actor: torch.Tensor) -> torch.Tensor:
        has_follower = follower_actor.gt(0)
        follower = follower_actor.clamp(1, 6)
        leader = leader_actor.clamp_min(0)
        follower_state = torch.gather(
            states, 1, follower[..., None].expand(-1, -1, 6)
        )
        leader_state = torch.gather(
            states, 1, leader[..., None].expand(-1, -1, 6)
        )
        follower_speed = torch.linalg.vector_norm(follower_state[..., 2:4], dim=-1)
        leader_speed = torch.linalg.vector_norm(leader_state[..., 2:4], dim=-1)
        pair_length = 0.5 * (
            torch.gather(lengths, 1, follower) + torch.gather(lengths, 1, leader)
        )
        gap = leader_state[..., 0] - follower_state[..., 0] - pair_length
        follower_theta = torch.gather(
            theta, 1, (follower - 1)[..., None].expand(-1, -1, 5)
        )
        free = leader_actor.lt(0)
        value = torch_idm(
            torch.where(free, torch.full_like(gap, 1.0e6), gap),
            follower_speed,
            torch.where(free, follower_speed, leader_speed),
            follower_theta,
        )
        return torch.where(has_follower, value, torch.zeros_like(value))

    current_leader = relations.leader_index[:, 1:]
    current_acceleration = acceleration(actor_index, current_leader)
    incentives = torch.full((batch, 6, 2), -float("inf"), dtype=states.dtype, device=states.device)
    safe = torch.zeros((batch, 6, 2), dtype=torch.bool, device=states.device)
    for side_index in range(2):
        front = relations.adjacent_front_index[:, 1:, side_index]
        rear = relations.adjacent_rear_index[:, 1:, side_index]
        lane_available = relations.adjacent_lane_index[:, 1:, side_index].ge(0)
        own_after = acceleration(actor_index, front)
        rear_before_leader = torch.gather(
            relations.leader_index, 1, rear.clamp_min(0)
        )
        rear_before = acceleration(rear, rear_before_leader)
        rear_after = acceleration(rear, actor_index)

        # The nearest current-lane follower is the actor whose resolved leader
        # is this NPC.  Its benefit after the lane is vacated is included.
        followers = relations.leader_index[:, 1:, None].eq(actor_index[:, None])
        follower_dx = states[:, 1:, 0, None] - states[:, None, 1:, 0]
        old_rear_score = torch.where(
            followers & follower_dx.lt(0.0),
            -follower_dx,
            torch.full_like(follower_dx, float("inf")),
        )
        old_rear_local = old_rear_score.argmin(dim=1)
        has_old_rear = torch.isfinite(old_rear_score.amin(dim=1))
        old_rear = torch.where(has_old_rear, old_rear_local + 1, torch.full_like(old_rear_local, -1))
        old_before = acceleration(old_rear, actor_index)
        old_after = acceleration(old_rear, current_leader)

        value = (
            own_after - current_acceleration
            + politeness_value * ((rear_after - rear_before) + (old_after - old_before))
            - plan_keep_bias
        )
        candidate_safe = lane_available & (
            rear.lt(0) | (rear_after >= -safe_braking)
        )
        safe[..., side_index] = candidate_safe
        incentives[..., side_index] = torch.where(
            candidate_safe, value, torch.full_like(value, -float("inf"))
        )
    best_local = incentives.argmax(dim=-1)
    best_value = torch.gather(incentives, 2, best_local[..., None]).squeeze(-1)
    selected = valid[:, 1:] & torch.isfinite(best_value) & (
        best_value >= incentive_threshold
    )
    best_side = torch.where(
        selected,
        torch.where(best_local.eq(0), -torch.ones_like(best_local), torch.ones_like(best_local)),
        torch.zeros_like(best_local),
    )
    target_lane = torch.gather(
        relations.adjacent_lane_index[:, 1:], 2, best_local[..., None]
    ).squeeze(-1)
    target_lane = torch.where(selected, target_lane, torch.full_like(target_lane, -1))
    return MOBILDecision(incentives, safe, selected, best_side, target_lane)
