"""Causal, per-agent authority routing for online world-model responses."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .scene_geometry import (
    torch_adjacent_lane_index,
    torch_map_lane_geometry,
    torch_nearest_lane_geometry,
)

ROLE_NONE = 0
ROLE_SAME_LANE_FOLLOWER = 1
ROLE_CUTIN_CONFLICT = 2
ROLE_ADJACENT_FOLLOWER = 3


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


@dataclass(frozen=True)
class InfluenceGraphState:
    authority: torch.Tensor
    role: torch.Tensor
    parent: torch.Tensor
    phase: torch.Tensor
    age_frames: torch.Tensor
    safe_frames: torch.Tensor
    recovery_remaining: torch.Tensor
    direct: torch.Tensor
    secondary: torch.Tensor
    predicted_ttc_s: torch.Tensor
    predicted_min_gap_m: torch.Tensor

    @classmethod
    def empty(cls, batch: int, *, device: torch.device) -> "InfluenceGraphState":
        shape = (batch, 6)
        zeros = torch.zeros(shape, device=device)
        return cls(
            authority=zeros,
            role=zeros.long(),
            parent=torch.full(shape, -1, device=device, dtype=torch.long),
            phase=zeros.long(),
            age_frames=zeros.long(),
            safe_frames=zeros.long(),
            recovery_remaining=zeros.long(),
            direct=zeros.bool(),
            secondary=zeros.bool(),
            predicted_ttc_s=torch.full(shape, float("inf"), device=device),
            predicted_min_gap_m=torch.full(shape, float("inf"), device=device),
        )


class CausalInfluenceGraph:
    """Build direct influence from already-realized states.

    Coordinates in the current highD bridge are road aligned, so longitudinal
    and lateral tests use x/y directly.  The graph deliberately receives no
    intervention label or pending ego command: candidate authority is a
    function only of traffic already realised by the current rollout.
    """

    def __init__(
        self,
        *,
        radius_m: float = 50.0,
        prediction_horizon_s: float = 1.5,
        lane_half_width_m: float = 1.8,
        release_ttc_s: float = 4.0,
        stable_release_frames: int = 13,
        recovery_frames: int = 15,
        activation_brake_mps2: float = 0.5,
    ) -> None:
        self.radius_m = float(radius_m)
        self.prediction_horizon_s = float(prediction_horizon_s)
        self.lane_half_width_m = float(lane_half_width_m)
        self.release_ttc_s = float(release_ttc_s)
        self.stable_release_frames = int(stable_release_frames)
        self.recovery_frames = int(recovery_frames)
        self.activation_brake_mps2 = max(float(activation_brake_mps2), 0.0)

    @staticmethod
    def _pair_metrics(
        parent: torch.Tensor,
        child: torch.Tensor,
        parent_length: torch.Tensor | None = None,
        child_length: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        dx = child[..., 0] - parent[..., 0]
        dy = child[..., 1] - parent[..., 1]
        closing = child[..., 2] - parent[..., 2]
        pair_length = (
            torch.full_like(dx, 4.8)
            if parent_length is None or child_length is None
            else 0.5 * (parent_length + child_length)
        )
        gap = -dx - pair_length
        ttc = torch.where(
            (gap > 0.0) & (closing > 1.0e-4),
            gap / closing.clamp_min(1.0e-4),
            torch.full_like(gap, float("inf")),
        )
        return dx, dy, gap, closing, ttc

    def update(
        self,
        current: torch.Tensor,
        valid: torch.Tensor,
        history: torch.Tensor,
        previous: InfluenceGraphState | None,
        previous_background_actions: torch.Tensor | None = None,
        *,
        relations: DynamicInteractionRelations | None = None,
        vehicle_dimensions: torch.Tensor | None = None,
    ) -> InfluenceGraphState:
        batch = len(current)
        old = previous or InfluenceGraphState.empty(batch, device=current.device)
        ego = current[:, :1]
        npc = current[:, 1:]
        dimensions = (
            torch.full(
                (*current.shape[:2], 2), 0.0, dtype=current.dtype, device=current.device
            )
            if vehicle_dimensions is None
            else torch.as_tensor(
                vehicle_dimensions, dtype=current.dtype, device=current.device
            )
        )
        if vehicle_dimensions is None:
            dimensions[..., 0] = 4.8
            dimensions[..., 1] = 1.8
        if dimensions.shape != (*current.shape[:2], 2):
            raise ValueError("vehicle_dimensions must have shape [batch,7,2]")
        dx, dy, gap, closing, ttc = self._pair_metrics(
            ego, npc, dimensions[:, :1, 0], dimensions[:, 1:, 0]
        )
        distance = torch.sqrt(dx.square() + dy.square())
        same_lane = dy.abs() < 0.5 * (dimensions[:, :1, 1] + dimensions[:, 1:, 1])
        rear_sector = (dx <= 5.0) & (dx >= -self.radius_m) & (distance <= self.radius_m)

        # Constant-velocity swept-corridor test.  It catches adjacent cut-ins
        # without consuming a planned or future ego command.
        rel_vy = npc[..., 3] - ego[..., 3]
        future_dy = dy + rel_vy * self.prediction_horizon_s
        lane_overlap = torch.minimum(dy.abs(), future_dy.abs()) < 0.5 * (
            dimensions[:, :1, 1] + dimensions[:, 1:, 1]
        )
        future_dx = dx + (npc[..., 2] - ego[..., 2]) * self.prediction_horizon_s
        swept_gap = torch.minimum(dx.abs(), future_dx.abs()) - 0.5 * (
            dimensions[:, :1, 0] + dimensions[:, 1:, 0]
        )
        adjacent = (~same_lane) & lane_overlap & (distance <= self.radius_m)
        following_risk = (
            same_lane
            & (dx < 0.0)
            & (gap > 0.0)
            & ((ttc < self.release_ttc_s) | (gap < 30.0))
        )
        cutin_risk = adjacent & (swept_gap < 12.0)
        if relations is not None:
            relation_leader = relations.leader_index[:, 1:]
            relation_crossing = relations.crossing_conflict[:, 1:, 0]
            same_lane = relations.lane_index[:, 1:].eq(relations.lane_index[:, :1])
            adjacent = relation_crossing
            ego_is_leader = relation_leader.eq(0)
            gap = torch.where(ego_is_leader, relations.leader_gap_m[:, 1:], gap)
            ttc = torch.where(ego_is_leader, relations.leader_ttc_s[:, 1:], ttc)
            following_risk = relation_leader.eq(0) & (
                (relations.leader_ttc_s[:, 1:] < self.release_ttc_s)
                | (relations.leader_gap_m[:, 1:] < 30.0)
            )
            cutin_risk = relation_crossing & (swept_gap < 12.0)
        # Scope is intentionally broader than an emergency trigger.  It says
        # which local relations a residual policy may inspect; the policy can
        # still select an exact zero correction when HiQR is already suitable.
        # Crucially, no logged/synthetic/ADS experiment class enters here.
        geometric_direct = valid[:, 1:] & (
            following_risk | cutin_risk | (rear_sector & same_lane)
        )
        # A local follower relation is eligible, but a response episode opens
        # only after braking has appeared in the realized ego state.  This is
        # the preceding transition's committed control, never the pending
        # command for the current transition.
        realized_braking = current[:, :1, 4] <= -self.activation_brake_mps2
        direct = geometric_direct & realized_braking

        role = torch.where(
            direct & same_lane,
            torch.full_like(old.role, ROLE_SAME_LANE_FOLLOWER),
            torch.where(
                direct & adjacent,
                torch.full_like(old.role, ROLE_CUTIN_CONFLICT),
                torch.zeros_like(old.role),
            ),
        )
        role = torch.where(
            direct & rear_sector & ~same_lane & ~adjacent,
            torch.full_like(role, ROLE_ADJACENT_FOLLOWER),
            role,
        )
        parent = torch.where(
            direct, torch.zeros_like(old.parent), torch.full_like(old.parent, -1)
        )

        # NPC-to-NPC responses use physical leaders in the online controller,
        # not a second-level authority label in this ego stimulus graph.
        secondary = torch.zeros_like(direct)

        conflict = direct | secondary
        # Release only after conflict has cleared and distance has been opening
        # for 0.5 s.  History contains realized states only.
        if history.shape[1] >= 2:
            old_ego = history[:, -2, :1]
            old_npc = history[:, -2, 1:]
            old_distance = torch.linalg.vector_norm(
                old_npc[..., :2] - old_ego[..., :2], dim=-1
            )
            opening = distance > old_distance
        else:
            opening = torch.zeros_like(conflict)
        # Once the realized braking stimulus has ended, a non-closing or
        # comfortably large-TTC relation can accumulate release even when
        # Euclidean distance is momentarily flat.  Requiring strict opening
        # alone can retain authority for the rest of a finite rollout.
        safe = ~conflict & (opening | (closing <= 0.0) | (ttc >= self.release_ttc_s))
        # Phase 1 covers both engaged and clearing.  Keep it while safety is
        # accumulating; otherwise the first clear frame would reset the phase
        # and the configured release counter could never reach its threshold.
        safe_frames = torch.where(
            safe & old.phase.eq(1),
            old.safe_frames + 1,
            torch.where(conflict, torch.zeros_like(old.safe_frames), old.safe_frames),
        )
        begin_recovery = old.phase.eq(1) & (safe_frames >= self.stable_release_frames)
        continuing_recovery = old.phase.eq(2) & (old.recovery_remaining > 1)
        continuing_clearing = old.phase.eq(1) & ~conflict & ~begin_recovery
        phase = torch.where(
            conflict,
            torch.ones_like(old.phase),
            torch.where(
                begin_recovery | continuing_recovery,
                torch.full_like(old.phase, 2),
                torch.where(
                    continuing_clearing,
                    torch.ones_like(old.phase),
                    torch.zeros_like(old.phase),
                ),
            ),
        )
        recovery = torch.where(
            conflict,
            torch.zeros_like(old.recovery_remaining),
            torch.where(
                begin_recovery,
                torch.full_like(old.recovery_remaining, self.recovery_frames),
                torch.where(
                    continuing_recovery,
                    old.recovery_remaining - 1,
                    torch.zeros_like(old.recovery_remaining),
                ),
            ),
        )
        base_authority = direct.to(distance.dtype)
        authority = (
            torch.where(
                phase.eq(2),
                recovery.to(distance) / float(max(self.recovery_frames, 1)),
                torch.where(continuing_clearing, old.authority, base_authority),
            )
            * valid[:, 1:].float()
        )
        # Recovery is still part of the same causal episode.  Retaining the
        # last role and parent keeps its feature semantics stable while the
        # continuous authority envelope returns the vehicle to HiQR.
        retained_episode = phase.ne(0)
        role = torch.where(retained_episode & role.eq(ROLE_NONE), old.role, role)
        parent = torch.where(retained_episode & parent.lt(0), old.parent, parent)
        direct = torch.where(
            retained_episode & ~direct & old.direct, old.direct, direct
        )
        secondary = torch.where(
            retained_episode & ~secondary & old.secondary, old.secondary, secondary
        )
        age = torch.where(
            phase.ne(0), old.age_frames + 1, torch.zeros_like(old.age_frames)
        )
        predicted_ttc = ttc
        min_gap = torch.where(
            direct, torch.minimum(gap, swept_gap), torch.full_like(gap, float("inf"))
        )
        return InfluenceGraphState(
            authority=authority,
            role=role,
            parent=parent,
            phase=phase,
            age_frames=age,
            safe_frames=safe_frames,
            recovery_remaining=recovery,
            direct=direct,
            secondary=secondary,
            predicted_ttc_s=predicted_ttc,
            predicted_min_gap_m=min_gap,
        )
