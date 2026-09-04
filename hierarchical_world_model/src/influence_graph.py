"""Causal, per-agent authority assignment for longitudinal NPC reactions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


ROLE_NONE = 0
ROLE_SAME_LANE_FOLLOWER = 1
ROLE_CUTIN_CONFLICT = 2
ROLE_ADJACENT_FOLLOWER = 3
ROLE_SECONDARY_FOLLOWER = 4


def dynamic_candidate_scene_mask(
    states: np.ndarray,
    valid: np.ndarray,
    *,
    rows: np.ndarray | None = None,
    start_frame: int = 24,
    stop_frame: int = 173,
    radius_m: float = 50.0,
    prediction_horizon_s: float = 1.5,
    batch_size: int = 2048,
) -> np.ndarray:
    """Offline-only scene selection using the union over observed frames.

    Future logged states are used only to decide whether a recording belongs
    in the training/evaluation population.  They are never exposed to the
    online influence graph or controller.
    """
    values, present = np.asarray(states), np.asarray(valid, bool)
    indices = np.arange(len(values), dtype=np.int64) if rows is None else np.asarray(rows, np.int64)
    selected = np.zeros(len(indices), dtype=bool)
    stop = min(int(stop_frame), values.shape[1])
    for begin in range(0, len(indices), max(int(batch_size), 1)):
        end = min(begin + max(int(batch_size), 1), len(indices))
        source = indices[begin:end]
        window = values[source, int(start_frame):stop]
        mask = present[source, int(start_frame):stop, 1:]
        relative = window[:, :, 1:, :2] - window[:, :, :1, :2]
        distance = np.linalg.norm(relative, axis=-1)
        rear_sector = (
            (relative[..., 0] <= 5.0) & (relative[..., 0] >= -float(radius_m))
            & (distance <= float(radius_m))
        )
        rel_vy = window[:, :, 1:, 3] - window[:, :, :1, 3]
        future_dy = relative[..., 1] + rel_vy * float(prediction_horizon_s)
        swept = (
            (np.abs(relative[..., 1]) >= 1.8)
            & (np.minimum(np.abs(relative[..., 1]), np.abs(future_dy)) < 1.8)
            & (distance <= float(radius_m))
        )
        selected[begin:end] = (mask & (rear_sector | swept)).any(axis=(1, 2))
    return selected


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
    """Build a one-hop ego/NPC influence graph from already-realized states.

    Coordinates in the current highD bridge are road aligned, so longitudinal
    and lateral tests use x/y directly.  No pending ego command is accepted by
    this API: ``armed`` is set only after the preceding HighwayEnv step.
    """

    def __init__(
        self,
        *,
        radius_m: float = 50.0,
        secondary_radius_m: float = 35.0,
        prediction_horizon_s: float = 1.5,
        lane_half_width_m: float = 1.8,
        release_ttc_s: float = 4.0,
        stable_release_frames: int = 13,
        recovery_frames: int = 15,
    ) -> None:
        self.radius_m = float(radius_m)
        self.secondary_radius_m = float(secondary_radius_m)
        self.prediction_horizon_s = float(prediction_horizon_s)
        self.lane_half_width_m = float(lane_half_width_m)
        self.release_ttc_s = float(release_ttc_s)
        self.stable_release_frames = int(stable_release_frames)
        self.recovery_frames = int(recovery_frames)

    @staticmethod
    def _pair_metrics(parent: torch.Tensor, child: torch.Tensor) -> tuple[torch.Tensor, ...]:
        dx = child[..., 0] - parent[..., 0]
        dy = child[..., 1] - parent[..., 1]
        closing = child[..., 2] - parent[..., 2]
        gap = -dx - 4.8
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
        armed: torch.Tensor,
        previous: InfluenceGraphState | None,
        previous_background_actions: torch.Tensor | None = None,
    ) -> InfluenceGraphState:
        batch = len(current)
        old = previous or InfluenceGraphState.empty(batch, device=current.device)
        ego = current[:, :1]
        npc = current[:, 1:]
        dx, dy, gap, closing, ttc = self._pair_metrics(ego, npc)
        distance = torch.sqrt(dx.square() + dy.square())
        same_lane = dy.abs() < self.lane_half_width_m
        rear_sector = (dx <= 5.0) & (dx >= -self.radius_m) & (distance <= self.radius_m)

        # Constant-velocity swept-corridor test.  It catches adjacent cut-ins
        # without consuming a planned or future ego command.
        rel_vy = npc[..., 3] - ego[..., 3]
        future_dy = dy + rel_vy * self.prediction_horizon_s
        lane_overlap = torch.minimum(dy.abs(), future_dy.abs()) < self.lane_half_width_m
        future_dx = dx + (npc[..., 2] - ego[..., 2]) * self.prediction_horizon_s
        swept_gap = torch.minimum(dx.abs(), future_dx.abs()) - 4.8
        adjacent = (~same_lane) & lane_overlap & (distance <= self.radius_m)
        following_risk = same_lane & (dx < 0.0) & (gap > 0.0) & (
            (ttc < self.release_ttc_s) | (gap < 30.0)
        )
        cutin_risk = adjacent & (swept_gap < 12.0)
        direct = armed[:, None] & valid[:, 1:] & (following_risk | cutin_risk | (rear_sector & same_lane))

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
        parent = torch.where(direct, torch.zeros_like(old.parent), torch.full_like(old.parent, -1))

        # One-hop propagation from a directly influenced vehicle to a follower
        # behind it.  A previous realized brake is sufficient evidence that
        # the parent's corrected motion can matter to the child.
        secondary = torch.zeros_like(direct)
        secondary_ttc = torch.full_like(ttc, float("inf"))
        secondary_gap = torch.full_like(gap, float("inf"))
        for parent_slot in range(6):
            parent_active = direct[:, parent_slot]
            if not bool(parent_active.any()):
                continue
            leader = npc[:, parent_slot : parent_slot + 1]
            child_dx, child_dy, child_gap, child_closing, child_ttc = self._pair_metrics(leader, npc)
            parent_braking = (
                torch.zeros(batch, dtype=torch.bool, device=current.device)
                if previous_background_actions is None
                else previous_background_actions[:, parent_slot, 0] < -0.5
            )
            candidate = (
                parent_active[:, None]
                & valid[:, 1:]
                & (torch.arange(6, device=current.device)[None] != parent_slot)
                & (child_dx < 0.0)
                & (child_gap > 0.0)
                & (child_gap < self.secondary_radius_m)
                & (child_dy.abs() < self.lane_half_width_m)
                & ((child_ttc < self.release_ttc_s) | parent_braking[:, None])
            )
            new_edge = candidate & ~direct & ~secondary
            secondary |= new_edge
            parent = torch.where(new_edge, torch.full_like(parent, parent_slot + 1), parent)
            secondary_ttc = torch.where(new_edge, child_ttc, secondary_ttc)
            secondary_gap = torch.where(new_edge, child_gap, secondary_gap)
        role = torch.where(secondary, torch.full_like(role, ROLE_SECONDARY_FOLLOWER), role)

        conflict = direct | secondary
        # Release only after conflict has cleared and distance has been opening
        # for 0.5 s.  History contains realized states only.
        if history.shape[1] >= 2:
            old_ego = history[:, -2, :1]
            old_npc = history[:, -2, 1:]
            old_distance = torch.linalg.vector_norm(old_npc[..., :2] - old_ego[..., :2], dim=-1)
            opening = distance > old_distance
        else:
            opening = torch.zeros_like(conflict)
        safe = ~conflict & opening
        safe_frames = torch.where(safe & old.phase.eq(1), old.safe_frames + 1, torch.zeros_like(old.safe_frames))
        begin_recovery = old.phase.eq(1) & (safe_frames >= self.stable_release_frames)
        continuing_recovery = old.phase.eq(2) & (old.recovery_remaining > 1)
        phase = torch.where(
            conflict,
            torch.ones_like(old.phase),
            torch.where(begin_recovery | continuing_recovery, torch.full_like(old.phase, 2), torch.zeros_like(old.phase)),
        )
        recovery = torch.where(
            conflict,
            torch.zeros_like(old.recovery_remaining),
            torch.where(
                begin_recovery,
                torch.full_like(old.recovery_remaining, self.recovery_frames),
                torch.where(continuing_recovery, old.recovery_remaining - 1, torch.zeros_like(old.recovery_remaining)),
            ),
        )
        base_authority = torch.where(conflict, torch.ones_like(distance), torch.zeros_like(distance))
        authority = torch.where(
            phase.eq(2),
            recovery.to(distance) / float(max(self.recovery_frames, 1)),
            base_authority,
        ) * valid[:, 1:].float()
        # Recovery is still part of the same causal episode.  Retaining the
        # last role and parent keeps its feature semantics stable while the
        # continuous authority envelope returns the vehicle to HiQR.
        recovering = phase.eq(2)
        role = torch.where(recovering & role.eq(ROLE_NONE), old.role, role)
        parent = torch.where(recovering & parent.lt(0), old.parent, parent)
        age = torch.where(phase.ne(0), old.age_frames + 1, torch.zeros_like(old.age_frames))
        predicted_ttc = torch.where(secondary, secondary_ttc, ttc)
        min_gap = torch.where(
            secondary,
            secondary_gap,
            torch.where(direct, torch.minimum(gap, swept_gap), torch.full_like(gap, float("inf"))),
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
