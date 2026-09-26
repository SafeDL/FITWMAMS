"""Single-pass, causal MA-IDM safety response around the current HiQR action.

The controller never consumes a precomputed no-intervention trajectory. HiQR
remains the action source unless the *realized* leader/follower state predicts
an unsafe gap and its action does not already brake sufficiently.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from external_model_baselines.models.bayesian_ma_idm.src.model import (
    load_posterior,
    sample_driver_joint,
)

from ..reaction_controller import (
    ReactionController,
    ReactionControllerContext,
    ReactionControllerOutput,
)

DEFAULT_MA_IDM_POSTERIOR = (
    Path(__file__).resolve().parents[3]
    / "external_model_baselines/models/bayesian_ma_idm/evidence/deployment/ma_idm_all_251.npz"
)


def sample_population_theta_from_world(
    posterior_path: str | Path, exogenous_state: Any, *, slots: int = 6
) -> np.ndarray:
    """Draw replayable fixed episode parameters from explicit world randomness."""
    innovations = np.asarray(exogenous_state.agent_response_innovations)
    if innovations.ndim != 4 or innovations.shape[1] < 1 or innovations.shape[2] < slots + 1:
        raise ValueError("world innovations do not cover all NPC driver slots")
    posterior = load_posterior(posterior_path)
    theta = np.empty((innovations.shape[0], slots, 5), np.float32)
    for row in range(innovations.shape[0]):
        for slot in range(slots):
            key = np.asarray(innovations[row, 0, slot + 1], dtype="<f4").tobytes()
            digest = hashlib.sha256(b"online_ma_idm_theta_v1:" + key).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            theta[row, slot] = sample_driver_joint(posterior, rng)[:5]
    return theta


def nearest_leader_observation(
    states: torch.Tensor,
    valid: torch.Tensor,
    *,
    lane_half_width_m: float = 1.8,
    vehicle_length_m: float = 4.8,
    prediction_horizon_s: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return current gap, follower/leader speeds and leader slot per NPC."""
    if states.ndim != 3 or states.shape[1:] != (7, 6):
        raise ValueError("states must have shape [batch,7,6]")
    if valid.shape != states.shape[:2]:
        raise ValueError("valid must have shape [batch,7]")
    follower = states[:, 1:]
    dx = states[:, None, :, 0] - follower[:, :, None, 0]
    relative_y = states[:, None, :, 1] - follower[:, :, None, 1]
    future_relative_y = relative_y + (
        states[:, None, :, 3] - follower[:, :, None, 3]
    ) * float(prediction_horizon_s)
    lateral_overlap = (
        torch.minimum(relative_y.abs(), future_relative_y.abs()).lt(float(lane_half_width_m))
        | (relative_y * future_relative_y <= 0.0)
    )
    candidates = (
        valid[:, None, :]
        & valid[:, 1:, None]
        & dx.gt(0.0)
        & lateral_overlap
    )
    identity = torch.arange(6, device=states.device)[None, :, None] + 1
    candidates &= torch.arange(7, device=states.device)[None, None, :].ne(identity)
    speed_all = torch.linalg.vector_norm(states[..., 2:4], dim=-1)
    follower_speed = speed_all[:, 1:]
    horizon = float(prediction_horizon_s)
    projected_gap = (
        dx - float(vehicle_length_m)
        + (speed_all[:, None] - follower_speed[..., None]) * horizon
        + 0.5 * (states[:, None, :, 4] - follower[..., 4, None]) * horizon * horizon
    )
    unsafe_projection = candidates & projected_gap.lt(
        torch.maximum(
            torch.full_like(follower_speed[..., None], 2.0),
            0.1 * follower_speed[..., None],
        )
    )
    # The geometrically closest prospective leader need not be the dangerous
    # one: a target-lane car pulling away can mask a slightly farther but
    # hard-braking current-lane leader. Prefer the worst projected clearance
    # whenever one is unsafe, then fall back to nearest distance.
    nearest_distance = torch.where(candidates, dx, torch.full_like(dx, float("inf")))
    risk_distance = torch.where(
        unsafe_projection, projected_gap, torch.full_like(dx, float("inf"))
    )
    leader_index = torch.where(
        unsafe_projection.any(dim=-1),
        risk_distance.argmin(dim=-1), nearest_distance.argmin(dim=-1),
    )
    has_leader = candidates.any(dim=-1)
    selected_dx = torch.gather(dx, 2, leader_index[..., None]).squeeze(-1)
    leader_speed = torch.gather(speed_all, 1, leader_index)
    gap = torch.where(
        has_leader,
        (selected_dx - float(vehicle_length_m)).clamp_min(0.05),
        torch.full_like(selected_dx, 1.0e6),
    )
    leader_speed = torch.where(has_leader, leader_speed, follower_speed)
    leader_index = torch.where(has_leader, leader_index, torch.full_like(leader_index, -1))
    return gap, follower_speed, leader_speed, leader_index


def torch_idm(
    gap_m: torch.Tensor,
    speed_mps: torch.Tensor,
    leader_speed_mps: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """MA-IDM longitudinal equation for the sampled episode driver."""
    v0, s0, headway, alpha, beta = theta.unbind(dim=-1)
    speed = speed_mps.clamp_min(0.0)
    closing = speed - leader_speed_mps
    dynamic_gap = speed * headway + speed * closing / (
        2.0 * torch.sqrt((alpha * beta).clamp_min(1.0e-8))
    )
    desired_gap = s0 + dynamic_gap
    return alpha * (
        1.0
        - (speed / v0.clamp_min(1.0e-4)).pow(4)
        - (desired_gap / gap_m.clamp_min(1.0e-3)).pow(2)
    )


def planned_adjacent_lane_intent(
    origin_y: torch.Tensor,
    terminal_y: torch.Tensor,
    map_polylines: torch.Tensor | None,
    map_polyline_valid: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Infer a map-grounded adjacent lane from the generated terminal plan.

    A partially completed lateral manoeuvre is not automatically a command
    to snap a vehicle one nominal lane width from its *current* position.
    """
    if map_polylines is None or map_polyline_valid is None:
        displacement = terminal_y - origin_y
        intent = (displacement.abs() > 1.8) & (displacement.abs() < 5.4)
        return intent, origin_y, origin_y + torch.sign(displacement) * 3.6
    lane_y = map_polylines[..., 0, 1]
    lane_present = map_polyline_valid.any(dim=-1)
    missing = torch.full_like(lane_y, float("inf"))
    source_distance = torch.where(
        lane_present[:, None], (lane_y[:, None] - origin_y[..., None]).abs(),
        missing[:, None],
    )
    target_distance = torch.where(
        lane_present[:, None], (lane_y[:, None] - terminal_y[..., None]).abs(),
        missing[:, None],
    )
    source_index = source_distance.argmin(dim=-1)
    target_index = target_distance.argmin(dim=-1)
    source_y = torch.gather(lane_y, 1, source_index)
    target_y = torch.gather(lane_y, 1, target_index)
    lane_distance = (target_y - source_y).abs()
    intent = (
        lane_present.any(dim=1)[:, None]
        & source_index.ne(target_index)
        & lane_distance.gt(1.8)
        & lane_distance.lt(5.4)
        & (origin_y - source_y).abs().lt(1.2)
        & (terminal_y - target_y).abs().lt(1.2)
    )
    return intent, source_y, target_y


def planned_lateral_position_at_x(
    origin_xy: torch.Tensor,
    path_xy: torch.Tensor,
    query_x: torch.Tensor,
) -> torch.Tensor:
    """Interpolate a generated lane path at realized longitudinal progress."""
    if origin_xy.ndim != 3 or origin_xy.shape[-1] != 2:
        raise ValueError("origin_xy must have shape [batch,npc,2]")
    if path_xy.ndim != 4 or path_xy.shape[0] != origin_xy.shape[0] or path_xy.shape[2:] != origin_xy.shape[1:]:
        raise ValueError("path_xy must have shape [batch,time,npc,2]")
    if query_x.shape != origin_xy.shape[:2]:
        raise ValueError("query_x must have shape [batch,npc]")
    path = torch.cat((origin_xy[:, None], path_xy), dim=1).permute(0, 2, 1, 3)
    path_x, path_y = path[..., 0], path[..., 1]
    upper = (path_x <= query_x[..., None]).sum(dim=-1).clamp(1, path_x.shape[-1] - 1)
    lower = upper - 1
    lower_x = torch.gather(path_x, -1, lower[..., None]).squeeze(-1)
    upper_x = torch.gather(path_x, -1, upper[..., None]).squeeze(-1)
    lower_y = torch.gather(path_y, -1, lower[..., None]).squeeze(-1)
    upper_y = torch.gather(path_y, -1, upper[..., None]).squeeze(-1)
    fraction = ((query_x - lower_x) / (upper_x - lower_x).clamp_min(1.0e-4)).clamp(0.0, 1.0)
    lateral = lower_y + fraction * (upper_y - lower_y)
    return torch.where(query_x >= path_x[..., -1], path_y[..., -1], lateral)


class OnlineMAIDMController(ReactionController):
    """Online traffic-aware longitudinal response for every NPC slot.

    Posterior driver parameters are fixed within an episode. Every decision
    uses only the state at the current response boundary and the current HiQR
    action. No ADS action for a future tick or nominal rollout is available.
    """

    mode = "online_ma_idm"

    def __init__(
        self,
        theta: np.ndarray | torch.Tensor,
        *,
        lane_half_width_m: float = 1.8,
        vehicle_length_m: float = 4.8,
        prediction_horizon_s: float = 2.0,
        minimum_clearance_m: float = 2.0,
        speed_clearance_s: float = 0.1,
        minimum_acceleration_mps2: float = -8.0,
        maximum_acceleration_mps2: float = 4.0,
        correction_jerk_limit_mps3: float = 12.0,
        native_dt_s: float = 0.04,
    ) -> None:
        super().__init__()
        values = torch.as_tensor(theta, dtype=torch.float32)
        if values.ndim not in (2, 3) or tuple(values.shape[-2:]) != (6, 5):
            raise ValueError("theta must have shape [6,5] or [batch,6,5]")
        if min(prediction_horizon_s, minimum_clearance_m, speed_clearance_s,
               correction_jerk_limit_mps3, native_dt_s) <= 0.0:
            raise ValueError("response horizon, clearance and timing must be positive")
        self.register_buffer("theta", values.clone())
        self.lane_half_width_m = float(lane_half_width_m)
        self.vehicle_length_m = float(vehicle_length_m)
        self.prediction_horizon_s = float(prediction_horizon_s)
        self.minimum_clearance_m = float(minimum_clearance_m)
        self.speed_clearance_s = float(speed_clearance_s)
        self.minimum_acceleration_mps2 = float(minimum_acceleration_mps2)
        self.maximum_acceleration_mps2 = float(maximum_acceleration_mps2)
        self.correction_jerk_limit_mps3 = float(correction_jerk_limit_mps3)
        self.native_dt_s = float(native_dt_s)
        self._lane_repair_latched: torch.Tensor | None = None
        self._spatial_path_latched: torch.Tensor | None = None
        self._longitudinal_response_latched: torch.Tensor | None = None
        self._disturbance_latched: torch.Tensor | None = None
        self._disturbance_leader_index: torch.Tensor | None = None
        self._previous_base_ax: torch.Tensor | None = None
        self._passing_cutin_latched: torch.Tensor | None = None
        self._autonomous_lane_active: torch.Tensor | None = None
        self._autonomous_lane_ever: torch.Tensor | None = None
        self._autonomous_lane_target_y: torch.Tensor | None = None

    def snapshot_runtime(self) -> dict[str, torch.Tensor]:
        result = {}
        if self._lane_repair_latched is not None:
            result["lane_repair_latched"] = self._lane_repair_latched.detach().clone()
        if self._spatial_path_latched is not None:
            result["spatial_path_latched"] = self._spatial_path_latched.detach().clone()
        if self._longitudinal_response_latched is not None:
            result["longitudinal_response_latched"] = (
                self._longitudinal_response_latched.detach().clone()
            )
        if self._disturbance_latched is not None:
            result["disturbance_latched"] = self._disturbance_latched.detach().clone()
        if self._disturbance_leader_index is not None:
            result["disturbance_leader_index"] = (
                self._disturbance_leader_index.detach().clone()
            )
        if self._previous_base_ax is not None:
            result["previous_base_ax"] = self._previous_base_ax.detach().clone()
        if self._passing_cutin_latched is not None:
            result["passing_cutin_latched"] = self._passing_cutin_latched.detach().clone()
        if self._autonomous_lane_active is not None:
            result["autonomous_lane_active"] = self._autonomous_lane_active.detach().clone()
            result["autonomous_lane_ever"] = self._autonomous_lane_ever.detach().clone()
            result["autonomous_lane_target_y"] = self._autonomous_lane_target_y.detach().clone()
        return result

    def restore_runtime(self, state: dict[str, torch.Tensor]) -> None:
        value = state.get("lane_repair_latched")
        self._lane_repair_latched = (
            None if value is None else value.detach().clone().to(self.theta.device)
        )
        value = state.get("spatial_path_latched")
        self._spatial_path_latched = (
            None if value is None else value.detach().clone().to(self.theta.device)
        )
        value = state.get("longitudinal_response_latched")
        self._longitudinal_response_latched = (
            None if value is None else value.detach().clone().to(self.theta.device)
        )
        value = state.get("disturbance_latched")
        self._disturbance_latched = (
            None if value is None else value.detach().clone().to(self.theta.device)
        )
        value = state.get("disturbance_leader_index")
        self._disturbance_leader_index = (
            None if value is None else value.detach().clone().to(self.theta.device)
        )
        value = state.get("previous_base_ax")
        self._previous_base_ax = (
            None if value is None else value.detach().clone().to(self.theta.device)
        )
        value = state.get("passing_cutin_latched")
        self._passing_cutin_latched = (
            None if value is None else value.detach().clone().to(self.theta.device)
        )
        for key in (
            "autonomous_lane_active", "autonomous_lane_ever", "autonomous_lane_target_y",
        ):
            value = state.get(key)
            setattr(
                self, f"_{key}",
                None if value is None else value.detach().clone().to(self.theta.device),
            )

    def _planned_lane_escape(
        self,
        context: ReactionControllerContext,
        gap: torch.Tensor,
        speed: torch.Tensor,
        leader_speed: torch.Tensor,
        leader_index: torch.Tensor,
        leader_acceleration: torch.Tensor,
        base: torch.Tensor,
        safety_clearance: torch.Tensor,
    ) -> torch.Tensor:
        """Avoid braking for a gap that becomes unsafe only after a clear lane exit.

        This is a short-horizon collision check using the fixed Diffusion lane
        intent and currently realized traffic, not a no-intervention rollout.
        If the path has already fallen behind the realized longitudinal state
        or the target lane is occupied, ordinary longitudinal safety applies.
        """
        empty = torch.zeros_like(gap, dtype=torch.bool)
        if any(value is None for value in (
            context.planned_origin_xy, context.planned_current_xy,
            context.planned_terminal_xy, context.planned_path_xy,
        )):
            return empty
        npc = context.current[:, 1:]
        intent, _, target_y = planned_adjacent_lane_intent(
            context.planned_origin_xy[..., 1],
            context.planned_terminal_xy[..., 1],
            context.map_polylines, context.map_polyline_valid,
        )
        # planned_current_xy is the *next* state, so even on-plan traffic is
        # about one 25 Hz travel step behind it (roughly 1.2 m at 30 m/s).
        on_plan = (npc[..., 0] - context.planned_current_xy[..., 0]).abs() < 2.0
        horizon_steps = max(1, int(round(self.prediction_horizon_s / self.native_dt_s)))
        indices = (
            torch.arange(1, horizon_steps + 1, device=npc.device)
            + int(context.response_index)
        ).clamp_max(context.planned_path_xy.shape[1] - 1)
        times = torch.arange(1, horizon_steps + 1, device=npc.device, dtype=npc.dtype)
        times = times * self.native_dt_s
        planned_y = context.planned_path_xy.index_select(1, indices)[..., 1].permute(0, 2, 1)
        leader_y = torch.gather(context.current[..., 1], 1, leader_index.clamp_min(0))
        leader_vy = torch.gather(context.current[..., 3], 1, leader_index.clamp_min(0))
        predicted_lateral_gap = planned_y - (
            leader_y[..., None] + leader_vy[..., None] * times
        )
        predicted_longitudinal_gap = (
            gap[..., None]
            + (leader_speed - speed)[..., None] * times
            + 0.5 * (leader_acceleration - base)[..., None] * times.square()
        )
        collision_risk = (
            (predicted_longitudinal_gap < safety_clearance[..., None])
            & (predicted_lateral_gap.abs() < self.lane_half_width_m)
        ).any(dim=-1)
        # A planned exit can be later in the realized world than in the
        # frozen plan. During an observed hard brake, do not waive the rear
        # clearance merely because the planned lateral trace has already
        # cleared the leader; the current lateral motion must clear it too.
        realized_lateral_gap = (
            npc[..., 1, None] + npc[..., 3, None] * times
            - leader_y[..., None] - leader_vy[..., None] * times
        )
        collision_risk |= (
            leader_acceleration.le(-4.0)
            & (
                (predicted_longitudinal_gap < safety_clearance[..., None])
                & (realized_lateral_gap.abs() < self.lane_half_width_m)
            ).any(dim=-1)
        )
        other = context.current[:, None]
        dx = other[..., 0] - npc[:, :, None, 0]
        entry_mask = (planned_y - target_y[..., None]).abs() < self.lane_half_width_m
        steps = torch.arange(1, horizon_steps + 1, device=npc.device)
        entry_step = torch.where(
            entry_mask, steps, torch.full_like(steps, horizon_steps),
        ).amin(dim=-1)
        entry_time = entry_step.to(npc.dtype) * self.native_dt_s
        target_horizon = torch.full(
            (1, 1, 7), self.prediction_horizon_s,
            dtype=npc.dtype, device=npc.device,
        )
        # ADS may traverse a lane gradually over several seconds. Its
        # currently realized lateral velocity must reserve that crossing
        # corridor before the nearby NPC commits to the planned exit.
        target_horizon[..., 0] = max(4.0, self.prediction_horizon_s)
        future_dx = dx + target_horizon * (
            other[..., 2] - npc[:, :, None, 2]
        )
        other_y = other[..., 1]
        future_other_y = other_y + target_horizon * other[..., 3]
        entry_dx = dx + entry_time[..., None] * (
            other[..., 2] - npc[:, :, None, 2]
        )
        currently_in_target = (other_y - target_y[:, :, None]).abs() < 1.8
        target_band = (
            (torch.minimum(other_y, future_other_y) < target_y[:, :, None] + 1.8)
            & (torch.maximum(other_y, future_other_y) > target_y[:, :, None] - 1.8)
        )
        own = torch.arange(7, device=npc.device)[None, None] == (
            torch.arange(6, device=npc.device)[None, :, None] + 1
        )
        target_occupied = (
            context.current_valid[:, None]
            & ~own
            & (
                (
                    currently_in_target
                    & (entry_dx < self.vehicle_length_m + self.minimum_clearance_m)
                    & (entry_dx > -8.0)
                )
                | (
                    target_band & ~currently_in_target
                    & (torch.minimum(dx, future_dx) < self.vehicle_length_m + self.minimum_clearance_m)
                    & (torch.maximum(dx, future_dx) > -8.0)
                )
            )
        ).any(dim=-1)
        return (
            intent & on_plan & leader_index.ge(0) & ~target_occupied
            & ~collision_risk
            & (predicted_lateral_gap[..., -1].abs() >= self.lane_half_width_m)
        )

    def _departing_leader_clearance(
        self,
        context: ReactionControllerContext,
        gap: torch.Tensor,
        speed: torch.Tensor,
        leader_speed: torch.Tensor,
        leader_index: torch.Tensor,
        leader_acceleration: torch.Tensor,
        base: torch.Tensor,
        safety_clearance: torch.Tensor,
    ) -> torch.Tensor:
        """Keep a lane-keeping NPC on HiQR when its leader exits safely.

        The leader's observed lateral velocity must increase separation, and
        longitudinal clearance must remain safe until lateral separation is
        complete. Planned NPC lane changes use the stricter escape gate above
        so a slowly crossing ADS cannot be mistaken for a departing leader.
        """
        empty = torch.zeros_like(gap, dtype=torch.bool)
        if any(value is None for value in (
            context.planned_origin_xy, context.planned_terminal_xy,
            context.map_polylines, context.map_polyline_valid,
        )):
            return empty
        intent, _, _ = planned_adjacent_lane_intent(
            context.planned_origin_xy[..., 1],
            context.planned_terminal_xy[..., 1],
            context.map_polylines, context.map_polyline_valid,
        )
        npc = context.current[:, 1:]
        leader_y = torch.gather(context.current[..., 1], 1, leader_index.clamp_min(0))
        leader_vy = torch.gather(context.current[..., 3], 1, leader_index.clamp_min(0))
        lateral_gap = leader_y - npc[..., 1]
        lateral_rate = leader_vy - npc[..., 3]
        horizon_steps = max(1, int(round(self.prediction_horizon_s / self.native_dt_s)))
        times = torch.arange(
            1, horizon_steps + 1, device=npc.device, dtype=npc.dtype,
        ) * self.native_dt_s
        future_gap = (
            gap[..., None]
            + (leader_speed - speed)[..., None] * times
            + 0.5 * (leader_acceleration - base)[..., None] * times.square()
        )
        future_lateral_gap = lateral_gap[..., None] + lateral_rate[..., None] * times
        joint_risk = (
            (future_gap < safety_clearance[..., None])
            & (future_lateral_gap.abs() < self.lane_half_width_m)
        ).any(dim=-1)
        return (
            ~intent & context.current_valid[:, 1:] & leader_index.ge(0)
            & lateral_gap.abs().ge(0.5)
            & lateral_gap.abs().lt(self.lane_half_width_m)
            & (lateral_gap * lateral_rate > 0.2)
            & (future_lateral_gap[..., -1].abs() >= self.lane_half_width_m)
            & ~joint_risk
        )

    def _semantic_lane_repair(
        self, context: ReactionControllerContext, longitudinal_active: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Delay an unsafe planned change and finish it at its lane centre.

        The target comes from the generated Diffusion plan's start/end lane,
        not from a simulated nominal branch. On-plan HiQR steering is kept
        when the target gap is safe; an uncommitted change waits for a clear
        target-lane gap, then follows the same lane-level intent.
        """
        base_yaw = context.base_actions[:, 0, :, 1]
        if any(value is None for value in (
            context.planned_origin_xy, context.planned_current_xy,
            context.planned_terminal_xy,
        )):
            empty = torch.zeros_like(base_yaw, dtype=torch.bool)
            return base_yaw, empty, empty
        origin = context.planned_origin_xy[..., 1]
        planned = context.planned_current_xy[..., 1]
        terminal = context.planned_terminal_xy[..., 1]
        npc = context.current[:, 1:]
        intent, source_y, target_y = planned_adjacent_lane_intent(
            origin, terminal, context.map_polylines, context.map_polyline_valid
        )
        displaced = (npc[..., 1] - planned).abs() > 0.5
        speed = torch.linalg.vector_norm(npc[..., 2:4], dim=-1).clamp_min(0.5)
        heading = torch.atan2(npc[..., 3], npc[..., 2])
        error_y = target_y - npc[..., 1]
        settled = (error_y.abs() < 0.05) & (heading.abs() < 0.005)
        if context.planned_path_xy is None:
            path_y = target_y
            path_mode = torch.zeros_like(intent)
            spatial_mismatch = torch.zeros_like(intent)
        else:
            if (
                self._spatial_path_latched is None
                or self._spatial_path_latched.shape != intent.shape
            ):
                self._spatial_path_latched = torch.zeros_like(intent)
            progress_shift = (
                (npc[..., 0] - context.planned_current_xy[..., 0]).abs() > 2.0
            )
            path_mode = (
                self._spatial_path_latched | (longitudinal_active & progress_shift)
            ) & intent & ~settled
            self._spatial_path_latched = path_mode.detach().clone()
            lookahead = torch.minimum(torch.full_like(speed, 5.0), speed * 0.25)
            spatial_y = planned_lateral_position_at_x(
                context.planned_origin_xy, context.planned_path_xy,
                npc[..., 0] + lookahead,
            )
            path_y = torch.where(path_mode, spatial_y, target_y)
            spatial_mismatch = path_mode & progress_shift & ((path_y - planned).abs() > 0.2)
        near_source = (npc[..., 1] - source_y).abs() < 0.8
        other = context.current[:, None]
        dx = other[..., 0] - npc[:, :, None, 0]
        horizon_steps = max(1, int(round(self.prediction_horizon_s / self.native_dt_s)))
        if context.planned_path_xy is None:
            entry_time = torch.full_like(speed, self.prediction_horizon_s)
        else:
            indices = (
                torch.arange(1, horizon_steps + 1, device=npc.device)
                + int(context.response_index)
            ).clamp_max(context.planned_path_xy.shape[1] - 1)
            planned_future_y = context.planned_path_xy.index_select(1, indices)[..., 1].permute(0, 2, 1)
            entry_mask = (planned_future_y - target_y[..., None]).abs() < self.lane_half_width_m
            steps = torch.arange(1, horizon_steps + 1, device=npc.device)
            entry_step = torch.where(
                entry_mask, steps, torch.full_like(steps, horizon_steps),
            ).amin(dim=-1)
            entry_time = entry_step.to(npc.dtype) * self.native_dt_s
        entry_dx = dx + entry_time[..., None] * (
            other[..., 2] - npc[:, :, None, 2]
        )
        target_lane = (other[..., 1] - target_y[:, :, None]).abs() < 1.8
        own = torch.arange(7, device=npc.device)[None, None] == (
            torch.arange(6, device=npc.device)[None, :, None] + 1
        )
        unsafe = (
            context.current_valid[:, None]
            & ~own
            & target_lane
            & (entry_dx < self.vehicle_length_m + self.minimum_clearance_m)
            & (entry_dx > -8.0)
        ).any(dim=-1)
        target_horizon = torch.full(
            (1, 1, 7), self.prediction_horizon_s,
            dtype=npc.dtype, device=npc.device,
        )
        target_horizon[..., 0] = max(4.0, self.prediction_horizon_s)
        future_other_y = other[..., 1] + target_horizon * other[..., 3]
        future_dx = dx + target_horizon * (
            other[..., 2] - npc[:, :, None, 2]
        )
        swept_target = (
            ~target_lane
            & (torch.minimum(other[..., 1], future_other_y) < target_y[:, :, None] + 1.8)
            & (torch.maximum(other[..., 1], future_other_y) > target_y[:, :, None] - 1.8)
        )
        late_target_unsafe = unsafe | (
            context.current_valid[:, None] & ~own & swept_target
            & (torch.minimum(dx, future_dx) < self.vehicle_length_m + self.minimum_clearance_m)
            & (torch.maximum(dx, future_dx) > -8.0)
        ).any(dim=-1)
        # Once a disturbed driver has safely passed the blocked portion of a
        # lane change, its destination remains the map lane centre. A purely
        # spatial replay of the frozen path can otherwise leave a slowed NPC
        # indefinitely mid-change even when the target lane has cleared.
        late_clear_finish = (
            path_mode & (int(context.response_index) >= 75) & ~late_target_unsafe
        )
        path_y = torch.where(late_clear_finish, target_y, path_y)
        path_error_y = path_y - npc[..., 1]
        desired_vy = 2.0 * torch.tanh(0.9 * path_error_y / 2.0)
        target_heading = torch.asin((desired_vy / speed).clamp(-0.95, 0.95))
        heading_error = torch.atan2(
            torch.sin(target_heading - heading),
            torch.cos(target_heading - heading),
        )
        yaw = 6.0 * heading_error
        direction = torch.sign(target_y - origin)
        scheduled = (
            ((planned - origin) * direction > 0.2)
            | (path_mode & ((path_y - origin) * direction > 0.2))
        )
        unsafe_start = intent & scheduled & near_source & unsafe
        if (
            self._lane_repair_latched is None
            or self._lane_repair_latched.shape != intent.shape
        ):
            self._lane_repair_latched = torch.zeros_like(intent)
        repair = (
            (self._lane_repair_latched | displaced | unsafe_start | spatial_mismatch)
            & intent & ~settled & context.current_valid[:, 1:]
        )
        self._lane_repair_latched = repair.detach().clone()
        wait = repair & near_source & unsafe
        yaw = torch.where(wait, -6.0 * heading, yaw)
        limit = torch.minimum(torch.full_like(speed, 0.6), 4.0 / speed)
        yaw = torch.maximum(torch.minimum(yaw, limit), -limit)
        # Once already committed, an occupied target lane must not be rushed
        # by the recovery actuator. Keep HiQR's slower lateral progression
        # until the rear/front gap opens; the latch retains the intent.
        execute_repair = repair & (~unsafe | near_source)
        return torch.where(execute_repair, yaw, base_yaw), execute_repair, path_mode

    def _passing_cutin_clearance(self, context: ReactionControllerContext) -> torch.Tensor:
        """Let an NPC finish a clear pass before a visibly crossing ADS enters.

        A near-side ADS can be ahead *now* yet behind the NPC by the time its
        lateral motion reaches the occupied lane. In that case forcing an IDM
        brake can put the NPC back into the ADS cut-in corridor. This check
        uses realized ADS state, the current HiQR action and mapped lane centers.
        """
        npc = context.current[:, 1:]
        empty = torch.zeros_like(npc[..., 0], dtype=torch.bool)
        if context.map_polylines is None or context.map_polyline_valid is None:
            return empty
        ego = context.current[:, :1]
        dx = ego[..., 0] - npc[..., 0]
        dy = npc[..., 1] - ego[..., 1]
        lane_y = context.map_polylines[..., 0, 1]
        lane_present = context.map_polyline_valid.any(dim=-1)
        source_index = torch.where(
            lane_present, (lane_y - ego[..., 1]).abs(),
            torch.full_like(lane_y, float("inf")),
        ).argmin(dim=1, keepdim=True)
        source_y = torch.gather(lane_y, 1, source_index)
        # The next mapped centre, not a nominal 3.6 m displacement: highD
        # target lanes vary appreciably across recordings.
        target_candidates = torch.where(
            lane_present & (lane_y > source_y + self.lane_half_width_m),
            lane_y, torch.full_like(lane_y, float("inf")),
        )
        target_y = target_candidates.min(dim=1, keepdim=True).values
        target_npc = (
            lane_present & (lane_y - target_y).abs().lt(0.45)
        ).any(dim=1, keepdim=True) & (npc[..., 1] - target_y).abs().lt(0.8)
        time_to_entry = (
            (dy - self.lane_half_width_m).clamp_min(0.0)
            / ego[..., 3].clamp_min(1.0)
        ).clamp_max(self.prediction_horizon_s)
        base_ax = context.base_actions[:, 0, :, 0]
        dx_at_entry = (
            dx + (ego[..., 2] - npc[..., 2]) * time_to_entry
            + 0.5 * (ego[..., 4] - base_ax) * time_to_entry.square()
        )
        eligible = (
            context.current_valid[:, 1:]
            & (ego[..., 3] > 0.08)
            & (dy > self.lane_half_width_m)
            & (dy < 5.4)
            & (dx > 0.0)
            & (dx < 45.0)
            & target_npc
        )
        clear_start = eligible & (
            dx_at_entry < -(self.vehicle_length_m + 0.5)
        )
        if self._passing_cutin_latched is None or self._passing_cutin_latched.shape != dx.shape:
            self._passing_cutin_latched = torch.zeros_like(eligible)
        # Once a safe pass has started, abruptly switching to IDM braking
        # while still abreast can put the NPC directly into the ADS cut-in.
        # Keep the pass until longitudinal clearance is actually achieved.
        keep_pass = (
            self._passing_cutin_latched
            & context.current_valid[:, 1:]
            & target_npc
            & (dx > -(self.vehicle_length_m + self.minimum_clearance_m))
            & (dx < 45.0)
            & (ego[..., 3] > 0.08)
        )
        passing = clear_start | keep_pass
        self._passing_cutin_latched = passing.detach().clone()
        return passing

    def _autonomous_lane_response(
        self,
        context: ReactionControllerContext,
        actions: torch.Tensor,
        *,
        gap: torch.Tensor,
        speed: torch.Tensor,
        leader_speed: torch.Tensor,
        leader_index: torch.Tensor,
        leader_acceleration: torch.Tensor,
        leader_y: torch.Tensor,
        idm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Commit a new mapped lane intent only after a realized hard brake.

        Diffusion's existing lane intent has priority. A novel manoeuvre is
        proposed only when the actual leader's braking makes the current-lane
        short-horizon gap unsafe and the adjacent lane is clear at current,
        entry and recovery times. Once chosen, the destination is persistent:
        returning to the old stay-lane HiQR trace would undo the decision.
        """
        npc = context.current[:, 1:]
        active_shape = npc.shape[:2]
        if self._autonomous_lane_active is None or self._autonomous_lane_active.shape != active_shape:
            self._autonomous_lane_active = torch.zeros(active_shape, dtype=torch.bool, device=npc.device)
            self._autonomous_lane_ever = torch.zeros_like(self._autonomous_lane_active)
            self._autonomous_lane_target_y = torch.zeros_like(npc[..., 1])
        active = self._autonomous_lane_active & context.current_valid[:, 1:]
        if context.map_polylines is None or context.map_polyline_valid is None:
            return actions, active

        lane_y = context.map_polylines[..., 0, 1]
        lane_present = context.map_polyline_valid.any(dim=-1)
        source_index = torch.where(
            lane_present[:, None], (lane_y[:, None] - npc[..., 1, None]).abs(),
            torch.full_like(lane_y[:, None], float("inf")),
        ).argmin(dim=-1)
        source_y = torch.gather(
            lane_y[:, None].expand(-1, npc.shape[1], -1),
            2, source_index[..., None],
        ).squeeze(-1)
        planned_intent = torch.zeros_like(active)
        if context.planned_origin_xy is not None and context.planned_terminal_xy is not None:
            planned_intent = planned_adjacent_lane_intent(
                context.planned_origin_xy[..., 1],
                context.planned_terminal_xy[..., 1],
                context.map_polylines, context.map_polyline_valid,
            )[0]
        horizon = self.prediction_horizon_s
        projected_gap = (
            gap + (leader_speed - speed) * horizon
            + 0.5 * (leader_acceleration - actions[:, 0, :, 0]) * horizon * horizon
        )
        clearance = torch.maximum(
            torch.full_like(speed, self.minimum_clearance_m),
            self.speed_clearance_s * speed,
        )
        can_start = (
            context.current_valid[:, 1:] & ~self._autonomous_lane_ever
            & ~active & ~planned_intent & leader_index.ge(0)
            & (leader_y - npc[..., 1]).abs().lt(0.8)
            & leader_acceleration.le(-4.0)
            & projected_gap.lt(clearance)
            & (npc[..., 1] - source_y).abs().lt(0.6)
            & lane_present.any(dim=1)[:, None]
        )
        other = context.current[:, None]
        dx = other[..., 0] - npc[:, :, None, 0]
        own = torch.arange(7, device=npc.device)[None, None] == (
            torch.arange(6, device=npc.device)[None, :, None] + 1
        )
        npc_dx = (npc[..., 0, None] - npc[:, None, :, 0]).abs()
        different = torch.arange(6, device=npc.device)[:, None].ne(
            torch.arange(6, device=npc.device)[None, :]
        )
        lower_slot = torch.arange(6, device=npc.device)[None, :] < (
            torch.arange(6, device=npc.device)[:, None]
        )
        for side in (-1.0, 1.0):
            distance = (lane_y[:, None] - source_y[..., None]) * side
            adjacent = lane_present[:, None] & distance.gt(1.8) & distance.lt(5.4)
            target_index = torch.where(
                adjacent, (distance - 3.6).abs(),
                torch.full_like(distance, float("inf")),
            ).argmin(dim=-1)
            target_y = torch.gather(
                lane_y[:, None].expand(-1, npc.shape[1], -1),
                2, target_index[..., None],
            ).squeeze(-1)
            clear = adjacent.any(dim=-1)
            for time in (0.0, 1.5, 3.0):
                future_y = other[..., 1] + other[..., 3] * time
                future_dx = dx + (other[..., 2] - npc[:, :, None, 2]) * time
                occupied = (
                    context.current_valid[:, None] & ~own
                    & (future_y - target_y[..., None]).abs().lt(self.lane_half_width_m)
                    & future_dx.gt(-18.0) & future_dx.lt(12.0)
                ).any(dim=-1)
                clear &= ~occupied
            # Reservations prevent two nearby NPCs from choosing the same
            # target in one tick or entering an already committed corridor.
            committed_conflict = (
                active[:, None] & different[None] & npc_dx.lt(25.0)
                & (self._autonomous_lane_target_y[:, None] - target_y[..., None]).abs().lt(1.8)
            ).any(dim=-1)
            take = can_start & clear & ~committed_conflict
            simultaneous_conflict = (
                take[..., None] & take[:, None] & lower_slot[None]
                & npc_dx.lt(25.0)
                & (target_y[..., None] - target_y[:, None]).abs().lt(1.8)
            ).any(dim=-1)
            take &= ~simultaneous_conflict
            self._autonomous_lane_target_y = torch.where(
                take, target_y, self._autonomous_lane_target_y,
            )
            active |= take
            self._autonomous_lane_ever |= take
            can_start &= ~take
        self._autonomous_lane_active = active.detach().clone()
        self._autonomous_lane_ever = self._autonomous_lane_ever.detach().clone()
        self._autonomous_lane_target_y = self._autonomous_lane_target_y.detach().clone()

        heading = torch.atan2(npc[..., 3], npc[..., 2])
        error_y = self._autonomous_lane_target_y - npc[..., 1]
        desired_vy = 2.0 * torch.tanh(0.9 * error_y / 2.0)
        target_heading = torch.asin((desired_vy / speed.clamp_min(0.5)).clamp(-0.95, 0.95))
        heading_error = torch.atan2(
            torch.sin(target_heading - heading), torch.cos(target_heading - heading),
        )
        yaw_limit = torch.minimum(torch.full_like(speed, 0.6), 4.0 / speed.clamp_min(0.5))
        yaw = (6.0 * heading_error).clamp(-yaw_limit, yaw_limit)
        result = actions.clone()
        result[:, 0, :, 1] = torch.where(active, yaw, result[:, 0, :, 1])
        # The stay-lane HiQR timeline no longer describes a committed novel
        # manoeuvre. Use the same episode driver for its longitudinal motion.
        lane_ax = idm.clamp(self.minimum_acceleration_mps2, self.maximum_acceleration_mps2)
        if context.previous_background_actions is not None:
            previous_ax = context.previous_background_actions[..., 0]
            max_step = self.correction_jerk_limit_mps3 * self.native_dt_s
            lane_ax = lane_ax.clamp(previous_ax - max_step, previous_ax + max_step)
        result[:, 0, :, 0] = torch.where(active, lane_ax, result[:, 0, :, 0])
        return result, active

    def forward(
        self, context: ReactionControllerContext, *, deterministic: bool = False
    ) -> ReactionControllerOutput:
        del deterministic
        batch, _, slots, _ = context.base_actions.shape
        if slots != 6:
            raise ValueError("online MA-IDM expects six NPC slots")
        theta = self.theta.to(context.current)
        if theta.ndim == 2:
            theta = theta[None].expand(batch, -1, -1)
        elif theta.shape[0] != batch:
            raise ValueError("theta batch does not match the world batch")

        gap, speed, leader_speed, leader_index = nearest_leader_observation(
            context.current, context.current_valid,
            lane_half_width_m=self.lane_half_width_m,
            vehicle_length_m=self.vehicle_length_m,
            prediction_horizon_s=self.prediction_horizon_s,
        )
        base = context.base_actions[:, 0, :, 0]
        leader_acceleration = torch.gather(
            context.current[..., 4], 1, leader_index.clamp_min(0)
        )
        horizon = self.prediction_horizon_s
        # Constant-acceleration short-horizon clearance, evaluated strictly
        # from the already realized leader state and this tick's HiQR command.
        projected_gap = (
            gap + (leader_speed - speed) * horizon
            + 0.5 * (leader_acceleration - base) * horizon * horizon
        )
        safety_clearance = torch.maximum(
            torch.full_like(speed, self.minimum_clearance_m),
            self.speed_clearance_s * speed,
        )
        at_risk = projected_gap < safety_clearance
        planned_escape = self._planned_lane_escape(
            context, gap, speed, leader_speed, leader_index,
            leader_acceleration, base, safety_clearance,
        )
        departing_leader = self._departing_leader_clearance(
            context, gap, speed, leader_speed, leader_index,
            leader_acceleration, base, safety_clearance,
        )
        at_risk &= ~(planned_escape | departing_leader)
        # Detect an abrupt leader control change from the *realized* 1 s
        # history. This is a causal innovation, not a paired nominal world:
        # it lets a driver respond to an ADS or NPC leader's braking or
        # acceleration before clearance becomes critical while leaving smooth
        # factual traffic to HiQR. An opposite step cancels the stimulus.
        leader_history = torch.gather(
            context.history[..., 4], 2,
            leader_index.clamp_min(0)[:, None].expand(-1, context.history.shape[1], -1),
        )
        leader_history_valid = torch.gather(
            context.history_valid, 2,
            leader_index.clamp_min(0)[:, None].expand(-1, context.history.shape[1], -1),
        )
        if leader_history.shape[1] >= 2:
            changes = leader_history[:, 1:] - leader_history[:, :-1]
            sharpest = torch.gather(
                changes, 1, changes.abs().argmax(dim=1, keepdim=True)
            ).squeeze(1)
            net_change = leader_acceleration - leader_history[:, 0]
            abrupt = (
                sharpest.abs() >= 1.5
            ) & (
                net_change.abs() >= 1.0
            ) & (
                sharpest * net_change > 0.0
            ) & (leader_index.eq(0) | leader_history_valid.all(dim=1))
        else:
            sharpest = torch.zeros_like(base)
            abrupt = torch.zeros_like(base, dtype=torch.bool)
        leader_y = torch.gather(
            context.current[..., 1], 1, leader_index.clamp_min(0)
        )
        # A predicted lane crossing can nominate an adjacent NPC as a future
        # leader. Its abrupt longitudinal action is not yet a same-lane shock;
        # the swept-corridor risk gate below still handles unsafe cut-ins.
        aligned_npc_leader = leader_index.gt(0) & (
            (leader_y - context.current[:, 1:, 1]).abs() < 0.8
        )
        influence = (
            (leader_index.eq(0) | aligned_npc_leader)
            & gap.lt(45.0)
            & abrupt
        ) & ~(planned_escape | departing_leader)
        # A positive HiQR command can be a catch-up response to a frozen
        # timestamp, even though the realized ADS ahead is still slower.
        # Once the NPC is materially behind its plan, suppress that positive
        # pursuit without adding an IDM brake. Existing risk control remains
        # responsible for any braking that actual clearance requires.
        plan_pursuit = torch.zeros_like(at_risk)
        if context.planned_current_xy is not None:
            plan_lag = context.planned_current_xy[..., 0] - context.current[:, 1:, 0]
            plan_pursuit = (
                leader_index.eq(0)
                & (leader_y - context.current[:, 1:, 1]).abs().lt(self.lane_half_width_m)
                & gap.lt(45.0)
                & speed.gt(leader_speed)
                & plan_lag.gt(2.0)
                & base.gt(0.0)
                & ~planned_escape
                & ~departing_leader
            )
        active = context.current_valid[:, 1:] & leader_index.ge(0) & (
            at_risk | influence | plan_pursuit
        )
        idm = torch_idm(gap, speed, leader_speed, theta)
        proximity = (1.0 - gap / 45.0).clamp(0.0, 1.0)
        stimulus_gain = (theta[..., 2] / 2.0).clamp(0.3, 0.75)
        stimulus = sharpest * stimulus_gain * proximity
        desired = torch.where(influence, base + stimulus, base)
        desired = torch.where(at_risk, torch.minimum(desired, idm), desired)
        desired = torch.where(plan_pursuit, torch.minimum(desired, torch.zeros_like(desired)), desired)
        desired = desired.clamp(
            self.minimum_acceleration_mps2, self.maximum_acceleration_mps2
        )
        if (
            self._longitudinal_response_latched is None
            or self._longitudinal_response_latched.shape != active.shape
        ):
            self._longitudinal_response_latched = torch.zeros_like(active)
        if self._disturbance_latched is None or self._disturbance_latched.shape != active.shape:
            self._disturbance_latched = torch.zeros_like(active)
        if (
            self._disturbance_leader_index is None
            or self._disturbance_leader_index.shape != leader_index.shape
        ):
            self._disturbance_leader_index = torch.full_like(leader_index, -1)
        continued_disturbance = (
            self._disturbance_latched
            & leader_index.eq(self._disturbance_leader_index)
            & gap.lt(45.0)
        )
        # Keep shock-release memory for NPC leaders. ADS longitudinal doses
        # retain the tested instantaneous fallback: a persistent ADS latch
        # created large cross-dose inversions in the full held-out cohort.
        npc_shock = influence & leader_index.gt(0)
        disturbance_response = npc_shock | continued_disturbance
        longitudinal_active = active
        if context.previous_background_actions is not None:
            previous = context.previous_background_actions[..., 0]
            max_step = self.correction_jerk_limit_mps3 * self.native_dt_s
            # A driver that has already braked for a realized leader remains
            # affected while releasing that correction. Otherwise a one-frame
            # risk-gate change can jump from strong braking straight to HiQR's
            # acceleration and chatter on the next frame. The same holds for
            # releasing a positive response. Never delay a newly requested
            # stronger HiQR brake merely to satisfy this release.
            new_hiqr_brake = (
                torch.zeros_like(active) if self._previous_base_ax is None
                else base < self._previous_base_ax - max_step
            )
            release = (
                self._longitudinal_response_latched & ~active
                & disturbance_response
                & ((base - previous).abs() > max_step)
                & ~new_hiqr_brake
            )
            guarded = desired.clamp(previous - max_step, previous + max_step)
            # Never alter an unaffected car merely to enforce a guard: that
            # would defeat the exact HiQR reconstruction fallback.
            longitudinal_active = active | release
            desired = torch.where(longitudinal_active, guarded, base)
        self._longitudinal_response_latched = longitudinal_active.detach().clone()
        self._disturbance_latched = disturbance_response.detach().clone()
        self._disturbance_leader_index = torch.where(
            disturbance_response,
            torch.where(npc_shock, leader_index, self._disturbance_leader_index),
            torch.full_like(leader_index, -1),
        ).detach().clone()
        self._previous_base_ax = base.detach().clone()
        actions = context.base_actions.clone()
        passing_ads = self._passing_cutin_clearance(context) & leader_index.eq(0)
        if self._passing_cutin_latched is not None:
            # An intervening NPC leader takes priority, but does not erase
            # the ADS pass intent unless it actually requires stronger
            # braking. The leader selector may briefly switch to a harmless
            # nearby NPC before returning to the crossing ADS.
            self._passing_cutin_latched &= ~(
                leader_index.gt(0) & (desired < base - 0.05)
            )
        actions[:, 0, :, 0] = torch.where(passing_ads, base, desired)
        actions[:, 0, :, 1], lane_repair, path_mode = self._semantic_lane_repair(
            context, longitudinal_active
        )
        actions, autonomous_lane = self._autonomous_lane_response(
            context, actions,
            gap=gap, speed=speed, leader_speed=leader_speed,
            leader_index=leader_index, leader_acceleration=leader_acceleration,
            leader_y=leader_y, idm=idm,
        )
        delta = actions[:, 0, :, 0] - base
        overall_active = longitudinal_active | lane_repair | autonomous_lane
        return ReactionControllerOutput(
            actions=actions,
            alpha=overall_active.to(base.dtype),
            delta_ax=delta,
            active=overall_active,
            rule_action_ax=idm,
            desired_action_ax=actions[:, 0, :, 0],
            causal_delta_ax=delta,
            calibration_correction_ax=delta,
            correction_constraint_infeasible=torch.zeros_like(active),
            policy_active=lane_repair | autonomous_lane,
            spatial_path_active=path_mode,
            autonomous_lane_active=autonomous_lane,
        )
