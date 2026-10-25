"""Shared persistent IDM driver and MOBIL maneuver execution."""

import numpy as np
import torch
from torch import nn

from ..core.dynamics import KinematicTrafficDynamics
from ..reaction_controller import ManeuverPhase, ReactionControllerContext
from ..scene_geometry import (
    torch_adjacent_lane_index,
    torch_map_lane_geometry,
    torch_nearest_lane_geometry,
)
from .driver_policy import (
    anticipated_entry_features,
    causal_maneuver_features,
    corrected_acceleration,
    exclusive_lane_probabilities,
    physical_leader_observation,
    response_history_features,
    sample_lane_choice,
)
from .lane_choice import lane_choice_features
from .lane_trajectory import LearnedLaneTrajectory, path_clearance
from .mobil_response import MOBILDecision, evaluate_mobil_candidates


def calibrated_mobil_parameters(fit):
    """Use the same frozen decision parameters in GT and generated worlds."""
    return {
        "mobil_politeness": fit["politeness"],
        "mobil_plan_keep_bias_mps2": 0.0,
        "mobil_onset_calibration": (fit["bias"], fit["slope"]),
        "mobil_parameter_variation": False,
        "lane_changes_on_response_only": True,
        "driver_correction": fit.get("correction"),
        "lane_style": fit.get("lane_style"),
        "lane_model": fit.get("lane_model"),
    }


def mobil_parameters_from_maneuver_latent(
    maneuver_latent: np.ndarray | torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Map persistent standard-normal traits to bounded MOBIL parameters."""
    latent = torch.as_tensor(maneuver_latent, dtype=torch.float32)
    if latent.ndim not in (2, 3) or tuple(latent.shape[-2:]) != (6, 4):
        raise ValueError("maneuver_latent must have shape [6,4] or [batch,6,4]")
    probability = torch.sigmoid(latent)
    return {
        "politeness": 0.05 + 0.55 * probability[..., 0],
        "minimum_incentive_mps2": 0.05 + 0.75 * probability[..., 1],
        "safe_braking_mps2": 2.5 + 3.5 * probability[..., 2],
        "plan_keep_bias_mps2": 0.02 + 0.28 * probability[..., 3],
    }


def nearest_leader_observation(states, valid, *, vehicle_lengths_m, vehicle_widths_m):
    dimensions = torch.stack((vehicle_lengths_m, vehicle_widths_m), -1)
    gap, speed, leader_speed, leader = physical_leader_observation(
        states, valid, dimensions
    )
    return gap[:, 1:], speed[:, 1:], leader_speed[:, 1:], leader[:, 1:]


class NPCPolicy(nn.Module):
    def __init__(
        self,
        theta: np.ndarray | torch.Tensor,
        *,
        prediction_horizon_s: float = 2.0,
        minimum_clearance_m: float = 2.0,
        speed_clearance_s: float = 0.1,
        minimum_acceleration_mps2: float = -8.0,
        maximum_acceleration_mps2: float = 4.0,
        correction_jerk_limit_mps3: float = 12.0,
        native_dt_s: float = 0.04,
        mobil_politeness: float = 0.3,
        mobil_minimum_incentive_mps2: float = 0.3,
        mobil_safe_braking_mps2: float = 4.0,
        mobil_plan_keep_bias_mps2: float = 0.1,
        mobil_onset_calibration: tuple[float, float] | None = None,
        mobil_parameter_variation: bool = True,
        maneuver_latent: np.ndarray | torch.Tensor | None = None,
        autonomous_lane_changes_enabled: bool = True,
        lane_changes_on_response_only: bool = True,
        driver_correction: dict | None = None,
        lane_style: dict | None = None,
        lane_model: dict | None = None,
    ) -> None:
        super().__init__()
        values = torch.as_tensor(theta, dtype=torch.float32)
        if values.ndim not in (2, 3) or tuple(values.shape[-2:]) != (6, 5):
            raise ValueError("theta must have shape [6,5] or [batch,6,5]")
        if (
            min(
                prediction_horizon_s,
                minimum_clearance_m,
                speed_clearance_s,
                correction_jerk_limit_mps3,
                native_dt_s,
            )
            <= 0.0
        ):
            raise ValueError("response horizon, clearance and timing must be positive")
        self.register_buffer("theta", values.clone())
        self.autonomous_lane_changes_enabled = bool(autonomous_lane_changes_enabled)
        self.lane_changes_on_response_only = bool(lane_changes_on_response_only)
        self.mobil_onset_calibration = mobil_onset_calibration
        self.mobil_parameter_variation = bool(mobil_parameter_variation)
        self.driver_correction = driver_correction
        self.lane_style = lane_style
        self._driver_residual_state: torch.Tensor | None = None
        self._driver_source_y: torch.Tensor | None = None
        self.lane_trajectory = LearnedLaneTrajectory(
            correction=(
                None if driver_correction is None else driver_correction["trajectory"]
            ),
            model=lane_model,
        )
        maneuver_values = (
            None
            if maneuver_latent is None
            else torch.as_tensor(maneuver_latent, dtype=torch.float32)
        )
        if maneuver_values is not None and (
            maneuver_values.ndim not in (2, 3)
            or tuple(maneuver_values.shape[-2:]) != (6, 4)
        ):
            raise ValueError("maneuver_latent must have shape [6,4] or [batch,6,4]")
        self.register_buffer(
            "maneuver_latent",
            None if maneuver_values is None else maneuver_values.clone(),
        )
        self.prediction_horizon_s = float(prediction_horizon_s)
        self.minimum_clearance_m = float(minimum_clearance_m)
        self.speed_clearance_s = float(speed_clearance_s)
        self.minimum_acceleration_mps2 = float(minimum_acceleration_mps2)
        self.maximum_acceleration_mps2 = float(maximum_acceleration_mps2)
        self.correction_jerk_limit_mps3 = float(correction_jerk_limit_mps3)
        self.native_dt_s = float(native_dt_s)
        self.mobil_politeness = float(mobil_politeness)
        self.mobil_minimum_incentive_mps2 = float(mobil_minimum_incentive_mps2)
        self.mobil_safe_braking_mps2 = float(mobil_safe_braking_mps2)
        self.mobil_plan_keep_bias_mps2 = float(mobil_plan_keep_bias_mps2)
        if not 0.0 <= self.mobil_politeness <= 1.0:
            raise ValueError("MOBIL politeness must lie in [0,1]")
        if (
            min(
                self.mobil_minimum_incentive_mps2,
                self.mobil_safe_braking_mps2,
                self.mobil_plan_keep_bias_mps2,
            )
            < 0.0
        ):
            raise ValueError("MOBIL thresholds must be non-negative")
        self._autonomous_lane_active: torch.Tensor | None = None
        self._autonomous_lane_ever: torch.Tensor | None = None
        self._autonomous_lane_target_y: torch.Tensor | None = None
        self._lane_path_coefficients: torch.Tensor | None = None
        self._lane_path_origin: torch.Tensor | None = None
        self._lane_path_distance: torch.Tensor | None = None
        self._lane_path_waiting: torch.Tensor | None = None
        self._maneuver_phase: torch.Tensor | None = None
        self._maneuver_stable_frames: torch.Tensor | None = None
        self._maneuver_cooldown_frames: torch.Tensor | None = None

    def driver_acceleration(
        self, context, theta, gap, speed, leader_speed, leader_index
    ):
        """Evaluate the shared driver and advance its history even while inactive."""
        innovation = (
            None
            if context.policy_standard_normal is None
            else context.policy_standard_normal[..., 1]
        )
        longitudinal_correction = (
            self.driver_correction is not None
            and self.driver_correction.get("longitudinal") is not None
        )
        driver_history = (
            response_history_features(
                context.current,
                context.history,
                context.history_valid,
                leader_index,
                context.vehicle_dimensions,
            )
            if longitudinal_correction
            else None
        )
        maneuver_features = None
        if longitudinal_correction:
            lane_y, lane_present, lane_width = torch_map_lane_geometry(
                context.map_polylines, context.map_polyline_valid
            )
            _, center, width, _ = torch_nearest_lane_geometry(
                context.current[:, 1:, 1], lane_y, lane_present, lane_width
            )
            if self._driver_source_y is None:
                self._driver_source_y = center.clone()
                for past, valid in zip(
                    context.history.unbind(1), context.history_valid.unbind(1)
                ):
                    _, old_center, old_width, _ = torch_nearest_lane_geometry(
                        past[:, 1:, 1], lane_y, lane_present, lane_width
                    )
                    _, updated = causal_maneuver_features(
                        past[:, 1:], self._driver_source_y, old_center, old_width
                    )
                    self._driver_source_y = torch.where(
                        valid[:, 1:], updated, self._driver_source_y
                    )
            maneuver_features, self._driver_source_y = causal_maneuver_features(
                context.current[:, 1:], self._driver_source_y, center, width
            )
        acceleration, self._driver_residual_state = corrected_acceleration(
            gap,
            speed,
            leader_speed,
            theta,
            correction=self.driver_correction,
            residual_state=self._driver_residual_state,
            innovation=innovation,
            initial_innovation=context.policy_initial_standard_normal,
            history_features=driver_history,
            maneuver_features=maneuver_features,
            entry_features=anticipated_entry_features(
                context.current,
                context.current_valid,
                context.vehicle_dimensions,
                None if context.history.shape[1] < 2 else context.history[:, -2],
                None if context.history.shape[1] < 2 else context.history_valid[:, -2],
                history=context.history,
                history_valid=context.history_valid,
            )[:, 1:],
            dt=self.native_dt_s,
        )
        return acceleration, driver_history

    def mobil_candidates(self, context, theta):
        """Return the frozen driver's lane scores and one-second probabilities."""
        mobil = None
        lane_choice_probability = None
        if context.interaction_relations is not None:
            mobil_parameters: dict[str, float | torch.Tensor] = {
                "politeness": self.mobil_politeness,
                "minimum_incentive_mps2": self.mobil_minimum_incentive_mps2,
                "safe_braking_mps2": self.mobil_safe_braking_mps2,
                "plan_keep_bias_mps2": self.mobil_plan_keep_bias_mps2,
            }
            if (
                self.maneuver_latent is not None
                and self.mobil_onset_calibration is None
                and self.mobil_parameter_variation
            ):
                latent = self.maneuver_latent
                if latent.ndim == 2:
                    latent = latent[None].expand(len(context.current), -1, -1)
                elif latent.shape[0] != len(context.current):
                    raise ValueError(
                        "maneuver_latent batch does not match the current world"
                    )
                mobil_parameters = mobil_parameters_from_maneuver_latent(latent)
            mobil = evaluate_mobil_candidates(
                context.current,
                context.current_valid,
                theta,
                context.interaction_relations,
                vehicle_lengths_m=(
                    None
                    if context.vehicle_dimensions is None
                    else context.vehicle_dimensions[..., 0]
                ),
                **mobil_parameters,
            )
            if self.mobil_onset_calibration is not None:
                bias, slope = self.mobil_onset_calibration
                dimensions = context.vehicle_dimensions
                features = lane_choice_features(
                    context.current, dimensions[..., 0], context.interaction_relations
                )
                lane_choice_probability = exclusive_lane_probabilities(
                    mobil.incentive_mps2,
                    mobil.safe,
                    {"bias": bias, "slope": slope},
                    features=features,
                    correction=self.driver_correction,
                    style_offset=(
                        None
                        if self.lane_style is None
                        else self.maneuver_latent[..., 0].to(context.current)
                        * self.lane_style["std"]
                        + self.lane_style["mean"]
                    ),
                )
        return mobil, lane_choice_probability

    def _initialize_lane_runtime(self, npc):
        active_shape = npc.shape[:2]
        initial_controller_state = (
            self._autonomous_lane_active is None
            or self._autonomous_lane_active.shape != active_shape
        )
        if initial_controller_state:
            self._autonomous_lane_active = torch.zeros(
                active_shape, dtype=torch.bool, device=npc.device
            )
            self._autonomous_lane_ever = torch.zeros_like(self._autonomous_lane_active)
            self._autonomous_lane_target_y = torch.zeros_like(npc[..., 1])
            self._lane_path_coefficients = torch.zeros(
                (*active_shape, 8), device=npc.device
            )
            self._lane_path_origin = npc[..., 0].clone()
            self._lane_path_distance = torch.ones_like(npc[..., 0])
            self._lane_path_waiting = torch.zeros_like(self._autonomous_lane_active)
            self._maneuver_phase = torch.full(
                active_shape,
                int(ManeuverPhase.TRACK),
                dtype=torch.long,
                device=npc.device,
            )
            self._maneuver_stable_frames = torch.zeros(
                active_shape, dtype=torch.long, device=npc.device
            )
            self._maneuver_cooldown_frames = torch.zeros(
                active_shape, dtype=torch.long, device=npc.device
            )

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
        theta: torch.Tensor,
        longitudinal_active: torch.Tensor,
        lateral_response_active: torch.Tensor,
        maneuver_permission: torch.Tensor | None = None,
        lane_choice: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, MOBILDecision | None]:
        """Plan a Train-conditioned lane path and track it in Frenet coordinates."""
        npc = context.current[:, 1:]
        active_shape = npc.shape[:2]
        execution_dimensions = context.vehicle_dimensions[:, 1:]
        self._initialize_lane_runtime(npc)
        self._maneuver_cooldown_frames = (self._maneuver_cooldown_frames - 1).clamp_min(
            0
        )
        rebase_finished = self._maneuver_phase.eq(
            int(ManeuverPhase.REBASE)
        ) & self._maneuver_cooldown_frames.eq(0)
        self._maneuver_phase = torch.where(
            rebase_finished,
            torch.full_like(self._maneuver_phase, int(ManeuverPhase.TRACK)),
            self._maneuver_phase,
        )
        longitudinal_response_phase = (
            longitudinal_active
            & context.current_valid[:, 1:]
            & (
                self._maneuver_phase.eq(int(ManeuverPhase.TRACK))
                | self._maneuver_phase.eq(int(ManeuverPhase.RESPOND))
            )
        )
        self._maneuver_phase = torch.where(
            longitudinal_response_phase,
            torch.full_like(self._maneuver_phase, int(ManeuverPhase.RESPOND)),
            torch.where(
                self._maneuver_phase.eq(int(ManeuverPhase.RESPOND)),
                torch.full_like(self._maneuver_phase, int(ManeuverPhase.TRACK)),
                self._maneuver_phase,
            ),
        )
        active = (
            self._maneuver_phase.eq(int(ManeuverPhase.EXECUTE))
            & context.current_valid[:, 1:]
        )
        if context.map_polylines is None or context.map_polyline_valid is None:
            return actions, active, None

        mobil = None
        lane_choice_probability = None
        learned_draw = None
        learned_selected = torch.zeros_like(active)
        mobil, lane_choice_probability = self.mobil_candidates(context, theta)
        if lane_choice_probability is not None and lane_choice is None:
            noise = (
                torch.randn_like(lane_choice_probability[..., 0])
                if context.policy_standard_normal is None
                else context.policy_standard_normal[..., 0]
            )
            learned_draw = sample_lane_choice(
                lane_choice_probability, torch.special.ndtr(noise), self.native_dt_s
            )
            learned_selected = learned_draw.ne(0)

        if lane_choice is not None:
            learned_draw = lane_choice
            learned_selected = lane_choice.ne(0)

        lane_y, lane_present, lane_width = torch_map_lane_geometry(
            context.map_polylines, context.map_polyline_valid
        )
        road_bounds = (
            (lane_y - lane_width / 2).masked_fill(~lane_present, torch.inf).amin(-1),
            (lane_y + lane_width / 2).masked_fill(~lane_present, -torch.inf).amax(-1),
        )
        source_index, source_y, source_width, source_available = (
            torch_nearest_lane_geometry(npc[..., 1], lane_y, lane_present, lane_width)
        )
        horizon = self.prediction_horizon_s
        projected_gap = (
            gap
            + (leader_speed - speed) * horizon
            + 0.5 * (leader_acceleration - actions[:, 0, :, 0]) * horizon * horizon
        )
        clearance = torch.maximum(
            torch.full_like(speed, self.minimum_clearance_m),
            self.speed_clearance_s * speed,
        )
        if context.interaction_relations is None:
            leader_lane, _, _, _ = torch_nearest_lane_geometry(
                leader_y, lane_y, lane_present, lane_width
            )
        else:
            leader_lane = torch.gather(
                context.interaction_relations.lane_index,
                1,
                leader_index.clamp_min(0),
            )
        emergency = (
            leader_index.ge(0)
            & leader_lane.eq(source_index)
            & leader_acceleration.le(-4.0)
            & projected_gap.lt(clearance)
        )
        mobil_selected = (
            torch.zeros_like(active)
            if mobil is None or lane_choice_probability is not None
            else mobil.selected
        )
        can_start = (
            context.current_valid[:, 1:]
            & ~active
            & (
                (emergency | mobil_selected | learned_selected)
                if lane_choice is None
                else learned_selected
            )
            & self._maneuver_cooldown_frames.eq(0)
            & (
                self._maneuver_phase.eq(int(ManeuverPhase.TRACK))
                | self._maneuver_phase.eq(int(ManeuverPhase.RESPOND))
            )
            & (npc[..., 1] - source_y).abs().lt(0.2 * source_width)
            & (npc[..., 3].abs().lt(0.3) | emergency)
            & source_available
        )
        if self.lane_changes_on_response_only:
            can_start = can_start & (
                longitudinal_active | lateral_response_active | emergency
            )
        if maneuver_permission is not None:
            can_start = can_start & maneuver_permission
        if not self.autonomous_lane_changes_enabled:
            can_start = torch.zeros_like(can_start)
        if not (active | can_start).any():
            return actions, active, mobil
        preferred_side = torch.zeros_like(source_index)
        if lane_choice is not None:
            preferred_side = lane_choice
        elif lane_choice_probability is not None:
            fallback = torch.where(
                lane_choice_probability[..., 1] >= lane_choice_probability[..., 2],
                -torch.ones_like(source_index),
                torch.ones_like(source_index),
            )
            preferred_side = torch.where(
                learned_selected,
                learned_draw,
                fallback,
            )
        elif mobil is not None:
            fallback_side = torch.where(
                mobil.incentive_mps2.argmax(dim=-1).eq(0),
                -torch.ones_like(source_index),
                torch.ones_like(source_index),
            )
            preferred_side = torch.where(
                mobil.best_side.ne(0), mobil.best_side, fallback_side
            )
        npc_dx = (npc[..., 0, None] - npc[:, None, :, 0]).abs()
        different = torch.arange(6, device=npc.device)[:, None].ne(
            torch.arange(6, device=npc.device)[None, :]
        )
        lower_slot = (
            torch.arange(6, device=npc.device)[None, :]
            < (torch.arange(6, device=npc.device)[:, None])
        )
        active_target_index, _, _, _ = torch_nearest_lane_geometry(
            self._autonomous_lane_target_y,
            lane_y,
            lane_present,
            lane_width,
        )
        style_noise = (
            torch.zeros((*active_shape, 2), dtype=npc.dtype, device=npc.device)
            if self.maneuver_latent is None
            else self.maneuver_latent[..., 2:4]
        )
        if style_noise.ndim == 2:
            style_noise = style_noise[None].expand(len(npc), -1, -1)
        candidate_clear = None
        candidate_paths = {}
        if context.interaction_relations is not None:
            dimensions = context.vehicle_dimensions
            target_indices = context.interaction_relations.adjacent_lane_index[
                :, 1:
            ].clamp_min(0)
            targets = torch.gather(lane_y[:, None].expand(-1, 6, -1), 2, target_indices)
            candidate_clear = []
            for side_index in (0, 1):
                path, clear = self.lane_trajectory.safe_plan(
                    context.current,
                    context.current_valid,
                    dimensions,
                    targets[..., side_index],
                    style_noise,
                    actions[:, 0, :, 0],
                    road_bounds,
                )
                candidate_paths[side_index] = path
                candidate_clear.append(clear)
            candidate_clear = torch.stack(candidate_clear, -1)
        for side in (-1.0, 1.0):
            side_index = 0 if side < 0.0 else 1
            if context.interaction_relations is None:
                target_index, target_available = torch_adjacent_lane_index(
                    source_index, lane_y, lane_present, lane_width, side=int(side)
                )
            else:
                target_index = context.interaction_relations.adjacent_lane_index[
                    :, 1:, side_index
                ]
                target_available = target_index.ge(0)
            safe_target_index = target_index.clamp_min(0)
            target_y = torch.gather(
                lane_y[:, None].expand(-1, npc.shape[1], -1),
                2,
                safe_target_index[..., None],
            ).squeeze(-1)
            target_width = torch.gather(
                lane_width[:, None].expand(-1, npc.shape[1], -1),
                2,
                safe_target_index[..., None],
            ).squeeze(-1)
            if candidate_clear is None:
                other = context.current[:, None]
                dx = other[..., 0] - npc[:, :, None, 0]
                own = torch.arange(7, device=npc.device)[None, None] == (
                    torch.arange(6, device=npc.device)[None, :, None] + 1
                )
                clear = target_available.clone()
                for time in (0.0, 1.5, 3.0):
                    future_y = other[..., 1] + other[..., 3] * time
                    future_dx = dx + (other[..., 2] - npc[:, :, None, 2]) * time
                    occupied = (
                        context.current_valid[:, None]
                        & ~own
                        & (
                            (future_y - target_y[..., None]).abs()
                            < 0.5 * target_width[..., None]
                        )
                        & future_dx.gt(-18.0)
                        & future_dx.lt(12.0)
                    ).any(dim=-1)
                    clear = clear & ~occupied
            else:
                clear = candidate_clear[..., side_index] & target_available
            # Avoid simultaneous choices of, or entry into, the same corridor.
            committed_conflict = (
                active[:, None]
                & different[None]
                & npc_dx.lt(25.0)
                & active_target_index[:, None].eq(target_index[..., None])
            ).any(dim=-1)
            take = can_start & clear & ~committed_conflict
            if mobil is not None or lane_choice_probability is not None:
                take = take & preferred_side.eq(int(side))
            simultaneous_conflict = (
                take[..., None]
                & take[:, None]
                & lower_slot[None]
                & npc_dx.lt(25.0)
                & target_index[..., None].eq(target_index[:, None])
            ).any(dim=-1)
            take = take & ~simultaneous_conflict
            self._autonomous_lane_target_y = torch.where(
                take,
                target_y,
                self._autonomous_lane_target_y,
            )
            path = candidate_paths.get(side_index)
            if path is None:
                path = self.lane_trajectory.plan(
                    npc, execution_dimensions, target_y, style_noise
                )
            self._lane_path_coefficients = torch.where(
                take[..., None], path[0], self._lane_path_coefficients
            )
            self._lane_path_origin = torch.where(take, path[1], self._lane_path_origin)
            self._lane_path_distance = torch.where(
                take, path[2], self._lane_path_distance
            )
            active = active | take
            self._autonomous_lane_ever = self._autonomous_lane_ever | take
            self._maneuver_phase = torch.where(
                take,
                torch.full_like(self._maneuver_phase, int(ManeuverPhase.EXECUTE)),
                self._maneuver_phase,
            )
            self._maneuver_stable_frames = torch.where(
                take,
                torch.zeros_like(self._maneuver_stable_frames),
                self._maneuver_stable_frames,
            )
            can_start = can_start & ~take
        self._autonomous_lane_active = active.detach().clone()
        self._autonomous_lane_ever = self._autonomous_lane_ever.detach().clone()
        self._autonomous_lane_target_y = self._autonomous_lane_target_y.detach().clone()

        heading = KinematicTrafficDynamics.heading_from_velocity(
            npc[..., 3], npc[..., 2]
        )
        error_y = self._autonomous_lane_target_y - npc[..., 1]
        settled = active & error_y.abs().lt(0.25) & npc[..., 3].abs().lt(0.15)
        self._maneuver_stable_frames = torch.where(
            settled,
            self._maneuver_stable_frames + 1,
            torch.where(
                active,
                torch.zeros_like(self._maneuver_stable_frames),
                self._maneuver_stable_frames,
            ),
        )
        completed = active & self._maneuver_stable_frames.ge(10)
        self._maneuver_phase = torch.where(
            completed,
            torch.full_like(self._maneuver_phase, int(ManeuverPhase.REBASE)),
            self._maneuver_phase,
        )
        self._maneuver_cooldown_frames = torch.where(
            completed,
            torch.full_like(self._maneuver_cooldown_frames, 25),
            self._maneuver_cooldown_frames,
        )
        active = active & ~completed
        self._autonomous_lane_active = active.detach().clone()
        dimensions = context.vehicle_dimensions
        safe = path_clearance(
            context.current,
            context.current_valid,
            dimensions,
            self._lane_path_coefficients[..., None, :],
            self._lane_path_origin[..., None],
            self._lane_path_distance[..., None],
            idm.clamp(-8, 4),
            road_bounds=road_bounds,
            acceleration_limit=2 * self.lane_trajectory.acceleration_limit,
        ).squeeze(-1)
        replan = active & (~safe | self._lane_path_waiting)
        path, clear = self.lane_trajectory.safe_plan(
            context.current,
            context.current_valid,
            dimensions,
            self._autonomous_lane_target_y,
            style_noise,
            idm.clamp(-8, 4),
            road_bounds,
        )
        replace_path = replan & clear
        self._lane_path_coefficients = torch.where(
            replace_path[..., None], path[0], self._lane_path_coefficients
        )
        self._lane_path_origin = torch.where(
            replace_path, path[1], self._lane_path_origin
        )
        self._lane_path_distance = torch.where(
            replace_path, path[2], self._lane_path_distance
        )
        waiting = replan & ~clear
        self._lane_path_waiting = waiting.detach().clone()
        yaw = self.lane_trajectory.track(
            npc,
            self._lane_path_coefficients,
            self._lane_path_origin,
            self._lane_path_distance,
        )
        # A blocked path keeps its target intent while dissipating lateral velocity.
        previous_lateral = -npc[..., 4] * heading.sin() + npc[..., 5] * heading.cos()
        wait_lateral = (-2 * heading * speed).clamp(-4, 4)
        wait_lateral = wait_lateral.clamp(
            previous_lateral - 12 * self.native_dt_s,
            previous_lateral + 12 * self.native_dt_s,
        )
        wait_yaw = wait_lateral / speed.clamp_min(0.5)
        yaw = torch.where(waiting, wait_yaw, yaw)
        result = actions.clone()
        result[:, 0, :, 1] = torch.where(active, yaw, result[:, 0, :, 1])
        # Use the episode driver during a committed lane change.
        lane_ax = idm.clamp(
            self.minimum_acceleration_mps2, self.maximum_acceleration_mps2
        )
        if context.previous_background_actions is not None:
            previous_ax = context.previous_background_actions[..., 0]
            max_step = self.correction_jerk_limit_mps3 * self.native_dt_s
            lane_ax = lane_ax.clamp(previous_ax - max_step, previous_ax + max_step)
        result[:, 0, :, 0] = torch.where(active, lane_ax, result[:, 0, :, 0])
        return result, active, mobil

    runtime_fields = (
        "driver_residual_state",
        "driver_source_y",
        "autonomous_lane_active",
        "autonomous_lane_ever",
        "autonomous_lane_target_y",
        "lane_path_coefficients",
        "lane_path_origin",
        "lane_path_distance",
        "lane_path_waiting",
        "maneuver_phase",
        "maneuver_stable_frames",
        "maneuver_cooldown_frames",
    )

    def snapshot_runtime(self):
        return {
            name: getattr(self, "_" + name).detach().clone()
            for name in self.runtime_fields
            if getattr(self, "_" + name) is not None
        }

    def restore_runtime(self, state):
        for name in self.runtime_fields:
            value = state.get(name)
            setattr(
                self,
                "_" + name,
                None if value is None else value.clone().to(self.theta.device),
            )
