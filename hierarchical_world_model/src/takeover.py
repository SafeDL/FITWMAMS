"""Local, irreversible handover from reference tracking to the shared NPC policy."""

from dataclasses import dataclass, replace
from pathlib import Path

import torch

from .reaction_controller import ManeuverPhase, ReactionControllerOutput
from .scene_geometry import torch_map_lane_geometry, torch_nearest_lane_geometry
from .stochastic_drivers.lane_trajectory import frenet_feedback
from .stochastic_drivers.online import NPCPolicy, nearest_leader_observation
from .plan_applicability import conflict_scores, forecast_bundle, interaction_scores
from .error_trajectories import ErrorTrajectoryBank


@dataclass(frozen=True)
class TakeoverConfig:
    lookahead_s: float = 1.5
    support_threshold: float = 1 / 16
    error_trajectories_path: str | None = None
    interaction_threshold: float = 1.0
    following_scale_s: float = 1.0
    release_scale_per_s: float = 1.0


class TakeoverController(NPCPolicy):
    """One admission per vehicle; subsequent choices belong to the NPC driver."""

    mode = "tracking_npc_takeover"

    def __init__(self, theta, *, takeover_config=None, **kwargs):
        kwargs["lane_changes_on_response_only"] = False
        super().__init__(theta, **kwargs)
        self.takeover_config = takeover_config or TakeoverConfig()
        if (
            self.takeover_config.lookahead_s <= 0
            or not 0 < self.takeover_config.support_threshold <= 1
        ):
            raise ValueError("Positive horizon and support threshold in (0,1] required")
        path = self.takeover_config.error_trajectories_path
        if path is None:
            self.error_bank = ErrorTrajectoryBank.nominal(
                int(self.takeover_config.lookahead_s / self.native_dt_s + 1e-6)
            )
        else:
            path = Path(path)
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[2] / path
            self.error_bank = ErrorTrajectoryBank.load(path)
        if self.native_dt_s != self.error_bank.dt:
            raise ValueError(
                "Empirical trajectories and execution require the same physical timestep"
            )
        for name in self.bridge_runtime_fields:
            setattr(self, name, None)
        self.interaction_raw = None
        self.interaction_score = None
        self.interaction_reason = None
        self.interaction_time = None

    bridge_runtime_fields = (
        "owned",
        "admission_frame",
        "admission_reason",
        "response_executed",
        "previous_tracking_actions",
        "previous_actual",
        "previous_valid",
        "plan_source_y",
        "plan_target_y",
        "plan_progress",
        "plan_started",
        "admission_influencer",
        "admission_time_s",
    )

    def snapshot_runtime(self):
        result = super().snapshot_runtime()
        result.update(
            {
                name: getattr(self, name).clone()
                for name in self.bridge_runtime_fields
                if getattr(self, name) is not None
            }
        )
        return result

    def restore_runtime(self, state):
        super().restore_runtime(state)
        for name in self.bridge_runtime_fields:
            value = state.get(name)
            setattr(
                self,
                name,
                None if value is None else value.clone().to(self.theta.device),
            )

    def _update_plan_maneuver(self, context):
        lanes, present, widths = torch_map_lane_geometry(
            context.map_polylines, context.map_polyline_valid
        )
        y = context.current[:, 1:, 1]
        if self.plan_source_y is None:
            first = context.history_valid[:, :, 1:].long().argmax(1, keepdim=True)
            past = context.history[:, :, 1:, 1].gather(1, first).squeeze(1)
            _, source, _, _ = torch_nearest_lane_geometry(past, lanes, present, widths)
            self.plan_source_y = source
            self.plan_target_y = source.clone()
            self.plan_started = torch.zeros_like(y, dtype=torch.bool)
            self.plan_progress = torch.zeros_like(y)
            # A cold start between centers has no settled source in its history.
            # Use the direction of the first substantial own-plan displacement.
            plan = context.planned_path_xy
            if plan is not None and plan.shape[1]:
                delta = plan[..., 1] - y[:, None]
                substantial = delta.abs() > 0.25
                first = substantial.long().argmax(1, keepdim=True)
                direction = delta.gather(1, first).squeeze(1).sign()
                behind = (y[..., None] - lanes[:, None]) * direction[..., None] > 0
                distance = (
                    (y[..., None] - lanes[:, None])
                    .abs()
                    .masked_fill(~(behind & present[:, None]), torch.inf)
                )
                candidate = lanes.gather(1, distance.argmin(-1))
                cold_between = (
                    (past - source).abs().gt(0.25)
                    & substantial.any(1)
                    & distance.amin(-1).isfinite()
                )
                self.plan_source_y = torch.where(cold_between, candidate, source)
        completed = (
            self.plan_started
            & (y - self.plan_target_y).abs().lt(0.25)
            & context.current[:, 1:, 3].abs().lt(0.15)
        )
        self.plan_source_y = torch.where(
            completed, self.plan_target_y, self.plan_source_y
        )
        self.plan_started &= ~completed
        plan = context.planned_path_xy
        if plan is not None and context.response_index < plan.shape[1]:
            _, centers, _, _ = torch_nearest_lane_geometry(
                plan[:, context.response_index :, :, 1], lanes, present, widths
            )
            different = (centers - self.plan_source_y[:, None]).abs() > 0.25
            first = different.long().argmax(1, keepdim=True)
            target = centers.gather(1, first).squeeze(1)
            target = torch.where(different.any(1), target, self.plan_source_y)
            self.plan_target_y = torch.where(
                ~self.plan_started & ~self.owned, target, self.plan_target_y
            )
        distance = self.plan_target_y - self.plan_source_y
        progress = (y - self.plan_source_y) / torch.where(
            distance.abs() > 0.25, distance, torch.ones_like(distance)
        )
        start = distance.abs().gt(0.25) & (
            (y - self.plan_source_y) * distance.sign()
        ).gt(0.25)
        self.plan_started |= start & ~self.owned
        self.plan_progress = torch.where(distance.abs() > 0.25, progress.clamp(0, 1), 0)

    def _admission(self, context):
        cfg = self.takeover_config
        forecast = forecast_bundle(
            context,
            self.response_executed,
            self.error_bank,
            horizon_s=cfg.lookahead_s,
            dt=self.native_dt_s,
        )
        score, time, neighbour = conflict_scores(
            context,
            self.owned,
            self.error_bank,
            horizon_s=cfg.lookahead_s,
            dt=self.native_dt_s,
            support_threshold=cfg.support_threshold,
            forecast_owned=self.response_executed,
            forecast=forecast,
        )
        previous = None
        if self.previous_actual is not None:
            previous = self.previous_actual, self.previous_valid
        elif context.history.shape[1] > 1:
            previous = context.history[:, -2], context.history_valid[:, -2]
        raw, influence, event_time = interaction_scores(
            context, forecast, previous, self.plan_started
        )
        raw = torch.where(
            (context.current_valid[:, 1:] & ~self.owned)[..., None], raw, 0
        )
        normalized = raw / raw.new_tensor(
            [cfg.following_scale_s, cfg.release_scale_per_s]
        )
        self.interaction_score, which = normalized.max(-1)
        self.interaction_raw = raw
        noncollision = self.interaction_score.ge(
            cfg.interaction_threshold
        ) & self.interaction_score.gt(0)
        conflict = score.ge(cfg.support_threshold)
        self.interaction_reason = torch.where(noncollision & ~conflict, which + 4, 0)
        selected_time = event_time.gather(-1, which[..., None]).squeeze(-1)
        selected_influence = influence.gather(-1, which[..., None]).squeeze(-1)
        self.interaction_time = torch.where(
            conflict, time, torch.where(noncollision, selected_time, -1)
        )
        neighbour = torch.where(noncollision & ~conflict, selected_influence, neighbour)
        return (
            (conflict | noncollision) & ~self.owned & context.current_valid[:, 1:],
            score,
            time,
            neighbour,
        )

    def _adopt_current_maneuver(self, context, entered, centre):
        """Continue an underway lane change from its actual position and motion."""
        npc = context.current[:, 1:]
        self._initialize_lane_runtime(npc)
        target = torch.where(self.plan_started, self.plan_target_y, centre)
        self._autonomous_lane_target_y = torch.where(
            entered, target, self._autonomous_lane_target_y
        )
        ongoing = entered & self.plan_started
        if ongoing.any():
            style = (
                None if self.maneuver_latent is None else self.maneuver_latent[..., 2:4]
            )
            if style is None:
                style = torch.zeros((*npc.shape[:2], 2), device=npc.device)
            if style.ndim == 2:
                style = style[None].expand(len(npc), -1, -1)
            path = self.lane_trajectory.plan(
                npc, context.vehicle_dimensions[:, 1:], target, style
            )
            self._lane_path_coefficients = torch.where(
                ongoing[..., None], path[0], self._lane_path_coefficients
            )
            self._lane_path_origin = torch.where(
                ongoing, path[1], self._lane_path_origin
            )
            self._lane_path_distance = torch.where(
                ongoing, path[2], self._lane_path_distance
            )
            self._autonomous_lane_active |= ongoing
            self._maneuver_phase = torch.where(
                ongoing, int(ManeuverPhase.EXECUTE), self._maneuver_phase
            )
        self._autonomous_lane_ever |= entered
        self._autonomous_lane_initial_active |= ongoing

    def forward(self, context, *, deterministic=False):
        del deterministic
        valid = context.current_valid[:, 1:]
        if self.owned is None:
            self.owned = torch.zeros_like(valid)
            self.admission_frame = torch.full_like(valid, -1, dtype=torch.long)
            self.admission_reason = torch.zeros_like(valid, dtype=torch.long)
            self.response_executed = torch.zeros_like(valid)
            self.admission_influencer = torch.full_like(self.admission_frame, -1)
            self.admission_time_s = context.current.new_full(valid.shape, -1)
        if (
            self.previous_tracking_actions is not None
            and context.previous_background_actions is not None
        ):
            changed = ~torch.isclose(
                context.previous_background_actions,
                self.previous_tracking_actions,
                rtol=1e-5,
                atol=1e-6,
            ).all(-1)
            self.response_executed |= self.owned & changed
        # Retain the physically realised tracking control for the next tick's
        # execution check; an ownership label alone cannot propagate response.
        expected = context.base_actions[:, 0].detach().clone()
        speed = context.current[:, 1:, 2:4].norm(dim=-1)
        next_speed = speed + expected[..., 0] * self.native_dt_s
        expected[..., 0] = torch.where(
            (next_speed < 0) | (next_speed > 100), 0, expected[..., 0]
        )
        self.previous_tracking_actions = expected
        self.interaction_raw = context.current.new_zeros((*valid.shape, 2))
        self.interaction_score = context.current.new_zeros(valid.shape)
        self.interaction_reason = torch.zeros_like(self.admission_reason)
        self.interaction_time = context.current.new_full(valid.shape, -1)
        score = torch.zeros_like(valid, dtype=context.current.dtype)
        time = torch.full_like(score, -1)
        neighbour = torch.full_like(self.admission_frame, -1)
        entered = torch.zeros_like(valid)
        if (valid & ~self.owned).any():
            self._update_plan_maneuver(context)
            entered, score, time, neighbour = self._admission(context)
        reason = torch.where(entered, 1 if context.response_index == 0 else 2, 0)
        reason = torch.where(
            entered & self.interaction_reason.gt(0), self.interaction_reason, reason
        )
        self.previous_actual = context.current.detach().clone()
        self.previous_valid = context.current_valid.clone()
        # End-of-reference is a single handover too, never a return to tracking.
        if (
            context.planned_path_xy is not None
            and context.response_index >= context.planned_path_xy.shape[1]
        ):
            expired = valid & ~self.owned
            entered |= expired
            reason = torch.where(expired, 3, reason)
        self.owned |= entered
        self.admission_frame = torch.where(
            entered, context.response_index, self.admission_frame
        )
        self.admission_reason = torch.where(entered, reason, self.admission_reason)
        self.admission_influencer = torch.where(
            entered, neighbour, self.admission_influencer
        )
        self.admission_time_s = torch.where(
            entered, self.interaction_time, self.admission_time_s
        )
        diagnostics = dict(
            takeover_started=entered,
            takeover_owned=self.owned.clone(),
            takeover_motion_changed=self.response_executed.clone(),
            takeover_reason=reason,
            takeover_score=score,
            takeover_conflict_time_s=time,
            takeover_influencer=neighbour,
            takeover_interaction_score=self.interaction_score,
            takeover_following_score_s=self.interaction_raw[..., 0],
            takeover_release_score_per_s=self.interaction_raw[..., 1],
            takeover_interaction_time_s=self.interaction_time,
        )
        active = self.owned & valid
        base = context.base_actions
        theta = (
            self.theta
            if self.theta.ndim == 3
            else self.theta[None].expand(len(base), -1, -1)
        )
        gap, speed, leader_speed, leader = nearest_leader_observation(
            context.current,
            context.current_valid,
            vehicle_lengths_m=context.vehicle_dimensions[..., 0],
            vehicle_widths_m=context.vehicle_dimensions[..., 1],
        )
        acceleration, _ = self.driver_acceleration(
            context, theta, gap, speed, leader_speed, leader
        )
        if not active.any():
            return ReactionControllerOutput(
                actions=base,
                alpha=active.float(),
                delta_ax=torch.zeros_like(active, dtype=base.dtype),
                active=active,
                **diagnostics,
            )
        lanes, present, widths = torch_map_lane_geometry(
            context.map_polylines, context.map_polyline_valid
        )
        _, centre, _, _ = torch_nearest_lane_geometry(
            context.current[:, 1:, 1], lanes, present, widths
        )
        if entered.any():
            self._adopt_current_maneuver(context, entered, centre)
        yaw = frenet_feedback(
            context.current[:, 1:],
            self._autonomous_lane_target_y,
            torch.zeros_like(centre),
            torch.zeros_like(centre),
            frequency=self.lane_trajectory.frequency,
            acceleration_limit=self.lane_trajectory.acceleration_limit,
            jerk_limit=self.lane_trajectory.jerk_limit,
            dt=self.native_dt_s,
        )
        actions = torch.stack((acceleration, yaw), -1)[:, None]
        no_plan = replace(
            context,
            base_actions=actions,
            planned_path_xy=None,
            planned_current_xy=None,
            planned_terminal_xy=None,
        )
        actions, lane_active, mobil = self._autonomous_lane_response(
            no_plan,
            actions,
            gap=gap,
            speed=speed,
            leader_speed=leader_speed,
            leader_index=leader,
            leader_acceleration=context.current[..., 4].gather(1, leader.clamp_min(0)),
            leader_y=context.current[..., 1].gather(1, leader.clamp_min(0)),
            idm=acceleration,
            theta=theta,
            longitudinal_active=active,
            lateral_response_active=active,
            maneuver_permission=active,
        )
        previous = context.previous_background_actions
        if previous is None:
            from .core.dynamics import KinematicTrafficDynamics

            previous = KinematicTrafficDynamics.controls_from_highd_actions(
                context.current[:, 1:, 4:6], context.current[:, 1:]
            )
        step = self.correction_jerk_limit_mps3 * self.native_dt_s
        actions[:, 0, :, 0] = actions[:, 0, :, 0].clamp(
            previous[..., 0] - step, previous[..., 0] + step
        )
        actions[..., 0] = actions[..., 0].clamp(
            context.cfg.min_acceleration_mps2, context.cfg.max_acceleration_mps2
        )
        actions[..., 1] = actions[..., 1].clamp(
            -context.cfg.max_yaw_rate_rps, context.cfg.max_yaw_rate_rps
        )
        actions = (
            torch.where(active[:, None, :, None], actions, base)
            * valid[:, None, :, None]
        )
        return ReactionControllerOutput(
            actions=actions,
            alpha=active.float(),
            delta_ax=actions[:, 0, :, 0] - base[:, 0, :, 0],
            active=active,
            rule_action_ax=acceleration,
            autonomous_lane_active=lane_active,
            mobil_incentive_mps2=None if mobil is None else mobil.incentive_mps2,
            realized_leader_index=leader,
            **diagnostics,
        )
