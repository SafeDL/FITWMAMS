"""Prior-driven, two-time-scale HiQR-v2 world model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as functional

from world_model.src.core.dynamics import DynamicsConfig, KinematicTrafficDynamics
from world_model.src.core.initial_behavior_anchor import (
    start_state_from_flow_tensor,
    summarize_first_second_states,
)
from world_model.src.hiqr.encoder import UnifiedRelationalQueryEncoder

from .config import HiQRV2Config
from .decoder import StateAwarePlanContinuationDecoder
from .filter import FilterState, ObservedHierarchicalInteractionFilter, masked_mean


@dataclass
class _ResponseCoreResult:
    """Public plan output plus training-only distributions and encodings."""

    output: dict[str, Any]
    agents: torch.Tensor
    scene: torch.Tensor
    prior_scene: tuple[torch.Tensor, torch.Tensor]
    prior_agent_log_std: torch.Tensor
    background_valid: torch.Tensor


class HiQRV2WorldModel(torch.nn.Module):
    """HiQR-v2 with prior-driven closed-loop state and local posterior auxiliary."""

    model_type = "prior_driven_hierarchical_interaction_query_world_model_v2"

    def __init__(self, cfg: HiQRV2Config | None = None) -> None:
        super().__init__()
        self.cfg = cfg or HiQRV2Config()
        self.encoder = UnifiedRelationalQueryEncoder(self.cfg)
        self.filter = ObservedHierarchicalInteractionFilter(self.cfg)
        self.decoder = StateAwarePlanContinuationDecoder(self.cfg)
        self.dynamics = KinematicTrafficDynamics(
            DynamicsConfig(
                acceleration_min_mps2=self.cfg.min_acceleration,
                acceleration_max_mps2=self.cfg.max_acceleration,
            )
        )
        self.flow_schema_sha256: str | None = None

    @staticmethod
    def _ego_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        agents = int(batch["agent_states"].shape[2])
        return functional.one_hot(
            batch["ego_index"].long().clamp(0, agents - 1), agents
        ).bool()

    @staticmethod
    def flow_condition_to_scene(flow_condition: torch.Tensor, slot_valid: torch.Tensor):
        return start_state_from_flow_tensor(flow_condition, slot_valid)[:3]

    def initialize_start(
        self,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        ego_mask: torch.Tensor,
        map_polylines: torch.Tensor,
        map_polyline_valid: torch.Tensor,
        raw_b0: torch.Tensor,
        behavior_anchor_valid: torch.Tensor,
    ) -> FilterState:
        agents, scene, _, _ = self.encoder(
            None,
            None,
            current,
            current_valid,
            ego_mask,
            map_polylines,
            map_polyline_valid,
            mode="start",
        )
        return self.filter.initialize(
            scene, agents, raw_b0, behavior_anchor_valid.bool()
        )

    def _integrate_background_actions(
        self, current: torch.Tensor, actions: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        state = current[:, 1:]
        values: list[torch.Tensor] = []
        for frame in range(actions.shape[1]):
            state = self.dynamics.step(
                state, actions[:, frame], valid[:, 1:], self.cfg.simulation_dt_s
            )
            values.append(state)
        return torch.stack(values, dim=1)

    def _expected_ego(self, current: torch.Tensor) -> torch.Tensor:
        times = torch.arange(
            1, int(self.cfg.plan_frames) + 1, dtype=current.dtype, device=current.device
        )[:, None] * float(self.cfg.simulation_dt_s)
        current_ego = current[:, :1]
        expected = (
            current_ego[:, None].expand(-1, int(self.cfg.plan_frames), -1, -1).clone()
        )
        expected[..., :2] = (
            current_ego[:, None, :, :2]
            + times[None, :, None] * current_ego[:, None, :, 2:4]
        )
        return expected

    def _logged_ego_controls(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Recover exogenous ego controls from adjacent logged states.

        These controls are consumed only by the physics loop at their realized
        25 Hz tick.  They are never inputs to the response planner.
        """
        source_speed = torch.linalg.vector_norm(source[..., 2:4], dim=-1)
        target_speed = torch.linalg.vector_norm(target[..., 2:4], dim=-1)
        acceleration = (target_speed - source_speed) / float(self.cfg.simulation_dt_s)
        source_heading = self.dynamics._safe_heading(source[..., 3], source[..., 2])
        target_heading = self.dynamics._safe_heading(target[..., 3], target[..., 2])
        heading_delta = torch.atan2(
            torch.sin(target_heading - source_heading),
            torch.cos(target_heading - source_heading),
        )
        yaw_rate = heading_delta / float(self.cfg.simulation_dt_s)
        controls = torch.stack((acceleration, yaw_rate), dim=-1)
        return controls * valid[..., None].float()

    def _response_core(
        self,
        history: torch.Tensor | None,
        history_valid: torch.Tensor | None,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        ego_mask: torch.Tensor,
        map_polylines: torch.Tensor,
        map_polyline_valid: torch.Tensor,
        *,
        filter_state: FilterState,
        previous_buffer: torch.Tensor | None = None,
        previous_current: torch.Tensor | None = None,
        previous_background_states: torch.Tensor | None = None,
        previous_expected_ego: torch.Tensor | None = None,
        slow_scene: torch.Tensor | None = None,
        response_index: int = 0,
        deterministic: bool = True,
        scene_standard_normal: torch.Tensor | None = None,
        agent_standard_normal: torch.Tensor | None = None,
    ) -> _ResponseCoreResult:
        """Run the single causal response implementation used online and offline."""
        mode = "start" if previous_buffer is None else "roll"
        agents, scene, _, _ = self.encoder(
            history,
            history_valid,
            current,
            current_valid,
            ego_mask,
            map_polylines,
            map_polyline_valid,
            mode=mode,
        )
        observed = (
            filter_state
            if previous_buffer is None
            else self.filter.observe(
                filter_state,
                agents,
                scene,
                current,
                previous_current,
                current_valid,
            )
        )
        prior_scene = self.filter.prior_scene(observed, scene)
        refresh = (
            slow_scene is None
            or int(response_index) % int(self.cfg.scene_mode_responses) == 0
        )
        if refresh:
            mean_g, log_g = prior_scene
            if deterministic:
                slow_scene = mean_g
            else:
                noise = (
                    torch.randn_like(mean_g)
                    if scene_standard_normal is None
                    else scene_standard_normal.to(mean_g)
                )
                slow_scene = mean_g + noise * torch.exp(log_g)
        assert slow_scene is not None
        mean_z, log_z = self.filter.prior_agents(observed, agents, slow_scene)
        if deterministic:
            z = mean_z
        else:
            noise = (
                torch.randn_like(mean_z)
                if agent_standard_normal is None
                else agent_standard_normal.to(mean_z)
            )
            z = mean_z + noise * torch.exp(log_z) * float(self.cfg.residual_noise_scale)
        background = current_valid.clone()
        background[:, 0] = False
        z = z * background[..., None].float()
        decoded = self.decoder(
            agents,
            observed.global_hidden,
            observed.agent_hidden,
            slow_scene,
            z,
            current,
            current_valid,
            previous_buffer,
            previous_background_states,
            previous_expected_ego,
        )
        plan = decoded["background_future_actions"]
        return _ResponseCoreResult(
            output={
                **decoded,
                "filter_state": observed,
                "slow_scene": slow_scene,
                "scene_latent": slow_scene,
                "agent_residual": z,
                "background_future_states": self._integrate_background_actions(
                    current, plan, current_valid
                ),
                "expected_ego_states": self._expected_ego(current),
                "scene_refreshed": refresh,
            },
            agents=agents,
            scene=scene,
            prior_scene=prior_scene,
            prior_agent_log_std=log_z,
            background_valid=background,
        )

    def plan_step(
        self,
        history: torch.Tensor | None,
        history_valid: torch.Tensor | None,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        ego_mask: torch.Tensor,
        map_polylines: torch.Tensor,
        map_polyline_valid: torch.Tensor,
        *,
        filter_state: FilterState,
        previous_buffer: torch.Tensor | None = None,
        previous_current: torch.Tensor | None = None,
        previous_background_states: torch.Tensor | None = None,
        previous_expected_ego: torch.Tensor | None = None,
        slow_scene: torch.Tensor | None = None,
        response_index: int = 0,
        deterministic: bool = True,
        scene_standard_normal: torch.Tensor | None = None,
        agent_standard_normal: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Plan one causal V2 response without exposing training-only context."""
        return self._response_core(
            history,
            history_valid,
            current,
            current_valid,
            ego_mask,
            map_polylines,
            map_polyline_valid,
            filter_state=filter_state,
            previous_buffer=previous_buffer,
            previous_current=previous_current,
            previous_background_states=previous_background_states,
            previous_expected_ego=previous_expected_ego,
            slow_scene=slow_scene,
            response_index=response_index,
            deterministic=deterministic,
            scene_standard_normal=scene_standard_normal,
            agent_standard_normal=agent_standard_normal,
        ).output

    def _target_plan(
        self, batch: dict[str, torch.Tensor], response: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states, valid, recorded = (
            batch["agent_states"],
            batch["agent_valid"],
            batch["actions_highd"],
        )
        start, horizon = int(response) * int(self.cfg.execute_frames), int(
            self.cfg.plan_frames
        )
        count = min(horizon, int(recorded.shape[1]) - start)
        batch_size, agents = states.shape[0], states.shape[2]
        target = states.new_zeros((batch_size, horizon, agents, 6))
        target_valid = torch.zeros(
            (batch_size, horizon, agents), dtype=torch.bool, device=states.device
        )
        actions = states.new_zeros((batch_size, horizon, agents - 1, 2))
        action_valid = torch.zeros(
            (batch_size, horizon, agents - 1), dtype=torch.bool, device=states.device
        )
        if count:
            first = self.cfg.first_future_state_index + start
            target[:, :count] = states[:, first : first + count]
            target_valid[:, :count] = valid[:, first : first + count]
            source = states[
                :,
                self.cfg.anchor_state_index
                + start : self.cfg.anchor_state_index
                + start
                + count,
                1:,
            ]
            actions[:, :count] = self.dynamics.controls_from_highd_actions(
                recorded[:, start : start + count], source
            )
            action_valid[:, :count] = target_valid[:, :count, 1:]
        return target, target_valid, actions, action_valid

    @staticmethod
    def _interaction_loss(
        predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        predicted_bg, target_bg = predicted[:, :, 1:], target[:, :, 1:]
        ego, target_ego = predicted[:, :, :1], target[:, :, :1]
        gap = (
            (predicted_bg[..., 0] - ego[..., 0]).abs()
            - (target_bg[..., 0] - target_ego[..., 0]).abs()
        ).abs()
        velocity = (
            (predicted_bg[..., 2] - ego[..., 2])
            - (target_bg[..., 2] - target_ego[..., 2])
        ).abs()
        return masked_mean(gap + 0.25 * velocity, valid)

    @staticmethod
    def _target_aware_physical_loss(
        predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        def barrier(values: torch.Tensor) -> torch.Tensor:
            background, ego = values[:, :, 1:], values[:, :, :1]
            dx = (background[..., 0] - ego[..., 0]).abs()
            dy = (background[..., 1] - ego[..., 1]).abs()
            return functional.softplus((4.5 - dx) / 0.75) * functional.softplus(
                (1.0 - dy) / 0.25
            )

        return masked_mean(functional.relu(barrier(predicted) - barrier(target)), valid)

    def _gap_ttc_loss(
        self, predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        background, target_bg, target_ego = (
            predicted[:, :, 1:],
            target[:, :, 1:],
            target[:, :, :1],
        )
        target_dx = target_bg[..., 0] - target_ego[..., 0]
        same_lane = (target_bg[..., 1] - target_ego[..., 1]).abs() < 0.5 * float(
            self.cfg.lane_width_m
        )
        following = valid.bool() & same_lane & (target_dx > 0.0)
        pred_gap = (background[..., 0] - target_ego[..., 0] - 4.8).clamp_min(0.0)
        target_gap = (target_dx - 4.8).clamp_min(0.0)
        pred_closing = (target_ego[..., 2] - background[..., 2]).clamp_min(0.0)
        target_closing = (target_ego[..., 2] - target_bg[..., 2]).clamp_min(0.0)
        closing = following & (target_closing > 1.0e-3)
        pred_ttc = torch.where(
            closing,
            pred_gap / pred_closing.clamp_min(1.0e-3),
            pred_gap.new_full(pred_gap.shape, 10.0),
        ).clamp_max(10.0)
        target_ttc = torch.where(
            closing,
            target_gap / target_closing.clamp_min(1.0e-3),
            target_gap.new_full(target_gap.shape, 10.0),
        ).clamp_max(10.0)
        return masked_mean(
            (pred_gap - target_gap).abs() / 10.0, following
        ) + masked_mean((pred_ttc - target_ttc).abs() / 10.0, closing)

    def _lane_loss(
        self, states: torch.Tensor, current: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        lateral = (states[..., 1] - current[:, None, 1:, 1]).abs()
        return masked_mean(
            functional.relu(lateral - 1.25 * float(self.cfg.lane_width_m)), valid
        )

    def _execute_logged_response(
        self,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        plan: torch.Tensor,
        ego_controls: torch.Tensor,
        ego_control_valid: torch.Tensor,
        *,
        first_frame: int,
        frame_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute one response with causal logged ego controls."""
        generated = current.clone()
        frames: list[torch.Tensor] = []
        valid_frames: list[torch.Tensor] = []
        for frame in range(frame_count):
            rollout_frame = first_frame + frame
            physical = current.new_zeros((current.shape[0], current.shape[1], 2))
            physical[:, 0] = ego_controls[:, rollout_frame]
            physical[:, 1:] = plan[:, frame]
            current_valid = current_valid.clone()
            current_valid[:, 0] &= ego_control_valid[:, rollout_frame]
            generated = self.dynamics.step(
                generated, physical, current_valid, self.cfg.simulation_dt_s
            )
            # Logged ego control is an exogenous current-tick environment input.
            # Logged states and future background labels never enter the rollout.
            generated = generated * current_valid[..., None].float()
            frames.append(generated)
            valid_frames.append(current_valid)
        return (
            torch.stack(frames, dim=1),
            torch.stack(valid_frames, dim=1),
            current_valid,
        )

    def _posterior_auxiliary_terms(
        self,
        *,
        enabled: bool,
        filter_state: FilterState,
        agents: torch.Tensor,
        scene: torch.Tensor,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        target: torch.Tensor,
        target_valid: torch.Tensor,
        target_actions: torch.Tensor,
        target_action_valid: torch.Tensor,
        prior_scene: tuple[torch.Tensor, torch.Tensor],
        scene_refresh: bool,
        posterior_slow_scene: torch.Tensor | None,
        previous_buffer: torch.Tensor | None,
        previous_states: torch.Tensor | None,
        previous_expected_ego: torch.Tensor | None,
        execute_count: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        zero = current.new_zeros(())
        terms = {
            "posterior_position": zero,
            "posterior_action": zero,
            "scene_kl": zero,
            "agent_kl": zero,
            "prior_distillation": zero,
            "diversity_floor": zero,
            "posterior_scene_std": zero,
            "posterior_agent_std": zero,
        }
        if not enabled:
            return terms, posterior_slow_scene

        qg, qz, posterior = self.filter.posterior(
            filter_state,
            agents,
            scene,
            current,
            current_valid,
            target,
            target_valid,
            prior_scene,
            fixed_scene_latent=None if scene_refresh else posterior_slow_scene,
        )
        if scene_refresh:
            posterior_slow_scene = qg
        q_plan = self.decoder(
            agents,
            filter_state.global_hidden,
            filter_state.agent_hidden,
            qg,
            qz,
            current,
            current_valid,
            previous_buffer,
            previous_states,
            previous_expected_ego,
        )["background_future_actions"]
        q_states = self._integrate_background_actions(current, q_plan, current_valid)
        execute_valid = target_valid[:, :execute_count, 1:]
        terms.update(posterior)
        terms.update(
            {
                "posterior_position": masked_mean(
                    (
                        q_states[:, :execute_count, :, :2]
                        - target[:, :execute_count, 1:, :2]
                    )
                    .abs()
                    .mean(dim=-1),
                    execute_valid,
                ),
                "posterior_action": masked_mean(
                    (q_plan[:, :execute_count] - target_actions[:, :execute_count])
                    .abs()
                    .mean(dim=-1),
                    target_action_valid[:, :execute_count],
                ),
            }
        )
        return terms, posterior_slow_scene

    def _response_terms(
        self,
        *,
        predicted: torch.Tensor,
        target: torch.Tensor,
        target_valid: torch.Tensor,
        plan: torch.Tensor,
        plan_states: torch.Tensor,
        target_actions: torch.Tensor,
        target_action_valid: torch.Tensor,
        current: torch.Tensor,
        decoded: dict[str, Any],
        prior_scene_log_std: torch.Tensor,
        prior_agent_log_std: torch.Tensor,
        background: torch.Tensor,
        posterior: dict[str, torch.Tensor],
        execute_count: int,
    ) -> dict[str, torch.Tensor]:
        execute_valid = target_valid[:, :execute_count, 1:]
        full_valid = target_valid[:, :, 1:]
        target_executed = target[:, :execute_count]
        masks = decoded["background_future_action_masks"]
        carried = masks["carried"]
        return {
            "position": masked_mean(
                (predicted[:, :, 1:, :2] - target_executed[:, :, 1:, :2])
                .abs()
                .mean(dim=-1),
                execute_valid,
            ),
            "velocity": masked_mean(
                (predicted[:, :, 1:, 2:4] - target_executed[:, :, 1:, 2:4])
                .abs()
                .mean(dim=-1),
                execute_valid,
            ),
            "action": masked_mean(
                (plan[:, :execute_count] - target_actions[:, :execute_count])
                .abs()
                .mean(dim=-1),
                target_action_valid[:, :execute_count],
            ),
            "plan_position": masked_mean(
                (plan_states[..., :2] - target[:, :, 1:, :2]).abs().mean(dim=-1),
                full_valid,
            ),
            "plan_action": masked_mean(
                (plan - target_actions).abs().mean(dim=-1), target_action_valid
            ),
            "interaction": self._interaction_loss(
                predicted, target_executed, execute_valid
            ),
            "physical": (
                self._target_aware_physical_loss(
                    predicted, target_executed, execute_valid
                )
                if self.cfg.physical_mode == "target_aware"
                else current.new_zeros(())
            ),
            "jerk": (
                plan.new_zeros(())
                if plan.shape[1] < 2
                else (plan[:, 1:] - plan[:, :-1]).abs().mean()
            ),
            "lane": self._lane_loss(plan_states, current, full_valid),
            "gap_ttc": self._gap_ttc_loss(predicted, target_executed, execute_valid),
            "gate": masked_mean(decoded["continuation_gate"].squeeze(-1), carried),
            "revision_rate": masked_mean(masks["revised"].float(), carried),
            "emergency_rate": masked_mean(masks["emergency"].float(), carried),
            "prior_scene_std": torch.exp(prior_scene_log_std).mean(),
            "prior_agent_std": masked_mean(
                torch.exp(prior_agent_log_std).mean(dim=-1), background
            ),
            **posterior,
        }

    def _b0_summary_loss(
        self,
        batch: dict[str, torch.Tensor],
        initial_current: torch.Tensor,
        initial_valid: torch.Tensor,
        plan_states: torch.Tensor,
        target_valid: torch.Tensor,
    ) -> torch.Tensor:
        prefix_states = torch.cat((initial_current[:, None, 1:], plan_states), dim=1)
        prefix_valid = torch.cat(
            (
                initial_valid[:, None, 1:],
                target_valid[:, : int(self.cfg.plan_frames), 1:],
            ),
            dim=1,
        )
        summary, summary_valid = summarize_first_second_states(
            prefix_states, prefix_valid
        )
        summary_valid &= batch["behavior_anchor_valid"].bool()
        return masked_mean(
            (summary - batch["behavior_anchor_raw"]).abs().mean(dim=-1),
            summary_valid,
        )

    def _available_frames(self, batch: dict[str, torch.Tensor]) -> int:
        return min(
            int(self.cfg.rollout_frames),
            int(batch["actions_highd"].shape[1]),
            int(batch["agent_states"].shape[1]) - self.cfg.first_future_state_index,
        )

    def _rollout(
        self,
        batch: dict[str, torch.Tensor],
        *,
        response_steps: int | None,
        deterministic: bool,
        posterior_auxiliary: bool,
        tbptt_steps: int = 0,
    ) -> dict[str, Any]:
        states, valid = batch["agent_states"], batch["agent_valid"]
        requested = (
            self.cfg.response_steps if response_steps is None else int(response_steps)
        )
        total_frames = min(
            self.cfg.rollout_frames_for_responses(requested),
            self._available_frames(batch),
        )
        if total_frames < 1:
            raise ValueError("HiQR-v2 rollout has no supervised transition")
        steps = (total_frames + int(self.cfg.execute_frames) - 1) // int(
            self.cfg.execute_frames
        )
        ego_mask = self._ego_mask(batch)
        if not torch.all(ego_mask[:, 0]):
            raise ValueError("HiQR-v2 requires fixed [ego, six backgrounds] slots")
        current, current_valid = (
            states[:, self.cfg.anchor_state_index],
            valid[:, self.cfg.anchor_state_index],
        )
        initial_current, initial_valid = current, current_valid
        filter_state = self.initialize_start(
            current,
            current_valid,
            ego_mask,
            batch["map_polylines"],
            batch["map_polyline_valid"],
            batch["behavior_anchor_raw"],
            batch["behavior_anchor_valid"],
        )
        ego_source = states[
            :,
            self.cfg.anchor_state_index : self.cfg.anchor_state_index + total_frames,
            0,
        ]
        ego_target = states[
            :,
            self.cfg.first_future_state_index : self.cfg.first_future_state_index
            + total_frames,
            0,
        ]
        ego_control_valid = (
            valid[
                :,
                self.cfg.anchor_state_index : self.cfg.anchor_state_index
                + total_frames,
                0,
            ]
            & valid[
                :,
                self.cfg.first_future_state_index : self.cfg.first_future_state_index
                + total_frames,
                0,
            ]
        )
        ego_controls = self._logged_ego_controls(
            ego_source, ego_target, ego_control_valid
        )
        history = history_valid = None
        previous_buffer = previous_current = previous_states = previous_expected_ego = (
            None
        )
        slow_scene: torch.Tensor | None = None
        posterior_slow_scene: torch.Tensor | None = None
        predicted_frames: list[torch.Tensor] = []
        plans: list[torch.Tensor] = []
        scene_latents: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []
        gate_values: list[torch.Tensor] = []
        masks: dict[str, list[torch.Tensor]] = {
            "carried": [],
            "revised": [],
            "emergency": [],
            "valid": [],
        }
        term_rows: list[dict[str, torch.Tensor]] = []
        b0_summary = current.new_zeros(())
        for response in range(steps):
            target, target_valid, target_actions, target_action_valid = (
                self._target_plan(batch, response)
            )
            step_current, step_valid = current, current_valid
            core = self._response_core(
                history,
                history_valid,
                current,
                current_valid,
                ego_mask,
                batch["map_polylines"],
                batch["map_polyline_valid"],
                filter_state=filter_state,
                previous_buffer=previous_buffer,
                previous_current=previous_current,
                previous_background_states=previous_states,
                previous_expected_ego=previous_expected_ego,
                slow_scene=slow_scene,
                response_index=response,
                deterministic=deterministic,
            )
            decoded = core.output
            filter_state = decoded["filter_state"]
            slow_scene = decoded["slow_scene"]
            z = decoded["agent_residual"]
            plan = decoded["background_future_actions"]
            plan_states = decoded["background_future_states"]
            expected_ego = decoded["expected_ego_states"]
            scene_refresh = decoded["scene_refreshed"]
            execute_count = min(
                int(self.cfg.execute_frames),
                total_frames - response * int(self.cfg.execute_frames),
            )
            predicted, generated_valid, current_valid = self._execute_logged_response(
                current,
                current_valid,
                plan,
                ego_controls,
                ego_control_valid,
                first_frame=response * int(self.cfg.execute_frames),
                frame_count=execute_count,
            )
            posterior_terms, posterior_slow_scene = self._posterior_auxiliary_terms(
                enabled=posterior_auxiliary,
                filter_state=filter_state,
                agents=core.agents,
                scene=core.scene,
                current=step_current,
                current_valid=step_valid,
                target=target,
                target_valid=target_valid,
                target_actions=target_actions,
                target_action_valid=target_action_valid,
                prior_scene=core.prior_scene,
                scene_refresh=scene_refresh,
                posterior_slow_scene=posterior_slow_scene,
                previous_buffer=previous_buffer,
                previous_states=previous_states,
                previous_expected_ego=previous_expected_ego,
                execute_count=execute_count,
            )
            if response == 0:
                b0_summary = self._b0_summary_loss(
                    batch,
                    initial_current,
                    initial_valid,
                    plan_states,
                    target_valid,
                )
            term_rows.append(
                self._response_terms(
                    predicted=predicted,
                    target=target,
                    target_valid=target_valid,
                    plan=plan,
                    plan_states=plan_states,
                    target_actions=target_actions,
                    target_action_valid=target_action_valid,
                    current=current,
                    decoded=decoded,
                    prior_scene_log_std=core.prior_scene[1],
                    prior_agent_log_std=core.prior_agent_log_std,
                    background=core.background_valid,
                    posterior=posterior_terms,
                    execute_count=execute_count,
                )
            )
            predicted_frames.extend(predicted.unbind(dim=1))
            plans.append(plan)
            scene_latents.append(slow_scene)
            residuals.append(z)
            gate_values.append(decoded["continuation_gate"])
            for name, value in decoded["background_future_action_masks"].items():
                masks[name].append(value)
            (
                previous_buffer,
                previous_current,
                previous_states,
                previous_expected_ego,
            ) = (plan, current, plan_states, expected_ego)
            current = predicted[:, -1]
            history = torch.cat(
                (step_current[:, None] if history is None else history, predicted),
                dim=1,
            )[:, -int(self.cfg.plan_frames) :]
            history_valid = torch.cat(
                (
                    step_valid[:, None] if history_valid is None else history_valid,
                    generated_valid,
                ),
                dim=1,
            )[:, -int(self.cfg.plan_frames) :]
            if (
                tbptt_steps
                and (response + 1) % int(tbptt_steps) == 0
                and response + 1 < steps
            ):
                (
                    history,
                    history_valid,
                    current,
                    previous_buffer,
                    previous_current,
                    previous_states,
                    previous_expected_ego,
                    filter_state,
                    slow_scene,
                    posterior_slow_scene,
                ) = (
                    history.detach(),
                    history_valid.detach(),
                    current.detach(),
                    previous_buffer.detach(),
                    previous_current.detach(),
                    previous_states.detach(),
                    previous_expected_ego.detach(),
                    filter_state.detach(),
                    slow_scene.detach(),
                    (
                        None
                        if posterior_slow_scene is None
                        else posterior_slow_scene.detach()
                    ),
                )
        terms = {
            name: torch.stack([row[name] for row in term_rows]).mean()
            for name in term_rows[0]
        }
        terms["b0_summary"] = b0_summary
        return {
            "predicted_states": torch.stack(predicted_frames, dim=1),
            "target_states": states[
                :,
                self.cfg.first_future_state_index : self.cfg.first_future_state_index
                + total_frames,
            ],
            "target_valid": valid[
                :,
                self.cfg.first_future_state_index : self.cfg.first_future_state_index
                + total_frames,
            ],
            "background_future_actions": torch.stack(plans, dim=1),
            "scene_latent": torch.stack(scene_latents, dim=1),
            "agent_residual": torch.stack(residuals, dim=1),
            "continuation_gate": torch.stack(gate_values, dim=1),
            "applied_ego_controls": ego_controls,
            "applied_ego_control_valid": ego_control_valid,
            "background_future_action_masks": {
                name: torch.stack(values, dim=1) for name, values in masks.items()
            },
            "terms": terms,
        }

    def _objective(
        self,
        rollout: dict[str, Any],
        *,
        posterior_scale: float,
        kl_scale: float,
        diversity_scale: float,
    ) -> dict[str, torch.Tensor]:
        terms = rollout["terms"]
        loss = (
            self.cfg.position_weight * terms["position"]
            + self.cfg.velocity_weight * terms["velocity"]
            + self.cfg.action_weight * terms["action"]
            + self.cfg.plan_position_weight * terms["plan_position"]
            + self.cfg.plan_action_weight * terms["plan_action"]
            + self.cfg.interaction_weight * terms["interaction"]
            + self.cfg.physical_weight * terms["physical"]
            + self.cfg.jerk_weight * terms["jerk"]
            + self.cfg.lane_weight * terms["lane"]
            + self.cfg.gap_ttc_weight * terms["gap_ttc"]
            + self.cfg.b0_summary_weight * terms["b0_summary"]
            + float(posterior_scale)
            * self.cfg.posterior_aux_weight
            * (
                terms["posterior_position"]
                + self.cfg.action_weight * terms["posterior_action"]
            )
            + float(kl_scale)
            * (
                self.cfg.scene_kl_weight * terms["scene_kl"]
                + self.cfg.agent_kl_weight * terms["agent_kl"]
                + self.cfg.prior_distillation_weight * terms["prior_distillation"]
            )
            + float(diversity_scale)
            * self.cfg.diversity_weight
            * terms["diversity_floor"]
        )
        return {"loss": loss, **terms}

    def rollout_reconstruction(
        self,
        batch: dict[str, torch.Tensor],
        *,
        response_steps: int | None = None,
        deterministic: bool = True,
    ) -> dict[str, Any]:
        return self._rollout(
            batch,
            response_steps=response_steps,
            deterministic=deterministic,
            posterior_auxiliary=False,
        )

    def supervised_terms(
        self, batch: dict[str, torch.Tensor], *, response_steps: int | None = None
    ) -> dict[str, torch.Tensor]:
        return self._objective(
            self._rollout(
                batch,
                response_steps=response_steps,
                deterministic=True,
                posterior_auxiliary=False,
            ),
            posterior_scale=0.0,
            kl_scale=0.0,
            diversity_scale=0.0,
        )

    def diagnostic_rollout(
        self,
        batch: dict[str, torch.Tensor],
        *,
        response_steps: int | None = None,
    ) -> dict[str, Any]:
        """Return the deterministic prior rollout plus local posterior terms.

        Enabling the auxiliary branch does not alter the trajectory or any
        state carried to the next response; it only measures the prior/posterior
        reconstruction gap on the current response.
        """
        return self._rollout(
            batch,
            response_steps=response_steps,
            deterministic=True,
            posterior_auxiliary=True,
        )

    def forward_training(
        self,
        batch: dict[str, torch.Tensor],
        *,
        response_steps: int | None = None,
        tbptt_steps: int = 10,
        posterior_scale: float = 1.0,
        kl_scale: float = 1.0,
        diversity_scale: float = 1.0,
        deterministic_prior: bool = False,
        posterior_auxiliary: bool = True,
    ) -> dict[str, torch.Tensor]:
        rollout = self._rollout(
            batch,
            response_steps=response_steps,
            deterministic=deterministic_prior,
            posterior_auxiliary=posterior_auxiliary,
            tbptt_steps=tbptt_steps,
        )
        return self._objective(
            rollout,
            posterior_scale=posterior_scale,
            kl_scale=kl_scale,
            diversity_scale=diversity_scale,
        )

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "model_config": asdict(self.cfg),
            "state_dict": self.state_dict(),
            "flow_interface": {
                "input_dim": 76,
                "layout": "ego[vx,vy,ax,ay]+background_relative[6,6]+B0[6,6]",
                "b0_lifecycle": "per-agent filter initialization plus first-second summary consistency",
                "persistent_state": "observation-only global and per-agent filter; no posterior/prior latent intent",
                "scene_latent": "sampled every five 5 Hz responses and held within its one-second mode",
                "supervised_ego_control": "adjacent logged state transitions executed only by 25 Hz physics",
                "flow_schema_sha256": self.flow_schema_sha256,
            },
        }
