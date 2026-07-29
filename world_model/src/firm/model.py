"""FIRM-WM model, training rollout, and probabilistic control objective."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn

from world_model.src.core.dynamics import DynamicsConfig, KinematicTrafficDynamics
from world_model.src.core import ContinuousTrafficMemory
from world_model.src.core.initial_behavior_anchor import summarize_first_second_states

from .action_flow import JointActionFlow
from .config import FIRMConfig
from .encoder import CausalRelationEncoder
from .relational_plan_field import RelationalPlanField


def _masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid.float()
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


class PersistentWorldLatent(nn.Module):
    """START prior and low-frequency latent transition for one traffic world."""

    def __init__(self, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.prior = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, latent_dim * 2)
        )
        self.transition = nn.Sequential(
            nn.Linear(hidden_dim * 2 + latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )
        nn.init.zeros_(self.prior[-1].weight)
        nn.init.zeros_(self.prior[-1].bias)
        nn.init.zeros_(self.transition[-1].weight)
        nn.init.zeros_(self.transition[-1].bias)
        with torch.no_grad():
            self.transition[-1].bias[:latent_dim].fill_(2.0)
            self.transition[-1].bias[latent_dim:].fill_(-2.0)

    def prior_parameters(
        self, scene: torch.Tensor, start_memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_scale = self.prior(torch.cat((scene, start_memory), dim=-1)).chunk(2, -1)
        scale = 0.08 + 0.35 * torch.sigmoid(raw_scale)
        return mean, scale

    def initialize(
        self,
        scene: torch.Tensor,
        start_memory: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prior_mean, prior_scale = self.prior_parameters(scene, start_memory)
        return prior_mean + prior_scale * noise, prior_scale

    def forward(
        self,
        scene: torch.Tensor,
        memory: torch.Tensor,
        latent: torch.Tensor,
        innovation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rho_raw, sigma_raw = self.transition(
            torch.cat((scene, memory, latent), dim=-1)
        ).chunk(2, -1)
        rho = torch.sigmoid(rho_raw)
        sigma = 0.01 + 0.20 * torch.sigmoid(sigma_raw)
        return rho * latent + sigma * innovation, rho, sigma


class FIRMWorldModel(nn.Module):
    """Map-free relational world model with persistent innovations and action flow."""

    model_type = "firm_flow_initialized_relational_memory_world_model"
    def __init__(self, cfg: FIRMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_type = type(self).model_type
        self.encoder = CausalRelationEncoder(
            cfg.hidden_dim,
            dropout=cfg.dropout,
            lane_width_m=cfg.lane_width_m,
        )
        self.start_anchor = nn.Sequential(
            nn.Linear(42, cfg.hidden_dim), nn.SiLU(), nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        )
        self.flow_condition = nn.Sequential(
            nn.Linear(76, cfg.hidden_dim), nn.SiLU(), nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        )
        self.start_initializer = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 3, cfg.hidden_dim), nn.SiLU(), nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        )
        self.memory = ContinuousTrafficMemory(cfg.hidden_dim)
        self.world_latent = PersistentWorldLatent(cfg.hidden_dim, cfg.world_latent_dim)
        context_dim = cfg.hidden_dim * 2 + cfg.world_latent_dim
        # B0/zF is a reset condition, not a disposable initial-memory token.
        # It is concatenated to every action-flow context so its behavioural
        # constraint cannot be washed out by a long closed-loop memory path.
        conditioned_context_dim = context_dim + cfg.hidden_dim
        self.action_context = nn.Sequential(
            nn.Linear(conditioned_context_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.action_flow = JointActionFlow(
            cfg.hidden_dim,
            execute_frames=cfg.execute_frames,
            layers=cfg.action_flow_layers,
            max_jerk=(cfg.max_longitudinal_jerk, cfg.max_yaw_jerk),
        )
        self.plan_field = RelationalPlanField(
            cfg.hidden_dim,
            cfg.world_latent_dim,
            cfg.plan_frames,
        )
        self.dynamics = KinematicTrafficDynamics(
            DynamicsConfig(
                acceleration_min_mps2=cfg.min_acceleration,
                acceleration_max_mps2=cfg.max_acceleration,
            )
        )

    @staticmethod
    def _ego_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        agents = batch["agent_states"].shape[2]
        return torch.nn.functional.one_hot(
            batch["ego_index"].long().clamp(0, agents - 1), agents
        ).bool()

    @staticmethod
    def _start_anchor_features(
        anchor: torch.Tensor | None,
        anchor_valid: torch.Tensor | None,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if anchor is None:
            return torch.zeros((batch, 42), dtype=dtype, device=device)
        valid = (
            torch.ones(anchor.shape[:-1], dtype=torch.bool, device=device)
            if anchor_valid is None
            else anchor_valid.bool()
        )
        return torch.cat((anchor * valid[..., None].float(), valid[..., None].float()), dim=-1).reshape(batch, -1)

    @staticmethod
    def _flow_condition_from_start(
        current: torch.Tensor,
        current_valid: torch.Tensor,
        anchor: torch.Tensor | None,
        anchor_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        """Reconstruct Flow's 76-D C0/B0 condition without future frames.

        The condition is used when training highD sequences, for which an
        external Flow draw is not available.  At deployment callers pass the
        exact sampled Flow condition through ``flow_latent`` instead.
        """
        ego = current[:, :1]
        background = current[:, 1:]
        relative = torch.cat(
            (
                background[..., :2] - ego[..., :2],
                background[..., 2:4] - ego[..., 2:4],
                background[..., 4:6],
            ),
            dim=-1,
        )
        background_valid = current_valid[:, 1:]
        relative = relative * background_valid[..., None].float()
        if anchor is None:
            anchor = current.new_zeros((current.shape[0], 6, 6))
        if anchor_valid is not None:
            anchor = anchor * anchor_valid[..., None].float()
        return torch.cat((ego[:, 0, 2:6], relative.reshape(current.shape[0], -1), anchor.reshape(current.shape[0], -1)), dim=-1)

    def initialize(
        self,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        ego_mask: torch.Tensor,
        *,
        behavior_anchor: torch.Tensor | None = None,
        behavior_anchor_valid: torch.Tensor | None = None,
        flow_latent: torch.Tensor | None = None,
        world_noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """START from C0 and B0 only; no copied or fabricated history."""
        agents, scene = self.encoder(
            current[:, None], current_valid[:, None], current, current_valid, ego_mask
        )
        anchor = self.start_anchor(
            self._start_anchor_features(
                behavior_anchor,
                behavior_anchor_valid,
                batch=current.shape[0],
                device=current.device,
                dtype=current.dtype,
            )
        )
        flow_input = (
            self._flow_condition_from_start(
                current, current_valid, behavior_anchor, behavior_anchor_valid
            )
            if flow_latent is None
            else flow_latent
        )
        if flow_input.shape != (current.shape[0], 76):
            raise ValueError("FIRM START flow_latent must have shape [batch, 76]")
        flow = self.flow_condition(flow_input)
        memory = self.start_initializer(torch.cat((scene, anchor, flow), dim=-1))
        if world_noise is None:
            world_noise = current.new_zeros((current.shape[0], self.cfg.world_latent_dim))
        latent, scale = self.world_latent.initialize(scene, memory, world_noise)
        output = {
            "agent_context": agents,
            "scene_context": scene,
            "continuous_memory": memory,
            "world_latent": latent,
            "world_latent_scale": scale,
            "flow_condition": flow_input,
            "flow_embedding": flow,
        }
        return output

    def _controls_from_jerks(
        self, current: torch.Tensor, jerks: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        previous = self.dynamics.controls_from_highd_actions(
            current[:, 1:, 4:6], current[:, 1:]
        )
        controls: list[torch.Tensor] = []
        for frame in range(jerks.shape[1]):
            previous = previous + jerks[:, frame] * self.cfg.simulation_dt_s
            previous = torch.stack(
                (
                    previous[..., 0].clamp(
                        self.cfg.min_acceleration, self.cfg.max_acceleration
                    ),
                    previous[..., 1].clamp(
                        -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate
                    ),
                ),
                dim=-1,
            )
            controls.append(previous * valid[:, 1:, None].float())
        return torch.stack(controls, dim=1)

    def _target_controls(
        self, batch: dict[str, torch.Tensor], response: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start = response * self.cfg.execute_frames
        frames = self.cfg.plan_frames
        actions = batch["actions_highd"]
        count = min(frames, actions.shape[1] - start)
        controls = actions.new_zeros((actions.shape[0], frames, 6, 2))
        valid = torch.zeros(
            (actions.shape[0], frames, 6), dtype=torch.bool, device=actions.device
        )
        if count <= 0:
            return controls, valid
        states = batch["agent_states"][:, 24 + start : 24 + start + count, 1:]
        current_valid = batch["agent_valid"][:, 25 + start : 25 + start + count, 1:]
        converted = self.dynamics.controls_from_highd_actions(actions[:, start : start + count], states)
        controls[:, :count] = converted
        valid[:, :count] = current_valid
        return controls, valid

    def _target_jerks(
        self,
        current: torch.Tensor,
        controls: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        previous = self.dynamics.controls_from_highd_actions(
            current[:, 1:, 4:6], current[:, 1:]
        )
        output: list[torch.Tensor] = []
        for frame in range(controls.shape[1]):
            target = controls[:, frame]
            output.append((target - previous) / self.cfg.simulation_dt_s)
            previous = torch.where(valid[:, frame, :, None], target, previous)
        return torch.stack(output, dim=1)

    def _target_plan_states(
        self, batch: dict[str, torch.Tensor], response: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start = 25 + response * self.cfg.execute_frames
        stop = min(start + self.cfg.plan_frames, batch["agent_states"].shape[1])
        state = batch["agent_states"].new_zeros(
            (batch["agent_states"].shape[0], self.cfg.plan_frames, 7, 6)
        )
        valid = torch.zeros(state.shape[:-1], dtype=torch.bool, device=state.device)
        state[:, : stop - start] = batch["agent_states"][:, start:stop]
        valid[:, : stop - start] = batch["agent_valid"][:, start:stop]
        return state, valid

    def _integrate_background_plan(
        self,
        current: torch.Tensor,
        controls: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Integrate the complete proposed 1 s background-control plan.

        This is used only by the horizon-consistency objective.  It has no
        access to future states at rollout time: the predicted controls are
        integrated from the current generated background state exactly as the
        first executed prefix is.
        """
        state = current[:, 1:]
        predicted: list[torch.Tensor] = []
        for frame in range(controls.shape[1]):
            state = self.dynamics.step(
                state, controls[:, frame], valid[:, frame], self.cfg.simulation_dt_s
            )
            predicted.append(state)
        return torch.stack(predicted, dim=1)

    def plan_step(
        self,
        history: torch.Tensor,
        history_valid: torch.Tensor,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        ego_mask: torch.Tensor,
        memory: torch.Tensor,
        world_latent: torch.Tensor,
        flow_embedding: torch.Tensor,
        previous_plan: torch.Tensor | None,
        previous_current: torch.Tensor | None,
        action_noise: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        agents, scene = self.encoder(
            history, history_valid, current, current_valid, ego_mask
        )
        delta = torch.zeros_like(current) if previous_current is None else current - previous_current
        memory_next = self.memory(scene, agents, previous_plan, delta, memory)
        raw_context = torch.cat((scene, memory_next, world_latent), dim=-1)
        conditioned_context = torch.cat((raw_context, flow_embedding), dim=-1)
        action_context = self.action_context(conditioned_context)
        limit = current.new_tensor(
            (self.cfg.max_longitudinal_jerk, self.cfg.max_yaw_jerk)
        )
        raw_centre = self.plan_field(
            agents,
            scene,
            memory_next,
            world_latent,
            flow_embedding,
            current_valid,
            previous_plan,
        )
        prefix_jerk = self.action_flow.sample(
            action_context,
            action_noise,
            center=raw_centre[:, : self.cfg.execute_frames],
        )
        suffix_raw = raw_centre[:, self.cfg.execute_frames :]
        suffix = torch.tanh(suffix_raw) * limit
        jerks = torch.cat((prefix_jerk, suffix), dim=1)
        plans = self._controls_from_jerks(current, jerks, current_valid)
        output = {
            "agent_context": agents,
            "scene_context": scene,
            "continuous_memory_next": memory_next,
            "world_latent": world_latent,
            "action_context": action_context,
            "raw_joint_jerk_centre": raw_centre,
            "joint_jerk_plan": jerks,
            "joint_control_plan": plans,
        }
        return output

    @staticmethod
    def _interaction_loss(
        predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        pred_bg, true_bg = predicted[:, :, 1:], target[:, :, 1:]
        pred_ego, true_ego = predicted[:, :, :1], target[:, :, :1]
        pred_gap = (pred_bg[..., 0] - pred_ego[..., 0]).abs()
        true_gap = (true_bg[..., 0] - true_ego[..., 0]).abs()
        pred_relative = pred_bg[..., 2] - pred_ego[..., 2]
        true_relative = true_bg[..., 2] - true_ego[..., 2]
        pred_closing = (-pred_relative).clamp_min(0.0)
        true_closing = (-true_relative).clamp_min(0.0)
        pred_ttc = torch.where(pred_closing > 1.0e-3, pred_gap / pred_closing.clamp_min(1.0e-3), pred_gap.new_full(pred_gap.shape, 10.0)).clamp_max(10.0)
        true_ttc = torch.where(true_closing > 1.0e-3, true_gap / true_closing.clamp_min(1.0e-3), true_gap.new_full(true_gap.shape, 10.0)).clamp_max(10.0)
        pred_drac = torch.where(pred_closing > 1.0e-3, pred_closing.square() / (2.0 * pred_gap.clamp_min(1.0e-3)), torch.zeros_like(pred_gap)).clamp_max(20.0)
        true_drac = torch.where(true_closing > 1.0e-3, true_closing.square() / (2.0 * true_gap.clamp_min(1.0e-3)), torch.zeros_like(true_gap)).clamp_max(20.0)
        value = (pred_gap - true_gap).abs() + (pred_relative - true_relative).abs()
        value = value + 0.1 * (pred_ttc - true_ttc).abs() + 0.05 * (pred_drac - true_drac).abs()
        return _masked_mean(value, valid)

    @staticmethod
    def _physical_loss(
        predicted: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        background, ego = predicted[:, :, 1:], predicted[:, :, :1]
        dx = (background[..., 0] - ego[..., 0]).abs()
        dy = (background[..., 1] - ego[..., 1]).abs()
        # A smooth body-clearance barrier is non-zero before the rectangular
        # collision set is reached.  It trains the *sampled, executed* joint
        # controls away from an unsafe state without rejecting high-risk
        # episodes after generation or projecting controls outside the flow.
        def clearance_barrier(longitudinal: torch.Tensor, lateral: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.softplus((4.5 - longitudinal) / 0.75) * torch.nn.functional.softplus((1.0 - lateral) / 0.25)

        ego_overlap = clearance_barrier(dx, dy)
        pair_dx = (background[..., None, :, 0] - background[..., :, None, 0]).abs()
        pair_dy = (background[..., None, :, 1] - background[..., :, None, 1]).abs()
        pair_overlap = clearance_barrier(pair_dx, pair_dy)
        agents = background.shape[2]
        upper = torch.triu(
            torch.ones((agents, agents), dtype=torch.bool, device=background.device),
            diagonal=1,
        )
        pair_valid = valid[..., :, None] & valid[..., None, :] & upper
        speed = torch.linalg.vector_norm(background[..., 2:4], dim=-1)
        return (
            _masked_mean(ego_overlap + torch.relu(speed - 50.0), valid)
            + _masked_mean(pair_overlap, pair_valid)
        )

    def _closed_loop(
        self,
        batch: dict[str, torch.Tensor],
        response_steps: int,
        *,
        deterministic: bool,
        world_noise: torch.Tensor | None = None,
        innovation_noise: torch.Tensor | None = None,
        action_noise: torch.Tensor | None = None,
        tbptt_steps: int = 0,
    ) -> dict[str, Any]:
        states, valid = batch["agent_states"], batch["agent_valid"]
        batch_size = states.shape[0]
        response_steps = min(int(response_steps), self.cfg.response_steps)
        ego_mask = self._ego_mask(batch)
        current, current_valid = states[:, 24], valid[:, 24]
        history, history_valid = current[:, None], current_valid[:, None]
        if world_noise is None:
            world_noise = states.new_zeros((batch_size, self.cfg.world_latent_dim))
        initial = self.initialize(
            current,
            current_valid,
            ego_mask,
            behavior_anchor=batch.get("behavior_anchor_raw"),
            behavior_anchor_valid=batch.get("behavior_anchor_valid"),
            flow_latent=batch.get("flow_latent"),
            world_noise=world_noise,
        )
        memory, world_latent = initial["continuous_memory"], initial["world_latent"]
        flow_embedding = initial["flow_embedding"]
        previous_plan = previous_current = None
        predicted_frames: list[torch.Tensor] = []
        plans: list[torch.Tensor] = []
        jerks: list[torch.Tensor] = []
        flow_contexts: list[torch.Tensor] = []
        raw_centres: list[torch.Tensor] = []
        latents: list[torch.Tensor] = []
        latent_rho: list[torch.Tensor] = []
        latent_sigma: list[torch.Tensor] = []
        loss_terms: list[dict[str, torch.Tensor]] = []
        for response in range(response_steps):
            if action_noise is None:
                action_draw = states.new_zeros(
                    (batch_size, self.cfg.execute_frames, 6, 2)
                )
                if not deterministic:
                    action_draw.normal_()
            else:
                action_draw = action_noise[:, response]
            out = self.plan_step(
                history,
                history_valid,
                current,
                current_valid,
                ego_mask,
                memory,
                world_latent,
                flow_embedding,
                previous_plan,
                previous_current,
                action_draw,
            )
            plan = out["joint_control_plan"]
            target_controls, target_control_valid = self._target_controls(batch, response)
            target_jerks = self._target_jerks(current, target_controls, target_control_valid)
            prefix_nll = self.action_flow.nll(
                target_jerks[:, : self.cfg.execute_frames],
                target_control_valid[:, : self.cfg.execute_frames],
                out["action_context"],
                center=out["raw_joint_jerk_centre"][:, : self.cfg.execute_frames],
            )
            generated = current.clone()
            response_frames: list[torch.Tensor] = []
            for frame in range(self.cfg.execute_frames):
                control = current.new_zeros((batch_size, 7, 2))
                control[:, 1:] = plan[:, frame]
                generated = self.dynamics.step(
                    generated, control, current_valid, self.cfg.simulation_dt_s
                )
                target_frame = 25 + response * self.cfg.execute_frames + frame
                generated = torch.cat((states[:, target_frame, :1], generated[:, 1:]), dim=1)
                current_valid = torch.cat(
                    (valid[:, target_frame, :1], current_valid[:, 1:]), dim=1
                )
                response_frames.append(generated)
            predicted = torch.stack(response_frames, dim=1)
            target = states[
                :, 25 + response * self.cfg.execute_frames : 25 + (response + 1) * self.cfg.execute_frames
            ]
            target_valid = valid[
                :, 25 + response * self.cfg.execute_frames : 25 + (response + 1) * self.cfg.execute_frames, 1:
            ]
            position = _masked_mean(
                (predicted[:, :, 1:, :2] - target[:, :, 1:, :2]).abs().mean(-1), target_valid
            )
            velocity = _masked_mean(
                (predicted[:, :, 1:, 2:4] - target[:, :, 1:, 2:4]).abs().mean(-1), target_valid
            )
            control = _masked_mean(
                (plan[:, : self.cfg.execute_frames] - target_controls[:, : self.cfg.execute_frames]).abs().mean(-1),
                target_control_valid[:, : self.cfg.execute_frames],
            )
            horizon_control = plan.new_zeros(())
            horizon_position = plan.new_zeros(())
            if (
                self.cfg.plan_horizon_control_weight > 0.0
                or self.cfg.plan_horizon_state_weight > 0.0
            ):
                # The whole plan is supervised in training, although only
                # its causal prefix is ever executed.  This prevents the
                # 0.8 s overlap suffix from being an ungrounded by-product.
                horizon_control = _masked_mean(
                    (plan - target_controls).abs().mean(-1), target_control_valid
                )
                if (
                    self.cfg.plan_horizon_state_weight > 0.0
                    and response % max(1, self.cfg.plan_horizon_state_interval_responses) == 0
                ):
                    target_plan, target_plan_valid = self._target_plan_states(batch, response)
                    plan_states = self._integrate_background_plan(
                        current, plan, target_control_valid
                    )
                    horizon_valid = target_control_valid & target_plan_valid[:, :, 1:]
                    horizon_position = _masked_mean(
                        (plan_states[..., :2] - target_plan[:, :, 1:, :2]).abs().mean(-1),
                        horizon_valid,
                    )
            overlap = plan.new_zeros(())
            if previous_plan is not None:
                overlap = (
                    previous_plan[:, self.cfg.execute_frames :]
                    - plan[:, : -self.cfg.execute_frames]
                ).abs().mean()
            interaction = self._interaction_loss(predicted, target, target_valid)
            physical = self._physical_loss(predicted, target_valid)
            loss_terms.append(
                {
                    "prefix_nll": prefix_nll,
                    "prefix_position": position,
                    "prefix_velocity": velocity,
                    "prefix_control": control,
                    "plan_horizon_control": horizon_control,
                    "plan_horizon_position": horizon_position,
                    "overlap": overlap,
                    "interaction": interaction,
                    "physical": physical,
                }
            )
            predicted_frames.extend(response_frames)
            plans.append(plan)
            jerks.append(out["joint_jerk_plan"])
            flow_contexts.append(out["action_context"])
            raw_centres.append(out["raw_joint_jerk_centre"])
            latents.append(world_latent)
            if innovation_noise is None:
                innovation = states.new_zeros((batch_size, self.cfg.world_latent_dim))
                if not deterministic:
                    innovation.normal_()
            else:
                innovation = innovation_noise[:, response]
            world_latent, rho, sigma = self.world_latent(
                out["scene_context"], out["continuous_memory_next"], world_latent, innovation
            )
            latent_rho.append(rho)
            latent_sigma.append(sigma)
            previous_plan, previous_current = plan, current
            current = predicted[:, -1]
            history = torch.cat((history, predicted), dim=1)[:, -25:]
            history_valid = torch.cat(
                (history_valid, current_valid[:, None].expand(-1, self.cfg.execute_frames, -1)), dim=1
            )[:, -25:]
            memory = out["continuous_memory_next"]
            if tbptt_steps and (response + 1) % tbptt_steps == 0 and response + 1 < response_steps:
                history = history.detach()
                current = current.detach()
                memory = memory.detach()
                world_latent = world_latent.detach()
                previous_plan = previous_plan.detach()
        return {
            "predicted_states": torch.stack(predicted_frames, dim=1),
            "target_states": states[:, 25 : 25 + response_steps * self.cfg.execute_frames],
            "target_valid": valid[:, 25 : 25 + response_steps * self.cfg.execute_frames],
            "joint_control_plans": torch.stack(plans, dim=1),
            "joint_jerk_plans": torch.stack(jerks, dim=1),
            "action_contexts": torch.stack(flow_contexts, dim=1),
            "raw_joint_jerk_centres": torch.stack(raw_centres, dim=1),
            "world_latent_path": torch.stack(latents, dim=1),
            "world_latent_rho": torch.stack(latent_rho, dim=1),
            "world_latent_sigma": torch.stack(latent_sigma, dim=1),
            "world_latent_scale": initial["world_latent_scale"],
            "loss_terms": loss_terms,
        }

    def forward_training(
        self,
        batch: dict[str, torch.Tensor],
        *,
        response_steps: int | None = None,
        tbptt_steps: int = 5,
    ) -> dict[str, torch.Tensor]:
        steps = response_steps or self.cfg.response_steps
        world_noise = torch.randn(
            (batch["agent_states"].shape[0], self.cfg.world_latent_dim),
            dtype=batch["agent_states"].dtype,
            device=batch["agent_states"].device,
        )
        rollout = self._closed_loop(
            batch,
            steps,
            deterministic=True,
            world_noise=world_noise,
            tbptt_steps=tbptt_steps,
        )
        means = {
            key: torch.stack([term[key] for term in rollout["loss_terms"]]).mean()
            for key in rollout["loss_terms"][0]
        }
        anchor = batch.get("behavior_anchor_raw")
        anchor_valid = batch.get("behavior_anchor_valid")
        behavior = rollout["predicted_states"].new_zeros(())
        if anchor is not None and rollout["predicted_states"].shape[1] >= 25:
            initial = batch["agent_states"][:, 24:25]
            initial_valid = batch["agent_valid"][:, 24:25]
            first_second = torch.cat((initial, rollout["predicted_states"][:, :25]), dim=1)
            first_valid = torch.cat(
                (initial_valid, rollout["target_valid"][:, :25]), dim=1
            )
            summary, summary_valid = summarize_first_second_states(first_second, first_valid)
            mask = summary_valid[:, 1:]
            if anchor_valid is not None:
                mask = mask & anchor_valid.bool()
            behavior = _masked_mean((summary[:, 1:] - anchor).abs().mean(-1), mask)
        variance_floor = torch.relu(0.10 - rollout["world_latent_scale"]).mean()
        sampled_physical = rollout["predicted_states"].new_zeros(())
        if self.cfg.sampled_physical_weight > 0.0 and steps >= self.cfg.response_steps:
            # The likelihood alone calibrates control density but does not
            # constrain random flow draws in a multi-step closed loop.  This
            # second rollout draws the same action-flow variable that is
            # executed at test time and applies only a differentiable
            # clearance loss. It is deliberately enabled only for the full
            # 5 s stage, where long-horizon Flow composition is evaluated.
            # A short TBPTT span keeps the safety term local to its causal
            # response while avoiding a second full 5 s graph.
            sampled_count = min(
                int(batch["agent_states"].shape[0]),
                max(1, int(self.cfg.sampled_physical_batch_size)),
            )
            sampled_batch = {
                key: (
                    value[:sampled_count]
                    if torch.is_tensor(value)
                    and value.ndim
                    and value.shape[0] == batch["agent_states"].shape[0]
                    else value
                )
                for key, value in batch.items()
            }
            sampled = self._closed_loop(
                sampled_batch,
                steps,
                deterministic=False,
                world_noise=world_noise[:sampled_count],
                tbptt_steps=max(1, min(5, tbptt_steps or 5)),
            )
            sampled_physical = torch.stack(
                [term["physical"] for term in sampled["loss_terms"]]
            ).mean()
        total = (
            self.cfg.prefix_nll_weight * means["prefix_nll"]
            + self.cfg.roll_weight * means["prefix_position"]
            + self.cfg.velocity_weight * means["prefix_velocity"]
            + self.cfg.control_weight * means["prefix_control"]
            + self.cfg.plan_horizon_control_weight * means["plan_horizon_control"]
            + self.cfg.plan_horizon_state_weight * means["plan_horizon_position"]
            + self.cfg.behavior_anchor_weight * behavior
            + self.cfg.overlap_weight * means["overlap"]
            + self.cfg.interaction_weight * means["interaction"]
            + self.cfg.physical_weight * means["physical"]
            + self.cfg.sampled_physical_weight * sampled_physical
            + self.cfg.latent_variance_weight * variance_floor
        )
        return {
            "loss": total,
            **means,
            "behavior_anchor": behavior,
            "latent_variance_floor": variance_floor,
            "sampled_physical": sampled_physical,
            **{key: value for key, value in rollout.items() if key != "loss_terms"},
        }

    @torch.no_grad()
    def rollout_roll_mode(
        self,
        batch: dict[str, torch.Tensor],
        *,
        seed: int = 123,
        deterministic: bool = True,
    ) -> dict[str, Any]:
        generator = torch.Generator(device=batch["agent_states"].device).manual_seed(int(seed))
        shape = batch["agent_states"].shape[0]
        device, dtype = batch["agent_states"].device, batch["agent_states"].dtype
        if deterministic:
            world_noise = torch.zeros((shape, self.cfg.world_latent_dim), device=device, dtype=dtype)
            innovation = torch.zeros((shape, self.cfg.response_steps, self.cfg.world_latent_dim), device=device, dtype=dtype)
            action = torch.zeros((shape, self.cfg.response_steps, self.cfg.execute_frames, 6, 2), device=device, dtype=dtype)
        else:
            world_noise = torch.randn((shape, self.cfg.world_latent_dim), generator=generator, device=device, dtype=dtype)
            innovation = torch.randn((shape, self.cfg.response_steps, self.cfg.world_latent_dim), generator=generator, device=device, dtype=dtype)
            action = torch.randn((shape, self.cfg.response_steps, self.cfg.execute_frames, 6, 2), generator=generator, device=device, dtype=dtype)
        return self._closed_loop(
            batch,
            self.cfg.response_steps,
            deterministic=deterministic,
            world_noise=world_noise,
            innovation_noise=innovation,
            action_noise=action,
        )

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "model_config": asdict(self.cfg),
            "state_dict": self.state_dict(),
        }
