"""QR-WM planner, persistent memory, and receding-horizon refinement buffer."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

from world_model.src.core.dynamics import DynamicsConfig, KinematicTrafficDynamics
from world_model.src.core.traffic_memory import ContinuousTrafficMemory

from .config import QRWorldModelConfig
from .encoder import QueryRelationalSceneEncoder


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weight = valid.float()
    return (values * weight).sum() / weight.sum().clamp_min(1.0)


class BehaviorPrior(nn.Module):
    """Conditional per-agent Gaussian behavior prior with a future posterior."""

    def __init__(self, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.prior = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, latent_dim * 2)
        )
        self.future = nn.Sequential(nn.Linear(12, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.posterior = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, latent_dim * 2)
        )
        self.reconstruction = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 12)
        )

    @staticmethod
    def _distribution_parameters(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw = value.chunk(2, dim=-1)
        return mean, -3.0 + 2.5 * torch.sigmoid(raw)

    @staticmethod
    def _future_features(
        current: torch.Tensor, future: torch.Tensor, future_valid: torch.Tensor
    ) -> torch.Tensor:
        """Encode supervised future motion without exposing it to the prior."""
        weight = future_valid.float()[..., None]
        denominator = weight.sum(dim=1).clamp_min(1.0)
        delta = future - current[:, None]
        mean_delta = (delta * weight).sum(dim=1) / denominator
        last_index = future_valid.long().sum(dim=1).sub(1).clamp_min(0)
        gathered = future.gather(
            1, last_index[:, None, :, None].expand(-1, 1, -1, future.shape[-1])
        ).squeeze(1)
        endpoint = gathered - current
        return torch.cat((mean_delta[..., :6], endpoint[..., :6]), dim=-1)

    def prior_parameters(
        self, agents: torch.Tensor, scene: torch.Tensor, memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared = torch.cat((scene, memory), dim=-1)[:, None].expand(-1, agents.shape[1], -1)
        return self._distribution_parameters(self.prior(torch.cat((agents, shared), dim=-1)))

    def posterior_parameters(
        self,
        agents: torch.Tensor,
        scene: torch.Tensor,
        memory: torch.Tensor,
        current: torch.Tensor,
        future: torch.Tensor,
        future_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_feature = self._future_features(current, future, future_valid)
        feature = self.future(raw_feature)
        shared = torch.cat((scene, memory), dim=-1)[:, None].expand(-1, agents.shape[1], -1)
        mean, log_scale = self._distribution_parameters(self.posterior(torch.cat((agents, shared, feature), dim=-1)))
        return mean, log_scale, raw_feature


class QueryRefineWorldModel(nn.Module):
    """Multimodal query-centric traffic world model with a refinement buffer."""

    model_type = "query_refine_world_model_v1"

    def __init__(self, cfg: QRWorldModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or QRWorldModelConfig()
        h, z = self.cfg.hidden_dim, self.cfg.behavior_latent_dim
        self.encoder = QueryRelationalSceneEncoder(self.cfg)
        self.memory = ContinuousTrafficMemory(h)
        self.world_initializer = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
        self.world_update = nn.GRUCell(h * 2, h)
        self.behavior = BehaviorPrior(h, z)
        self.plan_time = nn.Parameter(torch.randn(self.cfg.plan_frames, h) * 0.02)
        self.plan_agent = nn.Sequential(
            nn.Linear(h * 4 + z, h), nn.SiLU(), nn.Linear(h, h)
        )
        self.plan_head = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))
        self.refine_state = nn.Sequential(nn.Linear(6, h), nn.SiLU(), nn.Linear(h, h))
        self.refine_head = nn.Sequential(nn.Linear(h * 2, h), nn.SiLU(), nn.Linear(h, 2))
        self.dynamics = KinematicTrafficDynamics(
            DynamicsConfig(
                acceleration_min_mps2=self.cfg.min_acceleration,
                acceleration_max_mps2=self.cfg.max_acceleration,
            )
        )

    @staticmethod
    def _ego_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        agents = int(batch["agent_states"].shape[2])
        return functional.one_hot(batch["ego_index"].long().clamp(0, agents - 1), agents).bool()

    @staticmethod
    def flow_condition_to_scene(
        flow_condition: torch.Tensor, slot_valid: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode Flow's fixed 76-D C0+B0 representation into QR tensors.

        Layout is ``ego[vx,vy,ax,ay] + six relative 6-D states + B0[6,6]``.
        The returned current scene uses the same ``[B,7,6]`` raw-state schema
        consumed by QR-WM training; ego position is the local origin.  B0 is
        returned separately for callers that retain the Flow provenance.
        """
        if flow_condition.ndim != 2 or flow_condition.shape[-1] != 76:
            raise ValueError("flow_condition must have shape [batch, 76]")
        batch = flow_condition.shape[0]
        ego = flow_condition.new_zeros((batch, 1, 6))
        ego[:, 0, 2:6] = flow_condition[:, :4]
        background = flow_condition[:, 4:40].reshape(batch, 6, 6)
        scene = torch.cat((ego, background), dim=1)
        valid = torch.ones((batch, 7), dtype=torch.bool, device=flow_condition.device)
        if slot_valid is not None:
            if slot_valid.shape != (batch, 6):
                raise ValueError("slot_valid must have shape [batch, 6]")
            valid[:, 1:] = slot_valid.bool()
            scene[:, 1:] = scene[:, 1:] * valid[:, 1:, None].float()
        return scene, valid, flow_condition[:, 40:].reshape(batch, 6, 6)

    def _world_context(
        self,
        scene: torch.Tensor,
        agents: torch.Tensor,
        previous_plan: torch.Tensor | None,
        current: torch.Tensor,
        previous_current: torch.Tensor | None,
        previous_memory: torch.Tensor | None,
        previous_world: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta = torch.zeros_like(current) if previous_current is None else current - previous_current
        memory = self.memory(scene, agents, previous_plan, delta, previous_memory)
        initial = self.world_initializer(scene)
        world = self.world_update(torch.cat((scene, memory), dim=-1), initial if previous_world is None else previous_world)
        return memory, world

    def _fresh_controls(
        self,
        agents: torch.Tensor,
        scene: torch.Tensor,
        memory: torch.Tensor,
        world: torch.Tensor,
        behavior: torch.Tensor,
    ) -> torch.Tensor:
        background = agents[:, 1:]
        shared = torch.cat((scene, memory, world), dim=-1)[:, None].expand(-1, background.shape[1], -1)
        token = self.plan_agent(torch.cat((background, shared, behavior[:, 1:]), dim=-1))
        value = token[:, None] + self.plan_time[None, :, None]
        raw = self.plan_head(value)
        acceleration = torch.tanh(raw[..., 0]) * max(abs(self.cfg.min_acceleration), abs(self.cfg.max_acceleration))
        acceleration = acceleration.clamp(self.cfg.min_acceleration, self.cfg.max_acceleration)
        yaw_rate = torch.tanh(raw[..., 1]) * self.cfg.max_yaw_rate
        return torch.stack((acceleration, yaw_rate), dim=-1)

    def _integrate_background_plan(
        self, current: torch.Tensor, controls: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        state = current[:, 1:]
        frames: list[torch.Tensor] = []
        for frame in range(controls.shape[1]):
            state = self.dynamics.step(state, controls[:, frame], valid[:, 1:], self.cfg.simulation_dt_s)
            frames.append(state)
        return torch.stack(frames, dim=1)

    def _refine_controls(
        self,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        agents: torch.Tensor,
        controls: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Iteratively correct a complete future trajectory buffer."""
        initial_states = self._integrate_background_plan(current, controls, current_valid)
        refined = controls
        context = agents[:, None, 1:].expand(-1, self.cfg.plan_frames, -1, -1)
        residual_total = torch.zeros_like(controls)
        for _ in range(max(1, int(self.cfg.refinement_iterations))):
            states = self._integrate_background_plan(current, refined, current_valid)
            relative = states.clone()
            relative[..., :2] = relative[..., :2] - current[:, None, 1:, :2]
            token = self.refine_state(relative) + context + self.plan_time[None, :, None]
            residual = self.refine_head(torch.cat((token, context), dim=-1))
            residual = torch.stack((1.5 * torch.tanh(residual[..., 0]), 0.15 * torch.tanh(residual[..., 1])), dim=-1)
            refined = torch.stack(
                (
                    (refined[..., 0] + residual[..., 0]).clamp(self.cfg.min_acceleration, self.cfg.max_acceleration),
                    (refined[..., 1] + residual[..., 1]).clamp(-self.cfg.max_yaw_rate, self.cfg.max_yaw_rate),
                ),
                dim=-1,
            )
            residual_total = residual_total + residual
        return refined, initial_states, self._integrate_background_plan(current, refined, current_valid)

    def plan_step(
        self,
        history: torch.Tensor,
        history_valid: torch.Tensor,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        ego_mask: torch.Tensor,
        map_polylines: torch.Tensor,
        map_polyline_valid: torch.Tensor,
        lane_graph_edges: torch.Tensor,
        behavior_latent: torch.Tensor,
        *,
        previous_buffer: torch.Tensor | None = None,
        previous_current: torch.Tensor | None = None,
        previous_memory: torch.Tensor | None = None,
        previous_world: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        agents, scene = self.encoder(
            history, history_valid, current, current_valid, ego_mask,
            map_polylines, map_polyline_valid, lane_graph_edges,
        )
        memory, world = self._world_context(
            scene, agents, previous_buffer, current, previous_current, previous_memory, previous_world
        )
        fresh = self._fresh_controls(agents, scene, memory, world, behavior_latent)
        carried = fresh
        if previous_buffer is not None:
            execute = int(self.cfg.execute_frames)
            carried = torch.cat((previous_buffer[:, execute:], fresh[:, -execute:]), dim=1)
            controls = (1.0 - self.cfg.buffer_carry_mix) * fresh + self.cfg.buffer_carry_mix * carried
        else:
            controls = fresh
        refined, initial_states, refined_states = self._refine_controls(current, current_valid, agents, controls)
        return {
            "agent_context": agents,
            "scene_context": scene,
            "persistent_memory": memory,
            "world_memory": world,
            "fresh_buffer": fresh,
            "carried_buffer": carried,
            "pre_refinement_buffer": controls,
            "refined_buffer": refined,
            "initial_plan_states": initial_states,
            "refined_plan_states": refined_states,
        }

    def _target_plan(
        self, batch: dict[str, torch.Tensor], response: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states, valid, actions = batch["agent_states"], batch["agent_valid"], batch["actions_highd"]
        start = int(response) * self.cfg.execute_frames
        count = min(self.cfg.plan_frames, int(actions.shape[1]) - start)
        batch_size, agents = states.shape[0], states.shape[2]
        target_states = states.new_zeros((batch_size, self.cfg.plan_frames, agents, 6))
        target_valid = torch.zeros((batch_size, self.cfg.plan_frames, agents), dtype=torch.bool, device=states.device)
        controls = states.new_zeros((batch_size, self.cfg.plan_frames, agents - 1, 2))
        control_valid = torch.zeros((batch_size, self.cfg.plan_frames, agents - 1), dtype=torch.bool, device=states.device)
        if count:
            target_states[:, :count] = states[:, 25 + start : 25 + start + count]
            target_valid[:, :count] = valid[:, 25 + start : 25 + start + count]
            current = states[:, 24 + start : 24 + start + count, 1:]
            controls[:, :count] = self.dynamics.controls_from_highd_actions(actions[:, start : start + count], current)
            control_valid[:, :count] = valid[:, 25 + start : 25 + start + count, 1:]
        return target_states, target_valid, controls, control_valid

    @staticmethod
    def _interaction_loss(predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        pred_bg, true_bg = predicted[:, :, 1:], target[:, :, 1:]
        pred_ego, true_ego = predicted[:, :, :1], target[:, :, :1]
        pred_gap = (pred_bg[..., 0] - pred_ego[..., 0]).abs()
        true_gap = (true_bg[..., 0] - true_ego[..., 0]).abs()
        pred_relative = pred_bg[..., 2] - pred_ego[..., 2]
        true_relative = true_bg[..., 2] - true_ego[..., 2]
        return _masked_mean((pred_gap - true_gap).abs() + 0.25 * (pred_relative - true_relative).abs(), valid)

    @staticmethod
    def _physical_loss(predicted: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        background, ego = predicted[:, :, 1:], predicted[:, :, :1]
        dx = (background[..., 0] - ego[..., 0]).abs()
        dy = (background[..., 1] - ego[..., 1]).abs()
        ego_barrier = functional.softplus((4.5 - dx) / 0.75) * functional.softplus((1.0 - dy) / 0.25)
        pair_dx = (background[..., None, :, 0] - background[..., :, None, 0]).abs()
        pair_dy = (background[..., None, :, 1] - background[..., :, None, 1]).abs()
        pair_barrier = functional.softplus((4.5 - pair_dx) / 0.75) * functional.softplus((1.0 - pair_dy) / 0.25)
        agents = background.shape[2]
        upper = torch.triu(torch.ones((agents, agents), dtype=torch.bool, device=background.device), diagonal=1)
        pair_valid = valid[..., :, None] & valid[..., None, :] & upper
        return _masked_mean(ego_barrier, valid) + _masked_mean(pair_barrier, pair_valid)

    def _behavior_latent(
        self,
        batch: dict[str, torch.Tensor],
        agents: torch.Tensor,
        scene: torch.Tensor,
        memory: torch.Tensor,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        *,
        deterministic: bool,
        use_posterior: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prior_mean, prior_log = self.behavior.prior_parameters(agents, scene, memory)
        terms: dict[str, torch.Tensor] = {
            "behavior_kl": current.new_zeros(()),
            "behavior_reconstruction": current.new_zeros(()),
            "diversity_floor": current.new_zeros(()),
        }
        mean, log_scale = prior_mean, prior_log
        if use_posterior:
            future = batch["agent_states"][:, 25:50]
            future_valid = batch["agent_valid"][:, 25:50]
            posterior_mean, posterior_log, future_feature = self.behavior.posterior_parameters(
                agents, scene, memory, current, future, future_valid
            )
            mean, log_scale = posterior_mean, posterior_log
            variance_ratio = torch.exp(2.0 * (posterior_log - prior_log))
            kl = prior_log - posterior_log + 0.5 * (variance_ratio + (posterior_mean - prior_mean).square() * torch.exp(-2.0 * prior_log) - 1.0)
            terms["behavior_kl"] = _masked_mean(kl.mean(dim=-1), current_valid)
            reconstruction = self.behavior.reconstruction(posterior_mean)
            terms["behavior_reconstruction"] = _masked_mean(
                (reconstruction - future_feature).abs().mean(dim=-1), current_valid
            )
        terms["diversity_floor"] = functional.relu(0.12 - torch.exp(log_scale).mean())
        sample = mean if deterministic else mean + torch.randn_like(mean) * torch.exp(log_scale)
        return sample * current_valid[..., None].float(), terms

    def rollout(
        self,
        batch: dict[str, torch.Tensor],
        *,
        response_steps: int | None = None,
        deterministic: bool = True,
        use_posterior: bool = False,
        tbptt_steps: int = 0,
    ) -> dict[str, Any]:
        """Execute a receding-horizon QR buffer without future encoder leakage."""
        states, valid = batch["agent_states"], batch["agent_valid"]
        steps = min(int(response_steps or self.cfg.response_steps), self.cfg.response_steps)
        ego_mask = self._ego_mask(batch)
        current, current_valid = states[:, 24], valid[:, 24]
        history, history_valid = states[:, :25], valid[:, :25]
        # Construct the initial posterior only for the training objective.
        initial_agents, initial_scene = self.encoder(
            history, history_valid, current, current_valid, ego_mask,
            batch["map_polylines"], batch["map_polyline_valid"], batch["lane_graph_edges"],
        )
        initial_memory = self.memory(initial_scene, initial_agents, None, torch.zeros_like(current), None)
        behavior_latent, behavior_terms = self._behavior_latent(
            batch, initial_agents, initial_scene, initial_memory, current, current_valid,
            deterministic=deterministic, use_posterior=use_posterior,
        )
        predicted_frames: list[torch.Tensor] = []
        plans: list[torch.Tensor] = []
        pre_refinement: list[torch.Tensor] = []
        plan_states: list[torch.Tensor] = []
        initial_plan_states: list[torch.Tensor] = []
        term_rows: list[dict[str, torch.Tensor]] = []
        previous_buffer = previous_current = previous_memory = previous_world = None
        for response in range(steps):
            out = self.plan_step(
                history, history_valid, current, current_valid, ego_mask,
                batch["map_polylines"], batch["map_polyline_valid"], batch["lane_graph_edges"], behavior_latent,
                previous_buffer=previous_buffer, previous_current=previous_current,
                previous_memory=previous_memory, previous_world=previous_world,
            )
            target, target_valid, target_controls, target_control_valid = self._target_plan(batch, response)
            plan = out["refined_buffer"]
            generated = current.clone()
            response_frames: list[torch.Tensor] = []
            response_valid: list[torch.Tensor] = []
            for frame in range(self.cfg.execute_frames):
                control = current.new_zeros((current.shape[0], current.shape[1], 2))
                control[:, 1:] = plan[:, frame]
                generated = self.dynamics.step(generated, control, current_valid, self.cfg.simulation_dt_s)
                target_frame = 25 + response * self.cfg.execute_frames + frame
                generated = torch.where(ego_mask[..., None], states[:, target_frame], generated)
                current_valid = torch.where(ego_mask, valid[:, target_frame], current_valid)
                response_frames.append(generated)
                response_valid.append(current_valid)
            predicted = torch.stack(response_frames, dim=1)
            exec_valid = target_valid[:, : self.cfg.execute_frames, 1:]
            position = _masked_mean((predicted[:, :, 1:, :2] - target[:, : self.cfg.execute_frames, 1:, :2]).abs().mean(dim=-1), exec_valid)
            velocity = _masked_mean((predicted[:, :, 1:, 2:4] - target[:, : self.cfg.execute_frames, 1:, 2:4]).abs().mean(dim=-1), exec_valid)
            control_loss = _masked_mean((plan[:, : self.cfg.execute_frames] - target_controls[:, : self.cfg.execute_frames]).abs().mean(dim=-1), target_control_valid[:, : self.cfg.execute_frames])
            full_valid = target_valid[:, :, 1:]
            full_position = _masked_mean((out["refined_plan_states"][..., :2] - target[:, :, 1:, :2]).abs().mean(dim=-1), full_valid)
            initial_position = _masked_mean((out["initial_plan_states"][..., :2] - target[:, :, 1:, :2]).abs().mean(dim=-1), full_valid)
            full_control = _masked_mean((plan - target_controls).abs().mean(dim=-1), target_control_valid)
            overlap = plan.new_zeros(())
            if previous_buffer is not None:
                overlap = (previous_buffer[:, self.cfg.execute_frames:] - plan[:, : -self.cfg.execute_frames]).abs().mean()
            interaction = self._interaction_loss(predicted, target[:, : self.cfg.execute_frames], exec_valid)
            physical = self._physical_loss(predicted, exec_valid)
            term_rows.append({
                "position": position, "velocity": velocity, "control": control_loss,
                "plan_position": full_position, "initial_plan_position": initial_position,
                "plan_control": full_control, "overlap": overlap,
                "interaction": interaction, "physical": physical,
            })
            predicted_frames.extend(response_frames)
            plans.append(plan)
            pre_refinement.append(out["pre_refinement_buffer"])
            plan_states.append(out["refined_plan_states"])
            initial_plan_states.append(out["initial_plan_states"])
            previous_buffer, previous_current = plan, current
            previous_memory, previous_world = out["persistent_memory"], out["world_memory"]
            current = predicted[:, -1]
            appended_valid = torch.stack(response_valid, dim=1)
            history = torch.cat((history, predicted), dim=1)[:, -25:]
            history_valid = torch.cat((history_valid, appended_valid), dim=1)[:, -25:]
            if tbptt_steps and (response + 1) % int(tbptt_steps) == 0 and response + 1 < steps:
                history, history_valid = history.detach(), history_valid.detach()
                current, previous_buffer = current.detach(), previous_buffer.detach()
                previous_memory, previous_world = previous_memory.detach(), previous_world.detach()
        return {
            "predicted_states": torch.stack(predicted_frames, dim=1),
            "target_states": states[:, 25 : 25 + steps * self.cfg.execute_frames],
            "target_valid": valid[:, 25 : 25 + steps * self.cfg.execute_frames],
            "control_buffers": torch.stack(plans, dim=1),
            "pre_refinement_buffers": torch.stack(pre_refinement, dim=1),
            "refined_plan_states": torch.stack(plan_states, dim=1),
            "initial_plan_states": torch.stack(initial_plan_states, dim=1),
            "behavior_latent": behavior_latent,
            "behavior_terms": behavior_terms,
            "loss_terms": term_rows,
        }

    def forward_training(
        self, batch: dict[str, torch.Tensor], *, response_steps: int | None = None, tbptt_steps: int = 5
    ) -> dict[str, torch.Tensor]:
        rollout = self.rollout(
            batch, response_steps=response_steps, deterministic=False, use_posterior=True, tbptt_steps=tbptt_steps
        )
        terms = {key: torch.stack([value[key] for value in rollout["loss_terms"]]).mean() for key in rollout["loss_terms"][0]}
        behavior = rollout["behavior_terms"]
        loss = (
            self.cfg.position_weight * terms["position"]
            + self.cfg.velocity_weight * terms["velocity"]
            + self.cfg.control_weight * terms["control"]
            + self.cfg.plan_position_weight * terms["plan_position"]
            + self.cfg.plan_control_weight * terms["plan_control"]
            + self.cfg.refinement_weight * functional.relu(terms["plan_position"] - terms["initial_plan_position"] + 0.01)
            + self.cfg.overlap_weight * terms["overlap"]
            + self.cfg.interaction_weight * terms["interaction"]
            + self.cfg.physical_weight * terms["physical"]
            + self.cfg.behavior_kl_weight * behavior["behavior_kl"]
            + self.cfg.behavior_reconstruction_weight * behavior["behavior_reconstruction"]
            + self.cfg.diversity_weight * behavior["diversity_floor"]
        )
        return {"loss": loss, **terms, **behavior}

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "model_config": asdict(self.cfg),
            "state_dict": self.state_dict(),
            "flow_interface": {
                "input_dim": 76,
                "layout": "ego[vx,vy,ax,ay]+background_relative[6,6]+behavior_anchor[6,6]",
                "scene_tensor_shape": [7, 6],
            },
        }
