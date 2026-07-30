"""QR-WM: joint multi-agent control buffers with START-only Flow conditioning."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

from world_model.src.core.dynamics import DynamicsConfig, KinematicTrafficDynamics
from world_model.src.core.initial_behavior_anchor import (
    BehaviorAnchorControlPlan,
    start_state_from_flow_tensor,
    summarize_first_second_states,
)

from .config import QRWorldModelConfig
from .encoder import QueryRelationalSceneEncoder
from .joint_refiner import JointAgentTimeRefiner
from .memory import PersistentSceneMemory


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weight = valid.float()
    return (values * weight).sum() / weight.sum().clamp_min(1.0)


BUFFER_MASK_NAMES = ("carried", "appended", "refinable", "valid")


class BehaviorPrior(nn.Module):
    """Conditional per-agent Gaussian behavior prior with a training-only posterior."""

    def __init__(self, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.prior = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, latent_dim * 2))
        self.future = nn.Sequential(nn.Linear(12, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.posterior = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, latent_dim * 2))
        self.reconstruction = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 12))

    @staticmethod
    def _distribution_parameters(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw = value.chunk(2, dim=-1)
        return mean, -3.0 + 2.5 * torch.sigmoid(raw)

    @staticmethod
    def _future_features(current: torch.Tensor, future: torch.Tensor, future_valid: torch.Tensor) -> torch.Tensor:
        weight = future_valid.float()[..., None]
        denominator = weight.sum(dim=1).clamp_min(1.0)
        mean_delta = ((future - current[:, None]) * weight).sum(dim=1) / denominator
        last_index = future_valid.long().sum(dim=1).sub(1).clamp_min(0)
        gathered = future.gather(1, last_index[:, None, :, None].expand(-1, 1, -1, future.shape[-1])).squeeze(1)
        return torch.cat((mean_delta[..., :6], (gathered - current)[..., :6]), dim=-1)

    def prior_parameters(
        self, agents: torch.Tensor, scene: torch.Tensor, memory: torch.Tensor, start_seed: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared = torch.cat((scene, memory), dim=-1)[:, None].expand(-1, agents.shape[1], -1)
        mean, log_scale = self._distribution_parameters(self.prior(torch.cat((agents, shared), dim=-1)))
        return (mean if start_seed is None else mean + start_seed), log_scale

    def posterior_parameters(
        self,
        agents: torch.Tensor,
        scene: torch.Tensor,
        memory: torch.Tensor,
        current: torch.Tensor,
        future: torch.Tensor,
        future_valid: torch.Tensor,
        start_seed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_feature = self._future_features(current, future, future_valid)
        feature = self.future(raw_feature)
        shared = torch.cat((scene, memory), dim=-1)[:, None].expand(-1, agents.shape[1], -1)
        mean, log_scale = self._distribution_parameters(self.posterior(torch.cat((agents, shared, feature), dim=-1)))
        return (mean if start_seed is None else mean + start_seed), log_scale, raw_feature


class QueryRefineWorldModel(nn.Module):
    """Joint query-refine traffic world model with one persistent scene memory."""

    model_type = "query_refine_world_model"

    def __init__(self, cfg: QRWorldModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or QRWorldModelConfig()
        h, z = self.cfg.hidden_dim, self.cfg.behavior_latent_dim
        self.encoder = QueryRelationalSceneEncoder(self.cfg)
        self.scene_memory = PersistentSceneMemory(h)
        self.behavior = BehaviorPrior(h, z)
        self.start_behavior = nn.Sequential(nn.Linear(6, h), nn.SiLU(), nn.Linear(h, z))
        self.start_anchor_plan = BehaviorAnchorControlPlan(self.cfg.plan_frames)
        self.joint_refiner = JointAgentTimeRefiner(
            hidden_dim=h, behavior_latent_dim=z, plan_frames=self.cfg.plan_frames,
            attention_layers=self.cfg.attention_layers, num_heads=self.cfg.num_heads, dropout=self.cfg.dropout,
            min_acceleration=self.cfg.min_acceleration, max_acceleration=self.cfg.max_acceleration,
            max_yaw_rate=self.cfg.max_yaw_rate, acceleration_noise_std=self.cfg.denoising_acceleration_std,
            yaw_noise_std=self.cfg.denoising_yaw_rate_std,
        )
        self.dynamics = KinematicTrafficDynamics(
            DynamicsConfig(acceleration_min_mps2=self.cfg.min_acceleration, acceleration_max_mps2=self.cfg.max_acceleration)
        )
        self.flow_schema_sha256: str | None = None

    @staticmethod
    def _ego_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        agents = int(batch["agent_states"].shape[2])
        return functional.one_hot(batch["ego_index"].long().clamp(0, agents - 1), agents).bool()

    @staticmethod
    def flow_condition_to_scene(
        flow_condition: torch.Tensor, slot_valid: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode Flow's ``C0+B0`` row into initial state, validity, and raw B0."""
        return start_state_from_flow_tensor(flow_condition, slot_valid)[:3]

    def _clamp_controls(self, controls: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                controls[..., 0].clamp(self.cfg.min_acceleration, self.cfg.max_acceleration),
                controls[..., 1].clamp(-self.cfg.max_yaw_rate, self.cfg.max_yaw_rate),
            ), dim=-1
        )

    def _mix_start_controls(self, fresh: torch.Tensor, anchor_controls: torch.Tensor) -> torch.Tensor:
        """Blend B0 controls once, with a decaying convex weight over one second."""
        alpha = torch.linspace(
            float(self.cfg.start_anchor_mix), 0.0, self.cfg.plan_frames,
            device=fresh.device, dtype=fresh.dtype,
        )[None, :, None, None]
        return self._clamp_controls((1.0 - alpha) * fresh + alpha * anchor_controls)

    def _start_anchor(
        self,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        raw_anchor: torch.Tensor | None,
        anchor_valid: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Consume raw B0 exactly once to seed controls, memory, and behavior."""
        batch, agents = current.shape[:2]
        background = agents - 1
        empty_plan = current.new_zeros((batch, self.cfg.plan_frames, background, 2))
        empty_seed = current.new_zeros((batch, agents, self.cfg.behavior_latent_dim))
        if raw_anchor is None:
            return empty_plan, empty_seed
        if raw_anchor.shape != (batch, background, 6):
            raise ValueError("B0 must have shape [batch, six background slots, 6]")
        valid = (current_valid[:, 1:] if anchor_valid is None else anchor_valid.bool()) & current_valid[:, 1:]
        highd = self.start_anchor_plan(current, raw_anchor, valid)
        controls = self.dynamics.controls_from_highd_actions(highd, current[:, None, 1:])
        seed = empty_seed.clone()
        seed[:, 1:] = self.start_behavior(raw_anchor) * valid[..., None].float()
        return self._clamp_controls(controls) * valid[:, None, :, None].float(), seed

    def initialize_start(
        self,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        ego_mask: torch.Tensor,
        map_polylines: torch.Tensor,
        map_polyline_valid: torch.Tensor,
        lane_graph_edges: torch.Tensor,
        raw_anchor: torch.Tensor,
        anchor_valid: torch.Tensor,
        *,
        deterministic: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Initialize the shared START state once from C0, B0, and map."""
        agents, scene, _, _ = self.encoder.encode_start(
            current, current_valid, ego_mask, map_polylines, map_polyline_valid, lane_graph_edges,
        )
        anchor_controls, start_seed = self._start_anchor(current, current_valid, raw_anchor, anchor_valid)
        memory = self.scene_memory(
            scene, agents, anchor_controls, torch.zeros_like(current), current.new_zeros((current.shape[0], 2)), None,
        )
        behavior, _ = self._behavior_latent(
            {}, agents, scene, memory, current, current_valid, start_seed,
            deterministic=deterministic, use_posterior=False,
        )
        return {"scene_memory": memory, "behavior_latent": behavior, "start_anchor_controls": anchor_controls}

    @staticmethod
    def _stack_masks(masks: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {name: torch.stack(masks[name], dim=1) for name in BUFFER_MASK_NAMES}

    def _integrate_background_plan(self, current: torch.Tensor, controls: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        state = current[:, 1:]
        frames: list[torch.Tensor] = []
        for frame in range(controls.shape[1]):
            state = self.dynamics.step(state, controls[:, frame], valid[:, 1:], self.cfg.simulation_dt_s)
            frames.append(state)
        return torch.stack(frames, dim=1)

    def _refine_controls(
        self,
        controls: torch.Tensor,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        agents: torch.Tensor,
        scene: torch.Tensor,
        memory: torch.Tensor,
        behavior: torch.Tensor,
        map_tokens: torch.Tensor,
        map_valid: torch.Tensor,
        ego_controls: torch.Tensor,
        ego_states: torch.Tensor,
        refine_mask: torch.Tensor,
        noise_level: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        initial = self._integrate_background_plan(current, controls, current_valid)
        refined = controls
        for _ in range(max(1, int(self.cfg.refinement_iterations))):
            states = self._integrate_background_plan(current, refined, current_valid)
            residual = self.joint_refiner.residual(
                refined, states, agents, scene, memory, behavior, map_tokens, map_valid,
                ego_controls, ego_states, refine_mask, noise_level,
            )
            refined = self._clamp_controls(refined + residual) * refine_mask[..., None].float()
        return refined, initial, self._integrate_background_plan(current, refined, current_valid)

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
        ego_controls: torch.Tensor,
        ego_states: torch.Tensor,
        *,
        previous_buffer: torch.Tensor | None = None,
        previous_current: torch.Tensor | None = None,
        previous_memory: torch.Tensor | None = None,
        previous_ego_control: torch.Tensor | None = None,
        start_anchor_controls: torch.Tensor | None = None,
        start_mode: bool = False,
    ) -> dict[str, Any]:
        """One ROLL step; raw B0 is intentionally absent from this interface."""
        if start_mode:
            agents, scene, map_tokens, map_valid = self.encoder.encode_start(
                current, current_valid, ego_mask, map_polylines, map_polyline_valid, lane_graph_edges,
            )
        else:
            agents, scene, map_tokens, map_valid = self.encoder(
                history, history_valid, current, current_valid, ego_mask,
                map_polylines, map_polyline_valid, lane_graph_edges,
            )
        delta = torch.zeros_like(current) if previous_current is None else current - previous_current
        ego_previous = current.new_zeros((current.shape[0], 2)) if previous_ego_control is None else previous_ego_control
        memory = previous_memory
        if memory is None:
            memory = self.scene_memory(scene, agents, previous_buffer, delta, ego_previous, None)
        elif previous_buffer is not None:
            memory = self.scene_memory(scene, agents, previous_buffer, delta, ego_previous, memory)
        fresh = self.joint_refiner.fresh_plan(agents, memory, behavior_latent, ego_controls, ego_states)
        valid_mask = current_valid[:, None, 1:].expand(-1, self.cfg.plan_frames, -1)
        carried_mask = torch.zeros_like(valid_mask)
        appended_mask = torch.ones_like(valid_mask)
        if previous_buffer is None:
            pre_refinement = fresh
            if start_anchor_controls is not None:
                pre_refinement = self._mix_start_controls(fresh, start_anchor_controls)
        else:
            execute = int(self.cfg.execute_frames)
            carried = torch.cat((previous_buffer[:, execute:], fresh[:, -execute:]), dim=1)
            pre_refinement = (1.0 - self.cfg.buffer_carry_mix) * fresh + self.cfg.buffer_carry_mix * carried
            carried_mask[:, : -execute] = valid_mask[:, : -execute]
            appended_mask[:, : -execute] = False
        noise = current.new_zeros((current.shape[0],))
        refined, initial_states, refined_states = self._refine_controls(
            pre_refinement, current, current_valid, agents, scene, memory, behavior_latent,
            map_tokens, map_valid, ego_controls, ego_states, valid_mask, noise,
        )
        return {
            "agent_context": agents, "scene_context": scene, "map_context": map_tokens,
            "map_context_valid": map_valid, "scene_memory": memory,
            "pre_refinement_buffer": pre_refinement,
            "refined_buffer": refined, "initial_plan_states": initial_states,
            "refined_plan_states": refined_states,
            "buffer_masks": {
                "carried": carried_mask, "appended": appended_mask & valid_mask,
                "refinable": valid_mask, "valid": valid_mask,
            },
        }

    def _target_plan(self, batch: dict[str, torch.Tensor], response: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        pred_bg, true_bg, ego = predicted[:, :, 1:], target[:, :, 1:], predicted[:, :, :1]
        target_ego = target[:, :, :1]
        gap_error = ((pred_bg[..., 0] - ego[..., 0]).abs() - (true_bg[..., 0] - target_ego[..., 0]).abs()).abs()
        speed_error = ((pred_bg[..., 2] - ego[..., 2]) - (true_bg[..., 2] - target_ego[..., 2])).abs()
        return _masked_mean(gap_error + 0.25 * speed_error, valid)

    @staticmethod
    def _physical_loss(predicted: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        background, ego = predicted[:, :, 1:], predicted[:, :, :1]
        dx, dy = (background[..., 0] - ego[..., 0]).abs(), (background[..., 1] - ego[..., 1]).abs()
        ego_barrier = functional.softplus((4.5 - dx) / 0.75) * functional.softplus((1.0 - dy) / 0.25)
        pair_dx = (background[..., None, :, 0] - background[..., :, None, 0]).abs()
        pair_dy = (background[..., None, :, 1] - background[..., :, None, 1]).abs()
        pair = functional.softplus((4.5 - pair_dx) / 0.75) * functional.softplus((1.0 - pair_dy) / 0.25)
        count = background.shape[2]
        upper = torch.triu(torch.ones((count, count), dtype=torch.bool, device=background.device), diagonal=1)
        pair_valid = valid[..., :, None] & valid[..., None, :] & upper
        return _masked_mean(ego_barrier, valid) + _masked_mean(pair, pair_valid)

    def _behavior_latent(
        self,
        batch: dict[str, torch.Tensor],
        agents: torch.Tensor,
        scene: torch.Tensor,
        memory: torch.Tensor,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        start_seed: torch.Tensor,
        *,
        deterministic: bool,
        use_posterior: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prior_mean, prior_log = self.behavior.prior_parameters(agents, scene, memory, start_seed)
        terms = {name: current.new_zeros(()) for name in ("behavior_kl", "behavior_reconstruction", "diversity_floor")}
        mean, log_scale = prior_mean, prior_log
        if use_posterior:
            future, future_valid = batch["agent_states"][:, 25:50], batch["agent_valid"][:, 25:50]
            posterior_mean, posterior_log, future_feature = self.behavior.posterior_parameters(
                agents, scene, memory, current, future, future_valid, start_seed
            )
            mean, log_scale = posterior_mean, posterior_log
            ratio = torch.exp(2.0 * (posterior_log - prior_log))
            kl = prior_log - posterior_log + 0.5 * (ratio + (posterior_mean - prior_mean).square() * torch.exp(-2.0 * prior_log) - 1.0)
            terms["behavior_kl"] = _masked_mean(kl.mean(dim=-1), current_valid)
            terms["behavior_reconstruction"] = _masked_mean(
                (self.behavior.reconstruction(posterior_mean) - future_feature).abs().mean(dim=-1), current_valid
            )
        terms["diversity_floor"] = functional.relu(0.12 - torch.exp(log_scale).mean())
        sample = mean if deterministic else mean + torch.randn_like(mean) * torch.exp(log_scale)
        return sample * current_valid[..., None].float(), terms

    def _ego_controls_from_replay(self, states: torch.Tensor, *, frames: int) -> torch.Tensor:
        """Adapt externally replayed highD ego acceleration to ADS control input."""
        ego_states = states[:, 24 : 24 + frames, 0]
        return self.dynamics.controls_from_highd_actions(ego_states[..., 4:6], ego_states)

    def _ego_window(self, controls: torch.Tensor, response: int) -> torch.Tensor:
        start, frames = int(response) * self.cfg.execute_frames, self.cfg.plan_frames
        value = controls[:, start : start + frames]
        if value.shape[1] == frames:
            return value
        if not value.shape[1]:
            raise ValueError("ego controls do not cover the requested response")
        return torch.cat((value, value[:, -1:].expand(-1, frames - value.shape[1], -1)), dim=1)

    def _integrate_ego_plan(
        self, current: torch.Tensor, current_valid: torch.Tensor, controls: torch.Tensor,
    ) -> torch.Tensor:
        """Convert ADS controls into the immutable ego state token sequence."""
        state, valid = current[:, :1], current_valid[:, :1]
        frames: list[torch.Tensor] = []
        for frame in range(controls.shape[1]):
            state = self.dynamics.step(state, controls[:, frame, None], valid, self.cfg.simulation_dt_s)
            frames.append(state[:, 0])
        return torch.stack(frames, dim=1)

    def _replay_ego_window(self, states: torch.Tensor, response: int) -> torch.Tensor:
        start, frames = 25 + int(response) * self.cfg.execute_frames, self.cfg.plan_frames
        value = states[:, start : start + frames, 0]
        if value.shape[1] == frames:
            return value
        if not value.shape[1]:
            raise ValueError("logged ego replay does not cover the requested response")
        return torch.cat((value, value[:, -1:].expand(-1, frames - value.shape[1], -1)), dim=1)

    def _denoising_loss(
        self, target_controls: torch.Tensor, target_valid: torch.Tensor, out: dict[str, Any], current: torch.Tensor, current_valid: torch.Tensor,
        behavior: torch.Tensor, ego_controls: torch.Tensor, ego_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        levels = target_controls.new_tensor(self.cfg.refinement_noise_levels)
        choice = torch.randint(len(levels), (target_controls.shape[0],), device=target_controls.device)
        noise_level = levels[choice]
        noise = torch.randn_like(target_controls) * target_controls.new_tensor(
            (self.cfg.denoising_acceleration_std, self.cfg.denoising_yaw_rate_std)
        )
        corrupt = self._clamp_controls(target_controls + noise_level[:, None, None, None] * noise)
        corrupt = torch.where(target_valid[..., None], corrupt, target_controls)
        states = self._integrate_background_plan(current, corrupt, current_valid)
        residual = self.joint_refiner.residual(
            corrupt, states, out["agent_context"], out["scene_context"], out["scene_memory"], behavior,
            out["map_context"], out["map_context_valid"], ego_controls, ego_states, target_valid, noise_level,
        )
        denoised = self._clamp_controls(corrupt + residual) * target_valid[..., None].float()
        return _masked_mean((denoised - target_controls).abs().mean(dim=-1), target_valid), noise_level.mean()

    def _rollout(
        self,
        batch: dict[str, torch.Tensor],
        *,
        response_steps: int | None = None,
        deterministic: bool = True,
        use_posterior: bool = False,
        tbptt_steps: int = 0,
        ego_future_controls: torch.Tensor,
        training_denoising: bool = False,
        replay_ego: bool,
        start_mode: bool,
    ) -> dict[str, Any]:
        """Shared rollout body; only reconstruction is permitted to replay ego."""
        states, valid = batch["agent_states"], batch["agent_valid"]
        steps = min(int(response_steps or self.cfg.response_steps), self.cfg.response_steps)
        total_frames = steps * self.cfg.execute_frames
        ego_mask = self._ego_mask(batch)
        if not torch.all(ego_mask[:, 0]):
            raise ValueError("QR-WM uses the fixed [ego, six background slots] tensor schema")
        controls = ego_future_controls
        if controls.shape[:2] != (states.shape[0], total_frames) or controls.shape[-1] != 2:
            raise ValueError("ego_future_controls must have shape [batch, response_steps * execute_frames, 2]")
        current, current_valid = states[:, 24], valid[:, 24]
        history, history_valid = (
            (current[:, None], current_valid[:, None]) if start_mode else (states[:, :25], valid[:, :25])
        )
        if start_mode:
            initial_agents, initial_scene, _, _ = self.encoder.encode_start(
                current, current_valid, ego_mask,
                batch["map_polylines"], batch["map_polyline_valid"], batch["lane_graph_edges"],
            )
        else:
            initial_agents, initial_scene, _, _ = self.encoder(
                history, history_valid, current, current_valid, ego_mask,
                batch["map_polylines"], batch["map_polyline_valid"], batch["lane_graph_edges"],
            )
        anchor_controls, start_seed = self._start_anchor(
            current, current_valid, batch.get("behavior_anchor_raw"), batch.get("behavior_anchor_valid")
        )
        initial_memory = self.scene_memory(
            initial_scene, initial_agents, anchor_controls, torch.zeros_like(current), current.new_zeros((current.shape[0], 2)), None
        )
        behavior, behavior_terms = self._behavior_latent(
            batch, initial_agents, initial_scene, initial_memory, current, current_valid, start_seed,
            deterministic=deterministic, use_posterior=use_posterior,
        )
        predicted_frames: list[torch.Tensor] = []
        plans: list[torch.Tensor] = []
        pre_refinement: list[torch.Tensor] = []
        plan_states: list[torch.Tensor] = []
        initial_plan_states: list[torch.Tensor] = []
        masks: dict[str, list[torch.Tensor]] = {name: [] for name in BUFFER_MASK_NAMES}
        term_rows: list[dict[str, torch.Tensor]] = []
        generated_valid_frames: list[torch.Tensor] = []
        previous_buffer = previous_current = previous_ego = None
        previous_memory: torch.Tensor | None = initial_memory
        for response in range(steps):
            target, target_valid, target_controls, target_control_valid = self._target_plan(batch, response)
            step_current, step_valid = current, current_valid
            ego_controls = self._ego_window(controls, response)
            ego_states = self._replay_ego_window(states, response) if replay_ego else self._integrate_ego_plan(
                current, current_valid, ego_controls
            )
            out = self.plan_step(
                history, history_valid, current, current_valid, ego_mask,
                batch["map_polylines"], batch["map_polyline_valid"], batch["lane_graph_edges"], behavior, ego_controls, ego_states,
                previous_buffer=previous_buffer, previous_current=previous_current, previous_memory=previous_memory,
                previous_ego_control=previous_ego, start_anchor_controls=anchor_controls if response == 0 else None,
                start_mode=start_mode and response == 0,
            )
            plan = out["refined_buffer"]
            generated, response_frames, response_valid = current.clone(), [], []
            for frame in range(self.cfg.execute_frames):
                physical = current.new_zeros((current.shape[0], current.shape[1], 2))
                physical[:, 0] = controls[:, response * self.cfg.execute_frames + frame]
                physical[:, 1:] = plan[:, frame]
                generated = self.dynamics.step(generated, physical, current_valid, self.cfg.simulation_dt_s)
                if replay_ego:
                    target_frame = 25 + response * self.cfg.execute_frames + frame
                    generated = torch.where(ego_mask[..., None], states[:, target_frame], generated)
                    current_valid = torch.where(ego_mask, valid[:, target_frame], current_valid)
                response_frames.append(generated)
                response_valid.append(current_valid)
            predicted = torch.stack(response_frames, dim=1)
            execute_valid, full_valid = target_valid[:, : self.cfg.execute_frames, 1:], target_valid[:, :, 1:]
            position = _masked_mean((predicted[:, :, 1:, :2] - target[:, : self.cfg.execute_frames, 1:, :2]).abs().mean(dim=-1), execute_valid)
            velocity = _masked_mean((predicted[:, :, 1:, 2:4] - target[:, : self.cfg.execute_frames, 1:, 2:4]).abs().mean(dim=-1), execute_valid)
            control_loss = _masked_mean((plan[:, : self.cfg.execute_frames] - target_controls[:, : self.cfg.execute_frames]).abs().mean(dim=-1), target_control_valid[:, : self.cfg.execute_frames])
            full_position = _masked_mean((out["refined_plan_states"][..., :2] - target[:, :, 1:, :2]).abs().mean(dim=-1), full_valid)
            initial_position = _masked_mean((out["initial_plan_states"][..., :2] - target[:, :, 1:, :2]).abs().mean(dim=-1), full_valid)
            full_control = _masked_mean((plan - target_controls).abs().mean(dim=-1), target_control_valid)
            overlap = plan.new_zeros(()) if previous_buffer is None else (previous_buffer[:, self.cfg.execute_frames:] - plan[:, : -self.cfg.execute_frames]).abs().mean()
            denoising, noise_level = plan.new_zeros(()), plan.new_zeros(())
            if training_denoising:
                denoising, noise_level = self._denoising_loss(
                    target_controls, target_control_valid, out, step_current, step_valid, behavior, ego_controls, ego_states
                )
            term_rows.append({
                "position": position, "velocity": velocity, "control": control_loss, "plan_position": full_position,
                "initial_plan_position": initial_position, "plan_control": full_control, "overlap": overlap,
                "interaction": self._interaction_loss(predicted, target[:, : self.cfg.execute_frames], execute_valid),
                "physical": self._physical_loss(predicted, execute_valid), "denoising": denoising,
                "denoising_noise_level": noise_level, "refinable_fraction": out["buffer_masks"]["refinable"].float().mean(),
            })
            predicted_frames.extend(response_frames)
            generated_valid_frames.extend(response_valid)
            plans.append(plan); pre_refinement.append(out["pre_refinement_buffer"])
            plan_states.append(out["refined_plan_states"]); initial_plan_states.append(out["initial_plan_states"])
            for name, value in out["buffer_masks"].items():
                masks[name].append(value)
            previous_buffer, previous_current, previous_memory = plan, step_current, out["scene_memory"]
            previous_ego = controls[:, (response + 1) * self.cfg.execute_frames - 1]
            current = predicted[:, -1]
            appended_valid = torch.stack(response_valid, dim=1)
            history, history_valid = torch.cat((history, predicted), dim=1)[:, -25:], torch.cat((history_valid, appended_valid), dim=1)[:, -25:]
            if tbptt_steps and (response + 1) % int(tbptt_steps) == 0 and response + 1 < steps:
                history, history_valid, current, previous_buffer, previous_memory = (
                    history.detach(), history_valid.detach(), current.detach(), previous_buffer.detach(), previous_memory.detach()
                )
        start_summary = current.new_zeros(())
        if start_mode and batch.get("behavior_anchor_raw") is not None and len(predicted_frames) >= self.cfg.plan_frames:
            prefix = torch.cat((states[:, 24:25], torch.stack(predicted_frames[: self.cfg.plan_frames], dim=1)), dim=1)
            prefix_valid = torch.cat((valid[:, 24:25], torch.stack(generated_valid_frames[: self.cfg.plan_frames], dim=1)), dim=1)
            generated_anchor, generated_anchor_valid = summarize_first_second_states(prefix[:, :, 1:], prefix_valid[:, :, 1:])
            anchor_valid = generated_anchor_valid & batch.get("behavior_anchor_valid", current_valid[:, 1:]).bool()
            start_summary = _masked_mean(
                (generated_anchor - batch["behavior_anchor_raw"]).abs().mean(dim=-1), anchor_valid
            )
        return {
            "predicted_states": torch.stack(predicted_frames, dim=1),
            "target_states": states[:, 25 : 25 + total_frames], "target_valid": valid[:, 25 : 25 + total_frames],
            "control_buffers": torch.stack(plans, dim=1), "pre_refinement_buffers": torch.stack(pre_refinement, dim=1),
            "refined_plan_states": torch.stack(plan_states, dim=1), "initial_plan_states": torch.stack(initial_plan_states, dim=1),
            "control_buffer_masks": self._stack_masks(masks),
            "executed_control_masks": torch.stack([item[:, : self.cfg.execute_frames] for item in masks["valid"]], dim=1),
            "behavior_latent": behavior, "behavior_terms": behavior_terms, "loss_terms": term_rows,
            "start_summary": start_summary, "start_mode": start_mode,
        }

    def rollout_reconstruction(
        self,
        batch: dict[str, torch.Tensor],
        *,
        response_steps: int | None = None,
        deterministic: bool = True,
        use_posterior: bool = False,
        tbptt_steps: int = 0,
        training_denoising: bool = False,
        start_mode: bool = False,
    ) -> dict[str, Any]:
        """Logged highD ego replay, used only for training and reconstruction."""
        steps = min(int(response_steps or self.cfg.response_steps), self.cfg.response_steps)
        return self._rollout(
            batch, response_steps=steps, deterministic=deterministic, use_posterior=use_posterior,
            tbptt_steps=tbptt_steps, ego_future_controls=self._ego_controls_from_replay(
                batch["agent_states"], frames=steps * self.cfg.execute_frames
            ), training_denoising=training_denoising, replay_ego=True, start_mode=start_mode,
        )

    def rollout(
        self,
        batch: dict[str, torch.Tensor],
        *,
        ego_future_controls: torch.Tensor | None = None,
        response_steps: int | None = None,
        deterministic: bool = True,
        use_posterior: bool = False,
        tbptt_steps: int = 0,
        training_denoising: bool = False,
    ) -> dict[str, Any]:
        """Dynamic ADS rollout; ego controls are required and never replayed."""
        if ego_future_controls is None:
            raise ValueError("rollout requires ADS ego_future_controls; use rollout_reconstruction for logged highD replay")
        return self._rollout(
            batch, response_steps=response_steps, deterministic=deterministic, use_posterior=use_posterior,
            tbptt_steps=tbptt_steps, ego_future_controls=ego_future_controls,
            training_denoising=training_denoising, replay_ego=False, start_mode=False,
        )

    @torch.no_grad()
    def rollout_from_flow(
        self,
        flow_condition: torch.Tensor,
        *,
        slot_valid: torch.Tensor | None,
        map_polylines: torch.Tensor,
        map_polyline_valid: torch.Tensor,
        lane_graph_edges: torch.Tensor,
        ego_future_controls: torch.Tensor,
        response_steps: int | None = None,
        deterministic: bool = True,
        flow_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """START from Flow's ``C0+B0`` and ROLL using ADS-provided ego controls."""
        current, current_valid, raw_anchor = self.flow_condition_to_scene(flow_condition, slot_valid)
        steps = min(int(response_steps or self.cfg.response_steps), self.cfg.response_steps)
        total = steps * self.cfg.execute_frames
        if ego_future_controls.shape != (current.shape[0], total, 2):
            raise ValueError("ego_future_controls must have shape [batch, response_steps * execute_frames, 2]")
        ego_mask = torch.zeros_like(current_valid); ego_mask[:, 0] = True
        history, history_valid = current[:, None], current_valid[:, None]
        start = self.initialize_start(
            current, current_valid, ego_mask, map_polylines, map_polyline_valid, lane_graph_edges,
            raw_anchor, current_valid[:, 1:], deterministic=deterministic,
        )
        memory, behavior, anchor_controls = start["scene_memory"], start["behavior_latent"], start["start_anchor_controls"]
        frames: list[torch.Tensor] = []
        plans: list[torch.Tensor] = []
        masks: dict[str, list[torch.Tensor]] = {name: [] for name in BUFFER_MASK_NAMES}
        previous_buffer = previous_current = previous_ego = None
        for response in range(steps):
            step_current = current
            ego_controls = self._ego_window(ego_future_controls, response)
            ego_states = self._integrate_ego_plan(current, current_valid, ego_controls)
            out = self.plan_step(
                history, history_valid, current, current_valid, ego_mask, map_polylines, map_polyline_valid, lane_graph_edges,
                behavior, ego_controls, ego_states, previous_buffer=previous_buffer, previous_current=previous_current,
                previous_memory=memory, previous_ego_control=previous_ego,
                start_anchor_controls=anchor_controls if response == 0 else None,
                start_mode=response == 0,
            )
            plan = out["refined_buffer"]
            for frame in range(self.cfg.execute_frames):
                physical = current.new_zeros((current.shape[0], current.shape[1], 2))
                physical[:, 0] = ego_future_controls[:, response * self.cfg.execute_frames + frame]
                physical[:, 1:] = plan[:, frame]
                current = self.dynamics.step(current, physical, current_valid, self.cfg.simulation_dt_s)
                frames.append(current)
            appended_valid = current_valid[:, None].expand(-1, self.cfg.execute_frames, -1)
            history, history_valid = torch.cat((history, torch.stack(frames[-self.cfg.execute_frames:], dim=1)), dim=1)[:, -25:], torch.cat((history_valid, appended_valid), dim=1)[:, -25:]
            previous_buffer, previous_current, memory = plan, step_current, out["scene_memory"]
            previous_ego = ego_future_controls[:, (response + 1) * self.cfg.execute_frames - 1]
            plans.append(plan)
            for name, value in out["buffer_masks"].items():
                masks[name].append(value)
        return {
            "predicted_states": torch.stack(frames, dim=1), "control_buffers": torch.stack(plans, dim=1),
            "control_buffer_masks": self._stack_masks(masks), "behavior_latent": behavior,
            "flow_metadata": {} if flow_metadata is None else dict(flow_metadata),
        }

    def _objective(self, rollout: dict[str, Any]) -> dict[str, torch.Tensor]:
        terms = {key: torch.stack([value[key] for value in rollout["loss_terms"]]).mean() for key in rollout["loss_terms"][0]}
        behavior = rollout["behavior_terms"]
        loss = (
            self.cfg.position_weight * terms["position"] + self.cfg.velocity_weight * terms["velocity"] + self.cfg.control_weight * terms["control"]
            + self.cfg.plan_position_weight * terms["plan_position"] + self.cfg.plan_control_weight * terms["plan_control"]
            + self.cfg.refinement_weight * functional.relu(terms["plan_position"] - terms["initial_plan_position"] + 0.01)
            + self.cfg.denoising_weight * terms["denoising"] + self.cfg.overlap_weight * terms["overlap"]
            + self.cfg.interaction_weight * terms["interaction"] + self.cfg.physical_weight * terms["physical"]
            + self.cfg.behavior_kl_weight * behavior["behavior_kl"] + self.cfg.behavior_reconstruction_weight * behavior["behavior_reconstruction"]
            + self.cfg.diversity_weight * behavior["diversity_floor"]
            + self.cfg.start_summary_weight * rollout["start_summary"]
        )
        return {"loss": loss, **terms, **behavior, "start_summary": rollout["start_summary"]}

    @staticmethod
    def _select_batch(batch: dict[str, torch.Tensor], index: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = batch["agent_states"].shape[0]
        return {
            key: value.index_select(0, index) if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == batch_size else value
            for key, value in batch.items()
        }

    def supervised_terms(
        self,
        batch: dict[str, torch.Tensor],
        *,
        response_steps: int | None = None,
        start_mode: bool,
        training: bool = False,
        tbptt_steps: int = 0,
    ) -> dict[str, torch.Tensor]:
        rollout = self.rollout_reconstruction(
            batch, response_steps=response_steps, deterministic=not training, use_posterior=training,
            tbptt_steps=tbptt_steps, training_denoising=training, start_mode=start_mode,
        )
        return self._objective(rollout)

    def forward_training(self, batch: dict[str, torch.Tensor], *, response_steps: int | None = None, tbptt_steps: int = 5) -> dict[str, torch.Tensor]:
        """Balance no-history START and history-aware ROLL samples in every batch."""
        batch_size = batch["agent_states"].shape[0]
        if batch_size < 2:
            return self.supervised_terms(
                batch, response_steps=response_steps, start_mode=True, training=True, tbptt_steps=tbptt_steps,
            )
        start_count = min(batch_size - 1, max(1, round(batch_size * float(self.cfg.start_training_fraction))))
        order = torch.randperm(batch_size, device=batch["agent_states"].device)
        start = self.supervised_terms(
            self._select_batch(batch, order[:start_count]), response_steps=response_steps,
            start_mode=True, training=True, tbptt_steps=tbptt_steps,
        )
        roll = self.supervised_terms(
            self._select_batch(batch, order[start_count:]), response_steps=response_steps,
            start_mode=False, training=True, tbptt_steps=tbptt_steps,
        )
        out = {key: 0.5 * (start[key] + roll[key]) for key in start}
        out.update({"start_loss": start["loss"], "roll_loss": roll["loss"], "start_fraction": out["loss"].new_tensor(start_count / batch_size)})
        return out

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type, "model_config": asdict(self.cfg), "state_dict": self.state_dict(),
            "architecture_version": 3,
            "flow_interface": {
                "input_dim": 76, "layout": "ego[vx,vy,ax,ay]+background_relative[6,6]+B0[6,6]",
                "scene_tensor_shape": [7, 6], "b0_lifecycle": "START-only: initializes latent, scene memory, and first control buffer",
                "start_encoder": "C0 plus map only; no synthetic history", "ego_condition": "future ego state token from controls",
                "flow_schema_sha256": self.flow_schema_sha256,
            },
        }
