"""QR-WM: joint background future-action sequences with START-only Flow conditioning."""

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            max_yaw_rate=self.cfg.max_yaw_rate,
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

    def _clamp_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                actions[..., 0].clamp(self.cfg.min_acceleration, self.cfg.max_acceleration),
                actions[..., 1].clamp(-self.cfg.max_yaw_rate, self.cfg.max_yaw_rate),
            ), dim=-1
        )

    def _mix_start_actions(self, fresh: torch.Tensor, anchor_actions: torch.Tensor) -> torch.Tensor:
        """Blend B0 actions once, with a decaying convex weight over one second."""
        alpha = torch.linspace(
            float(self.cfg.start_anchor_mix), 0.0, self.cfg.plan_frames,
            device=fresh.device, dtype=fresh.dtype,
        )[None, :, None, None]
        return self._clamp_actions((1.0 - alpha) * fresh + alpha * anchor_actions)

    def _start_anchor(
        self,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        raw_anchor: torch.Tensor | None,
        anchor_valid: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Consume raw B0 exactly once to seed actions, memory, and behavior."""
        batch, agents = current.shape[:2]
        background = agents - 1
        empty_seed = current.new_zeros((batch, agents, self.cfg.behavior_latent_dim))
        if raw_anchor is None:
            return None, empty_seed
        if raw_anchor.shape != (batch, background, 6):
            raise ValueError("B0 must have shape [batch, six background slots, 6]")
        if anchor_valid is not None and anchor_valid.shape != (batch, background):
            raise ValueError("B0 validity must have shape [batch, six background slots]")
        valid = (current_valid[:, 1:] if anchor_valid is None else anchor_valid.bool()) & current_valid[:, 1:]
        highd = self.start_anchor_plan(current, raw_anchor, valid)
        actions = self.dynamics.controls_from_highd_actions(highd, current[:, None, 1:])
        seed = empty_seed.clone()
        seed[:, 1:] = self.start_behavior(raw_anchor) * valid[..., None].float()
        return self._clamp_actions(actions) * valid[:, None, :, None].float(), seed

    def _initialize_episode_state(
        self,
        batch: dict[str, torch.Tensor],
        agents: torch.Tensor,
        scene: torch.Tensor,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        raw_anchor: torch.Tensor | None,
        anchor_valid: torch.Tensor | None,
        *,
        deterministic: bool,
        use_posterior: bool,
        behavior_standard_normal: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Initialize B0-derived actions, memory, and behavior once per rollout."""
        anchor_actions, start_seed = self._start_anchor(current, current_valid, raw_anchor, anchor_valid)
        memory = self.scene_memory(scene, agents, anchor_actions, torch.zeros_like(current), None)
        behavior, terms = self._behavior_latent(
            batch, agents, scene, memory, current, current_valid, start_seed,
            deterministic=deterministic, use_posterior=use_posterior,
            behavior_standard_normal=behavior_standard_normal,
        )
        return anchor_actions, memory, behavior, terms

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
        behavior_standard_normal: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Initialize the shared START state once from C0, B0, and map."""
        agents, scene, _, _ = self.encoder.encode_start(
            current, current_valid, ego_mask, map_polylines, map_polyline_valid, lane_graph_edges,
        )
        anchor_actions, memory, behavior, _ = self._initialize_episode_state(
            {}, agents, scene, current, current_valid, raw_anchor, anchor_valid,
            deterministic=deterministic, use_posterior=False,
            behavior_standard_normal=behavior_standard_normal,
        )
        if anchor_actions is None:
            raise ValueError("START initialization requires B0 actions")
        return {"scene_memory": memory, "behavior_latent": behavior, "start_anchor_actions": anchor_actions}

    @staticmethod
    def _stack_masks(masks: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {name: torch.stack(masks[name], dim=1) for name in BUFFER_MASK_NAMES}

    def _integrate_background_actions(self, current: torch.Tensor, actions: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        state = current[:, 1:]
        frames: list[torch.Tensor] = []
        for frame in range(actions.shape[1]):
            state = self.dynamics.step(state, actions[:, frame], valid[:, 1:], self.cfg.simulation_dt_s)
            frames.append(state)
        return torch.stack(frames, dim=1)

    def _refine_actions(
        self,
        actions: torch.Tensor,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        agents: torch.Tensor,
        scene: torch.Tensor,
        memory: torch.Tensor,
        behavior: torch.Tensor,
        map_tokens: torch.Tensor,
        map_valid: torch.Tensor,
        refine_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        initial = self._integrate_background_actions(current, actions, current_valid)
        refined = actions
        for _ in range(max(1, int(self.cfg.refinement_iterations))):
            states = self._integrate_background_actions(current, refined, current_valid)
            residual = self.joint_refiner.residual(
                refined, states, agents, scene, memory, behavior, map_tokens, map_valid,
                refine_mask,
            )
            refined = self._clamp_actions(refined - residual) * refine_mask[..., None].float()
        return refined, initial, self._integrate_background_actions(current, refined, current_valid)

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
        start_anchor_actions: torch.Tensor | None = None,
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
        memory = previous_memory
        if memory is None:
            memory = self.scene_memory(scene, agents, previous_buffer, delta, None)
        elif previous_buffer is not None:
            memory = self.scene_memory(scene, agents, previous_buffer, delta, memory)
        fresh = self.joint_refiner.fresh_plan(agents, memory, behavior_latent)
        valid_mask = current_valid[:, None, 1:].expand(-1, self.cfg.plan_frames, -1)
        carried_mask = torch.zeros_like(valid_mask)
        appended_mask = torch.ones_like(valid_mask)
        if previous_buffer is None:
            pre_refinement = fresh
            if start_anchor_actions is not None:
                pre_refinement = self._mix_start_actions(fresh, start_anchor_actions)
        else:
            execute = int(self.cfg.execute_frames)
            carried = torch.cat((previous_buffer[:, execute:], fresh[:, -execute:]), dim=1)
            pre_refinement = (1.0 - self.cfg.buffer_carry_mix) * fresh + self.cfg.buffer_carry_mix * carried
            carried_mask[:, : -execute] = valid_mask[:, : -execute]
            appended_mask[:, : -execute] = False
        refined, initial_states, refined_states = self._refine_actions(
            pre_refinement, current, current_valid, agents, scene, memory, behavior_latent,
            map_tokens, map_valid, valid_mask,
        )
        return {
            "agent_context": agents, "scene_context": scene, "map_context": map_tokens,
            "map_context_valid": map_valid, "scene_memory": memory,
            "background_future_actions_before_refinement": pre_refinement,
            "background_future_actions": refined,
            "initial_background_future_states": initial_states,
            "refined_background_future_states": refined_states,
            "background_future_action_masks": {
                "carried": carried_mask, "appended": appended_mask & valid_mask,
                "refinable": valid_mask, "valid": valid_mask,
            },
        }

    def _target_plan(self, batch: dict[str, torch.Tensor], response: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states, valid, recorded_actions = batch["agent_states"], batch["agent_valid"], batch["actions_highd"]
        start = int(response) * self.cfg.execute_frames
        count = min(self.cfg.plan_frames, int(recorded_actions.shape[1]) - start)
        batch_size, agents = states.shape[0], states.shape[2]
        target_states = states.new_zeros((batch_size, self.cfg.plan_frames, agents, 6))
        target_valid = torch.zeros((batch_size, self.cfg.plan_frames, agents), dtype=torch.bool, device=states.device)
        target_actions = states.new_zeros((batch_size, self.cfg.plan_frames, agents - 1, 2))
        action_valid = torch.zeros((batch_size, self.cfg.plan_frames, agents - 1), dtype=torch.bool, device=states.device)
        if count:
            target_states[:, :count] = states[:, 25 + start : 25 + start + count]
            target_valid[:, :count] = valid[:, 25 + start : 25 + start + count]
            current = states[:, 24 + start : 24 + start + count, 1:]
            target_actions[:, :count] = self.dynamics.controls_from_highd_actions(recorded_actions[:, start : start + count], current)
            action_valid[:, :count] = valid[:, 25 + start : 25 + start + count, 1:]
        return target_states, target_valid, target_actions, action_valid

    def _response_count(self, response_steps: int | None) -> int:
        steps = self.cfg.response_steps if response_steps is None else int(response_steps)
        if steps < 1:
            raise ValueError("response_steps must be positive")
        return min(steps, self.cfg.response_steps)

    @staticmethod
    def _available_rollout_frames(batch: dict[str, torch.Tensor]) -> int:
        """Number of supervised physical transitions represented by a batch."""
        states = batch["agent_states"]
        actions = batch["actions_highd"]
        # Cache index 24 is C0; the first predicted state is index 25/S1.
        state_transitions = max(0, int(states.shape[1]) - 25)
        return min(state_transitions, int(actions.shape[1]))

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
        behavior_standard_normal: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prior_mean, prior_log = self.behavior.prior_parameters(agents, scene, memory, start_seed)
        terms = {name: current.new_zeros(()) for name in ("behavior_kl", "behavior_reconstruction", "diversity_floor")}
        mean, log_scale = prior_mean, prior_log
        if use_posterior:
            future = batch["agent_states"][:, 25:50].clone()
            future_valid = batch["agent_valid"][:, 25:50].clone()
            future[:, :, 0] = current[:, None, 0]
            future_valid[:, :, 0] = current_valid[:, None, 0]
            posterior_mean, posterior_log, future_feature = self.behavior.posterior_parameters(
                agents, scene, memory, current, future, future_valid, start_seed
            )
            mean, log_scale = posterior_mean, posterior_log
            background = current_valid.clone()
            background[:, 0] = False
            ratio = torch.exp(2.0 * (posterior_log - prior_log))
            kl = prior_log - posterior_log + 0.5 * (ratio + (posterior_mean - prior_mean).square() * torch.exp(-2.0 * prior_log) - 1.0)
            terms["behavior_kl"] = _masked_mean(kl.mean(dim=-1), background)
            terms["behavior_reconstruction"] = _masked_mean(
                (self.behavior.reconstruction(posterior_mean) - future_feature).abs().mean(dim=-1), background
            )
        terms["diversity_floor"] = functional.relu(0.12 - torch.exp(log_scale).mean())
        if deterministic:
            if behavior_standard_normal is not None:
                raise ValueError("deterministic QR rollout must not receive behavior_standard_normal")
            sample = mean
        elif behavior_standard_normal is None:
            sample = mean + torch.randn_like(mean) * torch.exp(log_scale)
        else:
            if behavior_standard_normal.shape != mean.shape:
                raise ValueError(
                    "behavior_standard_normal must have shape "
                    "[batch, agents, behavior_latent_dim]"
                )
            sample = mean + behavior_standard_normal.to(device=mean.device, dtype=mean.dtype) * torch.exp(log_scale)
        return sample * current_valid[..., None].float(), terms

    def _rollout(
        self,
        batch: dict[str, torch.Tensor],
        *,
        response_steps: int | None = None,
        deterministic: bool = True,
        use_posterior: bool = False,
        tbptt_steps: int = 0,
        start_mode: bool,
        behavior_standard_normal: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Logged-state rollout; planning sees only ego state observed so far."""
        states, valid = batch["agent_states"], batch["agent_valid"]
        requested_steps = self._response_count(response_steps)
        available_frames = self._available_rollout_frames(batch)
        total_frames = min(
            self.cfg.rollout_frames_for_responses(requested_steps),
            available_frames,
        )
        if total_frames < 1:
            raise ValueError("QR rollout batch does not contain a supervised future transition")
        steps = (total_frames + self.cfg.execute_frames - 1) // self.cfg.execute_frames
        ego_mask = self._ego_mask(batch)
        if not torch.all(ego_mask[:, 0]):
            raise ValueError("QR-WM uses the fixed [ego, six background slots] tensor schema")
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
        anchor_actions, initial_memory, behavior, behavior_terms = self._initialize_episode_state(
            batch, initial_agents, initial_scene, current, current_valid,
            batch.get("behavior_anchor_raw"), batch.get("behavior_anchor_valid"),
            deterministic=deterministic, use_posterior=use_posterior,
            behavior_standard_normal=behavior_standard_normal,
        )
        predicted_frames: list[torch.Tensor] = []
        plans: list[torch.Tensor] = []
        pre_refinement: list[torch.Tensor] = []
        plan_states: list[torch.Tensor] = []
        initial_future_states: list[torch.Tensor] = []
        masks: dict[str, list[torch.Tensor]] = {name: [] for name in BUFFER_MASK_NAMES}
        term_rows: list[dict[str, torch.Tensor]] = []
        generated_valid_frames: list[torch.Tensor] = []
        executed_action_masks: list[torch.Tensor] = []
        previous_buffer = previous_current = None
        previous_memory: torch.Tensor | None = initial_memory
        for response in range(steps):
            target, target_valid, target_actions, target_action_valid = self._target_plan(batch, response)
            step_current = current
            out = self.plan_step(
                history, history_valid, current, current_valid, ego_mask,
                batch["map_polylines"], batch["map_polyline_valid"], batch["lane_graph_edges"], behavior,
                previous_buffer=previous_buffer, previous_current=previous_current, previous_memory=previous_memory,
                start_anchor_actions=anchor_actions if response == 0 else None,
                start_mode=start_mode and response == 0,
            )
            plan = out["background_future_actions"]
            execute_count = min(
                self.cfg.execute_frames,
                total_frames - response * self.cfg.execute_frames,
            )
            generated, response_frames, response_valid = current.clone(), [], []
            for frame in range(execute_count):
                physical = current.new_zeros((current.shape[0], current.shape[1], 2))
                physical[:, 1:] = plan[:, frame]
                generated = self.dynamics.step(generated, physical, current_valid, self.cfg.simulation_dt_s)
                target_frame = 25 + response * self.cfg.execute_frames + frame
                generated = torch.where(ego_mask[..., None], states[:, target_frame], generated)
                current_valid = torch.where(ego_mask, valid[:, target_frame], current_valid)
                response_frames.append(generated)
                response_valid.append(current_valid)
            predicted = torch.stack(response_frames, dim=1)
            execute_valid, full_valid = target_valid[:, :execute_count, 1:], target_valid[:, :, 1:]
            position = _masked_mean((predicted[:, :, 1:, :2] - target[:, :execute_count, 1:, :2]).abs().mean(dim=-1), execute_valid)
            velocity = _masked_mean((predicted[:, :, 1:, 2:4] - target[:, :execute_count, 1:, 2:4]).abs().mean(dim=-1), execute_valid)
            action_loss = _masked_mean((plan[:, :execute_count] - target_actions[:, :execute_count]).abs().mean(dim=-1), target_action_valid[:, :execute_count])
            full_position = _masked_mean((out["refined_background_future_states"][..., :2] - target[:, :, 1:, :2]).abs().mean(dim=-1), full_valid)
            initial_position = _masked_mean((out["initial_background_future_states"][..., :2] - target[:, :, 1:, :2]).abs().mean(dim=-1), full_valid)
            full_action = _masked_mean((plan - target_actions).abs().mean(dim=-1), target_action_valid)
            overlap = plan.new_zeros(()) if previous_buffer is None else (previous_buffer[:, self.cfg.execute_frames:] - plan[:, : -self.cfg.execute_frames]).abs().mean()
            term_rows.append({
                "position": position, "velocity": velocity, "action": action_loss, "plan_position": full_position,
                "initial_plan_position": initial_position, "plan_action": full_action, "overlap": overlap,
                "interaction": self._interaction_loss(predicted, target[:, :execute_count], execute_valid),
                "physical": self._physical_loss(predicted, execute_valid),
                "refinable_fraction": out["background_future_action_masks"]["refinable"].float().mean(),
            })
            predicted_frames.extend(response_frames)
            generated_valid_frames.extend(response_valid)
            plans.append(plan); pre_refinement.append(out["background_future_actions_before_refinement"])
            plan_states.append(out["refined_background_future_states"]); initial_future_states.append(out["initial_background_future_states"])
            for name, value in out["background_future_action_masks"].items():
                masks[name].append(value)
            executed_mask = out["background_future_action_masks"]["valid"][:, : self.cfg.execute_frames].clone()
            if execute_count < self.cfg.execute_frames:
                executed_mask[:, execute_count:] = False
            executed_action_masks.append(executed_mask)
            previous_buffer, previous_current, previous_memory = plan, step_current, out["scene_memory"]
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
            "background_future_actions": torch.stack(plans, dim=1),
            "background_future_actions_before_refinement": torch.stack(pre_refinement, dim=1),
            "refined_background_future_states": torch.stack(plan_states, dim=1),
            "initial_background_future_states": torch.stack(initial_future_states, dim=1),
            "background_future_action_masks": self._stack_masks(masks),
            "executed_background_action_masks": torch.stack(executed_action_masks, dim=1),
            "behavior_latent": behavior, "behavior_terms": behavior_terms, "loss_terms": term_rows,
            "start_summary": start_summary, "start_mode": start_mode,
            "start_reconstruction_frames": min(int(self.cfg.start_reconstruction_frames), total_frames),
            "roll_frames": max(0, total_frames - int(self.cfg.start_reconstruction_frames)),
            "total_frames": total_frames,
        }

    def rollout_reconstruction(
        self,
        batch: dict[str, torch.Tensor],
        *,
        response_steps: int | None = None,
        deterministic: bool = True,
        use_posterior: bool = False,
        tbptt_steps: int = 0,
        start_mode: bool = False,
        behavior_standard_normal: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Logged-state reconstruction; future ego states never enter planning."""
        return self._rollout(
            batch, response_steps=response_steps, deterministic=deterministic, use_posterior=use_posterior,
            tbptt_steps=tbptt_steps, start_mode=start_mode,
            behavior_standard_normal=behavior_standard_normal,
        )

    def _objective(self, rollout: dict[str, Any]) -> dict[str, torch.Tensor]:
        terms = {key: torch.stack([value[key] for value in rollout["loss_terms"]]).mean() for key in rollout["loss_terms"][0]}
        behavior = rollout["behavior_terms"]
        loss = (
            self.cfg.position_weight * terms["position"] + self.cfg.velocity_weight * terms["velocity"] + self.cfg.action_weight * terms["action"]
            + self.cfg.plan_position_weight * terms["plan_position"] + self.cfg.plan_action_weight * terms["plan_action"]
            + self.cfg.refinement_weight * functional.relu(terms["plan_position"] - terms["initial_plan_position"] + 0.01)
            + self.cfg.overlap_weight * terms["overlap"]
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
            tbptt_steps=tbptt_steps, start_mode=start_mode,
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
            "flow_interface": {
                "input_dim": 76, "layout": "ego[vx,vy,ax,ay]+background_relative[6,6]+B0[6,6]",
                "scene_tensor_shape": [7, 6], "b0_lifecycle": "START-only: initializes latent, scene memory, and first background future-action sequence",
                "start_encoder": "C0 plus map only; no synthetic history", "ego_condition": "observed ego state/history only; no ADS action input",
                "flow_schema_sha256": self.flow_schema_sha256,
            },
        }
