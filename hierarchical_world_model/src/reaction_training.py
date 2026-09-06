"""Training and rollout utilities for the calibrated reaction policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.nn import functional

from diffusion.src.data import ANCHOR_INDEX
from world_model.src.core.utils import ensure_dir, save_json

from .data import ego_controls
from .highway import HighwayEnvClosedLoopWorld
from .influence_graph import dynamic_candidate_scene_mask
from .randomness import WorldExogenousState
from .reaction_controller import (
    CalibratedResidualReactionController, IDMResidualReactionController,
    RLResidualReactionController, ReactionController,
)
from .reaction_evidence import (
    EVALUATION_FRAMES, FEATURE_NAMES, PRE_EVENT_FRAMES,
    ReactionEventReference, energy_score, event_window,
)
from .rule_models import RuleModelBundle


EpisodeKind = Literal["event", "non_event", "synthetic"]


@dataclass(frozen=True)
class PolicyTrainingConfig:
    rollout_steps: int = 149
    episodes_per_rollout: int = 64
    event_episodes: int = 32
    non_event_episodes: int = 16
    synthetic_episodes: int = 16
    pretrain_epochs: int = 3
    objective_check_events: int = 4
    objective_check_futures: int = 8
    epochs_per_update: int = 4
    minibatch_size: int = 2048
    updates: int = 100
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 1.0
    physical_safety_weight: float = 2.0
    physical_jerk_weight: float = 0.1
    invalid_penalty: float = 10.0
    jerk_limit_mps3: float = 12.0
    safety_ttc_s: float = 2.0
    reaction_min_frames: int = 5
    reaction_max_frames: int = 149
    reaction_recovery_frames: int = 15
    reaction_release_ttc_s: float = 4.0
    influence_radius_m: float = 50.0
    influence_secondary_radius_m: float = 35.0
    influence_prediction_horizon_s: float = 1.5
    influence_stable_release_frames: int = 13
    validation_interval_updates: int = 10
    validation_events: int = 32
    validation_futures: int = 32
    validation_patience: int = 4
    validation_minimum_improvement: float = 0.002
    longitudinal_reference_offset_weight: float = 1.0
    lateral_reference_offset_weight: float = 0.0
    seed: int = 20260906

    def __post_init__(self) -> None:
        total = self.event_episodes + self.non_event_episodes + self.synthetic_episodes
        if total != self.episodes_per_rollout:
            raise ValueError("event, non-event and synthetic counts must equal episodes_per_rollout")


@dataclass(frozen=True)
class ReactionEpisode:
    row_index: int
    kind: EpisodeKind
    event_index: int = -1


@dataclass(frozen=True)
class ReactionRollout:
    states: np.ndarray
    background_actions: np.ndarray
    base_background_actions: np.ndarray
    ego_actions: np.ndarray
    controller_diagnostics: dict[str, np.ndarray]
    collision: np.ndarray
    crashed: np.ndarray


def _world(
    model, controller: ReactionController | str | None, device: torch.device,
    config: PolicyTrainingConfig,
) -> HighwayEnvClosedLoopWorld:
    return HighwayEnvClosedLoopWorld(
        model, device=device, controller=controller,
        reaction_min_frames=config.reaction_min_frames,
        reaction_max_frames=config.reaction_max_frames,
        reaction_recovery_frames=config.reaction_recovery_frames,
        reaction_safety_ttc_s=config.safety_ttc_s,
        reaction_release_ttc_s=config.reaction_release_ttc_s,
        influence_radius_m=config.influence_radius_m,
        influence_secondary_radius_m=config.influence_secondary_radius_m,
        influence_prediction_horizon_s=config.influence_prediction_horizon_s,
        influence_stable_release_frames=config.influence_stable_release_frames,
        reference_rebase_weights=(
            config.longitudinal_reference_offset_weight,
            config.lateral_reference_offset_weight,
        ),
    )


@torch.no_grad()
def reaction_controller_rollout(
    model, *, states: np.ndarray, valid: np.ndarray, soft_plans: np.ndarray,
    maps: np.ndarray, map_valid: np.ndarray, controller: ReactionController | str | None,
    device: torch.device, motion_seed: int, intervention: str | None = None,
    dose: float = 0.0, intervention_start: int = 25,
    intervention_duration_frames: int = 25, config: PolicyTrainingConfig | None = None,
    deterministic_response: bool = True,
    ego_acceleration_offset: np.ndarray | None = None,
) -> ReactionRollout:
    """Run the same label-free world for factual or synthetic ego actions."""
    config = config or PolicyTrainingConfig()
    values, present = np.asarray(states, np.float32), np.asarray(valid, bool)
    logged_ego = ego_controls(values[:, ANCHOR_INDEX:173, 0], values[:, ANCHOR_INDEX + 1:174, 0], 0.04)
    prior_ego = ego_controls(values[:, :ANCHOR_INDEX, 0], values[:, 1:ANCHOR_INDEX + 1, 0], 0.04)
    exogenous = WorldExogenousState.sample(
        len(values), seed=motion_seed, response_steps=149,
        scene_dim=model.cfg.scene_latent_dim, agent_dim=model.cfg.agent_latent_dim,
    )
    world = _world(model, controller, device, config)
    world.reset(
        torch.from_numpy(values[:, ANCHOR_INDEX]), torch.from_numpy(present[:, ANCHOR_INDEX]),
        torch.from_numpy(np.asarray(soft_plans, np.float32)), torch.from_numpy(np.asarray(maps, np.float32)),
        torch.from_numpy(np.asarray(map_valid, bool)), exogenous_state=exogenous,
        initial_history=torch.from_numpy(values[:, :ANCHOR_INDEX + 1]),
        initial_history_valid=torch.from_numpy(present[:, :ANCHOR_INDEX + 1]),
        committed_ego_controls=torch.from_numpy(prior_ego), deterministic_response=deterministic_response,
    )
    names = (
        "agent_state_frames", "background_actions", "base_background_actions", "ego_actions",
        "controller_alpha", "controller_delta_ax", "controller_active", "controller_phase",
        "controller_rule_action_ax", "collision", "crashed", "influence_authority",
        "influence_role", "influence_parent", "influence_direct",
        "influence_secondary", "influence_predicted_ttc_s",
        "influence_predicted_min_gap_m", "controller_desired_action_ax",
    )
    recorded = {name: [] for name in names}
    for step in range(149):
        ego = torch.from_numpy(logged_ego[:, step]).to(device)
        if intervention == "brake" and intervention_start <= step < intervention_start + intervention_duration_frames:
            ego[:, 0] = (ego[:, 0] - dose).clamp_min(-8.0)
        if ego_acceleration_offset is not None:
            ego[:, 0] = (ego[:, 0] + float(ego_acceleration_offset[step])).clamp(-8.0, 4.0)
        transition = world.advance_response(ego)
        for name in names:
            value = transition[name]
            if value is None:
                value = torch.zeros((len(values), 6), device=device)
            recorded[name].append(value.detach().cpu())
    return ReactionRollout(
        states=torch.cat(recorded["agent_state_frames"], 1).numpy(),
        background_actions=torch.cat(recorded["background_actions"], 1).numpy(),
        base_background_actions=torch.cat(recorded["base_background_actions"], 1).numpy(),
        ego_actions=torch.cat(recorded["ego_actions"], 1).numpy(),
        controller_diagnostics={
            "alpha": torch.stack(recorded["controller_alpha"], 1).numpy(),
            "delta_ax": torch.stack(recorded["controller_delta_ax"], 1).numpy(),
            "active": torch.stack(recorded["controller_active"], 1).numpy(),
            "phase": torch.stack(recorded["controller_phase"], 1).numpy(),
            "rule_action_ax": torch.stack(recorded["controller_rule_action_ax"], 1).numpy(),
            "influence_authority": torch.stack(recorded["influence_authority"], 1).numpy(),
            "influence_role": torch.stack(recorded["influence_role"], 1).numpy(),
            "influence_parent": torch.stack(recorded["influence_parent"], 1).numpy(),
            "influence_direct": torch.stack(recorded["influence_direct"], 1).numpy(),
            "influence_secondary": torch.stack(recorded["influence_secondary"], 1).numpy(),
            "influence_predicted_ttc_s": torch.stack(recorded["influence_predicted_ttc_s"], 1).numpy(),
            "influence_predicted_min_gap_m": torch.stack(recorded["influence_predicted_min_gap_m"], 1).numpy(),
            "desired_action_ax": torch.stack(recorded["controller_desired_action_ax"], 1).numpy(),
        },
        collision=torch.stack(recorded["collision"], 1).numpy(),
        crashed=torch.stack(recorded["crashed"], 1).numpy(),
    )


class ReactionTrainingEnvironment:
    """Three-stream HighwayEnv training batches with explicit target gates."""

    def __init__(
        self, model, *, arrays: dict[str, np.ndarray], soft_plans: np.ndarray,
        controller: ReactionController, device: torch.device,
        config: PolicyTrainingConfig, event_reference: ReactionEventReference | None = None,
        deterministic_response: bool = False,
    ) -> None:
        self.model = model
        self.arrays = arrays
        self.states = np.asarray(arrays["agent_states"], np.float32)
        self.valid = np.asarray(arrays["agent_valid"], bool)
        self.maps = np.asarray(arrays["map_polylines"], np.float32)
        self.map_valid = np.asarray(arrays["map_polyline_valid"], bool)
        self.rows = np.asarray(arrays.get("row_index", np.arange(len(self.states))), np.int64)
        self.soft_plans = np.asarray(soft_plans, np.float32)
        self.controller = controller
        self.device = device
        self.config = config
        self.reference = event_reference
        self.deterministic_response = deterministic_response
        self.rng = np.random.default_rng(config.seed)
        self.row_lookup = {int(row): index for index, row in enumerate(self.rows)}
        self.eligible = np.flatnonzero(dynamic_candidate_scene_mask(
            self.states, self.valid, radius_m=config.influence_radius_m,
            prediction_horizon_s=config.influence_prediction_horizon_s,
        ))
        if not len(self.eligible):
            raise RuntimeError("training split has no autonomous response candidates")
        self._order = np.empty(0, np.int64)
        self._cursor = 0
        self._prepare_event_pools()
        self.world: HighwayEnvClosedLoopWorld | None = None
        self.specs: list[ReactionEpisode] = []
        self.logged_ego: torch.Tensor | None = None
        self.logged_background: torch.Tensor | None = None
        self.onset: np.ndarray | None = None
        self.stop: np.ndarray | None = None
        self.dose: np.ndarray | None = None
        self.previous_actions: torch.Tensor | None = None
        self.terminated: torch.Tensor | None = None
        self.step_index = 0

    def _prepare_event_pools(self) -> None:
        if self.reference is None:
            self.event_pool = np.empty(0, np.int64)
            self.non_event_pool = self.eligible
            self.non_event_features = np.empty((len(self.eligible), 3), np.float32)
            self.scales = np.ones(len(FEATURE_NAMES), np.float32)
            return
        events = self.reference.events
        self.event_pool = np.asarray([
            index for index in events.indices(self.reference.supported_cells)
            if events.leader_slot[index] == 0 and int(events.row_index[index]) in self.row_lookup
        ], np.int64)
        event_rows = {self.row_lookup[int(events.row_index[index])] for index in self.event_pool}
        candidates = np.asarray([index for index in self.eligible if index not in event_rows], np.int64)
        anchor = self.states[candidates, ANCHOR_INDEX]
        anchor_valid = self.valid[candidates, ANCHOR_INDEX]
        gap = anchor[:, 0, 0] - anchor[:, 2, 0] - 4.8
        closing = anchor[:, 2, 2] - anchor[:, 0, 2]
        follower_acceleration = (
            self.states[candidates, ANCHOR_INDEX, 2, 2]
            - self.states[candidates, ANCHOR_INDEX - 1, 2, 2]
        ) / 0.04
        following = (
            anchor_valid[:, 0] & anchor_valid[:, 2]
            & (np.abs(anchor[:, 0, 1] - anchor[:, 2, 1]) < 1.8)
            & (gap > 0.0)
        )
        self.non_event_pool = candidates[following]
        self.non_event_features = np.stack(
            (gap[following], closing[following], follower_acceleration[following]), axis=-1,
        )
        supported = event_window(events)[events.indices(self.reference.supported_cells)]
        if len(supported):
            flat = supported.reshape(-1, len(FEATURE_NAMES))
            self.scales = np.maximum(np.quantile(flat, 0.75, axis=0) - np.quantile(flat, 0.25, axis=0), 1.0e-3)
        else:
            self.scales = np.ones(len(FEATURE_NAMES), np.float32)

    def _matched_non_events(self, event_indices: np.ndarray, count: int) -> np.ndarray:
        if count == 0:
            return np.empty(0, np.int64)
        if not len(self.non_event_pool):
            raise RuntimeError("training split has no matched ordinary-following non-events")
        event_conditions = self.reference.events.initial_conditions[event_indices][:, (1, 2, 4)]
        targets = event_conditions[self.rng.integers(len(event_conditions), size=count)]
        scale = np.maximum(np.quantile(event_conditions, 0.75, axis=0) - np.quantile(event_conditions, 0.25, axis=0), 1.0e-3)
        selected = []
        for target in targets:
            distance = np.square((self.non_event_features - target) / scale).sum(-1)
            nearest = np.argsort(distance)[:min(8, len(distance))]
            selected.append(int(self.non_event_pool[self.rng.choice(nearest)]))
        return np.asarray(selected, np.int64)

    def _sample_rows(self, count: int, pool: np.ndarray | None = None) -> np.ndarray:
        if count == 0:
            return np.empty(0, np.int64)
        if pool is not None:
            if not len(pool):
                raise RuntimeError("requested reaction training stream has no eligible rows")
            return self.rng.choice(pool, size=count, replace=len(pool) < count)
        selected: list[int] = []
        while len(selected) < count:
            if self._cursor >= len(self._order):
                self._order = self.eligible.copy()
                self.rng.shuffle(self._order)
                self._cursor = 0
            take = min(count - len(selected), len(self._order) - self._cursor)
            selected.extend(self._order[self._cursor:self._cursor + take].tolist())
            self._cursor += take
        return np.asarray(selected, np.int64)

    def sample_episodes(self, *, human_targets: bool = True) -> list[ReactionEpisode]:
        if not human_targets or self.reference is None:
            return [ReactionEpisode(int(row), "synthetic") for row in self._sample_rows(self.config.episodes_per_rollout)]
        if not len(self.event_pool):
            raise RuntimeError("no replayable supported events with ego as leader")
        event_indices = self.rng.choice(
            self.event_pool, size=self.config.event_episodes,
            replace=len(self.event_pool) < self.config.event_episodes,
        )
        episodes = [
            ReactionEpisode(self.row_lookup[int(self.reference.events.row_index[index])], "event", int(index))
            for index in event_indices
        ]
        episodes.extend(
            ReactionEpisode(int(row), "non_event")
            for row in self._matched_non_events(event_indices, self.config.non_event_episodes)
        )
        episodes.extend(
            ReactionEpisode(int(row), "synthetic")
            for row in self._sample_rows(self.config.synthetic_episodes)
        )
        return episodes

    def reset(self, episodes: list[ReactionEpisode]) -> dict[str, torch.Tensor]:
        self.specs = episodes
        choice = np.asarray([episode.row_index for episode in episodes], np.int64)
        values, present = self.states[choice], self.valid[choice]
        self.logged_ego = torch.from_numpy(ego_controls(
            values[:, ANCHOR_INDEX:173, 0], values[:, ANCHOR_INDEX + 1:174, 0], 0.04,
        )).to(self.device)
        self.logged_background = torch.from_numpy(
            (values[:, ANCHOR_INDEX + 1:174, 1:, 2] - values[:, ANCHOR_INDEX:173, 1:, 2]) / 0.04
        ).to(self.device)
        prior_ego = ego_controls(values[:, :ANCHOR_INDEX, 0], values[:, 1:ANCHOR_INDEX + 1, 0], 0.04)
        self.onset = self.rng.integers(8, 49, len(episodes))
        for index, episode in enumerate(episodes):
            if episode.kind == "event":
                self.onset[index] = int(self.reference.events.local_onset_frame[episode.event_index]) - ANCHOR_INDEX
        self.stop = self.onset + self.rng.integers(5, 26, len(episodes))
        self.dose = self.rng.choice(np.asarray((-2.0, -4.0, -6.0, -8.0), np.float32), len(episodes))
        exogenous = WorldExogenousState.sample(
            len(episodes), seed=int(self.rng.integers(2**31 - 1)), response_steps=149,
            scene_dim=self.model.cfg.scene_latent_dim, agent_dim=self.model.cfg.agent_latent_dim,
        )
        self.world = _world(self.model, self.controller, self.device, self.config)
        self.world.reset(
            torch.from_numpy(values[:, ANCHOR_INDEX]), torch.from_numpy(present[:, ANCHOR_INDEX]),
            torch.from_numpy(self.soft_plans[choice]), torch.from_numpy(self.maps[choice]),
            torch.from_numpy(self.map_valid[choice]), exogenous_state=exogenous,
            initial_history=torch.from_numpy(values[:, :ANCHOR_INDEX + 1]),
            initial_history_valid=torch.from_numpy(present[:, :ANCHOR_INDEX + 1]),
            committed_ego_controls=torch.from_numpy(prior_ego),
            deterministic_response=self.deterministic_response,
        )
        self.step_index = 0
        self.previous_actions = None
        self.terminated = torch.zeros(len(episodes), dtype=torch.bool, device=self.device)
        return self.world.observe()

    @torch.no_grad()
    def step(self):
        if self.world is None or self.logged_ego is None or self.logged_background is None:
            raise RuntimeError("reset the reaction environment before stepping")
        assert self.onset is not None and self.stop is not None and self.dose is not None
        assert self.terminated is not None
        alive = ~self.terminated
        ego = self.logged_ego[:, self.step_index].clone()
        synthetic = np.asarray([episode.kind == "synthetic" for episode in self.specs])
        intervene = synthetic & (self.step_index >= self.onset) & (self.step_index < self.stop)
        if intervene.any():
            mask = torch.from_numpy(intervene).to(self.device)
            ego[mask, 0] = (ego[mask, 0] + torch.from_numpy(self.dose[intervene]).to(self.device)).clamp_min(-8.0)
        transition = self.world.advance_response(ego)
        final = transition["background_actions"][:, 0, :, 0]
        base = transition["base_background_actions"][:, 0, :, 0]
        active = transition["controller_active"].bool()
        phase = transition["controller_phase"]
        state = transition["agent_state_frames"][:, 0]
        ttc = transition["influence_predicted_ttc_s"]

        target_action = self.logged_background[:, self.step_index].clone()
        human_gate = torch.zeros_like(active)
        human_weight = torch.zeros_like(final)
        reward = torch.zeros_like(final)
        selected_features = torch.zeros((len(self.specs), len(FEATURE_NAMES)), device=self.device)
        for index, episode in enumerate(self.specs):
            if episode.kind == "event":
                event = self.reference.events
                offset = self.step_index - self.onset[index]
                if 0 <= offset < EVALUATION_FRAMES:
                    follower = int(event.follower_slot[episode.event_index]) - 1
                    target = torch.from_numpy(event.trajectory[episode.event_index, PRE_EVENT_FRAMES + offset]).to(self.device)
                    target_action[index, follower] = target[0]
                    human_gate[index, follower] = active[index, follower]
                    human_weight[index, follower] = 1.0 / EVALUATION_FRAMES
                    selected_features[index] = self._physical_features(state, final, index, int(event.leader_slot[episode.event_index]), follower + 1)
                    scale = torch.from_numpy(self.scales).to(self.device)
                    reward[index, follower] = (
                        -(selected_features[index] - target).abs().div(scale).mean()
                        / EVALUATION_FRAMES
                    )
            elif episode.kind == "non_event":
                human_gate[index] = active[index]
                human_weight[index] = active[index].float() / self.config.rollout_steps
                reward[index] = (
                    -(final[index] - target_action[index]).abs()
                    / max(float(self.scales[0]), 1.0e-3)
                    / self.config.rollout_steps
                )

        physical_gate = torch.from_numpy(synthetic).to(self.device)[:, None] & active
        safety = -(self.config.safety_ttc_s - ttc).clamp_min(0.0) / self.config.safety_ttc_s
        reward += self.config.physical_safety_weight * safety * physical_gate
        if self.previous_actions is not None:
            jerk = (final - self.previous_actions).abs() / 0.04
            reward -= self.config.physical_jerk_weight * (jerk / self.config.jerk_limit_mps3 - 1.0).clamp_min(0.0) * physical_gate
        collision = (transition["crashed"][:, 1:] & active).any(1)
        reward -= collision[:, None].float() * self.config.invalid_penalty
        reward *= alive[:, None]
        self.previous_actions = final.detach()
        self.terminated |= collision
        self.step_index += 1
        done = torch.full_like(reward, self.step_index >= min(self.config.rollout_steps, 149), dtype=torch.bool)
        done |= self.terminated[:, None]
        info = {
            "features": transition["controller_features"],
            "raw_action": transition["controller_raw_action"],
            "log_prob": transition["controller_log_prob"],
            "value": transition["controller_value"],
            "active": active & alive[:, None],
            "authority": transition["influence_authority"],
            "base_action": base,
            "final_action": final,
            "target_action": target_action,
            "human_target_gate": human_gate & alive[:, None],
            "human_target_weight": human_weight * alive[:, None],
            "physical_target_gate": physical_gate & alive[:, None],
            "selected_features": selected_features,
            "collision": collision,
            "phase": phase,
        }
        return self.world.observe(), reward, done, info

    def _physical_features(
        self, state: torch.Tensor, action: torch.Tensor,
        batch: int, leader: int, follower: int,
    ) -> torch.Tensor:
        leader_state = state[batch, leader]
        follower_state = state[batch, follower]
        follower_action = action[batch, follower - 1]
        previous = follower_action if self.previous_actions is None else self.previous_actions[batch, follower - 1]
        gap = leader_state[0] - follower_state[0] - 4.8
        closing = follower_state[2] - leader_state[2]
        ttc = torch.where(closing > 1.0e-4, gap / closing.clamp_min(1.0e-4), gap.new_tensor(10.0)).clamp(0.0, 10.0)
        return torch.stack((
            follower_action, (follower_action - previous).abs() / 0.04,
            follower_state[2], gap, closing, ttc,
        ))

    def sampler_state(self) -> dict[str, Any]:
        return {"rng": self.rng.bit_generator.state, "order": self._order, "cursor": self._cursor}

    def restore_sampler_state(self, state: dict[str, Any]) -> None:
        self.rng.bit_generator.state = state["rng"]
        self._order = np.asarray(state["order"], np.int64)
        self._cursor = int(state["cursor"])


def _gae(reward: torch.Tensor, value: torch.Tensor, done: torch.Tensor, gamma: float, lam: float):
    advantage = torch.zeros_like(reward)
    tail = torch.zeros_like(reward[0])
    for index in range(len(reward) - 1, -1, -1):
        next_value = torch.zeros_like(tail) if index == len(reward) - 1 else value[index + 1]
        live = (~done[index]).float()
        tail = reward[index] + gamma * next_value * live - value[index] + gamma * lam * live * tail
        advantage[index] = tail
    return advantage, advantage + value


def pretrain_final_action(
    controller: CalibratedResidualReactionController,
    environment: ReactionTrainingEnvironment,
    config: PolicyTrainingConfig,
) -> list[float]:
    """Fit the mapped final desired action on events and matched non-events."""
    optimizer = torch.optim.Adam(controller.parameters(), lr=config.learning_rate)
    history: list[float] = []
    for _ in range(config.pretrain_epochs):
        episodes = environment.sample_episodes(human_targets=True)
        environment.reset(episodes)
        losses: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        for _ in range(config.rollout_steps):
            _, _, _, info = environment.step()
            gate = info["human_target_gate"]
            if not gate.any():
                continue
            features = info["features"].detach()
            distribution, _ = controller.distribution_and_value(features)
            desired, _ = controller.mapped_action(
                info["base_action"].detach(), info["authority"].detach(), gate,
                distribution.mean, -8.0, 4.0,
            )
            losses.append(functional.smooth_l1_loss(
                desired[gate], info["target_action"][gate], reduction="none",
            ))
            weights.append(info["human_target_weight"][gate])
        if not losses:
            raise RuntimeError("final-action pretraining produced no human-supervised actions")
        weight = torch.cat(weights)
        loss = (torch.cat(losses) * weight).sum() / weight.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), config.max_grad_norm)
        optimizer.step()
        history.append(float(loss.detach()))
    return history


@torch.no_grad()
def validation_energy_score(
    model, controller: ReactionController, *, arrays: dict[str, np.ndarray],
    soft_plans: np.ndarray, reference: ReactionEventReference,
    device: torch.device, config: PolicyTrainingConfig,
) -> float:
    """Generate equally weighted stochastic futures for held-out events."""
    available = [
        index for index in reference.events.indices(reference.supported_cells)
        if reference.events.leader_slot[index] == 0
        and int(reference.events.row_index[index]) in set(np.asarray(arrays["row_index"], np.int64).tolist())
    ][:config.validation_events]
    if not available:
        raise RuntimeError("validation has no replayable supported events")
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed + 1)
        environment = ReactionTrainingEnvironment(
            model, arrays=arrays, soft_plans=soft_plans, controller=controller,
            device=device, config=replace(config, seed=config.seed + 1),
            event_reference=reference,
        )
        scores = []
        for event_index in available:
            row = environment.row_lookup[int(reference.events.row_index[event_index])]
            episodes = [ReactionEpisode(row, "event", event_index) for _ in range(config.validation_futures)]
            environment.reset(episodes)
            futures = np.zeros((config.validation_futures, EVALUATION_FRAMES, 2), np.float32)
            onset = int(reference.events.local_onset_frame[event_index]) - ANCHOR_INDEX
            for step in range(onset + EVALUATION_FRAMES):
                _, _, _, info = environment.step()
                if step >= onset:
                    follower = int(reference.events.follower_slot[event_index]) - 1
                    futures[:, step - onset, 0] = info["final_action"][:, follower].cpu().numpy()
                    if step > onset:
                        futures[:, step - onset, 1] = np.abs(futures[:, step - onset, 0] - futures[:, step - onset - 1, 0]) / 0.04
            observed = event_window(reference.events)[event_index, :, :2]
            scores.append(energy_score(futures, observed))
    return float(np.mean(scores))


def _controller(mode: str, rule_model: RuleModelBundle | None, device: torch.device) -> ReactionController:
    if mode == "rl_residual":
        return RLResidualReactionController().to(device)
    if rule_model is None:
        raise ValueError(f"{mode} requires the rule model")
    if mode == "rl_residual_idm":
        return IDMResidualReactionController(rule_model).to(device)
    if mode == "calibrated_residual":
        return CalibratedResidualReactionController(rule_model).to(device)
    raise ValueError(f"unsupported training mode: {mode}")


def initialise_calibrated_actor_features(
    controller: CalibratedResidualReactionController,
    source_state_dict: dict[str, torch.Tensor] | None,
) -> tuple[str, ...]:
    """Copy only A2 actor hidden layers with identical feature semantics.

    A2's output head, variance and critic encode its IDM-constrained action
    mapping.  They must never initialise the signed calibrated-residual
    mapping, whose zero-mean head is the explicit zero-correction baseline.
    """
    if source_state_dict is None:
        return ()
    names = ("actor.0.weight", "actor.0.bias", "actor.2.weight", "actor.2.bias")
    target = controller.state_dict()
    for name in names:
        if name not in source_state_dict or source_state_dict[name].shape != target[name].shape:
            raise ValueError(f"A2 actor feature layer {name} is incompatible with calibrated residual")
    with torch.no_grad():
        for name in names:
            target[name].copy_(source_state_dict[name].to(target[name]))
    return names


def _policy_payload(
    controller: ReactionController,
    config: PolicyTrainingConfig,
    artifact_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_name": "reaction_policy", "schema_version": 1,
        "controller_mode": controller.mode, "config": config.__dict__,
        "state_dict": controller.state_dict(), "artifact_metadata": artifact_metadata or {},
    }


def train_reaction_policy(
    model, *, train_arrays: dict[str, np.ndarray], train_plans: np.ndarray,
    output_dir: str | Path, config: PolicyTrainingConfig, device: torch.device,
    controller_mode: str = "calibrated_residual", rule_model: RuleModelBundle | None = None,
    initial_actor_hidden_state_dict: dict[str, torch.Tensor] | None = None,
    train_events: ReactionEventReference | None = None,
    validation_arrays: dict[str, np.ndarray] | None = None,
    validation_plans: np.ndarray | None = None,
    validation_events: ReactionEventReference | None = None,
    resume: bool = True, artifact_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    controller = _controller(controller_mode, rule_model, device)
    copied_layers: tuple[str, ...] = ()
    if isinstance(controller, CalibratedResidualReactionController):
        copied_layers = initialise_calibrated_actor_features(
            controller, initial_actor_hidden_state_dict,
        )
    environment = ReactionTrainingEnvironment(
        model, arrays=train_arrays, soft_plans=train_plans, controller=controller,
        device=device, config=config, event_reference=train_events,
    )
    optimizer = torch.optim.Adam(controller.parameters(), lr=config.learning_rate)
    target = ensure_dir(output_dir)
    progress_path = target / "training_progress.pt"
    history: list[dict[str, float]] = []
    start = 0
    best_state = None
    best_score = float("inf")
    stale = 0
    pretraining: list[float] = []
    objective_check: dict[str, float] | None = None
    if resume and progress_path.exists():
        progress = torch.load(progress_path, map_location=device, weights_only=False)
        if progress.get("schema_name") == "reaction_policy_training" and progress.get("controller_mode") == controller_mode:
            controller.load_state_dict(progress["state_dict"])
            optimizer.load_state_dict(progress["optimizer_state"])
            environment.restore_sampler_state(progress["sampler_state"])
            torch.set_rng_state(progress["torch_rng_state"].cpu())
            if torch.cuda.is_available() and progress.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in progress["cuda_rng_state"]])
            history = list(progress["history"])
            start = int(progress["next_update"])
            best_state = progress.get("best_state")
            best_score = float(progress.get("best_score", best_score))
            stale = int(progress.get("stale", 0))
            pretraining = list(progress.get("pretraining", []))
            objective_check = progress.get("objective_check")
    elif isinstance(controller, CalibratedResidualReactionController) and train_events is not None:
        initial_checkpoint = target / "initial.pt"
        torch.save(_policy_payload(controller, config, artifact_metadata), initial_checkpoint)
        check_config = replace(
            config,
            validation_events=config.objective_check_events,
            validation_futures=config.objective_check_futures,
        )
        before = validation_energy_score(
            model, controller, arrays=train_arrays, soft_plans=train_plans,
            reference=train_events, device=device, config=check_config,
        )
        pretraining = pretrain_final_action(controller, environment, config)
        supervised_checkpoint = target / "supervised.pt"
        torch.save(_policy_payload(controller, config, artifact_metadata), supervised_checkpoint)
        after = validation_energy_score(
            model, controller, arrays=train_arrays, soft_plans=train_plans,
            reference=train_events, device=device, config=check_config,
        )
        objective_check = {"before_energy_score": before, "after_energy_score": after}
        if after >= before:
            raise RuntimeError(
                "fixed-scene objective check did not improve executed acceleration and jerk"
            )
    for update in range(start, config.updates):
        environment.reset(environment.sample_episodes(human_targets=train_events is not None))
        buffer = {name: [] for name in ("features", "raw_action", "log_prob", "value", "reward", "done", "active")}
        human_count = physical_count = 0
        for _ in range(config.rollout_steps):
            _, reward, done, info = environment.step()
            for name in buffer:
                value = reward if name == "reward" else done if name == "done" else info[name]
                buffer[name].append(value.detach())
            human_count += int(info["human_target_gate"].sum())
            physical_count += int(info["physical_target_gate"].sum())
        rewards = torch.stack(buffer["reward"])
        values = torch.stack(buffer["value"])
        dones = torch.stack(buffer["done"])
        advantages, returns = _gae(rewards, values, dones, config.gamma, config.gae_lambda)
        features = torch.stack(buffer["features"]).reshape(-1, buffer["features"][0].shape[-1])
        raw = torch.stack(buffer["raw_action"]).reshape(-1, 2)
        old_log_prob = torch.stack(buffer["log_prob"]).reshape(-1)
        active = torch.stack(buffer["active"]).reshape(-1).bool()
        if not active.any():
            raise RuntimeError("PPO batch contains no autonomous policy actions")
        features, raw, old_log_prob = features[active], raw[active], old_log_prob[active]
        advantages = advantages.reshape(-1)[active]
        returns = returns.reshape(-1)[active]
        advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1.0e-6)
        losses = []
        for _ in range(config.epochs_per_update):
            for batch in torch.randperm(len(features), device=device).split(config.minibatch_size):
                log_prob, entropy, estimate = controller.evaluate_raw_action(features[batch], raw[batch])
                ratio = (log_prob - old_log_prob[batch]).exp()
                clipped = ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio)
                policy_loss = -torch.minimum(ratio * advantages[batch], clipped * advantages[batch]).mean()
                value_loss = functional.mse_loss(estimate, returns[batch])
                loss = policy_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(controller.parameters(), config.max_grad_norm)
                optimizer.step()
                losses.append(float(loss.detach()))
        entry = {
            "update": float(update), "loss": float(np.mean(losses)),
            "reward": float(rewards.mean()), "human_target_actions": float(human_count),
            "physical_target_actions": float(physical_count),
        }
        should_validate = (
            validation_arrays is not None and validation_plans is not None and validation_events is not None
            and ((update + 1) % config.validation_interval_updates == 0 or update + 1 == config.updates)
        )
        if should_validate:
            controller.eval()
            score = validation_energy_score(
                model, controller, arrays=validation_arrays, soft_plans=validation_plans,
                reference=validation_events, device=device, config=config,
            )
            controller.train()
            entry["validation_energy_score"] = score
            if score < best_score - config.validation_minimum_improvement:
                best_score = score
                best_state = {name: value.detach().cpu().clone() for name, value in controller.state_dict().items()}
                stale = 0
            else:
                stale += 1
        history.append(entry)
        torch.save({
            "schema_name": "reaction_policy_training", "schema_version": 1,
            "controller_mode": controller_mode, "next_update": update + 1,
            "state_dict": controller.state_dict(), "optimizer_state": optimizer.state_dict(),
            "sampler_state": environment.sampler_state(), "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "history": history, "best_state": best_state, "best_score": best_score,
            "stale": stale, "pretraining": pretraining,
            "objective_check": objective_check,
        }, progress_path)
        if should_validate and stale >= config.validation_patience:
            break
    if best_state is not None:
        controller.load_state_dict(best_state)
    checkpoint = target / "reaction_policy.pt"
    torch.save(_policy_payload(controller, config, artifact_metadata), checkpoint)
    summary = {
        "checkpoint": str(checkpoint), "controller_mode": controller_mode,
        "pretraining_loss": pretraining, "history": history,
        "objective_check": objective_check,
        "best_validation_energy_score": None if best_state is None else best_score,
        "frozen_world_model": True,
        "a2_actor_hidden_layers_copied": list(copied_layers),
        "initial_checkpoint": str(target / "initial.pt"),
        "supervised_checkpoint": str(target / "supervised.pt"),
    }
    save_json(summary, target / "training_summary.json")
    return summary
