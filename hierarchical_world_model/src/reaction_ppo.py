"""Small native-PyTorch PPO loop for post-HiQR longitudinal reactions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as functional

from diffusion.src.data import ANCHOR_INDEX
from world_model.src.core.utils import ensure_dir, save_json

from .data import ego_controls
from .highway import HighwayEnvClosedLoopWorld
from .human_prior import HumanActionPrior
from .influence_graph import dynamic_candidate_scene_mask
from .reaction_controller import IDMResidualReactionController, RLResidualReactionController, ReactionController
from .randomness import WorldExogenousState
from .reaction_realism import FEATURE_NAMES, ReactionRealismReference, mloo_rewards, realism_metric
from .rule_models import RuleModelBundle


@dataclass(frozen=True)
class PPOConfig:
    rollout_steps: int = 64
    episodes_per_rollout: int = 16
    epochs_per_update: int = 4
    minibatch_size: int = 512
    updates: int = 100
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 1.0
    plan_coefficient: float = 0.20
    invalid_penalty: float = 2.0
    jerk_limit_mps3: float = 12.0
    safety_ttc_s: float = 2.0
    safety_coefficient: float = 0.25
    excess_brake_coefficient: float = 0.02
    response_floor: float = 0.85
    response_floor_coefficient: float = 3.0
    # Preserve the full observed ADS intervention dose in the causal reward.
    # The former hard cap at 4 m/s² made an -8 m/s² intervention look solved
    # after a mild brake, which allowed the GAIL regularizer to weaken the
    # very response that A3 is meant to preserve.
    response_target_max_mps2: float = 8.0
    jerk_coefficient: float = 0.10
    reaction_min_frames: int = 5
    reaction_max_frames: int = 75
    reaction_recovery_frames: int = 15
    reaction_release_ttc_s: float = 4.0
    influence_radius_m: float = 50.0
    influence_secondary_radius_m: float = 35.0
    influence_prediction_horizon_s: float = 1.5
    influence_stable_release_frames: int = 13
    validation_interval_updates: int = 10
    validation_scenes: int = 128
    validation_patience: int = 4
    validation_minimum_improvement: float = 0.002
    recovery_plan_coefficient: float = 0.50
    recovery_residual_coefficient: float = 0.50
    naturalness_weight: float = 0.0
    naturalness_kl_scale: float = 1.0
    naturalness_ttc_relax_s: float = 2.0
    naturalness_ttc_full_s: float = 4.0
    # A3/A4 use observed highD braking replays for human-prior and rollout
    # alignment.  Synthetic interventions stay deliberately out-of-support.
    supported_replay_fraction: float = 0.5
    support_minimum_events: int = 100
    realism_group_size: int = 8
    realism_window_frames: int = 25
    realism_advantage_weight: float = 0.10
    seed: int = 20260830


@dataclass(frozen=True)
class HighwayControllerRollout:
    """One HighwayEnv-backed causal rollout for PPO evaluation or playback."""

    states: np.ndarray
    background_actions: np.ndarray
    base_background_actions: np.ndarray
    ego_actions: np.ndarray
    controller_diagnostics: dict[str, np.ndarray]
    collision: np.ndarray
    crashed: np.ndarray


@dataclass(frozen=True)
class ReactionEpisodeSpec:
    """One PPO reset: synthetic intervention or observed supported replay."""

    row_index: int
    mode: str = "legacy"
    onset_step: int = -1
    child_slot: int = -1
    support_cell: int = -1


def rear_ttc_and_gap(states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal same-lane rear TTC and gap from physical post-step state.

    ``states`` is the state synchronized from HighwayEnv, not an internal
    model integration.  Infinite TTC denotes non-closing or non-same-lane.
    """
    ego, rear = states[:, 0], states[:, 2]
    gap = ego[:, 0] - rear[:, 0]
    closing = rear[:, 2] - ego[:, 2]
    same_lane = (ego[:, 1] - rear[:, 1]).abs() < 1.8
    valid = same_lane & (gap > 0.0) & (closing > 1.0e-4)
    ttc = torch.where(valid, gap / closing.clamp_min(1.0e-4), torch.full_like(gap, float("inf")))
    return ttc, gap


@torch.no_grad()
def highway_controller_rollout(
    model, *, states: np.ndarray, valid: np.ndarray, soft_plans: np.ndarray,
    maps: np.ndarray, map_valid: np.ndarray, controller: ReactionController | str,
    device: torch.device, motion_seed: int, intervention: str | None = None,
    dose: float = 0.0, deterministic_response: bool = True,
    intervention_start: int = 25, intervention_duration_frames: int = 25,
    reaction_min_frames: int = 5, reaction_max_frames: int = 75,
    reaction_recovery_frames: int = 15, reaction_safety_ttc_s: float = 2.0,
    reaction_release_ttc_s: float = 4.0,
    influence_radius_m: float = 50.0, influence_secondary_radius_m: float = 35.0,
    influence_prediction_horizon_s: float = 1.5, influence_stable_release_frames: int = 13,
) -> HighwayControllerRollout:
    """Advance HiQR actions exclusively through ``HighwayEnvClosedLoopWorld``.

    This is the authoritative PPO evaluation path: it applies the actual
    HighwayEnv vehicle geometry and collision state, then feeds its realized
    state back into HiQR at the next 25 Hz response boundary.

    Time contract: ``result.states[:, 0]`` is the post-step state produced by
    the first command after the highD anchor, and thus corresponds to logged
    frame ``ANCHOR_INDEX + 1``.  The anchor itself is not included.
    """
    values, present = np.asarray(states, np.float32), np.asarray(valid, bool)
    history = values[:, ANCHOR_INDEX - 24 : ANCHOR_INDEX + 1]
    history_valid = present[:, ANCHOR_INDEX - 24 : ANCHOR_INDEX + 1]
    logged_ego = ego_controls(values[:, ANCHOR_INDEX:173, 0], values[:, ANCHOR_INDEX + 1:174, 0], .04)
    historical_ego = ego_controls(values[:, :ANCHOR_INDEX, 0], values[:, 1:ANCHOR_INDEX + 1, 0], .04)
    exogenous = WorldExogenousState.sample(
        len(values), seed=int(motion_seed), response_steps=149,
        scene_dim=model.cfg.scene_latent_dim, agent_dim=model.cfg.agent_latent_dim,
    )
    world = HighwayEnvClosedLoopWorld(
        model, device=device, controller=controller,
        reaction_min_frames=reaction_min_frames, reaction_max_frames=reaction_max_frames,
        reaction_recovery_frames=reaction_recovery_frames, reaction_safety_ttc_s=reaction_safety_ttc_s,
        reaction_release_ttc_s=reaction_release_ttc_s,
        influence_radius_m=influence_radius_m,
        influence_secondary_radius_m=influence_secondary_radius_m,
        influence_prediction_horizon_s=influence_prediction_horizon_s,
        influence_stable_release_frames=influence_stable_release_frames,
    )
    world.reset(
        torch.from_numpy(values[:, ANCHOR_INDEX]), torch.from_numpy(present[:, ANCHOR_INDEX]),
        torch.from_numpy(np.asarray(soft_plans, np.float32)), torch.from_numpy(np.asarray(maps, np.float32)),
        torch.from_numpy(np.asarray(map_valid, bool)), exogenous_state=exogenous,
        initial_history=torch.from_numpy(history), initial_history_valid=torch.from_numpy(history_valid),
        committed_ego_controls=torch.from_numpy(historical_ego), deterministic_response=deterministic_response,
    )
    names = ("agent_state_frames", "background_actions", "base_background_actions", "ego_actions",
             "controller_alpha", "controller_delta_ax", "controller_active", "controller_phase",
             "controller_age_frames", "controller_rule_action_ax", "controller_natural_kl", "controller_desired_action_ax", "collision", "crashed")
    influence_names = ("influence_authority", "influence_role", "influence_parent", "influence_direct",
                       "influence_secondary", "influence_predicted_ttc_s", "influence_predicted_min_gap_m")
    names = names + influence_names
    recorded: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    for step in range(149):
        ego = torch.from_numpy(logged_ego[:, step]).to(device)
        applied = intervention == "brake" and int(intervention_start) <= step < int(intervention_start) + int(intervention_duration_frames)
        if applied:
            ego[:, 0] = (ego[:, 0] - float(dose)).clamp_min(-8.0)
        transition = world.advance_response(ego)
        # The current command has physically executed.  It may only affect
        # the following response boundary; authority then persists while the
        # *realized* rear-end risk remains and has a bounded recovery phase.
        world.register_executed_ego_intervention(applied)
        for name in names:
            value = transition[name]
            if value is None:
                shape = (len(values), 6)
                value = torch.zeros(shape, dtype=torch.bool if name == "controller_active" else torch.float32, device=device)
            recorded[name].append(value.detach().cpu())
    return HighwayControllerRollout(
        states=torch.cat(recorded["agent_state_frames"], dim=1).numpy(),
        background_actions=torch.cat(recorded["background_actions"], dim=1).numpy(),
        base_background_actions=torch.cat(recorded["base_background_actions"], dim=1).numpy(),
        ego_actions=torch.cat(recorded["ego_actions"], dim=1).numpy(),
        controller_diagnostics={
            "alpha": torch.stack(recorded["controller_alpha"], dim=1).numpy(),
            "delta_ax": torch.stack(recorded["controller_delta_ax"], dim=1).numpy(),
            "active": torch.stack(recorded["controller_active"], dim=1).numpy(),
            "phase": torch.stack(recorded["controller_phase"], dim=1).numpy(),
            "age_frames": torch.stack(recorded["controller_age_frames"], dim=1).numpy(),
            "rule_action_ax": torch.stack(recorded["controller_rule_action_ax"], dim=1).numpy(),
            "natural_kl": torch.stack(recorded["controller_natural_kl"], dim=1).numpy(),
            "desired_action_ax": torch.stack(recorded["controller_desired_action_ax"], dim=1).numpy(),
            "influence_authority": torch.stack(recorded["influence_authority"], dim=1).numpy(),
            "influence_role": torch.stack(recorded["influence_role"], dim=1).numpy(),
            "influence_parent": torch.stack(recorded["influence_parent"], dim=1).numpy(),
            "influence_direct": torch.stack(recorded["influence_direct"], dim=1).numpy(),
            "influence_secondary": torch.stack(recorded["influence_secondary"], dim=1).numpy(),
            "influence_predicted_ttc_s": torch.stack(recorded["influence_predicted_ttc_s"], dim=1).numpy(),
            "influence_predicted_min_gap_m": torch.stack(recorded["influence_predicted_min_gap_m"], dim=1).numpy(),
        },
        collision=torch.stack(recorded["collision"], dim=1).numpy(),
        crashed=torch.stack(recorded["crashed"], dim=1).numpy(),
    )


class ReactionPPOEnvironment:
    """Vectorized highD-reset HighwayEnv wrapper with causal ego interventions."""

    def __init__(
        self, model, *, states: np.ndarray, valid: np.ndarray, soft_plans: np.ndarray,
        maps: np.ndarray, map_valid: np.ndarray, controller: ReactionController,
        device: torch.device, config: PPOConfig, deterministic_response: bool = False,
        realism_reference: ReactionRealismReference | None = None,
    ) -> None:
        self.model, self.states_source = model, np.asarray(states, np.float32)
        self.valid_source, self.soft_plans = np.asarray(valid, bool), np.asarray(soft_plans, np.float32)
        self.maps, self.map_valid = np.asarray(maps, np.float32), np.asarray(map_valid, bool)
        self.controller, self.device, self.config = controller, device, config
        self.deterministic_response = bool(deterministic_response)
        self.realism_reference = realism_reference
        self.rng = np.random.default_rng(config.seed)
        self.world: HighwayEnvClosedLoopWorld | None = None
        self.logged_ego: torch.Tensor | None = None
        self.factual: np.ndarray | None = None
        self.onset: np.ndarray | None = None
        self.stop: np.ndarray | None = None
        self.dose: np.ndarray | None = None
        self.episode_mode: np.ndarray | None = None
        self.primary_child_slot: np.ndarray | None = None
        self.support_cell: np.ndarray | None = None
        self.step_index = 0
        self.previous_actions: torch.Tensor | None = None
        self.terminated: torch.Tensor | None = None
        anchor = self.states_source[:, ANCHOR_INDEX]
        present = self.valid_source[:, ANCHOR_INDEX, 1:]
        distance = np.abs(anchor[:, 1:, 0] - anchor[:, :1, 0])
        distance[~present] = np.inf
        nearest = distance.argmin(1)
        gap = distance[np.arange(len(distance)), nearest]
        relative_speed = anchor[np.arange(len(anchor)), nearest + 1, 2] - anchor[:, 0, 2]
        finite = np.isfinite(gap)
        gap = np.where(finite, gap, np.nanmedian(gap[finite]) if finite.any() else 0.)
        self.strata = np.digitize(gap, np.quantile(gap, (1 / 3, 2 / 3))) * 3 + np.digitize(relative_speed, np.quantile(relative_speed, (1 / 3, 2 / 3)))
        # Membership is the union over the recorded training horizon, so a
        # vehicle that becomes relevant after the anchor is not discarded.
        # This future information is never part of an online observation.
        self.eligible = np.flatnonzero(dynamic_candidate_scene_mask(
            self.states_source, self.valid_source,
            radius_m=config.influence_radius_m,
            prediction_horizon_s=config.influence_prediction_horizon_s,
        ))
        if not len(self.eligible):
            raise RuntimeError("training split has no dynamic causal-influence candidates")
        self._eligible_order = np.empty(0, np.int64)
        self._eligible_cursor = 0

    def _renew_eligible_epoch(self) -> None:
        """Create a shuffled, gap/speed-stratified pass over every eligible reset.

        PPO samples are consequently not an opaque with-replacement subset:
        one pass consumes every causally eligible highD training scene once.
        """
        groups = [self.eligible[self.strata[self.eligible] == index].copy() for index in range(9)]
        for group in groups:
            self.rng.shuffle(group)
        ordered: list[int] = []
        cursor = [0] * len(groups)
        while True:
            emitted = False
            for index, group in enumerate(groups):
                if cursor[index] < len(group):
                    ordered.append(int(group[cursor[index]])); cursor[index] += 1; emitted = True
            if not emitted:
                break
        self._eligible_order = np.asarray(ordered, np.int64)
        self._eligible_cursor = 0

    def sample_indices(self, count: int) -> np.ndarray:
        """Balance highD reset scenes over observed gap/relative-speed tertiles."""
        selected: list[int] = []
        while len(selected) < int(count):
            if self._eligible_cursor >= len(self._eligible_order):
                self._renew_eligible_epoch()
            take = min(int(count) - len(selected), len(self._eligible_order) - self._eligible_cursor)
            selected.extend(self._eligible_order[self._eligible_cursor:self._eligible_cursor + take].tolist())
            self._eligible_cursor += take
        return np.asarray(selected, np.int64)

    def sample_episode_specs(self, count: int, *, use_supported_replay: bool) -> list[ReactionEpisodeSpec]:
        """Make 8-rollout same-cell groups without changing synthetic sampling.

        Supported rows are sampled with replacement across updates, but every
        group is homogeneous in its train-derived support cell.  Synthetic
        rows continue to consume the existing stratified epoch order.
        """
        if not use_supported_replay or self.realism_reference is None:
            return [ReactionEpisodeSpec(int(row)) for row in self.sample_indices(count)]
        group_size = int(self.config.realism_group_size)
        natural_count = int(round(int(count) * float(self.config.supported_replay_fraction)))
        natural_count = max(0, min(int(count), natural_count - natural_count % group_size))
        eligible = set(int(row) for row in self.eligible.tolist())
        choices = {
            cell: indices for cell in self.realism_reference.supported_cells
            if len(indices := np.asarray([
                index for index, row in enumerate(self.realism_reference.events.row_index)
                if int(row) in eligible and int(self.realism_reference.events.cell[index]) == int(cell)
            ], np.int64)) >= group_size
        }
        if natural_count and not choices:
            raise RuntimeError("no supported natural-event replay rows remain in the PPO eligibility scope")
        specs: list[ReactionEpisodeSpec] = []
        cells = np.asarray(sorted(choices), np.int64)
        for _ in range(natural_count // group_size):
            cell = int(cells[int(self.rng.integers(len(cells)))])
            selected = self.rng.choice(choices[cell], size=group_size, replace=len(choices[cell]) < group_size)
            for event_index in selected:
                specs.append(ReactionEpisodeSpec(
                    row_index=int(self.realism_reference.events.row_index[event_index]), mode="supported_replay",
                    onset_step=int(self.realism_reference.events.onset_step[event_index]),
                    child_slot=int(self.realism_reference.events.child_slot[event_index]), support_cell=cell,
                ))
        specs.extend(ReactionEpisodeSpec(int(row), mode="synthetic") for row in self.sample_indices(int(count) - len(specs)))
        # Keep groups contiguous for cheap, auditable MLOO computation.
        return specs

    def reset(self, indices: np.ndarray | list[ReactionEpisodeSpec]) -> dict[str, torch.Tensor]:
        specs = (
            list(indices) if isinstance(indices, list) and (not indices or isinstance(indices[0], ReactionEpisodeSpec))
            else [ReactionEpisodeSpec(int(row)) for row in np.asarray(indices, np.int64)]
        )
        choice = np.asarray([spec.row_index for spec in specs], np.int64)
        values, present = self.states_source[choice], self.valid_source[choice]
        history = values[:, ANCHOR_INDEX - 24 : ANCHOR_INDEX + 1]
        history_valid = present[:, ANCHOR_INDEX - 24 : ANCHOR_INDEX + 1]
        ego = ego_controls(values[:, ANCHOR_INDEX:173, 0], values[:, ANCHOR_INDEX + 1:174, 0], .04)
        historical = ego_controls(values[:, :ANCHOR_INDEX, 0], values[:, 1:ANCHOR_INDEX + 1, 0], .04)
        self.logged_ego = torch.from_numpy(ego).to(self.device)
        n = len(choice)
        self.episode_mode = np.asarray([spec.mode for spec in specs], dtype=object)
        self.primary_child_slot = np.asarray([spec.child_slot for spec in specs], np.int8)
        self.support_cell = np.asarray([spec.support_cell for spec in specs], np.int16)
        legacy = self.episode_mode == "legacy"
        self.factual = legacy & (np.arange(n) % 2 == 0)
        self.onset = self.rng.integers(8, 49, size=n)
        replay = self.episode_mode == "supported_replay"
        if replay.any():
            self.onset[replay] = np.asarray([spec.onset_step for spec in specs], np.int64)[replay]
        self.stop = self.onset + self.rng.integers(5, 26, size=n)
        self.dose = self.rng.choice(np.asarray((-2., -4., -6., -8.), np.float32), size=n)
        exogenous = WorldExogenousState.sample(n, seed=int(self.rng.integers(2**31 - 1)), response_steps=149,
            scene_dim=self.model.cfg.scene_latent_dim, agent_dim=self.model.cfg.agent_latent_dim)
        self.world = HighwayEnvClosedLoopWorld(
            self.model, device=self.device, controller=self.controller,
            reaction_min_frames=self.config.reaction_min_frames,
            reaction_max_frames=self.config.reaction_max_frames,
            reaction_recovery_frames=self.config.reaction_recovery_frames,
            reaction_safety_ttc_s=self.config.safety_ttc_s,
            reaction_release_ttc_s=self.config.reaction_release_ttc_s,
            influence_radius_m=self.config.influence_radius_m,
            influence_secondary_radius_m=self.config.influence_secondary_radius_m,
            influence_prediction_horizon_s=self.config.influence_prediction_horizon_s,
            influence_stable_release_frames=self.config.influence_stable_release_frames,
        )
        self.world.reset(torch.from_numpy(values[:, ANCHOR_INDEX]), torch.from_numpy(present[:, ANCHOR_INDEX]),
            torch.from_numpy(self.soft_plans[choice]), torch.from_numpy(self.maps[choice]),
            torch.from_numpy(self.map_valid[choice]), exogenous_state=exogenous,
            initial_history=torch.from_numpy(history), initial_history_valid=torch.from_numpy(history_valid),
            committed_ego_controls=torch.from_numpy(historical),
            deterministic_response=self.deterministic_response)
        self.step_index, self.previous_actions = 0, None
        self.terminated = torch.zeros(n, dtype=torch.bool, device=self.device)
        return self.world.observe()

    @torch.no_grad()
    def step(self) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if (self.world is None or self.logged_ego is None or self.factual is None or self.terminated is None
                or self.episode_mode is None or self.onset is None or self.stop is None or self.dose is None):
            raise RuntimeError("reset PPO environment before stepping")
        alive = ~self.terminated
        ego = self.logged_ego[:, self.step_index].clone()
        replay = self.episode_mode == "supported_replay"
        synthetic = (~replay) & (~self.factual)
        active_intervention = synthetic & (self.step_index >= self.onset) & (self.step_index < self.stop)
        if active_intervention.any():
            mask = torch.from_numpy(active_intervention).to(self.device)
            ego[mask, 0] = (ego[mask, 0] + torch.from_numpy(self.dose[active_intervention]).to(self.device)).clamp_min(-8.)
        transition = self.world.advance_response(ego)
        # Register only after HighwayEnv has committed the ADS command.  The
        # next response sees the event; afterwards authority is maintained by
        # realized rear TTC and a recovery phase, not the command window.
        # A supported replay arms only after the observed highD parent brake
        # has physically executed.  It therefore preserves the same one-tick
        # causal latency as synthetic ADS interventions.
        replay_event = replay & (self.step_index == self.onset)
        self.world.register_executed_ego_intervention(torch.from_numpy(active_intervention | replay_event).to(self.device))
        final = transition["background_actions"][:, 0, :, 0]
        base = transition["base_background_actions"][:, 0, :, 0]
        previous_final = self.previous_actions
        correction = final - base
        selected = transition["controller_active"].float()
        # Reward a dose *floor*, not a brittle exact target.  The observed
        # ego intervention specifies the minimum causal response; if Highway
        # TTC is poor, PPO may use the remaining legal brake range rather
        # than being punished for exceeding that minimum.
        # The target is derived only from already committed ego control
        # evidence retained by the frozen model.  A follower earns its best
        # response reward when its residual brake tracks that observed dose;
        # this gives PPO a reason to use a large legal adjustment for a large
        # ADS change instead of exploiting a clipped direction-only reward.
        phase = transition["controller_phase"]
        engaged = phase == 1
        recovery = phase == 2
        desired_brake = (-transition["intervention_memory"]).clamp(
            .5, float(self.config.response_target_max_mps2)
        )[:, None]
        applied_brake = (-correction).clamp_min(0.)
        response = (applied_brake / desired_brake).clamp(0., 1.) * selected * engaged
        excess_brake = (applied_brake - desired_brake).relu() / 8.0 * selected
        ttc = transition["influence_predicted_ttc_s"]
        safety = ((ttc.clamp(max=self.config.safety_ttc_s) / self.config.safety_ttc_s) - 1.0)
        safety = safety * self.config.safety_coefficient * selected * engaged
        factual = torch.from_numpy(self.factual).to(self.device)[:, None]
        plan_coefficient = torch.where(recovery, torch.full_like(phase, self.config.recovery_plan_coefficient, dtype=torch.float32),
            torch.full_like(phase, self.config.plan_coefficient, dtype=torch.float32))
        plan = -plan_coefficient * torch.square(correction / 12.) * selected
        recovery_residual = self.config.recovery_residual_coefficient * (correction.abs() / 8.0) * selected * recovery
        reward = torch.where(factual, plan, response + safety + plan - recovery_residual - self.config.excess_brake_coefficient * excess_brake * engaged)
        natural_kl = transition.get("controller_natural_kl")
        episode_cells = self.support_cell if self.support_cell is not None else np.full(len(replay), -1, np.int16)
        supported_replay = replay & np.asarray([
            self.realism_reference is not None and self.realism_reference.is_supported(int(cell))
            for cell in episode_cells
        ], bool)
        support_gate = torch.from_numpy(supported_replay).to(self.device)[:, None].float()
        if natural_kl is not None and self.config.naturalness_weight > 0.0:
            # Human-likeness is a regularizer, never an emergency brake cap:
            # below the realized rear-TTC safety threshold it is fully
            # relaxed; it transitions back only once a safe following margin
            # has actually been recovered in HighwayEnv.
            relax = float(self.config.naturalness_ttc_relax_s)
            full = max(float(self.config.naturalness_ttc_full_s), relax + 1.e-4)
            natural_gate = ((ttc - relax) / (full - relax)).clamp(0., 1.)
            # A3/A4 add a zero-baseline cost on the *executed* action density.
            # The train-derived support mask is deliberately zero for every
            # synthetic -2..-8 intervention: no counterfactual human target is
            # claimed in those cells.
            normalized_kl = (natural_kl / float(max(self.config.naturalness_kl_scale, 1.e-4))).clamp(0., 5.)
            reward = reward - float(self.config.naturalness_weight) * normalized_kl * natural_gate * support_gate * selected
        floor_shortfall = (float(self.config.response_floor) - response).clamp_min(0.)
        reward = reward - float(self.config.response_floor_coefficient) * floor_shortfall * selected * engaged
        if previous_final is not None:
            jerk = (final - previous_final).abs() / .04
            jerk_excess = (jerk / self.config.jerk_limit_mps3 - 1.0).relu()
            reward = reward - self.config.jerk_coefficient * jerk_excess * selected
        # This policy controls only the intended rear follower.  An ego/front
        # collision is reported by evaluation, but it is not a learnable
        # target for a same-rear brake policy and must not terminate its
        # trajectory or drown the rear-TTC signal in counterfactual noise.
        crashed = transition["crashed"].bool()
        collision = (crashed[:, 1:] & transition["controller_active"]).any(1)
        reward = reward - collision.float()[:, None] * self.config.invalid_penalty
        reward = reward * alive[:, None].float()
        self.previous_actions = final.detach()
        self.step_index += 1
        self.terminated |= collision
        done = torch.full_like(reward, self.step_index >= min(self.config.rollout_steps, 149), dtype=torch.bool)
        done |= self.terminated[:, None]
        info = {key: transition[key] for key in ("controller_features", "controller_raw_action", "controller_log_prob", "controller_value", "controller_active", "controller_natural_kl", "controller_rule_action_ax")}
        info["controller_active"] = info["controller_active"] & alive[:, None]
        info["correction_ax"] = correction.detach()
        info["collision"] = collision.detach()
        info["phase"] = phase.detach()
        info["ttc_s"] = ttc.detach()
        info["support_gate"] = support_gate.detach()
        info["support_cell"] = torch.from_numpy(self.support_cell.copy()).to(self.device) if self.support_cell is not None else torch.full((len(replay),), -1, device=self.device, dtype=torch.long)
        # Per-rollout realism features are telemetry only until the PPO loop
        # converts grouped trajectories into a sequence-level MLOO advantage.
        state = transition["agent_state_frames"][:, 0]
        primary = np.asarray(self.primary_child_slot if self.primary_child_slot is not None else np.full(len(replay), -1), np.int64)
        safe_primary = torch.from_numpy(np.clip(primary, 1, 6)).to(self.device)
        batch = torch.arange(len(primary), device=self.device)
        child = state[batch, safe_primary]
        parent = state[:, 0]
        current_action = final[batch, safe_primary - 1]
        previous_action = current_action if previous_final is None else previous_final[batch, safe_primary - 1]
        gap = parent[:, 0] - child[:, 0] - 4.8
        closing = child[:, 2] - parent[:, 2]
        feature_ttc = torch.where(closing > 1.e-4, gap / closing.clamp_min(1.e-4), torch.full_like(gap, 10.)).clamp(0., 10.)
        info["realism_features"] = torch.stack((
            current_action, (current_action - previous_action).abs() / .04, child[:, 2], gap, closing, feature_ttc,
        ), dim=-1).detach()
        in_window = replay & (primary > 0) & (self.step_index - 1 >= self.onset + 1) & (self.step_index - 1 < self.onset + 1 + int(self.config.realism_window_frames))
        info["realism_feature_valid"] = torch.from_numpy(in_window).to(self.device) & alive
        return self.world.observe(), reward, done, info


def _gae(reward: torch.Tensor, value: torch.Tensor, done: torch.Tensor, gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    advantage = torch.zeros_like(reward)
    last = torch.zeros_like(reward[0])
    for index in range(len(reward) - 1, -1, -1):
        next_value = torch.zeros_like(last) if index == len(reward) - 1 else value[index + 1]
        nonterminal = (~done[index]).float()
        delta = reward[index] + gamma * next_value * nonterminal - value[index]
        last = delta + gamma * lam * nonterminal * last
        advantage[index] = last
    return advantage, advantage + value


def _stratified_mloo_advantages(
    features: torch.Tensor,
    valid: torch.Tensor,
    support_cells: torch.Tensor,
    reference: ReactionRealismReference | None,
    *, group_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Turn post-event rollout features into one MLOO advantage per episode.

    The scorer is intentionally detached from the critic return.  PPO gets a
    sequence-level policy-gradient signal while the critic continues to model
    the causal response reward it can observe step by step.
    """
    count = features.shape[1]
    result = torch.zeros(count, dtype=torch.float32, device=features.device)
    if reference is None:
        return result, {"mloo_groups": 0.0, "mloo_score_mean": 0.0, "mloo_reward_mean": 0.0}
    source = features.detach().cpu().numpy()
    mask = valid.detach().cpu().numpy().astype(bool)
    cells = support_cells.detach().cpu().numpy().astype(np.int64)
    scores: list[float] = []
    rewards: list[float] = []
    groups = 0
    for cell in sorted(set(cells.tolist())):
        if not reference.is_supported(int(cell)):
            continue
        indices = [index for index in np.flatnonzero(cells == cell) if int(mask[:, index].sum()) == int(reference.window_frames)]
        for start in range(0, len(indices) - int(group_size) + 1, int(group_size)):
            group = indices[start:start + int(group_size)]
            trajectories = np.stack([source[mask[:, index], index] for index in group])
            mloo, leave_one_out = mloo_rewards(trajectories, reference, int(cell))
            result[torch.as_tensor(group, device=result.device)] = torch.from_numpy(mloo).to(result.device)
            scores.extend(leave_one_out.tolist()); rewards.extend(mloo.tolist()); groups += 1
    return result, {
        "mloo_groups": float(groups),
        "mloo_score_mean": float(np.mean(scores)) if scores else 0.0,
        "mloo_reward_mean": float(np.mean(rewards)) if rewards else 0.0,
    }


@torch.no_grad()
def _validation_metrics(
    model,
    controller: ReactionController,
    *,
    arrays: dict[str, np.ndarray],
    plans: np.ndarray,
    config: PPOConfig,
    device: torch.device,
    realism_reference: ReactionRealismReference | None = None,
) -> dict[str, float]:
    validation_config = replace(config, seed=config.seed + 100003)
    environment = ReactionPPOEnvironment(
        model,
        states=arrays["agent_states"], valid=arrays["agent_valid"],
        soft_plans=plans, maps=arrays["map_polylines"],
        map_valid=arrays["map_polyline_valid"], controller=controller,
        device=device, config=validation_config, deterministic_response=True,
        realism_reference=realism_reference,
    )
    count = min(int(config.validation_scenes), len(environment.eligible))
    use_supported = realism_reference is not None and getattr(controller, "human_prior", None) is not None
    environment.reset(environment.sample_episode_specs(count, use_supported_replay=use_supported))
    rewards, corrections, active_values, collision_values = [], [], [], []
    feature_steps: list[torch.Tensor] = []
    feature_valid_steps: list[torch.Tensor] = []
    support_cells: torch.Tensor | None = None
    rebound_numerator = rebound_denominator = 0
    for _ in range(int(config.rollout_steps)):
        _, reward, _, info = environment.step()
        active = info["controller_active"].bool()
        rewards.append(reward[active].cpu())
        corrections.append(info["correction_ax"][active].cpu())
        active_values.append(active.float().mean().cpu())
        collision_values.append(info["collision"].cpu())
        feature_steps.append(info["realism_features"])
        feature_valid_steps.append(info["realism_feature_valid"])
        support_cells = info["support_cell"]
        unresolved = active & info["phase"].eq(1) & (info["ttc_s"] < config.reaction_release_ttc_s)
        rebound_numerator += int((unresolved & (info["correction_ax"] > .05)).sum().cpu())
        rebound_denominator += int(unresolved.sum().cpu())
    reward_values = torch.cat([item for item in rewards if len(item)]) if any(len(item) for item in rewards) else torch.zeros(1)
    correction_values = torch.cat([item for item in corrections if len(item)]) if any(len(item) for item in corrections) else torch.zeros(1)
    collisions = torch.stack(collision_values).any(0)
    realism: dict[str, float] = {
        "validation_composite_realism": 0.0,
        "validation_mloo_reward_mean": 0.0,
        **{f"validation_{name}_w1": float("inf") for name in FEATURE_NAMES},
    }
    if realism_reference is not None and support_cells is not None:
        feature_tensor, valid_tensor = torch.stack(feature_steps), torch.stack(feature_valid_steps)
        source, mask = feature_tensor.cpu().numpy(), valid_tensor.cpu().numpy().astype(bool)
        cells = support_cells.cpu().numpy().astype(np.int64)
        scores: list[float] = []
        distances: list[np.ndarray] = []
        for cell in sorted(set(cells.tolist())):
            indices = [index for index in np.flatnonzero(cells == cell)
                       if int(mask[:, index].sum()) == int(realism_reference.window_frames)]
            if realism_reference.is_supported(int(cell)) and indices:
                score, distance = realism_metric(
                    np.stack([source[mask[:, index], index] for index in indices]),
                    realism_reference, int(cell),
                )
                scores.append(score); distances.append(distance)
        mloo, _ = _stratified_mloo_advantages(
            feature_tensor, valid_tensor, support_cells, realism_reference,
            group_size=config.realism_group_size,
        )
        if scores:
            realism["validation_composite_realism"] = float(np.mean(scores))
            mean_distance = np.mean(distances, axis=0)
            realism.update({f"validation_{name}_w1": float(value) for name, value in zip(FEATURE_NAMES, mean_distance)})
            realism["validation_mloo_reward_mean"] = float(mloo.mean().cpu())
    return {
        "validation_reward": float(reward_values.mean()),
        "validation_response_dose_mps2": float((-correction_values).clamp_min(0.).mean()),
        "validation_collision_sequence_rate": float(collisions.float().mean()),
        "validation_active_fraction": float(torch.stack(active_values).mean()),
        "validation_positive_rebound_rate": float(rebound_numerator / max(rebound_denominator, 1)),
        "validation_scenes": float(count),
        **realism,
    }


@torch.no_grad()
def supported_rollout_realism_metrics(
    model,
    controller: ReactionController,
    *,
    arrays: dict[str, np.ndarray],
    plans: np.ndarray,
    reference: ReactionRealismReference,
    config: PPOConfig,
    device: torch.device,
    episodes: int = 128,
) -> dict[str, Any]:
    """Evaluate closed-loop natural-event realism without contaminating PPO.

    This is deliberately a separate protocol from synthetic interventions:
    only train-admitted support cells are sampled, parent actions are logged
    highD actions, and the scorer is built from the evaluation split alone.
    """
    group = int(config.realism_group_size)
    count = max(group, int(episodes) - int(episodes) % group)
    evaluation_config = replace(
        config, seed=config.seed + 300003, supported_replay_fraction=1.0,
    )
    environment = ReactionPPOEnvironment(
        model, states=arrays["agent_states"], valid=arrays["agent_valid"],
        soft_plans=plans, maps=arrays["map_polylines"],
        map_valid=arrays["map_polyline_valid"], controller=controller,
        device=device, config=evaluation_config, deterministic_response=True,
        realism_reference=reference,
    )
    try:
        environment.reset(environment.sample_episode_specs(count, use_supported_replay=True))
    except RuntimeError as error:
        return {
            "supported": False, "reason": str(error), "supported_rollouts": 0,
            "support_cells": [], "support_gate_rate": 0.0,
        }
    feature_steps: list[torch.Tensor] = []
    valid_steps: list[torch.Tensor] = []
    support_cells: torch.Tensor | None = None
    gate_steps: list[torch.Tensor] = []
    kl_values: list[torch.Tensor] = []
    collisions: list[torch.Tensor] = []
    for _ in range(int(config.rollout_steps)):
        _, _, _, info = environment.step()
        feature_steps.append(info["realism_features"])
        valid_steps.append(info["realism_feature_valid"])
        support_cells = info["support_cell"]
        gate_steps.append(info["support_gate"])
        active = info["controller_active"].bool() & info["support_gate"].bool()
        current_kl = info["controller_natural_kl"]
        if current_kl is not None and active.any():
            kl_values.append(current_kl[active].detach().cpu())
        collisions.append(info["collision"].detach().cpu())
    features = torch.stack(feature_steps)
    valid = torch.stack(valid_steps)
    cells = support_cells if support_cells is not None else torch.full((count,), -1, device=device, dtype=torch.long)
    mloo, mloo_summary = _stratified_mloo_advantages(
        features, valid, cells, reference, group_size=group,
    )
    source, mask = features.cpu().numpy(), valid.cpu().numpy().astype(bool)
    source_cells = cells.cpu().numpy().astype(np.int64)
    cell_reports: dict[str, Any] = {}
    scores: list[float] = []
    w1_values: list[np.ndarray] = []
    rollout_count = 0
    for cell in sorted(set(source_cells.tolist())):
        indices = [index for index in np.flatnonzero(source_cells == cell)
                   if int(mask[:, index].sum()) == int(reference.window_frames)]
        if not reference.is_supported(int(cell)) or not indices:
            continue
        trajectories = np.stack([source[mask[:, index], index] for index in indices])
        score, w1 = realism_metric(trajectories, reference, int(cell))
        cell_reports[str(cell)] = {
            "rollouts": int(len(indices)), "composite_realism": float(score),
            "feature_w1": {name: float(value) for name, value in zip(FEATURE_NAMES, w1)},
        }
        scores.append(score); w1_values.append(w1); rollout_count += len(indices)
    mean_w1 = np.mean(w1_values, axis=0) if w1_values else np.full(len(FEATURE_NAMES), np.inf)
    gate = torch.stack(gate_steps)
    return {
        "supported": bool(rollout_count), "supported_rollouts": int(rollout_count),
        "support_cells": sorted(cell_reports), "support_gate_rate": float(gate.float().mean().cpu()),
        "composite_realism": float(np.mean(scores)) if scores else 0.0,
        "feature_w1": {name: float(value) for name, value in zip(FEATURE_NAMES, mean_w1)},
        "mloo_reward_mean": float(mloo.mean().cpu()),
        "mloo_reward_sum": float(mloo.sum().cpu()),
        "mloo": mloo_summary, "natural_kl_mean": float(torch.cat(kl_values).mean()) if kl_values else 0.0,
        "rear_collision_sequence_rate": float(torch.stack(collisions).any(0).float().mean()),
        "cells": cell_reports,
    }


@torch.no_grad()
def _calibrate_naturalness_kl_scale(
    model,
    controller: ReactionController,
    *,
    arrays: dict[str, np.ndarray],
    plans: np.ndarray,
    config: PPOConfig,
    device: torch.device,
    realism_reference: ReactionRealismReference | None = None,
) -> tuple[float, dict[str, float]]:
    """Calibrate A3's fixed KL normalization on the frozen A2 policy.

    The controller already contains A2's actor weights and the frozen V4
    prior.  No optimizer step occurs here.  The 90th percentile keeps the
    bounded bonus informative for most validation states without changing
    the requested fixed naturalness coefficient.
    """
    calibration_config = replace(
        config, seed=config.seed + 200003,
        naturalness_weight=0.0, naturalness_kl_scale=1.0,
    )
    environment = ReactionPPOEnvironment(
        model,
        states=arrays["agent_states"], valid=arrays["agent_valid"],
        soft_plans=plans, maps=arrays["map_polylines"],
        map_valid=arrays["map_polyline_valid"], controller=controller,
        device=device, config=calibration_config, deterministic_response=True,
        realism_reference=realism_reference,
    )
    count = min(512, len(environment.eligible))
    environment.reset(environment.sample_episode_specs(count, use_supported_replay=realism_reference is not None))
    values: list[torch.Tensor] = []
    for _ in range(int(config.rollout_steps)):
        _, _, _, info = environment.step()
        active = info["controller_active"].bool() & info["support_gate"].bool()
        current = info["controller_natural_kl"]
        if current is not None and active.any():
            values.append(current[active].detach().float().cpu())
    if not values:
        raise RuntimeError("A3 KL calibration found no active validation actions")
    joined = torch.cat(values)
    joined = joined[torch.isfinite(joined)]
    if not len(joined):
        raise RuntimeError("A3 KL calibration produced no finite values")
    scale = max(float(torch.quantile(joined, .90)), 1.e-3)
    return scale, {
        "samples": float(len(joined)),
        "mean": float(joined.mean()),
        "median": float(joined.median()),
        "p90": scale,
        "maximum": float(joined.max()),
    }


def train_reaction_ppo(model, *, train_arrays: dict[str, np.ndarray], soft_plans: np.ndarray,
                       output_dir: str | Path, config: PPOConfig, device: torch.device,
                       controller_mode: str = "rl_residual", rule_model: RuleModelBundle | None = None,
                       human_prior: HumanActionPrior | None = None,
                       initial_state_dict: dict[str, torch.Tensor] | None = None,
                       resume: bool = True,
                       artifact_metadata: dict[str, Any] | None = None,
                       validation_arrays: dict[str, np.ndarray] | None = None,
                       validation_plans: np.ndarray | None = None,
                       realism_reference: ReactionRealismReference | None = None,
                       validation_realism_reference: ReactionRealismReference | None = None) -> dict[str, Any]:
    """Train only a residual actor/critic while retaining a frozen HiQR model."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if controller_mode == "rl_residual":
        controller: ReactionController = RLResidualReactionController().to(device)
    elif controller_mode in {"rl_residual_idm", "rl_residual_gail", "rl_residual_realism"}:
        if rule_model is None:
            raise ValueError(f"{controller_mode} requires rule_model")
        if controller_mode in {"rl_residual_gail", "rl_residual_realism"} and human_prior is None:
            raise ValueError(f"{controller_mode} requires human_prior")
        if controller_mode in {"rl_residual_gail", "rl_residual_realism"} and realism_reference is None:
            raise ValueError(f"{controller_mode} requires a train-only reaction realism reference")
        controller = IDMResidualReactionController(
            rule_model, human_prior if controller_mode in {"rl_residual_gail", "rl_residual_realism"} else None,
        ).to(device)
    else:
        raise ValueError(f"unsupported PPO controller mode {controller_mode!r}")
    if initial_state_dict is not None:
        # A3 starts from the independently validated IDM-PPO safety policy.
        # The frozen GAIL submodule is intentionally absent from A2's state
        # dict, hence non-strict loading is the expected contract here.
        controller.load_state_dict(initial_state_dict, strict=False)
    if human_prior is not None:
        # The prior is a fixed measurement distribution, never an A3/A4
        # optimizer target even though it is a controller submodule.
        human_prior.eval()
        for parameter in human_prior.parameters():
            parameter.requires_grad_(False)
    kl_calibration: dict[str, float] | None = None
    if (
        controller_mode in {"rl_residual_gail", "rl_residual_realism"}
        and float(config.naturalness_kl_scale) <= 0.0
    ):
        if validation_arrays is None or validation_plans is None:
            raise ValueError("automatic A3 KL calibration requires validation arrays and plans")
        calibrated_scale, kl_calibration = _calibrate_naturalness_kl_scale(
            model, controller, arrays=validation_arrays, plans=validation_plans,
            config=config, device=device, realism_reference=validation_realism_reference,
        )
        config = replace(config, naturalness_kl_scale=calibrated_scale)
    optimizer = torch.optim.Adam(
        [parameter for parameter in controller.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
    )
    environment = ReactionPPOEnvironment(model, states=train_arrays["agent_states"], valid=train_arrays["agent_valid"],
        soft_plans=soft_plans, maps=train_arrays["map_polylines"], map_valid=train_arrays["map_polyline_valid"],
        controller=controller, device=device, config=config, realism_reference=realism_reference)
    if (validation_arrays is None) != (validation_plans is None):
        raise ValueError("validation arrays and plans must be supplied together")
    full_pass_updates = int(np.ceil(len(environment.eligible) / max(config.episodes_per_rollout, 1)))
    if int(config.updates) < full_pass_updates:
        raise ValueError(
            f"PPO schedule has {config.updates} updates but a full dynamic-scene pass requires {full_pass_updates}"
        )
    target_dir = ensure_dir(output_dir)
    progress_path = target_dir / "reaction_ppo_progress.pt"
    history: list[dict[str, float]] = []
    start_update = 0
    resume_status = "fresh"
    best_validation_state: dict[str, torch.Tensor] | None = None
    best_validation_reward = -float("inf")
    validation_stale = 0
    progress_schema = (
        "reaction_residual_ppo_dynamic_progress_v4"
        if controller_mode in {"rl_residual_gail", "rl_residual_realism"}
        else "reaction_residual_ppo_dynamic_progress_v2"
    )
    checkpoint_schema = (
        "reaction_residual_ppo_dynamic_v4"
        if controller_mode in {"rl_residual_gail", "rl_residual_realism"}
        else "reaction_residual_ppo_dynamic_v2"
    )
    if resume and progress_path.exists():
        progress = torch.load(progress_path, map_location=device, weights_only=False)
        if progress.get("schema") != progress_schema:
            resume_status = "fresh_incompatible_fixed_scope_progress"
        elif progress.get("controller_mode") != controller_mode:
            raise ValueError("PPO progress checkpoint belongs to a different controller arm")
        elif progress.get("artifact_metadata") != (artifact_metadata or {}):
            resume_status = "fresh_incompatible_artifact_hashes"
        else:
          try:
            controller.load_state_dict(progress["state_dict"])
            optimizer.load_state_dict(progress["optimizer_state"])
            history = list(progress.get("history", [])); start_update = int(progress.get("next_update", 0))
            environment.rng.bit_generator.state = progress["environment_rng_state"]
            if progress.get("torch_rng_state") is not None:
                # ``map_location=device`` moves every tensor in the payload;
                # the default RNG is nevertheless a CPU generator.
                torch.set_rng_state(progress["torch_rng_state"].cpu())
            if torch.cuda.is_available() and progress.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(
                    [state.cpu() for state in progress["cuda_rng_state"]]
                )
            environment._eligible_order = np.asarray(progress.get("eligible_order", []), np.int64)
            environment._eligible_cursor = int(progress.get("eligible_cursor", 0))
            best_validation_state = progress.get("best_validation_state")
            best_validation_reward = float(progress.get("best_validation_reward", -float("inf")))
            validation_stale = int(progress.get("validation_stale", 0))
            resume_status = "resumed"
          except RuntimeError:
            # A new causal actuator feature changes the on-policy state
            # distribution.  Start clean instead of silently loading it.
            resume_status = "fresh_incompatible_progress"

    def save_progress(next_update: int) -> None:
        torch.save({
            "schema": progress_schema, "controller_mode": controller_mode,
            "next_update": int(next_update), "state_dict": controller.state_dict(),
            "optimizer_state": optimizer.state_dict(), "history": history,
            "environment_rng_state": environment.rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "eligible_order": environment._eligible_order, "eligible_cursor": environment._eligible_cursor,
            "artifact_metadata": artifact_metadata or {},
            "best_validation_state": best_validation_state,
            "best_validation_reward": best_validation_reward,
            "validation_stale": validation_stale,
        }, progress_path)

    try:
      for update in range(start_update, config.updates):
        use_supported_replay = controller_mode in {"rl_residual_gail", "rl_residual_realism"}
        episode_specs = environment.sample_episode_specs(config.episodes_per_rollout, use_supported_replay=use_supported_replay)
        environment.reset(episode_specs)
        buffer: dict[str, list[torch.Tensor]] = {name: [] for name in (
            "feature", "raw", "logp", "value", "reward", "done", "active", "natural_kl",
            "realism_features", "realism_feature_valid",
        )}
        support_cells: torch.Tensor | None = None
        for _ in range(config.rollout_steps):
            _, reward, done, info = environment.step()
            support_cells = info["support_cell"]
            for name, value in (("feature", info["controller_features"]), ("raw", info["controller_raw_action"]),
                                ("logp", info["controller_log_prob"]), ("value", info["controller_value"]),
                                ("reward", reward), ("done", done), ("active", info["controller_active"]),
                                ("natural_kl", info["controller_natural_kl"]),
                                ("realism_features", info["realism_features"]),
                                ("realism_feature_valid", info["realism_feature_valid"])):
                # `none`/pure/IDM controllers legitimately have no human
                # prior.  Retain a shared, explicit zero telemetry channel
                # so every PPO arm remains trainable and plot-compatible.
                if value is None:
                    value = torch.zeros_like(reward)
                buffer[name].append(value.detach())
        values, rewards, dones = (torch.stack(buffer[key]) for key in ("value", "reward", "done"))
        advantages, returns = _gae(rewards, values, dones, config.gamma, config.gae_lambda)
        mloo_by_episode, mloo_summary = _stratified_mloo_advantages(
            torch.stack(buffer["realism_features"]), torch.stack(buffer["realism_feature_valid"]),
            (support_cells if support_cells is not None else torch.full((config.episodes_per_rollout,), -1, device=device, dtype=torch.long)),
            realism_reference if controller_mode == "rl_residual_realism" else None,
            group_size=config.realism_group_size,
        )
        flat = {
            key: torch.stack(value).reshape(-1, *value[0].shape[2:]) if key in {"feature", "raw"} else torch.stack(value).reshape(-1)
            for key, value in buffer.items()
            if key not in {"realism_features", "realism_feature_valid"}
        }
        active = flat["active"].bool()
        if not active.any():
            # A dynamic-candidate recording may become relevant only after
            # the sampled anchor, while a natural half-batch can contain no
            # currently armed vehicle.  Do not abort a full scene pass: keep
            # the sampled bounded actions as a zero-authority baseline so the
            # optimizer/progress cursor remains well-defined.  Subsequent
            # intervention boundaries still provide the policy gradient.
            active = torch.ones_like(active)
        feature, raw, old_logp = flat["feature"][active], flat["raw"][active], flat["logp"][active]
        target, advantage = returns.reshape(-1)[active], advantages.reshape(-1)[active]
        advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1.e-6)
        if controller_mode == "rl_residual_realism":
            # MLOO is one scalar per trajectory.  Broadcast it over time and
            # background slots before applying the active-action mask; the
            # critic target remains the ordinary stepwise return.
            slot_count = int(buffer["active"][0].shape[-1])
            sequence_advantage = mloo_by_episode[None, :, None].expand(
                config.rollout_steps, config.episodes_per_rollout, slot_count
            ).reshape(-1)
            advantage = advantage + float(config.realism_advantage_weight) * sequence_advantage[active]
        losses: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        clip_fractions: list[float] = []
        for _ in range(config.epochs_per_update):
            order = torch.randperm(len(feature), device=device)
            for start in range(0, len(order), config.minibatch_size):
                batch = order[start:start + config.minibatch_size]
                logp, entropy, value = controller.evaluate_raw_action(feature[batch], raw[batch])
                ratio = (logp - old_logp[batch]).exp()
                policy = -torch.minimum(ratio * advantage[batch], ratio.clamp(1 - config.clip_ratio, 1 + config.clip_ratio) * advantage[batch]).mean()
                value_loss = functional.mse_loss(value, target[batch])
                loss = policy + config.value_coefficient * value_loss - config.entropy_coefficient * entropy.mean()
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(controller.parameters(), config.max_grad_norm); optimizer.step()
                losses.append(float(loss.detach()))
                policy_losses.append(float(policy.detach()))
                value_losses.append(float(value_loss.detach()))
                entropies.append(float(entropy.mean().detach()))
                clip_fractions.append(float((ratio.sub(1.).abs() > config.clip_ratio).float().mean().detach()))
        active_history = torch.stack(buffer["active"]).bool()
        natural_values = torch.stack(buffer["natural_kl"])
        entry: dict[str, float] = {
            "update": float(update), "loss": float(np.mean(losses)),
            "policy_loss": float(np.mean(policy_losses)), "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)), "clip_fraction": float(np.mean(clip_fractions)),
            "reward": float(rewards[active_history].mean()), "active_fraction": float(active_history.float().mean()),
            "natural_kl_mean": float(natural_values[active_history].float().mean()),
            "eligible_scenes_seen": int(min((update + 1) * config.episodes_per_rollout, len(environment.eligible))),
            **mloo_summary,
        }
        should_validate = (
            validation_arrays is not None
            and update + 1 >= full_pass_updates
            and (
                (update + 1 - full_pass_updates) % max(config.validation_interval_updates, 1) == 0
                or update + 1 == config.updates
            )
        )
        if should_validate:
            controller.eval()
            validation = _validation_metrics(
                model, controller, arrays=validation_arrays,
                plans=validation_plans, config=config, device=device,
                realism_reference=validation_realism_reference,
            )
            controller.train()
            entry.update(validation)
            score = (
                validation["validation_composite_realism"]
                if controller_mode == "rl_residual_realism"
                else validation["validation_reward"]
            )
            if score > best_validation_reward + config.validation_minimum_improvement:
                best_validation_reward, validation_stale = score, 0
                best_validation_state = {
                    name: value.detach().cpu().clone()
                    for name, value in controller.state_dict().items()
                }
            else:
                validation_stale += 1
        history.append(entry)
        # A complete HighwayEnv update can take minutes. Persist every
        # update, including optimizer/sampler state, so an interruption never
        # silently turns a formal run into an untraceable partial result.
        save_progress(update + 1)
        if should_validate and validation_stale >= int(config.validation_patience):
            break
    except KeyboardInterrupt:
        save_progress(len(history))
        raise
    if best_validation_state is not None:
        controller.load_state_dict(best_validation_state)
    checkpoint = target_dir / "reaction_ppo.pt"
    torch.save({
        "schema": checkpoint_schema,
        "config": config.__dict__, "state_dict": controller.state_dict(),
        "controller_mode": controller_mode,
        "objective_terms": {
            "reactive_reward": True,
            "gail_final_action_kl": controller_mode in {"rl_residual_gail", "rl_residual_realism"},
            "stratified_highd_mloo": controller_mode == "rl_residual_realism",
        },
        "reaction_realism_reference_sha256": None if realism_reference is None else realism_reference.source_rows_sha256,
        "artifact_metadata": artifact_metadata or {},
    }, checkpoint)
    summary = {"checkpoint": str(checkpoint), "updates": len(history), "configured_max_updates": config.updates,
               "full_pass_updates": full_pass_updates, "history": history,
               "frozen_world_model": True, "controller_mode": controller_mode,
               "uses_idm_reference": controller_mode != "rl_residual", "uses_gail_prior": controller_mode in {"rl_residual_gail", "rl_residual_realism"},
               "uses_rollout_realism": controller_mode == "rl_residual_realism",
               "reaction_realism_reference_sha256": None if realism_reference is None else realism_reference.source_rows_sha256,
               "resume_status": resume_status, "artifact_metadata": artifact_metadata or {},
               "naturalness_kl_scale": float(config.naturalness_kl_scale),
               "naturalness_kl_calibration": kl_calibration,
               "validation_selection_metric": (
                   "validation_composite_realism" if controller_mode == "rl_residual_realism" else "validation_reward"
               ),
               "best_validation_score": None if best_validation_state is None else best_validation_reward,
               "best_validation_reward": None if best_validation_state is None else best_validation_reward,
               "validation_selected": best_validation_state is not None}
    save_json(summary, target_dir / "training_summary.json")
    return summary
