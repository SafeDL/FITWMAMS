"""Fine-grained, auditable inference environment for the new world model."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .graph_schema import DynamicTrafficGraph
from .initial_behavior_anchor import behavior_anchor_from_flow_feature
from .semi_markov_model import SemiMarkovRelationalWorldModel


@dataclass
class WorldRandomness:
    """ADS-independent exogenous uniforms; a seed fills any omitted suffix."""

    seed: int = 123
    state_uniforms: list[float] = field(default_factory=list)
    duration_uniforms: list[float] = field(default_factory=list)
    # These audit fields are deliberately exogenous metadata.  They neither
    # affect the learned transition nor identify an ADS implementation.
    event_structure: Any | None = None
    flow_base_sample: Any | None = None
    map_adapter_version: str | None = None
    model_checkpoint_hash: str | None = None


class SemiMarkovBackgroundEnvironment:
    """Environment with response-scale ``step`` and one-second wrappers.

    The only exogenous random variables consumed are one state and one duration
    uniform when a latent state begins.  The logged trace is sufficient to
    replay the same latent path under another ADS.
    """

    def __init__(
        self,
        model: SemiMarkovRelationalWorldModel,
        *,
        device: str | torch.device = "cpu",
        model_checkpoint_hash: str | None = None,
        map_adapter_version: str = "caller_supplied_dynamic_graph",
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.graph: DynamicTrafficGraph | None = None
        self._history_states: list[np.ndarray] = []
        self._history_valid: list[np.ndarray] = []
        self._states: np.ndarray | None = None
        self._valid: np.ndarray | None = None
        self._latent_state: int | None = None
        self._remaining_duration = 0
        self._elapsed = 0
        self._rng: np.random.Generator | None = None
        self._state_uniforms: list[float] = []
        self._duration_uniforms: list[float] = []
        self._behavior_anchor: np.ndarray | None = None
        self._behavior_anchor_valid: np.ndarray | None = None
        self._behavior_anchor_active = False
        self.model_checkpoint_hash = model_checkpoint_hash or getattr(model, "checkpoint_hash", None) or "unbound_model"
        self.map_adapter_version = str(map_adapter_version)
        self.trace: dict[str, Any] = {}

    def reset(
        self,
        initial_graph: DynamicTrafficGraph,
        world_randomness: WorldRandomness | None = None,
        *,
        behavior_anchor: np.ndarray | None = None,
        behavior_anchor_valid: np.ndarray | None = None,
    ) -> dict[str, Any]:
        randomness = world_randomness or WorldRandomness()
        self.graph = initial_graph
        self._states = np.asarray(initial_graph.agent_states, np.float32).copy()
        self._valid = np.asarray(initial_graph.agent_valid, bool).copy()
        self._history_states = [self._states.copy() for _ in range(25)]
        self._history_valid = [self._valid.copy() for _ in range(25)]
        self._latent_state, self._remaining_duration, self._elapsed = None, 0, 0
        self._rng = np.random.default_rng(int(randomness.seed))
        self._state_uniforms, self._duration_uniforms = list(randomness.state_uniforms), list(randomness.duration_uniforms)
        self._behavior_anchor = None
        self._behavior_anchor_valid = None
        if behavior_anchor is not None:
            anchor = np.asarray(behavior_anchor, np.float32)
            if anchor.shape != (len(self._states) - 1, 6):
                raise ValueError("behavior_anchor must be [background_agents, 6]")
            self._behavior_anchor = anchor.copy()
            self._behavior_anchor_valid = np.asarray(
                self._valid[1:] if behavior_anchor_valid is None else behavior_anchor_valid,
                bool,
            ).reshape(len(self._states) - 1).copy()
        self._behavior_anchor_active = self._behavior_anchor is not None and self.model.uses_behavior_anchor
        self.trace = {
            "event_structure": deepcopy(randomness.event_structure),
            "z_flow_or_flow_base_sample": deepcopy(randomness.flow_base_sample),
            "initial_physical_state": self._states.tolist(),
            "world_random_seed": int(randomness.seed), "uniform_state_random_numbers": [], "uniform_duration_random_numbers": [],
            "realized_latent_states": [], "realized_durations": [], "latent_transition_times": [],
            "response_update_period": float(self.model.cfg.response_interval_s), "dynamics_version": self.model.dynamics.version,
            "model_checkpoint_hash": str(randomness.model_checkpoint_hash or self.model_checkpoint_hash),
            "map_adapter_version": str(randomness.map_adapter_version or self.map_adapter_version), "response_steps": 0,
            "initial_behavior_anchor": None if self._behavior_anchor is None else self._behavior_anchor.tolist(),
            "initial_behavior_anchor_valid": None if self._behavior_anchor_valid is None else self._behavior_anchor_valid.tolist(),
            "behavior_anchor_active": bool(self._behavior_anchor_active),
        }
        return self.snapshot()

    def reset_from_flow_sample(
        self,
        initial_graph: DynamicTrafficGraph,
        feature_row: np.ndarray,
        slot_mask: np.ndarray,
        world_randomness: WorldRandomness | None = None,
    ) -> dict[str, Any]:
        """Reset from a graph plus the frozen Flow's 76-D behavior condition."""
        anchor, anchor_valid = behavior_anchor_from_flow_feature(feature_row, slot_mask)
        return self.reset(
            initial_graph, world_randomness,
            behavior_anchor=anchor, behavior_anchor_valid=anchor_valid,
        )

    def snapshot(self) -> dict[str, Any]:
        if self._states is None or self._valid is None:
            raise RuntimeError("environment has not been reset")
        return {
            # Include the map graph and the latent/RNG state so an AMS branch
            # can be restored in a fresh environment and continue bitwise
            # deterministically, not merely inspect the visible positions.
            "graph": deepcopy(self.graph),
            "agent_states": self._states.copy(), "agent_valid": self._valid.copy(),
            "history_states": [item.copy() for item in self._history_states],
            "history_valid": [item.copy() for item in self._history_valid],
            "latent_state": self._latent_state, "remaining_duration": int(self._remaining_duration),
            "elapsed": int(self._elapsed),
            "behavior_anchor": None if self._behavior_anchor is None else self._behavior_anchor.copy(),
            "behavior_anchor_valid": None if self._behavior_anchor_valid is None else self._behavior_anchor_valid.copy(),
            "behavior_anchor_active": bool(self._behavior_anchor_active),
            "rng_state": None if self._rng is None else deepcopy(self._rng.bit_generator.state),
            "state_uniforms_remaining": list(self._state_uniforms),
            "duration_uniforms_remaining": list(self._duration_uniforms),
            "trace": deepcopy(self.trace),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        graph = snapshot.get("graph")
        if graph is not None:
            self.graph = deepcopy(graph)
        if self.graph is None:
            raise ValueError("snapshot does not contain a graph and this environment has not been reset")
        self._states = np.asarray(snapshot["agent_states"], np.float32).copy()
        self._valid = np.asarray(snapshot["agent_valid"], bool).copy()
        self.trace = deepcopy(snapshot["trace"])
        self._history_states = [np.asarray(item, np.float32).copy() for item in snapshot.get("history_states", [self._states] * 25)]
        self._history_valid = [np.asarray(item, bool).copy() for item in snapshot.get("history_valid", [self._valid] * 25)]
        self._latent_state = snapshot.get("latent_state")
        self._remaining_duration = int(snapshot.get("remaining_duration", 0))
        self._elapsed = int(snapshot.get("elapsed", 0))
        anchor = snapshot.get("behavior_anchor")
        anchor_valid = snapshot.get("behavior_anchor_valid")
        self._behavior_anchor = None if anchor is None else np.asarray(anchor, np.float32).copy()
        self._behavior_anchor_valid = None if anchor_valid is None else np.asarray(anchor_valid, bool).copy()
        self._behavior_anchor_active = bool(snapshot.get("behavior_anchor_active", False))
        self._state_uniforms = [float(value) for value in snapshot.get("state_uniforms_remaining", [])]
        self._duration_uniforms = [float(value) for value in snapshot.get("duration_uniforms_remaining", [])]
        rng_state = snapshot.get("rng_state")
        self._rng = None
        if rng_state is not None:
            self._rng = np.random.default_rng()
            self._rng.bit_generator.state = deepcopy(rng_state)

    def _next_uniform(self, kind: str) -> float:
        values = self._state_uniforms if kind == "state" else self._duration_uniforms
        if values:
            value = float(values.pop(0))
        else:
            if self._rng is None:
                raise RuntimeError("environment randomness not initialized")
            value = float(self._rng.random())
        self.trace[f"uniform_{kind}_random_numbers"].append(value)
        return value

    def _tensors(self) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if self.graph is None or self._states is None or self._valid is None:
            raise RuntimeError("environment has not been reset")
        n = len(self._states)
        map_polylines = np.asarray(self.graph.map_polylines, np.float32)
        map_valid = np.asarray(self.graph.map_polyline_valid, bool)
        batch = {
            "agent_states": torch.as_tensor(np.stack(self._history_states)[None], device=self.device),
            "agent_valid": torch.as_tensor(np.stack(self._history_valid)[None], device=self.device),
            "ego_index": torch.as_tensor([self.graph.ego_index], device=self.device),
            "map_polylines": torch.as_tensor(map_polylines[None], device=self.device),
            "map_polyline_valid": torch.as_tensor(map_valid[None], device=self.device),
            "lane_graph_edges": torch.as_tensor(np.asarray(self.graph.lane_graph_edges, np.int64)[None], device=self.device),
        }
        # Conflict regions are static map features, just like lane polylines.
        # Passing them through the live environment is essential for rounD:
        # otherwise the trained conflict cross-attention block would be
        # silently absent during closed-loop rollout even though it was active
        # in the sequence loader and evaluation path.
        if self.model.cfg.use_conflict_zones:
            batch["conflict_zone_features"] = torch.as_tensor(
                np.asarray(self.graph.conflict_zone_features, np.float32)[None], device=self.device,
            )
            batch["conflict_zone_valid"] = torch.as_tensor(
                np.asarray(self.graph.conflict_zone_valid, bool)[None], device=self.device,
            )
        current = torch.as_tensor(self._states[None], device=self.device)
        return batch, current

    @torch.no_grad()
    def step(self, ego_state: np.ndarray, ego_valid: bool = True, dt: float | None = None) -> dict[str, Any]:
        """Advance one configured response interval using an observed ego state."""
        if self.graph is None or self._states is None or self._valid is None:
            raise RuntimeError("Call reset(initial_graph, world_randomness) first")
        requested_dt = self.model.cfg.response_interval_s if dt is None else float(dt)
        if abs(requested_dt - self.model.cfg.response_interval_s) > 1.0e-7:
            raise ValueError("step dt must equal the model response_interval_s; instantiate a model configured for 0.1 or 0.2 seconds")
        ego_index = int(self.graph.ego_index)
        self._states[ego_index] = np.asarray(ego_state, np.float32)
        self._valid[ego_index] = bool(ego_valid)
        self._history_states[-1] = self._states.copy()
        self._history_valid[-1] = self._valid.copy()
        batch, current = self._tensors()
        ego_mask = self.model._ego_mask(batch)
        valid = torch.as_tensor(self._valid[None], device=self.device)
        agents, scene, _ = self.model.encode_step(
            batch["agent_states"], batch["agent_valid"], current, valid, ego_mask,
            batch["map_polylines"], batch["map_polyline_valid"], batch["lane_graph_edges"],
            batch.get("conflict_zone_features"), batch.get("conflict_zone_valid"),
        )
        anchor_agents = None
        anchor_scene = None
        if self._behavior_anchor_active:
            anchor_agents, anchor_scene = self.model.encode_behavior_anchor(
                torch.as_tensor(self._behavior_anchor[None], device=self.device),
                torch.as_tensor(self._behavior_anchor_valid[None], device=self.device),
            )
        if self._remaining_duration <= 0:
            latent_scene = self.model.initial_latent_scene(scene, anchor_scene) if (
                self._latent_state is None and self.model.conditions_initial_latent
            ) else scene
            z, duration, probabilities = self.model.latent.sample_state_and_duration(
                latent_scene[0], self._latent_state, self._next_uniform("state"), self._next_uniform("duration"),
            )
            self._latent_state, self._remaining_duration, self._elapsed = z, duration, 1
            self.trace["realized_latent_states"].append(int(z))
            self.trace["realized_durations"].append(int(duration))
            self.trace["latent_transition_times"].append(int(self.trace["response_steps"]))
        else:
            self._elapsed += 1
            probabilities = None
        state_prob = torch.nn.functional.one_hot(
            torch.tensor([self._latent_state], device=self.device), num_classes=self.model.cfg.num_latent_states
        ).float()
        decoded = self.model.decoder(
            agents, scene, self.model.latent.state_embedding(state_prob),
            torch.tensor([self._elapsed], device=self.device), valid & ~ego_mask,
            self.model.dynamics.controls_from_highd_actions(current[..., 4:6], current),
            anchor_agents if self._behavior_anchor_active else None,
        )
        controls = decoded["controls"]
        integration_controls = self.model._integration_controls(decoded)
        # The physical ego state is observed only at the current response time;
        # it is held during the internal 25-Hz substeps.  ``roll`` below uses a
        # causal constant-velocity extrapolation when no later ego observation exists.
        ego_future = current[:, ego_index : ego_index + 1].expand(-1, self.model.cfg.physics_steps_per_response, -1)
        ego_valid_tensor = valid[:, ego_index : ego_index + 1].expand(-1, self.model.cfg.physics_steps_per_response)
        next_state, physical = self.model._integrate_response(
            current, integration_controls, valid, ego_future, ego_valid_tensor, ego_index=ego_index,
        )
        outputs = torch.stack(physical, dim=1)[0].cpu().numpy()
        self._states = next_state[0].cpu().numpy()
        self._valid = valid[0].cpu().numpy().astype(bool)
        for state in outputs:
            self._history_states.append(state.copy())
            self._history_valid.append(self._valid.copy())
        self._history_states, self._history_valid = self._history_states[-25:], self._history_valid[-25:]
        self._remaining_duration -= 1
        self.trace["response_steps"] += 1
        if self.trace["response_steps"] >= self.model.cfg.behavior_anchor_response_steps:
            self._behavior_anchor_active = False
        self.trace["behavior_anchor_active"] = bool(self._behavior_anchor_active)
        highd_actions = self.model.dynamics.highd_actions(controls, current)[0].cpu().numpy()
        background_indices = np.flatnonzero(np.arange(len(self._states)) != ego_index)
        control_curve = decoded["control_plan"][0, background_indices].permute(1, 0, 2).cpu().numpy()
        return {
            "agent_states": self._states.copy(), "agent_valid": self._valid.copy(),
            "background_states": outputs[:, background_indices], "background_valid": np.repeat(self._valid[None, background_indices], len(outputs), axis=0),
            "controls": controls[0, background_indices].cpu().numpy(), "actions_mps2": highd_actions[background_indices],
            "control_curve": control_curve,
            "latent_state": int(self._latent_state), "remaining_duration": int(self._remaining_duration),
            "latent_probabilities": None if probabilities is None else probabilities.cpu().numpy(), "trace": dict(self.trace),
        }

    @staticmethod
    def _causal_ego_extrapolation(state: np.ndarray, steps: int, dt: float) -> list[np.ndarray]:
        current = np.asarray(state, np.float32).copy()
        outputs: list[np.ndarray] = []
        for _ in range(steps):
            current = current.copy()
            current[0] += current[2] * dt
            current[1] += current[3] * dt
            outputs.append(current)
        return outputs

    def roll(self, ego_history_states: np.ndarray, ego_history_valid: np.ndarray) -> dict[str, Any]:
        """Causal one-second compatibility wrapper executing five updates."""
        history = np.asarray(ego_history_states, np.float32)
        valid = np.asarray(ego_history_valid, bool)
        if history.shape[0] < 1 or valid.shape[0] != history.shape[0]:
            raise ValueError("ego history must be [H, state_dim] with matching validity")
        current = history[-1]
        chunks = []
        for next_ego in self._causal_ego_extrapolation(current, 5, self.model.cfg.response_interval_s):
            result = self.step(next_ego, bool(valid[-1]))
            chunks.append(result)
            current = next_ego
        return {
            "actions_mps2": np.concatenate([item["actions_mps2"][None].repeat(self.model.cfg.physics_steps_per_response, axis=0) for item in chunks], axis=0),
            "background_states": np.concatenate([item["background_states"] for item in chunks], axis=0),
            "background_valid": np.concatenate([item["background_valid"] for item in chunks], axis=0),
            "latent_trace": dict(self.trace),
        }
