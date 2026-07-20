"""Fine-grained, auditable inference environment for the new world model."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .graph_schema import DynamicTrafficGraph
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

    State and duration uniforms are exogenous. The logged trace is sufficient
    to replay the same latent path under another ADS.
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
        self._behavior_anchor_std: np.ndarray | None = None
        self._start_mode_controls: torch.Tensor | None = None
        self._behavior_anchor_initial: torch.Tensor | None = None
        self._behavior_anchor_generated: list[torch.Tensor] = []
        self._behavior_anchor_active = False
        self._previous_control_plan: np.ndarray | None = None
        self._previous_plan_valid: np.ndarray | None = None
        self._previous_relation_summary: np.ndarray | None = None
        self._previous_intent_state: int | None = None
        self._plan_response_index = 0
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
        self._behavior_anchor_std = None
        self._start_mode_controls = None
        self._behavior_anchor_initial = None
        self._behavior_anchor_generated = []
        self._previous_control_plan = None
        self._previous_plan_valid = None
        self._previous_relation_summary = None
        self._previous_intent_state = None
        self._plan_response_index = 0
        if behavior_anchor is not None:
            anchor = np.asarray(behavior_anchor, np.float32)
            if anchor.shape != (len(self._states) - 1, 6):
                raise ValueError("behavior_anchor must be [background_agents, 6]")
            self._behavior_anchor = anchor.copy()
            self._behavior_anchor_valid = np.asarray(
                self._valid[1:] if behavior_anchor_valid is None else behavior_anchor_valid,
                bool,
            ).reshape(len(self._states) - 1).copy()
            raw = torch.as_tensor(self._behavior_anchor[None], device=self.device)
            anchor_valid = torch.as_tensor(self._behavior_anchor_valid[None], device=self.device)
            standardized = self.model.frozen_flow_schema.standardize(raw, anchor_valid) if self.model.frozen_flow_schema else raw
            self._behavior_anchor_std = standardized[0].cpu().numpy()
            self._behavior_anchor_initial = torch.as_tensor(self._states[None], device=self.device)
            self._start_mode_controls = self.model._start_mode_controls(
                self._behavior_anchor_initial, raw, torch.as_tensor(self._valid[None], device=self.device),
            )
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
            "previous_control_plan": None if self._previous_control_plan is None else self._previous_control_plan.copy(),
            "previous_plan_valid": None if self._previous_plan_valid is None else self._previous_plan_valid.copy(),
            "previous_relation_summary": None if self._previous_relation_summary is None else self._previous_relation_summary.copy(),
            "previous_intent_state": self._previous_intent_state,
            "plan_response_index": int(self._plan_response_index),
            "behavior_anchor_initial": None if self._behavior_anchor_initial is None else self._behavior_anchor_initial.cpu().numpy().copy(),
            "behavior_anchor_generated": [item.cpu().numpy().copy() for item in self._behavior_anchor_generated],
        }
        return self.snapshot()

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
            "behavior_anchor_initial": None if self._behavior_anchor_initial is None else self._behavior_anchor_initial.cpu().numpy().copy(),
            "behavior_anchor_generated": [item.cpu().numpy().copy() for item in self._behavior_anchor_generated],
            "previous_control_plan": None if self._previous_control_plan is None else self._previous_control_plan.copy(),
            "previous_plan_valid": None if self._previous_plan_valid is None else self._previous_plan_valid.copy(),
            "previous_relation_summary": None if self._previous_relation_summary is None else self._previous_relation_summary.copy(),
            "previous_intent_state": self._previous_intent_state,
            "plan_response_index": int(self._plan_response_index),
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
        if self._behavior_anchor is None:
            self._behavior_anchor_std = None
        else:
            raw = torch.as_tensor(self._behavior_anchor[None], device=self.device)
            valid_anchor = torch.as_tensor(self._behavior_anchor_valid[None], device=self.device)
            standardized = self.model.frozen_flow_schema.standardize(raw, valid_anchor) if self.model.frozen_flow_schema else raw
            self._behavior_anchor_std = standardized[0].cpu().numpy()
        initial = snapshot.get("behavior_anchor_initial")
        self._behavior_anchor_initial = torch.as_tensor(
            self._states[None] if initial is None else np.asarray(initial, np.float32), device=self.device,
        )
        self._behavior_anchor_generated = [
            torch.as_tensor(np.asarray(item, np.float32), device=self.device)
            for item in snapshot.get("behavior_anchor_generated", [])
        ]
        self._behavior_anchor_active = bool(snapshot.get("behavior_anchor_active", False))
        if self._behavior_anchor_active and self._behavior_anchor is not None:
            initial_valid = torch.zeros((1, len(self._states)), dtype=torch.bool, device=self.device)
            initial_valid[:, 0] = True
            initial_valid[:, 1:] = torch.as_tensor(self._behavior_anchor_valid[None], device=self.device)
            raw = torch.as_tensor(self._behavior_anchor[None], device=self.device)
            self._start_mode_controls = self.model._start_mode_controls(self._behavior_anchor_initial, raw, initial_valid)
        else:
            self._start_mode_controls = None
        previous_plan = snapshot.get("previous_control_plan")
        previous_valid = snapshot.get("previous_plan_valid")
        previous_relation = snapshot.get("previous_relation_summary")
        self._previous_control_plan = None if previous_plan is None else np.asarray(previous_plan, np.float32).copy()
        self._previous_plan_valid = None if previous_valid is None else np.asarray(previous_valid, bool).copy()
        self._previous_relation_summary = None if previous_relation is None else np.asarray(previous_relation, np.float32).copy()
        self._previous_intent_state = snapshot.get("previous_intent_state")
        self._plan_response_index = int(snapshot.get("plan_response_index", self.trace.get("response_steps", 0)))
        self._state_uniforms = [float(value) for value in snapshot.get("state_uniforms_remaining", [])]
        self._duration_uniforms = [float(value) for value in snapshot.get("duration_uniforms_remaining", [])]
        rng_state = snapshot.get("rng_state")
        self._rng = None
        if rng_state is not None:
            self._rng = np.random.default_rng()
            self._rng.bit_generator.state = deepcopy(rng_state)

    def _next_uniform(self, kind: str) -> float:
        values = {
            "state": self._state_uniforms,
            "duration": self._duration_uniforms,
        }.get(kind)
        if values is None:
            raise ValueError(f"unknown world-uniform kind={kind!r}")
        if values:
            value = float(values.pop(0))
        else:
            if self._rng is None:
                raise RuntimeError("environment randomness not initialized")
            value = float(self._rng.random())
        key = f"uniform_{kind}_random_numbers"
        self.trace[key].append(value)
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
        # Keep them aligned between the prepared sequence cache and live
        # closed-loop rollout whenever the optional encoder block is enabled.
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
        if self._remaining_duration <= 0:
            z, duration, probabilities = self.model.latent.sample_state_and_duration(
                scene[0], self._latent_state, self._next_uniform("state"), self._next_uniform("duration"),
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
        start_actions = None
        anchor_residual = None
        if self._behavior_anchor_active and self._start_mode_controls is not None:
            start = int(self.trace["response_steps"]) * self.model.cfg.physics_steps_per_response
            start_actions = self._start_mode_controls[:, start : start + self.model.cfg.physics_steps_per_response]
            anchor_residual = self.model._anchor_residual_controls(
                agents, scene, self.model.latent.state_embedding(state_prob),
                torch.as_tensor(self._behavior_anchor_std[None], device=self.device), self._behavior_anchor_initial,
                self._behavior_anchor_generated, start_actions, int(self.trace["response_steps"]), valid & ~ego_mask,
            )
        decoded = self.model._decode_response(
            agents, scene, self.model.latent.state_embedding(state_prob), torch.tensor([self._elapsed], device=self.device),
            valid & ~ego_mask, current, anchor_residual=anchor_residual,
        )
        controls = decoded["controls"]
        b0_plan = self.model._b0_nominal_plan(current, start_actions, ego_index=ego_index)
        control_plan = decoded["control_plan"]
        forecast_control_plan = decoded["forecast_control_plan"]
        if b0_plan is not None:
            control_plan = self.model.decoder._bounded_controls(control_plan + b0_plan)
            forecast_control_plan = self.model.decoder._bounded_controls(forecast_control_plan + b0_plan)
        if self._plan_response_index > self.model.cfg.effective_plan_carry_start_response_steps:
            controls, control_plan = self.model._carry_clean_plan_prefix(
                controls, control_plan,
                None if self._previous_control_plan is None else torch.as_tensor(
                    self._previous_control_plan[None], dtype=controls.dtype, device=controls.device,
                ),
            )
        relation_summary = scene[0].detach().cpu().numpy()
        overlap_diagnostics: dict[str, float | int | bool] = {
            "available": False, "response_index": int(self._plan_response_index),
        }
        if self.model.uses_control_plan and self._previous_control_plan is not None and self._previous_relation_summary is not None:
            overlap = min(
                self._previous_control_plan.shape[0] - self.model.cfg.plan_execute_frames,
                forecast_control_plan.shape[1] - self.model.cfg.plan_execute_frames,
            )
            if overlap > 0:
                current_plan = forecast_control_plan[0].detach().cpu().numpy()
                valid_background = np.asarray(self._valid[1:], bool)
                if self._previous_plan_valid is not None:
                    valid_background &= np.asarray(self._previous_plan_valid[1:], bool)
                difference = np.abs(
                    self._previous_control_plan[self.model.cfg.plan_execute_frames : self.model.cfg.plan_execute_frames + overlap, 1:]
                    - current_plan[:overlap, 1:]
                )
                mask = valid_background[None, :, None]
                relation_change = float(np.linalg.norm(relation_summary - self._previous_relation_summary))
                intent_changed = self._previous_intent_state is not None and int(self._previous_intent_state) != int(self._latent_state)
                weight = float(np.exp(-relation_change / max(float(self.model.cfg.overlap_relation_scale), 1.0e-6)))
                if intent_changed:
                    weight *= 0.5
                overlap_diagnostics = {
                    "available": True,
                    "response_index": int(self._plan_response_index),
                    "frames": int(overlap),
                    "l1": float((difference * mask).sum() / max(float(mask.sum()) * 2.0 * overlap, 1.0)),
                    "weight": weight,
                    "relation_change": relation_change,
                    "intent_changed": bool(intent_changed),
                }
        # The physical ego state is observed only at the current response time;
        # it is held during the internal 25-Hz substeps.  ``roll`` below uses a
        # Constant-velocity extrapolation when no later ego observation exists.
        ego_future = current[:, ego_index : ego_index + 1].expand(-1, self.model.cfg.physics_steps_per_response, -1)
        ego_valid_tensor = valid[:, ego_index : ego_index + 1].expand(-1, self.model.cfg.physics_steps_per_response)
        next_state, physical = self.model._integrate_response(
            current, controls, valid, ego_future, ego_valid_tensor, ego_index=ego_index, start_actions=start_actions,
        )
        outputs = torch.stack(physical, dim=1)[0].cpu().numpy()
        self._states = next_state[0].cpu().numpy()
        self._valid = valid[0].cpu().numpy().astype(bool)
        for state in outputs:
            self._history_states.append(state.copy())
            self._history_valid.append(self._valid.copy())
            if self._behavior_anchor_active:
                self._behavior_anchor_generated.append(torch.as_tensor(state[None], device=self.device))
        self._history_states, self._history_valid = self._history_states[-25:], self._history_valid[-25:]
        self._remaining_duration -= 1
        self.trace["response_steps"] += 1
        if self.trace["response_steps"] >= self.model.cfg.behavior_anchor_response_steps:
            self._behavior_anchor_active = False
            # Enforce non-reachability rather than merely relying on a flag:
            # no B0 representation can leak into later decoder calls.
            self._behavior_anchor = None; self._behavior_anchor_valid = None
            self._behavior_anchor_std = None; self._start_mode_controls = None
            self._behavior_anchor_initial = None; self._behavior_anchor_generated = []
        self.trace["behavior_anchor_active"] = bool(self._behavior_anchor_active)
        response_controls = control_plan[:, : self.model.cfg.physics_steps_per_response].mean(dim=1) if self.model.uses_control_plan else (controls.mean(dim=1) if controls.ndim == 4 else controls)
        highd_actions = self.model.dynamics.highd_actions(response_controls, current)[0].cpu().numpy()
        background_indices = np.flatnonzero(np.arange(len(self._states)) != ego_index)
        applied_controls = control_plan[:, : self.model.cfg.physics_steps_per_response]
        control_curve = applied_controls[0, :, background_indices].cpu().numpy() if self.model.uses_control_plan else response_controls[0, background_indices].unsqueeze(0).expand(
            self.model.cfg.physics_steps_per_response, -1, -1,
        ).cpu().numpy()
        if self.model.uses_control_plan:
            self._previous_control_plan = forecast_control_plan[0].detach().cpu().numpy()
            self._previous_plan_valid = self._valid.copy()
            self._previous_relation_summary = relation_summary.copy()
            self._previous_intent_state = int(self._latent_state)
            self._plan_response_index += 1
        return {
            "agent_states": self._states.copy(), "agent_valid": self._valid.copy(),
            "background_states": outputs[:, background_indices], "background_valid": np.repeat(self._valid[None, background_indices], len(outputs), axis=0),
            "controls": response_controls[0, background_indices].cpu().numpy(), "actions_mps2": highd_actions[background_indices],
            "control_curve": control_curve,
            "control_plan": control_plan[0, :, background_indices].cpu().numpy(),
            "forecast_control_plan": forecast_control_plan[0, :, background_indices].cpu().numpy(),
            "applied_controls": applied_controls[0, :, background_indices].cpu().numpy(),
            "intent_plan": decoded["intent_plan"][0, :, background_indices].cpu().numpy(),
            "local_residual_plan": decoded["local_residual_plan"][0, :, background_indices].cpu().numpy(),
            "b0_nominal_plan": np.zeros_like(control_plan[0, :, background_indices].cpu().numpy()) if b0_plan is None else b0_plan[0, :, background_indices].cpu().numpy(),
            "predicted_plan_states": self.model._predict_plan_states(current, control_plan, valid, ego_index=ego_index)[0, :, background_indices].cpu().numpy(),
            "overlap_diagnostics": overlap_diagnostics,
            "latent_state": int(self._latent_state), "remaining_duration": int(self._remaining_duration),
            "latent_probabilities": None if probabilities is None else probabilities.cpu().numpy(), "trace": dict(self.trace),
        }

    @staticmethod
    def _constant_velocity_ego_extrapolation(state: np.ndarray, steps: int, dt: float) -> list[np.ndarray]:
        current = np.asarray(state, np.float32).copy()
        outputs: list[np.ndarray] = []
        for _ in range(steps):
            current = current.copy()
            current[0] += current[2] * dt
            current[1] += current[3] * dt
            outputs.append(current)
        return outputs

    def roll(self, ego_history_states: np.ndarray, ego_history_valid: np.ndarray) -> dict[str, Any]:
        """One-second compatibility wrapper executing five response updates."""
        history = np.asarray(ego_history_states, np.float32)
        valid = np.asarray(ego_history_valid, bool)
        if history.shape[0] < 1 or valid.shape[0] != history.shape[0]:
            raise ValueError("ego history must be [H, state_dim] with matching validity")
        current = history[-1]
        chunks = []
        for next_ego in self._constant_velocity_ego_extrapolation(current, 5, self.model.cfg.response_interval_s):
            result = self.step(next_ego, bool(valid[-1]))
            chunks.append(result)
            current = next_ego
        return {
            "actions_mps2": np.concatenate([item["actions_mps2"][None].repeat(self.model.cfg.physics_steps_per_response, axis=0) for item in chunks], axis=0),
            "background_states": np.concatenate([item["background_states"] for item in chunks], axis=0),
            "background_valid": np.concatenate([item["background_valid"] for item in chunks], axis=0),
            "latent_trace": dict(self.trace),
        }
