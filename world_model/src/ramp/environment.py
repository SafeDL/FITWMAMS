"""Replayable RAMP-WM runtime environment with explicit plan uniforms."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from world_model.src.traffic_graph.graph_schema import DynamicTrafficGraph
from .model import RAMPWorldModel


@dataclass
class RAMPWorldRandomness:
    seed: int = 123
    plan_uniforms: list[float] = field(default_factory=list)
    model_checkpoint_hash: str | None = None
    behavior_anchor_raw: np.ndarray | None = None
    behavior_anchor_valid: np.ndarray | None = None


class RAMPBackgroundEnvironment:
    """0.2 s response environment; state snapshots replay every candidate choice."""

    def __init__(
        self, model: RAMPWorldModel, *, device: str | torch.device = "cpu"
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.graph: DynamicTrafficGraph | None = None
        self._rng: np.random.Generator | None = None
        self._uniforms: list[float] = []
        self._states: np.ndarray | None = None
        self._valid: np.ndarray | None = None
        self._history: list[np.ndarray] = []
        self._history_valid: list[np.ndarray] = []
        self._memory: torch.Tensor | None = None
        self._previous_plan: torch.Tensor | None = None
        self._previous_current: torch.Tensor | None = None
        self._previous_scene: torch.Tensor | None = None
        self._behavior_anchor_raw: np.ndarray | None = None
        self._behavior_anchor_valid: np.ndarray | None = None
        self.response_index = 0
        self.trace: dict[str, Any] = {}

    def reset(
        self,
        initial_graph: DynamicTrafficGraph,
        world_randomness: RAMPWorldRandomness | None = None,
    ) -> dict[str, Any]:
        random = world_randomness or RAMPWorldRandomness()
        self.graph = deepcopy(initial_graph)
        self._states = np.asarray(initial_graph.agent_states, np.float32).copy()
        self._valid = np.asarray(initial_graph.agent_valid, bool).copy()
        self._rng = np.random.default_rng(int(random.seed))
        self._uniforms = list(random.plan_uniforms)
        self._history = [self._states.copy() for _ in range(25)]
        self._history_valid = [self._valid.copy() for _ in range(25)]
        self._memory = self._previous_plan = self._previous_current = (
            self._previous_scene
        ) = None
        self.response_index = 0
        self._behavior_anchor_raw = (
            None
            if random.behavior_anchor_raw is None
            else np.asarray(random.behavior_anchor_raw, np.float32).copy()
        )
        self._behavior_anchor_valid = (
            None
            if random.behavior_anchor_valid is None
            else np.asarray(random.behavior_anchor_valid, bool).copy()
        )
        self.trace = {
            "world_random_seed": int(random.seed),
            "plan_uniform_random_numbers": [],
            "candidate_indices": [],
            "candidate_probabilities": [],
            "response_steps": 0,
            "model_checkpoint_hash": str(
                random.model_checkpoint_hash
                or getattr(self.model, "checkpoint_hash", "unbound_model")
            ),
        }
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        if self._states is None or self._valid is None:
            raise RuntimeError("environment has not been reset")
        return {
            "graph": deepcopy(self.graph),
            "agent_states": self._states.copy(),
            "agent_valid": self._valid.copy(),
            "history_states": [x.copy() for x in self._history],
            "history_valid": [x.copy() for x in self._history_valid],
            "continuous_memory": (
                None if self._memory is None else self._memory.detach().cpu().numpy()
            ),
            "previous_selected_plan": (
                None
                if self._previous_plan is None
                else self._previous_plan.detach().cpu().numpy()
            ),
            "previous_current": (
                None
                if self._previous_current is None
                else self._previous_current.detach().cpu().numpy()
            ),
            "previous_relation_summary": (
                None
                if self._previous_scene is None
                else self._previous_scene.detach().cpu().numpy()
            ),
            "previous_candidate_index": (
                self.trace["candidate_indices"][-1]
                if self.trace["candidate_indices"]
                else None
            ),
            "previous_candidate_probabilities": (
                self.trace["candidate_probabilities"][-1]
                if self.trace["candidate_probabilities"]
                else None
            ),
            "behavior_anchor_state": {
                "raw": (
                    None
                    if self._behavior_anchor_raw is None
                    else self._behavior_anchor_raw.copy()
                ),
                "valid": (
                    None
                    if self._behavior_anchor_valid is None
                    else self._behavior_anchor_valid.copy()
                ),
            },
            "plan_uniforms_remaining": list(self._uniforms),
            "rng_state": (
                None if self._rng is None else deepcopy(self._rng.bit_generator.state)
            ),
            "response_index": self.response_index,
            "trace": deepcopy(self.trace),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.graph = deepcopy(snapshot["graph"])
        self._states = np.asarray(snapshot["agent_states"], np.float32).copy()
        self._valid = np.asarray(snapshot["agent_valid"], bool).copy()
        self._history = [
            np.asarray(x, np.float32).copy() for x in snapshot["history_states"]
        ]
        self._history_valid = [
            np.asarray(x, bool).copy() for x in snapshot["history_valid"]
        ]
        self._memory = (
            None
            if snapshot.get("continuous_memory") is None
            else torch.as_tensor(snapshot["continuous_memory"], device=self.device)
        )
        self._previous_plan = (
            None
            if snapshot.get("previous_selected_plan") is None
            else torch.as_tensor(snapshot["previous_selected_plan"], device=self.device)
        )
        self._previous_current = (
            None
            if snapshot.get("previous_current") is None
            else torch.as_tensor(snapshot["previous_current"], device=self.device)
        )
        self._previous_scene = (
            None
            if snapshot.get("previous_relation_summary") is None
            else torch.as_tensor(
                snapshot["previous_relation_summary"], device=self.device
            )
        )
        anchor = snapshot.get("behavior_anchor_state", {})
        self._behavior_anchor_raw = (
            None
            if anchor.get("raw") is None
            else np.asarray(anchor["raw"], np.float32).copy()
        )
        self._behavior_anchor_valid = (
            None
            if anchor.get("valid") is None
            else np.asarray(anchor["valid"], bool).copy()
        )
        self._uniforms = [float(x) for x in snapshot.get("plan_uniforms_remaining", [])]
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = deepcopy(snapshot["rng_state"])
        self.response_index = int(snapshot["response_index"])
        self.trace = deepcopy(snapshot["trace"])

    def _uniform(self) -> float:
        value = (
            float(self._uniforms.pop(0))
            if self._uniforms
            else float(self._rng.random())
        )
        self.trace["plan_uniform_random_numbers"].append(value)
        return value

    def step(self, ego_state: np.ndarray, ego_valid: bool = True) -> dict[str, Any]:
        if self.graph is None or self._states is None or self._valid is None:
            raise RuntimeError("environment has not been reset")
        states = torch.as_tensor(self._states[None], device=self.device)
        valid = torch.as_tensor(self._valid[None], device=self.device)
        ego_index = int(self.graph.ego_index)
        ego = torch.nn.functional.one_hot(
            torch.tensor([ego_index], device=self.device), states.shape[1]
        ).bool()
        batch = {
            "agent_states": states[:, None].expand(-1, 25, -1, -1),
            "ego_index": torch.tensor([ego_index], device=self.device),
            "map_polylines": torch.as_tensor(
                self.graph.map_polylines[None], device=self.device
            ),
            "map_polyline_valid": torch.as_tensor(
                self.graph.map_polyline_valid[None], device=self.device
            ),
            "lane_graph_edges": torch.as_tensor(
                self.graph.lane_graph_edges[None], device=self.device
            ),
        }
        if self._behavior_anchor_raw is not None:
            batch["behavior_anchor_raw"] = torch.as_tensor(
                self._behavior_anchor_raw[None], device=self.device
            )
            if self._behavior_anchor_valid is not None:
                batch["behavior_anchor_valid"] = torch.as_tensor(
                    self._behavior_anchor_valid[None], device=self.device
                )
        with torch.no_grad():
            out = self.model.plan_step(
                torch.as_tensor(np.stack(self._history)[None], device=self.device),
                torch.as_tensor(
                    np.stack(self._history_valid)[None], device=self.device
                ),
                states,
                valid,
                ego,
                batch,
                self._memory,
                self._previous_plan,
                self._previous_current,
                self.response_index,
            )
            probs = out["candidate_probabilities"][0]
            index = int(
                torch.searchsorted(
                    probs.cumsum(0), torch.tensor(self._uniform(), device=self.device)
                ).clamp_max(self.model.cfg.num_candidates - 1)
            )
            plan = out["candidate_control_plans"][0, index]
            current = states
            generated = []
            for frame in range(self.model.cfg.execute_frames):
                control = current.new_zeros((1, current.shape[1], 2))
                control[:, 1:] = plan[frame]
                current = self.model.dynamics.step(
                    current, control, valid, self.model.cfg.simulation_dt_s
                )
                current[:, ego_index] = torch.as_tensor(ego_state, device=self.device)
                valid[:, ego_index] = bool(ego_valid)
                generated.append(current[0].cpu().numpy())
        self._states, self._valid = current[0].cpu().numpy(), valid[
            0
        ].cpu().numpy().astype(bool)
        self._history.extend(generated)
        self._history_valid.extend([self._valid.copy()] * len(generated))
        self._history, self._history_valid = (
            self._history[-25:],
            self._history_valid[-25:],
        )
        (
            self._memory,
            self._previous_plan,
            self._previous_current,
            self._previous_scene,
        ) = (out["continuous_memory_next"], plan[None], states, out["scene_context"])
        self.response_index += 1
        self.trace["candidate_indices"].append(index)
        self.trace["candidate_probabilities"].append(probs.cpu().numpy().tolist())
        self.trace["response_steps"] = self.response_index
        return {
            "agent_states": self._states.copy(),
            "agent_valid": self._valid.copy(),
            "background_states": np.stack(generated)[:, 1:],
            "candidate_index": index,
            "candidate_probabilities": probs.cpu().numpy(),
            "selected_control_plan": plan.cpu().numpy(),
            "applied_controls": plan[: self.model.cfg.execute_frames].cpu().numpy(),
            "continuous_memory": self._memory.cpu().numpy(),
            "trace": deepcopy(self.trace),
        }
