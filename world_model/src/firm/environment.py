"""Replayable FIRM-WM environment with persistent world randomness."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from world_model.src.traffic_graph.graph_schema import DynamicTrafficGraph

from .model import FIRMWorldModel


@dataclass
class FIRMWorldRandomness:
    seed: int = 123
    world_noise: np.ndarray | None = None
    innovation_noises: list[np.ndarray] = field(default_factory=list)
    action_noises: list[np.ndarray] = field(default_factory=list)
    behavior_anchor_raw: np.ndarray | None = None
    behavior_anchor_valid: np.ndarray | None = None
    flow_latent: np.ndarray | None = None
    model_checkpoint_hash: str | None = None


class FIRMBackgroundEnvironment:
    """Closed-loop 0.2 s environment with replayable continuous randomness."""

    def __init__(self, model: FIRMWorldModel, *, device: str | torch.device = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.graph: DynamicTrafficGraph | None = None
        self._rng: np.random.Generator | None = None
        self._states: np.ndarray | None = None
        self._valid: np.ndarray | None = None
        self._history: list[np.ndarray] = []
        self._history_valid: list[np.ndarray] = []
        self._memory: torch.Tensor | None = None
        self._world_latent: torch.Tensor | None = None
        self._flow_embedding: torch.Tensor | None = None
        self._previous_plan: torch.Tensor | None = None
        self._previous_current: torch.Tensor | None = None
        self._behavior_anchor_raw: np.ndarray | None = None
        self._behavior_anchor_valid: np.ndarray | None = None
        self._world_noise: np.ndarray | None = None
        self._innovation_noises: list[np.ndarray] = []
        self._action_noises: list[np.ndarray] = []
        self.response_index = 0
        self.trace: dict[str, Any] = {}

    def reset(
        self,
        initial_graph: DynamicTrafficGraph,
        world_randomness: FIRMWorldRandomness | None = None,
    ) -> dict[str, Any]:
        """Initialize from C0, B0, zF and zW without synthesizing a history."""
        random = world_randomness or FIRMWorldRandomness()
        self.graph = deepcopy(initial_graph)
        self._states = np.asarray(initial_graph.agent_states, np.float32).copy()
        self._valid = np.asarray(initial_graph.agent_valid, bool).copy()
        if int(initial_graph.ego_index) != 0:
            raise ValueError("FIRM-WM requires the fixed highD ego slot at index 0")
        self._rng = np.random.default_rng(int(random.seed))
        self._history = [self._states.copy()]
        self._history_valid = [self._valid.copy()]
        self._behavior_anchor_raw = None if random.behavior_anchor_raw is None else np.asarray(random.behavior_anchor_raw, np.float32).copy()
        self._behavior_anchor_valid = None if random.behavior_anchor_valid is None else np.asarray(random.behavior_anchor_valid, bool).copy()
        self._world_noise = None if random.world_noise is None else np.asarray(random.world_noise, np.float32).copy()
        self._innovation_noises = [np.asarray(item, np.float32).copy() for item in random.innovation_noises]
        self._action_noises = [np.asarray(item, np.float32).copy() for item in random.action_noises]
        states = torch.as_tensor(self._states[None], device=self.device)
        valid = torch.as_tensor(self._valid[None], device=self.device)
        ego = torch.nn.functional.one_hot(
            torch.tensor([initial_graph.ego_index], device=self.device), states.shape[1]
        ).bool()
        noise = (
            torch.randn((1, self.model.cfg.world_latent_dim), device=self.device)
            if self._world_noise is None
            else torch.as_tensor(self._world_noise[None], device=self.device)
        )
        with torch.no_grad():
            start = self.model.initialize(
                states,
                valid,
                ego,
                behavior_anchor=(
                    None
                    if self._behavior_anchor_raw is None
                    else torch.as_tensor(self._behavior_anchor_raw[None], device=self.device)
                ),
                behavior_anchor_valid=(
                    None
                    if self._behavior_anchor_valid is None
                    else torch.as_tensor(self._behavior_anchor_valid[None], device=self.device)
                ),
                flow_latent=(
                    None
                    if random.flow_latent is None
                    else torch.as_tensor(random.flow_latent[None], device=self.device)
                ),
                world_noise=noise,
            )
        self._memory = start["continuous_memory"]
        self._world_latent = start["world_latent"]
        self._flow_embedding = start["flow_embedding"]
        self._previous_plan = self._previous_current = None
        self.response_index = 0
        self.trace = {
            "world_random_seed": int(random.seed),
            "world_noise": noise[0].cpu().numpy().tolist(),
            "flow_latent": None if random.flow_latent is None else np.asarray(random.flow_latent, np.float32).tolist(),
            "action_flow_noises": [],
            "innovation_noises": [],
            "response_steps": 0,
            "model_checkpoint_hash": str(random.model_checkpoint_hash or getattr(self.model, "checkpoint_hash", "unbound_model")),
        }
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        if self._states is None or self._valid is None or self._world_latent is None or self._flow_embedding is None:
            raise RuntimeError("environment has not been reset")
        return {
            "graph": deepcopy(self.graph),
            "agent_states": self._states.copy(),
            "agent_valid": self._valid.copy(),
            "history_states": [item.copy() for item in self._history],
            "history_valid": [item.copy() for item in self._history_valid],
            "continuous_memory": self._memory.detach().cpu().numpy(),
            "world_latent": self._world_latent.detach().cpu().numpy(),
            "flow_embedding": self._flow_embedding.detach().cpu().numpy(),
            "previous_selected_plan": None if self._previous_plan is None else self._previous_plan.detach().cpu().numpy(),
            "previous_current": None if self._previous_current is None else self._previous_current.detach().cpu().numpy(),
            "behavior_anchor_raw": None if self._behavior_anchor_raw is None else self._behavior_anchor_raw.copy(),
            "behavior_anchor_valid": None if self._behavior_anchor_valid is None else self._behavior_anchor_valid.copy(),
            "world_noise": None if self._world_noise is None else self._world_noise.copy(),
            "innovation_noises_remaining": [item.copy() for item in self._innovation_noises],
            "action_noises_remaining": [item.copy() for item in self._action_noises],
            "rng_state": None if self._rng is None else deepcopy(self._rng.bit_generator.state),
            "response_index": self.response_index,
            "trace": deepcopy(self.trace),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.graph = deepcopy(snapshot["graph"])
        self._states = np.asarray(snapshot["agent_states"], np.float32).copy()
        self._valid = np.asarray(snapshot["agent_valid"], bool).copy()
        self._history = [np.asarray(item, np.float32).copy() for item in snapshot["history_states"]]
        self._history_valid = [np.asarray(item, bool).copy() for item in snapshot["history_valid"]]
        self._memory = torch.as_tensor(snapshot["continuous_memory"], device=self.device)
        self._world_latent = torch.as_tensor(snapshot["world_latent"], device=self.device)
        self._flow_embedding = torch.as_tensor(snapshot["flow_embedding"], device=self.device)
        self._previous_plan = None if snapshot.get("previous_selected_plan") is None else torch.as_tensor(snapshot["previous_selected_plan"], device=self.device)
        self._previous_current = None if snapshot.get("previous_current") is None else torch.as_tensor(snapshot["previous_current"], device=self.device)
        self._behavior_anchor_raw = None if snapshot.get("behavior_anchor_raw") is None else np.asarray(snapshot["behavior_anchor_raw"], np.float32).copy()
        self._behavior_anchor_valid = None if snapshot.get("behavior_anchor_valid") is None else np.asarray(snapshot["behavior_anchor_valid"], bool).copy()
        self._world_noise = None if snapshot.get("world_noise") is None else np.asarray(snapshot["world_noise"], np.float32).copy()
        self._innovation_noises = [np.asarray(item, np.float32).copy() for item in snapshot.get("innovation_noises_remaining", [])]
        self._action_noises = [np.asarray(item, np.float32).copy() for item in snapshot.get("action_noises_remaining", [])]
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = deepcopy(snapshot["rng_state"])
        self.response_index = int(snapshot["response_index"])
        self.trace = deepcopy(snapshot["trace"])

    def _action_noise(self) -> np.ndarray:
        if self._action_noises:
            return self._action_noises.pop(0)
        if self._rng is None:
            raise RuntimeError("environment random generator is unavailable")
        return self._rng.standard_normal((self.model.cfg.execute_frames, 6, 2)).astype(np.float32)

    def _innovation_noise(self) -> np.ndarray:
        if self._innovation_noises:
            return self._innovation_noises.pop(0)
        if self._rng is None:
            raise RuntimeError("environment random generator is unavailable")
        return self._rng.standard_normal(self.model.cfg.world_latent_dim).astype(np.float32)

    def step(self, ego_state: np.ndarray, ego_valid: bool = True) -> dict[str, Any]:
        if self.graph is None or self._states is None or self._valid is None:
            raise RuntimeError("environment has not been reset")
        if self._memory is None or self._world_latent is None or self._flow_embedding is None:
            raise RuntimeError("environment latent state is unavailable")
        states = torch.as_tensor(self._states[None], device=self.device)
        valid = torch.as_tensor(self._valid[None], device=self.device)
        ego_index = int(self.graph.ego_index)
        ego = torch.nn.functional.one_hot(
            torch.tensor([ego_index], device=self.device), states.shape[1]
        ).bool()
        action_noise = self._action_noise()
        innovation = self._innovation_noise()
        with torch.no_grad():
            out = self.model.plan_step(
                torch.as_tensor(np.stack(self._history)[None], device=self.device),
                torch.as_tensor(np.stack(self._history_valid)[None], device=self.device),
                states,
                valid,
                ego,
                self._memory,
                self._world_latent,
                self._flow_embedding,
                self._previous_plan,
                self._previous_current,
                torch.as_tensor(action_noise[None], device=self.device),
            )
            plan = out["joint_control_plan"][0]
            current = states
            generated: list[np.ndarray] = []
            current_valid = valid
            for frame in range(self.model.cfg.execute_frames):
                control = current.new_zeros((1, current.shape[1], 2))
                control[:, 1:] = plan[frame]
                current = self.model.dynamics.step(
                    current, control, current_valid, self.model.cfg.simulation_dt_s
                )
                current[:, ego_index] = torch.as_tensor(ego_state, device=self.device)
                current_valid[:, ego_index] = bool(ego_valid)
                generated.append(current[0].cpu().numpy())
            next_latent, rho, sigma = self.model.world_latent(
                out["scene_context"],
                out["continuous_memory_next"],
                self._world_latent,
                torch.as_tensor(innovation[None], device=self.device),
            )
        self._states = current[0].cpu().numpy()
        self._valid = current_valid[0].cpu().numpy().astype(bool)
        self._history.extend(generated)
        self._history_valid.extend([self._valid.copy()] * len(generated))
        self._history, self._history_valid = self._history[-25:], self._history_valid[-25:]
        self._memory = out["continuous_memory_next"]
        self._world_latent = next_latent
        self._previous_plan = plan[None]
        self._previous_current = states
        self.response_index += 1
        self.trace["action_flow_noises"].append(action_noise.tolist())
        self.trace["innovation_noises"].append(innovation.tolist())
        self.trace["response_steps"] = self.response_index
        return {
            "agent_states": self._states.copy(),
            "agent_valid": self._valid.copy(),
            "background_states": np.stack(generated)[:, 1:],
            "selected_control_plan": plan.cpu().numpy(),
            "applied_controls": plan[: self.model.cfg.execute_frames].cpu().numpy(),
            "continuous_memory": self._memory.cpu().numpy(),
            "world_latent": self._world_latent.cpu().numpy(),
            "world_latent_rho": rho.cpu().numpy(),
            "world_latent_sigma": sigma.cpu().numpy(),
            "trace": deepcopy(self.trace),
        }
