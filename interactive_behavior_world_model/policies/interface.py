"""Policy boundary shared by every method in benchmark v1.

The evaluator owns logged futures.  Deliberately, none of the objects exposed
to a policy contains such a field.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class RandomKey:
    benchmark_id: str
    scenario_id: str
    fit_seed: int
    rollout_id: int
    stream_name: str = "policy"
    decision_index: int = 0
    agent_id: int = -1

    def seed(self) -> int:
        """Return a stable NumPy-compatible seed without Python hash salt."""
        import hashlib

        payload = "|".join(
            str(value)
            for value in (
                self.benchmark_id,
                self.scenario_id,
                self.fit_seed,
                self.rollout_id,
                self.stream_name,
                self.decision_index,
                self.agent_id,
            )
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


@dataclass(frozen=True)
class PolicyObservation:
    """Causal state available at the beginning of one 5 Hz decision."""

    states: np.ndarray
    valid: np.ndarray
    agent_ids: np.ndarray
    lengths_m: np.ndarray
    widths_m: np.ndarray
    map_polylines: np.ndarray
    map_polyline_valid: np.ndarray
    history_states: np.ndarray
    history_valid: np.ndarray
    previous_applied_action: np.ndarray
    decision_index: int
    time_s: float

    def validate(self) -> None:
        states = np.asarray(self.states)
        valid = np.asarray(self.valid)
        ids = np.asarray(self.agent_ids)
        if states.ndim != 2 or states.shape[-1] != 6:
            raise ValueError("states must be [agents,6]")
        if valid.shape != states.shape[:1] or ids.shape != states.shape[:1]:
            raise ValueError("valid and agent_ids must align with states")
        if np.asarray(self.history_states).shape[1:] != states.shape:
            raise ValueError("history_states must be [history,agents,6]")
        if (
            np.asarray(self.history_valid).shape
            != np.asarray(self.history_states).shape[:2]
        ):
            raise ValueError("history_valid must align with history_states")
        if np.asarray(self.previous_applied_action).shape != (len(states), 2):
            raise ValueError("previous_applied_action must be [agents,2]")


@dataclass(frozen=True)
class PolicyAction:
    requested_action: np.ndarray
    state: Any = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, num_agents: int) -> None:
        action = np.asarray(self.requested_action)
        if action.shape != (int(num_agents), 2):
            raise ValueError(f"requested_action must have shape ({num_agents},2)")
        if not np.isfinite(action).all():
            raise ValueError("requested_action contains NaN or Inf")


class NPCPolicy(ABC):
    """Stateful, snapshot-capable causal NPC policy."""

    @abstractmethod
    def reset(
        self,
        history: np.ndarray,
        history_valid: np.ndarray,
        agent_ids: np.ndarray,
        map_context: Mapping[str, np.ndarray],
        random_key: RandomKey,
        behavior_condition: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize from the common prefix only."""

    @abstractmethod
    def act(self, observation: PolicyObservation) -> PolicyAction:
        """Return requested ``[acceleration, yaw_rate]`` for every NPC."""

    @abstractmethod
    def snapshot(self) -> Any:
        """Return all mutable policy and random-stream state."""

    @abstractmethod
    def restore(self, snapshot: Any) -> None:
        """Restore a previously returned snapshot exactly."""
