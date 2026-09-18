"""A small causal 25 Hz boundary shared by all retained driver reproductions."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LongitudinalObservation:
    """Frame-start state available to a longitudinal background driver."""

    gap_m: float
    ego_speed_mps: float
    leader_speed_mps: float
    leader_acceleration_mps2: float = 0.0

    @classmethod
    def from_project_states(
        cls,
        follower_state: np.ndarray,
        leader_state: np.ndarray,
        *,
        length_sum_m: float,
    ) -> "LongitudinalObservation":
        """Convert the CIH-WM ``[x,y,vx,vy,ax,ay]`` state convention."""
        follower = np.asarray(follower_state, float)
        leader = np.asarray(leader_state, float)
        return cls(
            gap_m=float(leader[0] - follower[0] - length_sum_m),
            ego_speed_mps=float(np.linalg.norm(follower[2:4])),
            leader_speed_mps=float(np.linalg.norm(leader[2:4])),
            leader_acceleration_mps2=float(leader[4]),
        )


@dataclass(frozen=True)
class DriverCommand:
    """One plant-tick command; ``updated`` marks a new 5 Hz decision."""

    acceleration_mps2: float
    requested_acceleration_mps2: float
    updated: bool
    native_frame: int
    decision_index: int
    model_id: str


@dataclass
class _SessionSnapshot:
    driver: Any
    native_frame: int
    decision_index: int
    held_requested: float
    held_applied: float


class DriverSession:
    """Run a retained 5 Hz driver inside the project's 25 Hz plant contract.

    The session owns action holding and safety clipping.  Neither operation is
    written back into GP/AR/regime state.  Create a new session per simulated
    driver episode so the factory can draw a new joint posterior parameter set.
    """

    def __init__(
        self,
        model_id: str,
        driver: Any,
        *,
        native_dt_s: float = 0.04,
        decision_dt_s: float = 0.2,
        min_acceleration_mps2: float = -8.0,
        max_acceleration_mps2: float = 4.0,
        kind: str = "idm",
    ) -> None:
        ratio = decision_dt_s / native_dt_s
        self.hold_frames = int(round(ratio))
        if self.hold_frames <= 0 or not np.isclose(ratio, self.hold_frames):
            raise ValueError("decision_dt_s must be an integer number of native ticks")
        self.model_id = model_id
        self.driver = driver
        self.kind = kind
        self.min_acceleration_mps2 = float(min_acceleration_mps2)
        self.max_acceleration_mps2 = float(max_acceleration_mps2)
        self.native_frame = 0
        self.decision_index = 0
        self.held_requested = 0.0
        self.held_applied = 0.0

    def reset(
        self,
        observation: LongitudinalObservation,
        *,
        seed: int = 0,
        prefix_observations: np.ndarray | None = None,
    ) -> None:
        self.native_frame = 0
        self.decision_index = 0
        self.held_requested = 0.0
        self.held_applied = 0.0
        if self.kind == "multi_regime":
            prefix = (
                np.asarray(prefix_observations, float)
                if prefix_observations is not None
                else np.asarray([[observation.gap_m, observation.ego_speed_mps,
                                  observation.ego_speed_mps - observation.leader_speed_mps]])
            )
            self.driver.reset(prefix, seed=seed)
        elif self.kind == "active_inference":
            self.driver.reset(
                gap_m=max(0.05, observation.gap_m),
                ego_speed_mps=observation.ego_speed_mps,
                leader_speed_mps=observation.leader_speed_mps,
                leader_acceleration_mps2=observation.leader_acceleration_mps2,
                seed=seed,
            )
        elif self.kind == "official_active_inference":
            self.driver.reset(
                gap_m=max(0.05, observation.gap_m),
                ego_speed_mps=observation.ego_speed_mps,
                target_speed_mps=observation.leader_speed_mps,
                seed=seed,
            )
        else:
            self.driver.reset(seed=seed)

    def _decision(self, observation: LongitudinalObservation) -> float:
        gap = max(0.05, observation.gap_m)
        if self.kind == "multi_regime":
            if self.decision_index:
                self.driver.observe(
                    gap_m=gap,
                    speed_mps=observation.ego_speed_mps,
                    leader_speed_mps=observation.leader_speed_mps,
                )
            return float(self.driver.decision(
                gap_m=gap,
                speed_mps=observation.ego_speed_mps,
                leader_speed_mps=observation.leader_speed_mps,
            ).requested_acceleration_mps2)
        if self.kind == "active_inference":
            return float(self.driver.decide(
                gap_m=gap,
                ego_speed_mps=observation.ego_speed_mps,
                leader_speed_mps=observation.leader_speed_mps,
            ).action)
        if self.kind == "official_active_inference":
            value = self.driver.observation(
                ego_x_m=0.0,
                ego_speed_mps=observation.ego_speed_mps,
                target_x_m=gap + 4.2,
                target_speed_mps=observation.leader_speed_mps,
                target_acceleration_mps2=observation.leader_acceleration_mps2,
            )
            return float(self.driver.step(value).acceleration_mps2[0])
        result = self.driver.decision(
            gap_m=gap,
            speed_mps=observation.ego_speed_mps,
            leader_speed_mps=observation.leader_speed_mps,
        )
        return float(getattr(result, "requested_acceleration_mps2", result))

    def step(self, observation: LongitudinalObservation) -> DriverCommand:
        """Consume one frame-start observation and return one 25 Hz command."""
        updated = self.native_frame % self.hold_frames == 0
        if updated:
            self.held_requested = self._decision(observation)
            self.held_applied = float(np.clip(
                self.held_requested,
                self.min_acceleration_mps2,
                self.max_acceleration_mps2,
            ))
            self.decision_index += 1
        command = DriverCommand(
            acceleration_mps2=self.held_applied,
            requested_acceleration_mps2=self.held_requested,
            updated=updated,
            native_frame=self.native_frame,
            decision_index=self.decision_index - 1,
            model_id=self.model_id,
        )
        self.native_frame += 1
        return command

    def snapshot(self) -> _SessionSnapshot:
        """Copy NumPy-driver state for matched counterfactual branches."""
        if self.kind == "official_active_inference":
            raise NotImplementedError(
                "the external official PyTorch agent is replayed by reset+seed, not deep-copied"
            )
        return _SessionSnapshot(
            copy.deepcopy(self.driver), self.native_frame, self.decision_index,
            self.held_requested, self.held_applied,
        )

    def restore(self, snapshot: _SessionSnapshot) -> None:
        self.driver = copy.deepcopy(snapshot.driver)
        self.native_frame = int(snapshot.native_frame)
        self.decision_index = int(snapshot.decision_index)
        self.held_requested = float(snapshot.held_requested)
        self.held_applied = float(snapshot.held_applied)
