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
        """Convert the world model's ``[x,y,vx,vy,ax,ay]`` state convention."""
        follower = np.asarray(follower_state, float)
        leader = np.asarray(leader_state, float)
        return cls(
            gap_m=float(leader[0] - follower[0] - length_sum_m),
            ego_speed_mps=float(np.linalg.norm(follower[2:4])),
            leader_speed_mps=float(np.linalg.norm(leader[2:4])),
            leader_acceleration_mps2=float(leader[4]),
        )


@dataclass(frozen=True)
class LongitudinalPrefix:
    """Completed causal 5 Hz actions observed before a rollout origin.

    Every row describes a decision made at ``time_s`` from the corresponding
    gap and speeds, followed by the acceleration that was realized over the
    complete decision interval.  Consequently the final row must end no later
    than the rollout origin; no future sample is needed to construct it.
    """

    time_s: np.ndarray
    gap_m: np.ndarray
    ego_speed_mps: np.ndarray
    leader_speed_mps: np.ndarray
    acceleration_mps2: np.ndarray
    decision_dt_s: float = 0.2

    def __post_init__(self) -> None:
        arrays = tuple(
            np.asarray(getattr(self, name), dtype=float)
            for name in (
                "time_s",
                "gap_m",
                "ego_speed_mps",
                "leader_speed_mps",
                "acceleration_mps2",
            )
        )
        size = len(arrays[0])
        if any(value.ndim != 1 or len(value) != size for value in arrays):
            raise ValueError("longitudinal prefix fields must be equal-length vectors")
        if size and (not all(np.isfinite(value).all() for value in arrays)):
            raise ValueError("longitudinal prefix fields must be finite")
        if size > 1 and np.any(np.diff(arrays[0]) <= 0.0):
            raise ValueError("longitudinal prefix timestamps must be strictly increasing")
        if float(self.decision_dt_s) <= 0.0:
            raise ValueError("decision_dt_s must be positive")
        for name, value in zip(
            ("time_s", "gap_m", "ego_speed_mps", "leader_speed_mps", "acceleration_mps2"),
            arrays,
        ):
            object.__setattr__(self, name, value)

    @classmethod
    def from_native_series(
        cls,
        *,
        gap_m: np.ndarray,
        ego_speed_mps: np.ndarray,
        leader_speed_mps: np.ndarray,
        native_dt_s: float = 0.04,
        decision_dt_s: float = 0.2,
        memory_s: float | None = 5.0,
    ) -> "LongitudinalPrefix":
        """Build prefix rows using only completed native-rate transitions."""
        gap = np.asarray(gap_m, float)
        speed = np.asarray(ego_speed_mps, float)
        leader = np.asarray(leader_speed_mps, float)
        if gap.ndim != 1 or speed.shape != gap.shape or leader.shape != gap.shape:
            raise ValueError("native prefix series must be equal-length vectors")
        ratio = float(decision_dt_s) / float(native_dt_s)
        stride = int(round(ratio))
        if stride <= 0 or not np.isclose(ratio, stride):
            raise ValueError("decision_dt_s must be an integer number of native ticks")
        if len(gap) <= stride:
            return cls(*(np.empty(0, float) for _ in range(5)), decision_dt_s)
        first = 0
        if memory_s is not None:
            first = max(0, len(gap) - 1 - int(round(float(memory_s) / native_dt_s)))
            first += (-first) % stride
        indices = np.arange(first, len(gap) - stride, stride, dtype=int)
        acceleration = (speed[indices + stride] - speed[indices]) / float(decision_dt_s)
        return cls(
            indices.astype(float) * float(native_dt_s),
            gap[indices],
            speed[indices],
            leader[indices],
            acceleration,
            float(decision_dt_s),
        )

    def as_array(self) -> np.ndarray:
        return np.column_stack(
            (
                self.time_s,
                self.gap_m,
                self.ego_speed_mps,
                self.leader_speed_mps,
                self.acceleration_mps2,
            )
        )

    def multi_regime_observations(self) -> np.ndarray:
        return np.column_stack(
            (
                self.gap_m,
                self.ego_speed_mps,
                self.ego_speed_mps - self.leader_speed_mps,
            )
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
        metadata: dict[str, Any] | None = None,
        default_prefix_observations: LongitudinalPrefix | np.ndarray | None = None,
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
        self.metadata = dict(metadata or {})
        self.default_prefix_observations = default_prefix_observations
        self.native_frame = 0
        self.decision_index = 0
        self.held_requested = 0.0
        self.held_applied = 0.0

    def reset(
        self,
        observation: LongitudinalObservation,
        *,
        seed: int = 0,
        prefix_observations: LongitudinalPrefix | np.ndarray | None = None,
    ) -> None:
        self.native_frame = 0
        self.decision_index = 0
        self.held_requested = 0.0
        self.held_applied = 0.0
        if prefix_observations is None:
            prefix_observations = self.default_prefix_observations
        if self.kind == "multi_regime":
            prefix = (
                prefix_observations.multi_regime_observations()
                if isinstance(prefix_observations, LongitudinalPrefix)
                else np.asarray(prefix_observations, float)
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
        elif self.kind == "bayesian_idm":
            prefix = (
                prefix_observations.as_array()
                if isinstance(prefix_observations, LongitudinalPrefix)
                else prefix_observations
            )
            self.driver.reset(seed=seed, prefix_observations=prefix)
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
        """Copy NumPy-driver state for deterministic branch replay."""
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
