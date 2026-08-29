"""Batched IDM evaluation in the maintained hierarchical traffic world."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hierarchical_world_model.src.composition import HierarchicalWorldSampler
from hierarchical_world_model.src.execution import rollout_world
from hierarchical_world_model.src.randomness import WorldExogenousState
from tools.evt import GPDTailModel

from .idm_policy import HighwayEnvIDMPolicy
from .metrics import collision_and_min_gap


@dataclass(frozen=True)
class WorldEvaluation:
    """Per-world EVT, collision, gap and numerical-validity observations."""

    evt_score: np.ndarray
    event_risk: np.ndarray
    collision: np.ndarray
    min_gap_m: np.ndarray
    numerical_valid: np.ndarray


class CurrentWorldEvaluator:
    """Evaluate complete exogenous worlds in bounded vectorized batches."""

    def __init__(
        self,
        sampler: HierarchicalWorldSampler,
        policy: HighwayEnvIDMPolicy,
        evt_model: GPDTailModel,
        *,
        steps: int,
        batch_size: int,
    ) -> None:
        if int(steps) < 1:
            raise ValueError("steps must be positive")
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        self.sampler = sampler
        self.policy = policy
        self.evt_model = evt_model
        self.steps = int(steps)
        self.batch_size = int(batch_size)

    def evaluate(self, worlds: WorldExogenousState) -> WorldEvaluation:
        """Evaluate explicit worlds in bounded batches on HighwayEnv."""
        scores: list[np.ndarray] = []
        risks: list[np.ndarray] = []
        collisions: list[np.ndarray] = []
        gaps: list[np.ndarray] = []
        valid: list[np.ndarray] = []
        for start in range(0, worlds.batch_size, self.batch_size):
            batch = worlds.select(slice(start, start + self.batch_size))
            rollout = rollout_world(
                self.sampler,
                batch,
                self.policy,
                steps=self.steps,
                evt_model=self.evt_model,
            )
            collision, min_gap = collision_and_min_gap(
                rollout.states, rollout.initial_valid
            )
            if rollout.evt_score is None or not np.isfinite(rollout.evt_score).all():
                raise FloatingPointError("world rollout produced a non-finite EVT score")
            if not rollout.numerical_valid.all():
                raise FloatingPointError("world rollout produced non-finite states")
            scores.append(rollout.evt_score)
            risks.append(rollout.event_risk)
            collisions.append(collision)
            gaps.append(min_gap)
            valid.append(rollout.numerical_valid)
        return WorldEvaluation(
            evt_score=np.concatenate(scores),
            event_risk=np.concatenate(risks),
            collision=np.concatenate(collisions),
            min_gap_m=np.concatenate(gaps),
            numerical_valid=np.concatenate(valid),
        )
