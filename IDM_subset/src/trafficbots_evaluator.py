"""Batched IDM evaluation with TrafficBots as the background world model."""

from __future__ import annotations

import numpy as np

from hierarchical_world_model.src.execution import trajectory_event_risk
from tools.evt import GPDTailModel

from .metrics import collision_and_min_gap
from .trafficbots_randomness import TrafficBotsExogenousState
from .trafficbots_world import (
    TrafficBotsHighwayEnvWorld,
    TrafficBotsIDMRollout,
    TrafficBotsInitialSampler,
    rollout_trafficbots_idm,
)
from .world_evaluator import WorldEvaluation


class TrafficBotsIDMWorldEvaluator:
    """Evaluate explicit TrafficBots priors in bounded HighwayEnv batches."""

    def __init__(
        self,
        initial_sampler: TrafficBotsInitialSampler,
        world: TrafficBotsHighwayEnvWorld,
        evt_model: GPDTailModel,
        *,
        steps: int,
        batch_size: int,
    ) -> None:
        if int(steps) < 1 or int(batch_size) < 1:
            raise ValueError("steps and batch_size must be positive")
        self.initial_sampler = initial_sampler
        self.world = world
        self.evt_model = evt_model
        self.steps = int(steps)
        self.batch_size = int(batch_size)

    def rollout(self, worlds: TrafficBotsExogenousState) -> TrafficBotsIDMRollout:
        return rollout_trafficbots_idm(
            self.initial_sampler, self.world, worlds, steps=self.steps
        )

    def evaluate(self, worlds: TrafficBotsExogenousState) -> WorldEvaluation:
        scores: list[np.ndarray] = []
        risks: list[np.ndarray] = []
        collisions: list[np.ndarray] = []
        gaps: list[np.ndarray] = []
        numerical: list[np.ndarray] = []
        for start in range(0, worlds.batch_size, self.batch_size):
            selected = worlds.select(slice(start, start + self.batch_size))
            rollout = self.rollout(selected)
            finite = np.isfinite(rollout.states).all(axis=(1, 2, 3))
            if not finite.all():
                raise FloatingPointError("TrafficBots IDM rollout produced non-finite states")
            risk = trajectory_event_risk(rollout.states, rollout.initial_valid)
            score = np.asarray(self.evt_model.score(risk), np.float64)
            if not np.isfinite(score).all():
                raise FloatingPointError("TrafficBots IDM rollout produced non-finite EVT scores")
            collision, minimum = collision_and_min_gap(
                rollout.states, rollout.initial_valid
            )
            scores.append(score)
            risks.append(risk)
            collisions.append(collision)
            gaps.append(minimum)
            numerical.append(finite)
        return WorldEvaluation(
            evt_score=np.concatenate(scores),
            event_risk=np.concatenate(risks),
            collision=np.concatenate(collisions),
            min_gap_m=np.concatenate(gaps),
            numerical_valid=np.concatenate(numerical),
        )
