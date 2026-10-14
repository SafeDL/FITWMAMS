"""Batched IDM evaluation in the generated responsive traffic world."""

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
    interaction_descriptor: np.ndarray | None = None


class CurrentWorldEvaluator:
    """Evaluate complete tail-world samples in bounded vectorized batches."""

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
        descriptors: list[np.ndarray] = []
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
                rollout.states, rollout.initial_valid, rollout.vehicle_dimensions
            )
            if rollout.evt_score is None or not np.isfinite(rollout.evt_score).all():
                raise FloatingPointError(
                    "world rollout produced a non-finite EVT score"
                )
            if not rollout.numerical_valid.all():
                raise FloatingPointError("world rollout produced non-finite states")
            scores.append(rollout.evt_score)
            risks.append(rollout.event_risk)
            collisions.append(collision)
            gaps.append(min_gap)
            valid.append(rollout.numerical_valid)
            states = np.asarray(rollout.states)
            ego = states[:, :, :1]
            npc = states[:, :, 1:]
            active = np.asarray(rollout.initial_valid[:, None, 1:], bool)
            dx = npc[..., 0] - ego[..., 0]
            dy = np.abs(npc[..., 1] - ego[..., 1])
            clearance = np.where(active, np.hypot(dx, dy), np.inf)
            flat = clearance.reshape(len(states), -1).argmin(axis=1)
            frame = flat // 6
            slot = flat % 6
            row = np.arange(len(states))
            closing = states[row, frame, 0, 2] - states[row, frame, slot + 1, 2]
            first_risk = np.argmax((clearance < 12.0).any(axis=2), axis=1)
            participants = (clearance < 15.0).any(axis=1).sum(axis=1)
            descriptors.append(
                np.stack(
                    (
                        slot.astype(np.float32),
                        np.minimum(
                            clearance.reshape(len(states), -1).min(axis=1), 100.0
                        ),
                        closing,
                        first_risk / max(states.shape[1] - 1, 1),
                        collision.astype(np.float32),
                        participants.astype(np.float32),
                    ),
                    axis=1,
                ).astype(np.float32)
            )
        return WorldEvaluation(
            evt_score=np.concatenate(scores),
            event_risk=np.concatenate(risks),
            collision=np.concatenate(collisions),
            min_gap_m=np.concatenate(gaps),
            numerical_valid=np.concatenate(valid),
            interaction_descriptor=np.concatenate(descriptors),
        )
