"""Subset simulation in the current world's explicit prior space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from hierarchical_world_model.src.randomness import WorldExogenousState

from .world_evaluator import WorldEvaluation


EvaluateWorlds = Callable[[Any], WorldEvaluation]
SampleWorlds = Callable[[int, int], Any]


@dataclass(frozen=True)
class WorldSubsetLevel:
    """One AMS level, including its population and evaluated risk scores."""

    level: int
    worlds: Any
    evaluation: WorldEvaluation
    threshold: float
    accepted: np.ndarray
    proposal_acceptance_rate: float


@dataclass(frozen=True)
class WorldSubsetResult:
    """Complete AMS estimate and the populations used to obtain it."""

    levels: list[WorldSubsetLevel]
    probability: float
    final_failure_fraction: float
    failure_threshold: float
    total_evaluations: int
    proposal_evaluations: int
    stop_reason: str


def _take_evaluation(evaluation: WorldEvaluation, indices: np.ndarray) -> WorldEvaluation:
    return WorldEvaluation(
        evt_score=evaluation.evt_score[indices].copy(),
        event_risk=evaluation.event_risk[indices].copy(),
        collision=evaluation.collision[indices].copy(),
        min_gap_m=evaluation.min_gap_m[indices].copy(),
        numerical_valid=evaluation.numerical_valid[indices].copy(),
    )


def _replace_accepted(
    current: Any,
    proposal: Any,
    accepted: np.ndarray,
) -> Any:
    values: dict[str, np.ndarray] = {}
    for name, current_value in current.as_dict().items():
        proposal_value = proposal.as_dict()[name]
        mask = np.asarray(accepted, bool).reshape(
            (len(accepted),) + (1,) * (current_value.ndim - 1)
        )
        values[name] = np.where(mask, proposal_value, current_value)
    return current.replace_arrays(values)


def _replace_evaluation(
    current: WorldEvaluation,
    proposal: WorldEvaluation,
    accepted: np.ndarray,
) -> WorldEvaluation:
    mask = np.asarray(accepted, bool)
    return WorldEvaluation(
        evt_score=np.where(mask, proposal.evt_score, current.evt_score),
        event_risk=np.where(mask, proposal.event_risk, current.event_risk),
        collision=np.where(mask, proposal.collision, current.collision),
        min_gap_m=np.where(mask, proposal.min_gap_m, current.min_gap_m),
        numerical_valid=np.where(mask, proposal.numerical_valid, current.numerical_valid),
    )


def _mutated_world(
    world: Any,
    *,
    blocks: tuple[str, ...],
    beta: float,
    rng: np.random.Generator,
) -> Any:
    """Make one joint, prior-reversible pCN proposal.

    Mutating only one of the independently distributed blocks in a short
    chain left most trajectory-defining randomness unchanged.  A sweep over
    every requested block is still prior reversible (the block kernels are
    independent), but gives each accepted proposal a meaningful opportunity
    to decorrelate from its resampled parent.
    """
    proposal = world
    for block_index in rng.permutation(len(blocks)):
        proposal = proposal.pcn_mutate(
            blocks[int(block_index)],
            beta=beta,
            seed=int(rng.integers(0, np.iinfo(np.int64).max)),
        )
    return proposal


def run_world_subset_simulation(
    evaluate: EvaluateWorlds,
    *,
    num_samples: int,
    p0: float,
    max_levels: int,
    failure_threshold: float,
    response_steps: int,
    scene_dim: int,
    agent_dim: int,
    mutation_blocks: tuple[str, ...],
    pcn_beta: float,
    mcmc_steps: int,
    seed: int,
    sample_worlds: SampleWorlds | None = None,
) -> WorldSubsetResult:
    """Estimate a rare failure probability with prior-reversible kernels.

    Each proposal makes one randomized sweep over the selected independent
    base-randomness blocks.  The categorical refresh and Gaussian pCN updates
    preserve the joint world prior, so a proposal is accepted exactly when it
    remains above the current subset threshold.
    """
    if int(num_samples) < 2:
        raise ValueError("num_samples must be at least two")
    if not 0.0 < float(p0) < 1.0:
        raise ValueError("p0 must lie in (0, 1)")
    if int(max_levels) < 1 or int(mcmc_steps) < 1:
        raise ValueError("max_levels and mcmc_steps must be positive")
    if not mutation_blocks:
        raise ValueError("mutation_blocks must not be empty")
    if not 0.0 < float(pcn_beta) <= 1.0:
        raise ValueError("pcn_beta must lie in (0, 1]")

    rng = np.random.default_rng(int(seed))
    elite_count = max(1, int(round(int(num_samples) * float(p0))))
    if elite_count >= int(num_samples):
        raise ValueError("p0 leaves no non-elite samples")
    initial_seed = int(rng.integers(0, np.iinfo(np.int64).max))
    worlds = (
        WorldExogenousState.sample(
            int(num_samples),
            seed=initial_seed,
            response_steps=int(response_steps),
            scene_dim=int(scene_dim),
            agent_dim=int(agent_dim),
        )
        if sample_worlds is None
        else sample_worlds(int(num_samples), initial_seed)
    )
    evaluation = evaluate(worlds)
    levels: list[WorldSubsetLevel] = []
    total_evaluations = int(num_samples)
    proposal_evaluations = 0

    for level_index in range(int(max_levels)):
        scores = evaluation.evt_score
        threshold = float(np.quantile(scores, 1.0 - float(p0)))
        failure_mask = scores >= float(failure_threshold)
        levels.append(
            WorldSubsetLevel(
                level=level_index,
                worlds=worlds,
                evaluation=evaluation,
                threshold=threshold,
                accepted=np.ones(int(num_samples), dtype=bool),
                proposal_acceptance_rate=1.0 if level_index == 0 else float("nan"),
            )
        )
        if threshold >= float(failure_threshold) or level_index == int(max_levels) - 1:
            probability = float(p0) ** level_index * float(failure_mask.mean())
            return WorldSubsetResult(
                levels=levels,
                probability=probability,
                final_failure_fraction=float(failure_mask.mean()),
                failure_threshold=float(failure_threshold),
                total_evaluations=total_evaluations,
                proposal_evaluations=proposal_evaluations,
                stop_reason=(
                    "subset_threshold_reached_failure_threshold"
                    if threshold >= float(failure_threshold)
                    else "max_levels_reached"
                ),
            )

        elite_indices = np.argsort(scores)[-elite_count:][::-1]
        parent_indices = np.resize(elite_indices, int(num_samples))
        next_worlds = worlds.select(parent_indices)
        next_evaluation = _take_evaluation(evaluation, parent_indices)
        accepted = np.zeros(int(num_samples), dtype=bool)
        accepted_count = 0
        attempts = 0
        for _ in range(int(mcmc_steps)):
            proposal = _mutated_world(
                next_worlds,
                blocks=mutation_blocks,
                beta=float(pcn_beta),
                rng=rng,
            )
            proposal_evaluation = evaluate(proposal)
            proposal_evaluations += int(num_samples)
            total_evaluations += int(num_samples)
            attempts += int(num_samples)
            accepted_step = proposal_evaluation.evt_score >= threshold
            accepted_count += int(accepted_step.sum())
            accepted |= accepted_step
            next_worlds = _replace_accepted(
                next_worlds,
                proposal,
                accepted_step,
            )
            next_evaluation = _replace_evaluation(
                next_evaluation,
                proposal_evaluation,
                accepted_step,
            )

        worlds = next_worlds
        evaluation = next_evaluation
        levels[-1] = WorldSubsetLevel(
            level=level_index,
            worlds=levels[-1].worlds,
            evaluation=levels[-1].evaluation,
            threshold=levels[-1].threshold,
            accepted=accepted,
            proposal_acceptance_rate=float(accepted_count / max(attempts, 1)),
        )

    raise AssertionError("subset simulation must return from its final level")
