"""Random-walk Metropolis subset simulation in latent/context space."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .latent_evaluator import LatentEvaluation


EvaluateFn = Callable[[int, np.ndarray], LatentEvaluation]
EvaluateManyFn = Callable[
    [np.ndarray, np.ndarray, int],
    tuple[np.ndarray, list[np.ndarray], list[dict[str, float]], list[list[dict[str, float]]]],
]
logger = logging.getLogger(__name__)


@dataclass
class SubsetLevel:
    level: int
    context_indices: np.ndarray
    latents: np.ndarray
    scores: np.ndarray
    actions: list[np.ndarray]
    metrics: list[dict[str, float]]
    traces: list[list[dict[str, float]]]
    threshold: float
    accepted: np.ndarray
    acceptance_rate: float


@dataclass
class SubsetSimulationResult:
    levels: list[SubsetLevel]
    probability: float
    final_failure_fraction: float
    failure_threshold: float
    total_evaluations: int
    proposal_evaluations: int
    stop_reason: str
    stop_level: int


@dataclass
class _CachedEvaluation:
    context_index: int
    latent: np.ndarray
    score: float
    actions: np.ndarray
    metrics: dict[str, float]
    trace: list[dict[str, float]]


def standard_normal_log_prob(z: np.ndarray) -> float:
    value = np.asarray(z, dtype=np.float64)
    return float(-0.5 * np.sum(value * value))


def _cached_from_result(
    context_index: int,
    latent: np.ndarray,
    result: LatentEvaluation,
) -> _CachedEvaluation:
    metrics = dict(result.metrics)
    metrics["context_index"] = float(context_index)
    return _CachedEvaluation(
        context_index=int(context_index),
        latent=np.asarray(latent, dtype=np.float32).copy(),
        score=float(result.score),
        actions=result.actions.astype(np.float32, copy=True),
        metrics=metrics,
        trace=list(result.trace),
    )


def _population_from_cached(
    cached: list[_CachedEvaluation],
) -> tuple[np.ndarray, list[np.ndarray], list[dict[str, float]], list[list[dict[str, float]]]]:
    return (
        np.asarray([item.score for item in cached], dtype=np.float64),
        [item.actions for item in cached],
        [dict(item.metrics) for item in cached],
        [list(item.trace) for item in cached],
    )


def _evaluate_population(
    context_indices: np.ndarray,
    latents: np.ndarray,
    evaluate: EvaluateFn,
    *,
    level: int,
    evaluate_many: EvaluateManyFn | None = None,
) -> tuple[np.ndarray, list[np.ndarray], list[dict[str, float]], list]:
    if evaluate_many is not None:
        return evaluate_many(context_indices, latents, level)

    scores: list[float] = []
    actions: list[np.ndarray] = []
    metrics: list[dict[str, float]] = []
    traces: list[list[dict[str, float]]] = []
    total = int(latents.shape[0])
    interval = max(1, total // 10)
    for idx, latent in enumerate(latents):
        context_index = int(context_indices[idx])
        result = evaluate(context_index, latent)
        scores.append(float(result.score))
        actions.append(result.actions)
        item_metrics = dict(result.metrics)
        item_metrics["context_index"] = float(context_index)
        metrics.append(item_metrics)
        traces.append(result.trace)
        done = idx + 1
        if done == total or done % interval == 0:
            logger.info(
                "Subset level %d evaluated %d/%d samples",
                level,
                done,
                total,
            )
    return np.asarray(scores, dtype=np.float64), actions, metrics, traces


def _state_key(context: int, z: np.ndarray) -> tuple[int, bytes]:
    return int(context), np.ascontiguousarray(z).tobytes()


def _unique_state_count(
    context_indices: np.ndarray,
    latents: np.ndarray,
) -> int:
    return len(
        {
            _state_key(int(context_indices[idx]), latents[idx])
            for idx in range(int(latents.shape[0]))
        }
    )


def _standard_elite_indices(
    scores: np.ndarray,
    *,
    threshold: float,
    elite_count: int,
) -> np.ndarray:
    if elite_count <= 0:
        raise ValueError("elite_count must be positive")
    order = np.argsort(scores)[::-1]
    eligible = [int(idx) for idx in order if float(scores[idx]) >= threshold]
    return np.asarray(eligible[:elite_count], dtype=np.int64)


def _mh_proposal(
    current: _CachedEvaluation,
    evaluate: EvaluateFn,
    rng: np.random.Generator,
    *,
    context_count: int,
    proposal_std: float,
    context_refresh_prob: float,
    threshold: float,
) -> _CachedEvaluation | None:
    if rng.random() < context_refresh_prob:
        proposal_context = int(rng.integers(0, int(context_count)))
        proposal_z = rng.standard_normal(current.latent.shape).astype(np.float32)
        proposal_eval = evaluate(proposal_context, proposal_z)
        if float(proposal_eval.score) < threshold:
            return None
        return _cached_from_result(proposal_context, proposal_z, proposal_eval)

    proposal_z = current.latent + proposal_std * rng.standard_normal(
        current.latent.shape
    )
    proposal_z = proposal_z.astype(np.float32)
    proposal_eval = evaluate(current.context_index, proposal_z)
    if float(proposal_eval.score) < threshold:
        return None

    log_alpha = standard_normal_log_prob(proposal_z)
    log_alpha -= standard_normal_log_prob(current.latent)
    if np.log(rng.random()) <= min(0.0, log_alpha):
        return _cached_from_result(current.context_index, proposal_z, proposal_eval)
    return None


def _build_next_population(
    context_indices: np.ndarray,
    latents: np.ndarray,
    scores: np.ndarray,
    actions: list[np.ndarray],
    metrics: list[dict[str, float]],
    traces: list[list[dict[str, float]]],
    evaluate: EvaluateFn,
    rng: np.random.Generator,
    *,
    context_count: int,
    num_samples: int,
    threshold: float,
    elite_count: int,
    proposal_std: float,
    context_refresh_prob: float,
    mh_retries_per_sample: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    int,
    np.ndarray,
    list[np.ndarray],
    list[dict[str, float]],
    list[list[dict[str, float]]],
]:
    elite_idx = _standard_elite_indices(
        scores,
        threshold=threshold,
        elite_count=elite_count,
    )
    if elite_idx.size == 0:
        raise RuntimeError(
            "No elite samples met the subset threshold; cannot build next level"
        )

    chain_states: list[_CachedEvaluation] = [
        _CachedEvaluation(
            context_index=int(context_indices[idx]),
            latent=latents[idx].copy(),
            score=float(scores[idx]),
            actions=actions[idx],
            metrics=dict(metrics[idx]),
            trace=list(traces[idx]),
        )
        for idx in elite_idx
    ]
    next_contexts: list[int] = []
    next_latents: list[np.ndarray] = []
    next_accepted: list[float] = []
    next_cached: list[_CachedEvaluation] = []

    for state in chain_states:
        next_contexts.append(int(state.context_index))
        next_latents.append(state.latent.copy())
        next_accepted.append(0.0)
        next_cached.append(state)
        if len(next_latents) >= num_samples:
            break

    cursor = 0
    proposal_evaluations = 0
    accepted_count = 0
    build_interval = max(1, int(num_samples) // 10)
    logger.info(
        (
            "Building next subset population from %d elites "
            "threshold %.6f retries=%d context_refresh_prob=%.3f"
        ),
        len(chain_states),
        threshold,
        int(mh_retries_per_sample),
        context_refresh_prob,
    )
    while len(next_latents) < num_samples:
        chain_idx = cursor % len(chain_states)
        current = chain_states[chain_idx]
        accepted_state: _CachedEvaluation | None = None
        for _attempt in range(max(1, int(mh_retries_per_sample))):
            proposal_evaluations += 1
            accepted_state = _mh_proposal(
                current,
                evaluate,
                rng,
                context_count=context_count,
                proposal_std=proposal_std,
                context_refresh_prob=context_refresh_prob,
                threshold=threshold,
            )
            if accepted_state is not None:
                chain_states[chain_idx] = accepted_state
                break

        if accepted_state is None:
            accepted_state = chain_states[chain_idx]
            is_accepted = 0.0
        else:
            is_accepted = 1.0
            accepted_count += 1

        next_contexts.append(int(accepted_state.context_index))
        next_latents.append(accepted_state.latent.copy())
        next_accepted.append(is_accepted)
        next_cached.append(accepted_state)
        cursor += 1
        built = len(next_latents)
        if built == num_samples or built % build_interval == 0:
            logger.info(
                (
                    "Built next subset population %d/%d proposal_evals=%d "
                    "accepted=%d acceptance_rate=%.4f"
                ),
                built,
                num_samples,
                proposal_evaluations,
                accepted_count,
                accepted_count / max(proposal_evaluations, 1),
            )

    next_scores, next_actions, next_metrics, next_traces = _population_from_cached(
        next_cached
    )
    return (
        np.asarray(next_contexts, dtype=np.int64),
        np.asarray(next_latents, dtype=np.float32),
        np.asarray(next_accepted, dtype=np.float32),
        float(np.mean(next_accepted)),
        int(proposal_evaluations),
        next_scores,
        next_actions,
        next_metrics,
        next_traces,
    )


def run_subset_simulation(
    evaluate: EvaluateFn,
    *,
    context_count: int,
    latent_shape: tuple[int, ...],
    num_samples: int,
    p0: float,
    max_levels: int,
    proposal_std: float,
    context_refresh_prob: float,
    failure_threshold: float,
    seed: int,
    mh_retries_per_sample: int = 4,
    evaluate_many: EvaluateManyFn | None = None,
    adaptive_stop_enabled: bool = False,
    adaptive_stop_min_failure_count: int = 20,
    adaptive_stop_min_levels: int = 2,
) -> SubsetSimulationResult:
    if context_count <= 0:
        raise ValueError("context_count must be positive")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not 0.0 < p0 < 1.0:
        raise ValueError("p0 must be in (0, 1)")
    if max_levels <= 0:
        raise ValueError("max_levels must be positive")
    if proposal_std <= 0.0:
        raise ValueError("proposal_std must be positive")
    if not 0.0 <= context_refresh_prob <= 1.0:
        raise ValueError("context_refresh_prob must be in [0, 1]")
    if mh_retries_per_sample <= 0:
        raise ValueError("mh_retries_per_sample must be positive")
    if adaptive_stop_min_failure_count <= 0:
        raise ValueError("adaptive_stop_min_failure_count must be positive")
    if adaptive_stop_min_levels <= 0:
        raise ValueError("adaptive_stop_min_levels must be positive")

    rng = np.random.default_rng(int(seed))
    elite_count = max(1, int(round(float(num_samples) * float(p0))))
    if elite_count >= num_samples:
        raise ValueError("p0 leaves no non-elite samples")

    context_indices = rng.integers(
        0,
        int(context_count),
        size=int(num_samples),
        dtype=np.int64,
    )
    latents = rng.standard_normal((num_samples, *latent_shape))
    latents = latents.astype(np.float32)
    levels: list[SubsetLevel] = []
    probability = float("nan")
    final_failure_fraction = 0.0
    proposal_evaluations_total = 0
    total_evaluations = int(num_samples)
    stop_reason = "max_levels_reached"
    stop_level = -1
    cached_population: (
        tuple[
            np.ndarray,
            list[np.ndarray],
            list[dict[str, float]],
            list[list[dict[str, float]]],
        ]
        | None
    ) = None

    for level_idx in range(max_levels):
        logger.info("Subset level %d started", level_idx)
        if cached_population is None:
            scores, actions, metrics, traces = _evaluate_population(
                context_indices,
                latents,
                evaluate,
                level=level_idx,
                evaluate_many=evaluate_many,
            )
        else:
            scores, actions, metrics, traces = cached_population
            cached_population = None
            logger.info(
                "Subset level %d reused cached next-population evaluations",
                level_idx,
            )
        threshold = float(np.quantile(scores, 1.0 - float(p0)))
        accepted = np.ones(num_samples, dtype=np.float32)
        acceptance_rate = 1.0 if level_idx == 0 else float("nan")
        levels.append(
            SubsetLevel(
                level=level_idx,
                context_indices=context_indices.copy(),
                latents=latents.copy(),
                scores=scores.copy(),
                actions=actions,
                metrics=metrics,
                traces=traces,
                threshold=threshold,
                accepted=accepted,
                acceptance_rate=acceptance_rate,
            )
        )

        failures = scores >= float(failure_threshold)
        final_failure_count = int(np.sum(failures))
        final_failure_fraction = float(np.mean(failures))
        logger.info(
            (
                "Subset level %d threshold %.6f score_min %.6f "
                "score_mean %.6f score_max %.6f failure_fraction %.6f"
            ),
            level_idx,
            threshold,
            float(np.min(scores)),
            float(np.mean(scores)),
            float(np.max(scores)),
            final_failure_fraction,
        )
        reached_failure_threshold = bool(threshold >= failure_threshold)
        adaptive_stop = bool(
            adaptive_stop_enabled
            and (level_idx + 1) >= int(adaptive_stop_min_levels)
            and final_failure_count >= int(adaptive_stop_min_failure_count)
        )
        reached_max_levels = bool(level_idx == max_levels - 1)
        if reached_failure_threshold or adaptive_stop or reached_max_levels:
            probability = (float(p0) ** level_idx) * final_failure_fraction
            stop_level = int(level_idx)
            if reached_failure_threshold:
                stop_reason = "subset_threshold_reached_failure_threshold"
            elif adaptive_stop:
                stop_reason = "adaptive_failure_count_reached"
            else:
                stop_reason = "max_levels_reached"
            break

        (
            context_indices_next,
            latents_next,
            accepted,
            acceptance_rate,
            proposal_evaluations,
            next_scores,
            next_actions,
            next_metrics,
            next_traces,
        ) = (
            _build_next_population(
                context_indices,
                latents,
                scores,
                actions,
                metrics,
                traces,
                evaluate,
                rng,
                context_count=context_count,
                num_samples=num_samples,
                threshold=threshold,
                elite_count=elite_count,
                proposal_std=proposal_std,
                context_refresh_prob=context_refresh_prob,
                mh_retries_per_sample=mh_retries_per_sample,
            )
        )
        proposal_evaluations_total += int(proposal_evaluations)
        total_evaluations += int(proposal_evaluations)
        levels[-1].accepted = accepted
        levels[-1].acceptance_rate = acceptance_rate
        next_unique_contexts = int(np.unique(context_indices_next).shape[0])
        next_unique_states = _unique_state_count(
            context_indices_next,
            latents_next,
        )
        logger.info(
            (
                "Subset level %d MH acceptance_rate %.6f "
                "next_unique_contexts=%d next_unique_states=%d"
            ),
            level_idx,
            levels[-1].acceptance_rate,
            next_unique_contexts,
            next_unique_states,
        )
        context_indices = context_indices_next
        latents = latents_next
        cached_population = (next_scores, next_actions, next_metrics, next_traces)

    logger.info(
        (
            "Subset simulation finished probability %.8g after %d levels "
            "stop_reason=%s closed_loop_evaluations=%d proposal_evaluations=%d"
        ),
        probability,
        len(levels),
        stop_reason,
        total_evaluations,
        proposal_evaluations_total,
    )
    return SubsetSimulationResult(
        levels=levels,
        probability=float(probability),
        final_failure_fraction=float(final_failure_fraction),
        failure_threshold=float(failure_threshold),
        total_evaluations=int(total_evaluations),
        proposal_evaluations=int(proposal_evaluations_total),
        stop_reason=str(stop_reason),
        stop_level=int(stop_level),
    )
