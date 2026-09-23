"""Exact stochastic metrics for benchmark v1."""

from __future__ import annotations

import numpy as np


def _masked_normalized_vectors(
    samples: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    worlds = np.asarray(samples, np.float64)
    truth = np.asarray(target, np.float64)
    mask = np.asarray(valid, bool)
    scale = np.asarray(scales, np.float64)
    if worlds.ndim != truth.ndim + 1 or worlds.shape[1:] != truth.shape:
        raise ValueError("samples must be [K,...] aligned with target")
    if truth.shape[:-1] != mask.shape or scale.shape != truth.shape[-1:]:
        raise ValueError("target, valid and scales do not align")
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError("all normalization scales must be finite and positive")
    expanded = np.broadcast_to(mask[..., None], truth.shape)
    dimensions = int(expanded.sum())
    if dimensions == 0:
        raise ValueError("score has no valid dimensions")
    denominator = np.sqrt(float(dimensions))
    generated = (worlds / scale)[:, expanded].reshape(len(worlds), -1) / denominator
    observed = (truth / scale)[expanded].reshape(-1) / denominator
    return generated, observed


def fair_energy_score(
    samples: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    scales: np.ndarray,
) -> float:
    """Finite-ensemble fair energy score (lower is better)."""
    generated, observed = _masked_normalized_vectors(samples, target, valid, scales)
    first = np.linalg.norm(generated - observed[None], axis=-1).mean()
    k = len(generated)
    if k < 2:
        return float(first)
    distance = np.linalg.norm(generated[:, None] - generated[None, :], axis=-1)
    second = distance.sum() / (2.0 * k * (k - 1))
    return float(first - second)


def fair_crps(samples: np.ndarray, target: float) -> float:
    values = np.asarray(samples, np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all() or not np.isfinite(target):
        raise ValueError("CRPS inputs must be non-empty and finite")
    first = np.abs(values - float(target)).mean()
    if len(values) < 2:
        return float(first)
    pairwise = np.abs(values[:, None] - values[None, :])
    return float(first - pairwise.sum() / (2.0 * len(values) * (len(values) - 1)))


def trajectory_errors(
    samples: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float]:
    """Scene-first ADE/FDE with one joint best sample per scene."""
    worlds, truth, mask = (
        np.asarray(samples),
        np.asarray(target),
        np.asarray(valid, bool),
    )
    if worlds.ndim != 4 or truth.ndim != 3 or worlds.shape[1:] != truth.shape:
        raise ValueError("expected samples [K,T,N,D], target [T,N,D]")
    if mask.shape != truth.shape[:2]:
        raise ValueError("valid must be [T,N]")
    distance = np.linalg.norm(worlds[..., :2] - truth[None, ..., :2], axis=-1)
    per_sample_ade = (distance * mask).sum((1, 2)) / mask.sum().clip(1)
    last = np.max(np.where(mask, np.arange(len(mask))[:, None], -1), axis=0)
    agents = np.flatnonzero(last >= 0)
    fde = (
        np.asarray([distance[:, last[i], i] for i in agents]).T.mean(-1)
        if len(agents)
        else np.zeros(len(worlds))
    )
    best = int(np.argmin(per_sample_ade))
    if len(worlds) > 1:
        pairwise = np.linalg.norm(
            worlds[:, None, ..., :2] - worlds[None, :, ..., :2], axis=-1
        )
        off_diagonal = ~np.eye(len(worlds), dtype=bool)
        diversity = float(
            (pairwise * mask[None, None]).sum((2, 3))[off_diagonal].mean()
            / mask.sum().clip(1)
        )
    else:
        diversity = 0.0
    return {
        "sample_mean_ADE_m": float(per_sample_ade.mean()),
        "joint_min_ADE_m": float(per_sample_ade[best]),
        "sample_mean_FDE_m": float(fde.mean()),
        "joint_min_corresponding_FDE_m": float(fde[best]),
        "mean_pairwise_trajectory_distance_m": diversity,
    }


def central_interval(samples: np.ndarray, level: float = 0.9) -> tuple[float, float]:
    values = np.asarray(samples, np.float64)
    alpha = (1.0 - float(level)) * 0.5
    return tuple(
        float(x) for x in np.quantile(values, (alpha, 1.0 - alpha), method="linear")
    )


def ensemble_channel_metrics(
    samples: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    channel_names: tuple[str, ...],
    level: float = 0.9,
) -> dict[str, float]:
    """Mean fair CRPS, central coverage, and width for each scalar channel.

    Metrics are first evaluated at every valid time/agent cell and then
    averaged within the scene, so scenes keep equal weight at aggregation.
    """
    worlds = np.asarray(samples, np.float64)
    truth = np.asarray(target, np.float64)
    mask = np.asarray(valid, bool)
    if worlds.ndim != truth.ndim + 1 or worlds.shape[1:] != truth.shape:
        raise ValueError("samples must be [K,...,C] aligned with target")
    if truth.shape[:-1] != mask.shape or truth.shape[-1] != len(channel_names):
        raise ValueError("target, valid, and channel names do not align")
    if not mask.any():
        raise ValueError("calibration metrics have no valid cells")
    alpha = (1.0 - float(level)) * 0.5
    lower, upper = np.quantile(worlds, (alpha, 1.0 - alpha), axis=0, method="linear")
    first = np.abs(worlds - truth[None]).mean(axis=0)
    k = len(worlds)
    if k > 1:
        pairwise = np.abs(worlds[:, None] - worlds[None, :]).sum((0, 1))
        crps = first - pairwise / (2.0 * k * (k - 1))
    else:
        crps = first
    output: dict[str, float] = {}
    for channel, name in enumerate(channel_names):
        selected = mask
        output[f"fair_CRPS_{name}"] = float(crps[..., channel][selected].mean())
        output[f"coverage_{int(round(level * 100))}_{name}"] = float(
            (
                (truth[..., channel] >= lower[..., channel])
                & (truth[..., channel] <= upper[..., channel])
            )[selected].mean()
        )
        output[f"interval_width_{int(round(level * 100))}_{name}"] = float(
            (upper[..., channel] - lower[..., channel])[selected].mean()
        )
    return output


def multiclass_brier(probabilities: np.ndarray, target_class: int) -> float:
    probs = np.asarray(probabilities, np.float64)
    if probs.ndim != 1 or not np.isclose(probs.sum(), 1.0) or np.any(probs < 0):
        raise ValueError("probabilities must be a simplex vector")
    observed = np.zeros_like(probs)
    observed[int(target_class)] = 1.0
    return float(np.square(probs - observed).sum())
