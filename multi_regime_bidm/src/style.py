"""Train-only driver-style features and dependency-light K-means."""
from __future__ import annotations

import numpy as np


FEATURE_NAMES = ("mean_time_headway_s", "max_acceleration_mps2", "max_deceleration_mps2")


def decision_grid(pair: dict[str, np.ndarray], plant_dt: float = .04, stride: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return causal 5 Hz observations ``[gap, speed, closing]`` and actions.

    The action is the speed change over the following held-action interval.
    Its final state is never used as a covariate for the interval start.
    """
    count = len(pair["follower_v"])
    indices = np.arange(0, count - stride, stride, dtype=int)
    speed, leader = pair["follower_v"], pair["leader_v"]
    observation = np.column_stack((pair["gap"][indices], speed[indices], speed[indices] - leader[indices]))
    action = (speed[indices + stride] - speed[indices]) / (plant_dt * stride)
    return observation, action


def style_features(pair: dict[str, np.ndarray]) -> np.ndarray:
    """Paper-style per-event descriptors, evaluated only on a training event."""
    observation, action = decision_grid(pair)
    headway = observation[:, 0] / np.maximum(observation[:, 1], .1)
    return np.asarray((np.mean(headway), np.max(action), max(0., -np.min(action))), dtype=float)


def standardize_train(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, float)
    mean = np.mean(values, axis=0)
    scale = np.maximum(np.std(values, axis=0), 1.e-8)
    return (values - mean) / scale, mean, scale


def kmeans(values: np.ndarray, clusters: int = 3, *, seed: int = 20260915, restarts: int = 20,
           iterations: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic K-means, avoiding a hidden sklearn dependency."""
    values = np.asarray(values, float)
    if values.ndim != 2 or len(values) < clusters:
        raise ValueError("need [events, features] with at least one event per cluster")
    rng = np.random.default_rng(seed)
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for _ in range(restarts):
        centers = values[rng.choice(len(values), clusters, replace=False)].copy()
        for _ in range(iterations):
            distance = np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(distance, axis=1)
            candidate = centers.copy()
            for cluster in range(clusters):
                members = values[labels == cluster]
                candidate[cluster] = np.mean(members, axis=0) if len(members) else values[rng.integers(len(values))]
            if np.max(np.abs(candidate - centers)) < 1.e-10:
                centers = candidate
                break
            centers = candidate
        loss = float(np.sum((values - centers[labels]) ** 2))
        if best is None or loss < best[0]:
            best = loss, labels, centers
    assert best is not None
    # Stable IDs are ranks by train-only mean headway; do not attach the paper's
    # semantic labels aggressive/neutral/timid to a different highD population.
    order = np.argsort(best[2][:, 0])
    remap = np.empty(clusters, dtype=int); remap[order] = np.arange(clusters)
    return remap[best[1]], best[2][order]
