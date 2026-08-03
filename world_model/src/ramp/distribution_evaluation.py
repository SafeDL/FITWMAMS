"""Distribution diagnostics for RAMP-WM random rollouts (compact EVT protocol)."""

from __future__ import annotations

import numpy as np


def empirical_coverage(
    samples: np.ndarray,
    target: np.ndarray,
    levels: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
) -> dict[str, float]:
    """Per-coordinate central-interval coverage; caller supplies random samples."""
    out = {}
    for level in levels:
        low, high = np.quantile(samples, ((1 - level) / 2, 1 - (1 - level) / 2), axis=0)
        out[f"coverage_{int(level*100)}"] = float(
            np.mean((target >= low) & (target <= high))
        )
    return out


def energy_score(samples: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    """Masked multivariate energy score, averaged by condition."""
    values: list[float] = []
    for sample, observed, mask in zip(samples.transpose(1, 0, 2, 3, 4), target, valid):
        take = np.asarray(mask, bool)
        if not take.any():
            continue
        x = sample[..., :2][:, take].reshape(sample.shape[0], -1)
        y = observed[..., :2][take].reshape(-1)
        first = np.linalg.norm(x - y[None], axis=1).mean()
        if len(x) > 1:
            pair = np.linalg.norm(x[:, None] - x[None, :], axis=-1).mean()
        else:
            pair = 0.0
        values.append(float(first - 0.5 * pair))
    return float(np.mean(values)) if values else float("nan")


def one_dimensional_distance(
    generated: np.ndarray, observed: np.ndarray
) -> dict[str, float]:
    """Dependency-free Wasserstein-1/KS/quantile diagnostics."""
    left, right = np.sort(np.asarray(generated, np.float64).reshape(-1)), np.sort(
        np.asarray(observed, np.float64).reshape(-1)
    )
    if not len(left) or not len(right):
        return {
            "wasserstein_1": float("nan"),
            "ks_distance": float("nan"),
            "quantile_error": float("nan"),
        }
    q = np.linspace(0.0, 1.0, 401)
    lq, rq = np.quantile(left, q), np.quantile(right, q)
    # CDF mismatch on the pooled support is the empirical KS statistic.
    support = np.sort(np.concatenate((left, right)))
    ks = np.max(
        np.abs(
            np.searchsorted(left, support, side="right") / len(left)
            - np.searchsorted(right, support, side="right") / len(right)
        )
    )
    return {
        "wasserstein_1": float(np.mean(np.abs(lq - rq))),
        "ks_distance": float(ks),
        "quantile_error": float(np.mean(np.abs(lq - rq))),
    }


def pit_histogram(
    samples: np.ndarray, target: np.ndarray, valid: np.ndarray, *, bins: int = 20
) -> dict[str, object]:
    """PIT/rank histogram for masked scalar coordinates of a sample ensemble."""
    draws = np.asarray(samples, np.float32)
    observed = np.asarray(target, np.float32)
    mask = np.asarray(valid, bool)
    while mask.ndim < observed.ndim:
        mask = mask[..., None]
    mask = np.broadcast_to(mask, observed.shape)
    rank = (draws < observed[None]).sum(axis=0)
    values = rank[mask].reshape(-1)
    counts, _ = np.histogram(values, bins=bins, range=(0, draws.shape[0]))
    expected = len(values) / max(bins, 1)
    return {
        "bins": int(bins),
        "counts": counts.astype(int).tolist(),
        "chi_square_uniform": float(
            ((counts - expected) ** 2 / max(expected, 1.0)).sum()
        ),
        "samples": int(len(values)),
    }


def univariate_crps(
    samples: np.ndarray, target: np.ndarray, valid: np.ndarray
) -> float:
    """Empirical CRPS, a coordinate-wise counterpart to the energy score."""
    draws = np.asarray(samples, np.float32)
    observed = np.asarray(target, np.float32)
    mask = np.asarray(valid, bool)
    while mask.ndim < observed.ndim:
        mask = mask[..., None]
    mask = np.broadcast_to(mask, observed.shape)
    if not mask.any():
        return float("nan")
    first = np.abs(draws - observed[None]).mean(axis=0)
    pair = np.abs(draws[:, None] - draws[None, :]).mean(axis=(0, 1))
    return float((first - 0.5 * pair)[mask].mean())


def candidate_calibration(
    probabilities: np.ndarray, candidate_errors: np.ndarray, *, bins: int = 10
) -> dict[str, float]:
    """Use the lowest-error candidate as an observed discrete responsibility."""
    p = np.asarray(probabilities, np.float64).reshape(-1, probabilities.shape[-1])
    errors = np.asarray(candidate_errors, np.float64).reshape(
        -1, candidate_errors.shape[-1]
    )
    keep = np.isfinite(p).all(axis=1) & np.isfinite(errors).all(axis=1)
    if not keep.any():
        return {
            "candidate_probability_ece": float("nan"),
            "candidate_responsibility_cross_entropy": float("nan"),
        }
    p, errors = p[keep], errors[keep]
    target = errors.argmin(axis=1)
    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == target).astype(np.float64)
    ece = 0.0
    for low, high in zip(np.linspace(0.0, 0.9, bins), np.linspace(0.1, 1.0, bins)):
        selected = (confidence >= low) & (
            (confidence < high) if high < 1.0 else (confidence <= high)
        )
        if selected.any():
            ece += selected.mean() * abs(
                confidence[selected].mean() - correct[selected].mean()
            )
    nll = -np.log(np.maximum(p[np.arange(len(p)), target], 1e-8)).mean()
    return {
        "candidate_probability_ece": float(ece),
        "candidate_responsibility_cross_entropy": float(nll),
    }


def trajectory_feature_rows(
    states: np.ndarray, valid: np.ndarray, ego: np.ndarray
) -> dict[str, np.ndarray]:
    """Natural-driving feature rows, one row per valid vehicle trajectory."""
    x, mask, ego_state = (
        np.asarray(states, np.float32),
        np.asarray(valid, bool),
        np.asarray(ego, np.float32),
    )
    speed = np.linalg.norm(x[..., 2:4], axis=-1)
    acceleration = x[..., 4]
    rows: dict[str, list[float]] = {
        key: []
        for key in (
            "initial_speed",
            "final_speed",
            "speed_change",
            "mean_acceleration",
            "minimum_acceleration",
            "final_acceleration",
            "braking_start_s",
            "braking_duration_s",
            "longitudinal_displacement",
            "lateral_displacement",
            "minimum_gap",
            "minimum_ttc",
            "maximum_drac",
        )
    }
    for item in range(x.shape[0]):
        for agent in range(x.shape[2]):
            take = mask[item, :, agent]
            if not take.any():
                continue
            idx = np.flatnonzero(take)
            first, last = idx[0], idx[-1]
            acc = acceleration[item, take, agent]
            brake = acc < -0.5
            rel_x = x[item, take, agent, 0] - ego_state[item, take, 0]
            rel_v = x[item, take, agent, 2] - ego_state[item, take, 2]
            gap = np.abs(rel_x)
            closing = np.maximum(-rel_v, 0.0)
            ttc = np.where(closing > 1e-3, gap / np.maximum(closing, 1e-3), np.inf)
            drac = np.where(closing > 1e-3, closing**2 / np.maximum(2 * gap, 1e-3), 0.0)
            rows["initial_speed"].append(float(speed[item, first, agent]))
            rows["final_speed"].append(float(speed[item, last, agent]))
            rows["speed_change"].append(
                float(speed[item, last, agent] - speed[item, first, agent])
            )
            rows["mean_acceleration"].append(float(acc.mean()))
            rows["minimum_acceleration"].append(float(acc.min()))
            rows["final_acceleration"].append(float(acc[-1]))
            rows["braking_start_s"].append(
                float(np.flatnonzero(brake)[0] * 0.04) if brake.any() else 5.0
            )
            rows["braking_duration_s"].append(float(brake.mean() * len(acc) * 0.04))
            rows["longitudinal_displacement"].append(
                float(x[item, last, agent, 0] - x[item, first, agent, 0])
            )
            rows["lateral_displacement"].append(
                float(x[item, last, agent, 1] - x[item, first, agent, 1])
            )
            rows["minimum_gap"].append(float(gap.min()))
            rows["minimum_ttc"].append(float(np.minimum(ttc, 10.0).min()))
            rows["maximum_drac"].append(float(drac.max()))
    return {key: np.asarray(value, np.float32) for key, value in rows.items()}


def _feature_matrix(rows: dict[str, np.ndarray]) -> np.ndarray:
    values = [np.asarray(rows[key], np.float64) for key in sorted(rows)]
    length = min((len(item) for item in values), default=0)
    return (
        np.stack([item[:length] for item in values], axis=1)
        if length
        else np.zeros((0, len(values)))
    )


def multivariate_feature_distance(
    generated: dict[str, np.ndarray],
    observed: dict[str, np.ndarray],
    *,
    seed: int = 314159,
    maximum: int = 4096,
) -> dict[str, float]:
    """Stable, bounded-cost multivariate two-sample distances and correlations."""
    left, right = _feature_matrix(generated), _feature_matrix(observed)
    rng = np.random.default_rng(seed)
    if len(left) > maximum:
        left = left[rng.choice(len(left), maximum, replace=False)]
    if len(right) > maximum:
        right = right[rng.choice(len(right), maximum, replace=False)]
    if not len(left) or not len(right):
        return {
            "energy_distance": float("nan"),
            "mmd_rbf": float("nan"),
            "pearson_correlation_matrix_error": float("nan"),
            "spearman_correlation_matrix_error": float("nan"),
        }
    scale = np.maximum(np.std(np.concatenate((left, right)), axis=0), 1e-4)
    left, right = left / scale, right / scale

    def average_distance(a: np.ndarray, b: np.ndarray) -> float:
        value = 0.0
        count = 0
        for start in range(0, len(a), 256):
            distance = np.linalg.norm(a[start : start + 256, None] - b[None], axis=-1)
            value += float(distance.sum())
            count += distance.size
        return value / max(count, 1)

    cross, within_left, within_right = (
        average_distance(left, right),
        average_distance(left, left),
        average_distance(right, right),
    )
    bandwidth = np.median(
        np.linalg.norm(
            left[: min(len(left), 512), None] - right[None, : min(len(right), 512)],
            axis=-1,
        )
    ).clip(min=1e-4)

    def kernel(a: np.ndarray, b: np.ndarray) -> float:
        total = count = 0.0
        for start in range(0, len(a), 256):
            square = ((a[start : start + 256, None] - b[None]) ** 2).sum(axis=-1)
            total += float(np.exp(-square / (2 * bandwidth**2)).sum())
            count += square.size
        return total / max(count, 1.0)

    pearson = np.corrcoef(left, rowvar=False) - np.corrcoef(right, rowvar=False)
    rank_left = np.apply_along_axis(
        lambda value: np.argsort(np.argsort(value)), 0, left
    )
    rank_right = np.apply_along_axis(
        lambda value: np.argsort(np.argsort(value)), 0, right
    )
    spearman = np.corrcoef(rank_left, rowvar=False) - np.corrcoef(
        rank_right, rowvar=False
    )
    return {
        "energy_distance": float(2 * cross - within_left - within_right),
        "mmd_rbf": float(
            kernel(left, left) + kernel(right, right) - 2 * kernel(left, right)
        ),
        "pearson_correlation_matrix_error": float(np.mean(np.abs(pearson))),
        "spearman_correlation_matrix_error": float(np.mean(np.abs(spearman))),
    }


def temporal_and_relationship_diagnostics(
    states: np.ndarray, valid: np.ndarray, ego: np.ndarray
) -> dict[str, float | list[float]]:
    """Compact temporal/multi-agent dependence diagnostics for one trajectory set."""
    x, mask = (
        np.asarray(states, np.float32),
        np.asarray(valid, bool),
    )
    acceleration = x[..., 4]
    pair = mask[:, 1:] & mask[:, :-1]
    if pair.any():
        left, right = acceleration[:, 1:][pair], acceleration[:, :-1][pair]
        autocorrelation = (
            float(np.corrcoef(left, right)[0, 1])
            if len(left) > 1 and np.std(left) > 1e-7 and np.std(right) > 1e-7
            else 0.0
        )
        jerk_rms = float(np.sqrt(np.mean(((left - right) / 0.04) ** 2)))
    else:
        autocorrelation = jerk_rms = float("nan")
    # Cross-agent acceleration correlation on frames with at least two active slots.
    cross = []
    for sample, active in zip(acceleration, mask):
        for frame, frame_valid in zip(sample, active):
            value = frame[frame_valid]
            if len(value) > 1:
                cross.append(float(np.std(value)))
    lanes = np.round(x[..., 1] / 3.6).astype(np.int64)
    occupancy = np.zeros(3, np.float64)
    transitions = np.zeros((3, 3), np.float64)
    for item in range(x.shape[0]):
        relation = np.full((x.shape[1],), -1, np.int64)
        for frame in range(x.shape[1]):
            active = np.flatnonzero(mask[item, frame])
            if len(active) >= 2:
                delta = abs(
                    int(lanes[item, frame, active[0]])
                    - int(lanes[item, frame, active[1]])
                )
                relation[frame] = 0 if delta == 0 else 1 if delta == 1 else 2
                occupancy[relation[frame]] += 1
        for left, right in zip(relation[:-1], relation[1:]):
            if left >= 0 and right >= 0:
                transitions[left, right] += 1
    occupancy = occupancy / occupancy.sum() if occupancy.sum() else occupancy
    transition = transitions / np.maximum(transitions.sum(axis=1, keepdims=True), 1.0)
    return {
        "acceleration_lag1_autocorrelation": autocorrelation,
        "jerk_rms": jerk_rms,
        "cross_agent_acceleration_dispersion": (
            float(np.mean(cross)) if cross else float("nan")
        ),
        "relationship_occupancy": occupancy.tolist(),
        "relationship_transition": transition.tolist(),
    }
