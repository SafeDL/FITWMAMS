"""Numerical diagnostics shared by formal long-tail reconstruction studies.

The functions in this module deliberately depend only on NumPy.  They define
the common measurement space for every world model rather than relying on a
model's private latent representation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


DT_S = 0.04
EPS = 1.0e-6
QUANTILES = (0.90, 0.95, 0.99)


def finite(values: np.ndarray) -> np.ndarray:
    """Flatten finite values without silently replacing invalid observations."""
    value = np.asarray(values, np.float64).reshape(-1)
    return value[np.isfinite(value)]


def masked_mean(values: np.ndarray, valid: np.ndarray) -> float:
    value = np.asarray(values, np.float64)
    mask = np.asarray(valid, bool)
    if not mask.any():
        return float("nan")
    return float(value[mask].mean())


def empirical_distance(real: np.ndarray, generated: np.ndarray) -> dict[str, Any]:
    """Dependency-free W1, KS, and fixed high-quantile errors."""
    left, right = finite(real), finite(generated)
    if not len(left) or not len(right):
        return {"available": False}
    grid = np.linspace(0.0, 1.0, 1001)
    left_q, right_q = np.quantile(left, grid), np.quantile(right, grid)
    support = np.sort(np.concatenate((left, right)))
    left_cdf = np.searchsorted(np.sort(left), support, side="right") / len(left)
    right_cdf = np.searchsorted(np.sort(right), support, side="right") / len(right)
    return {
        "available": True,
        "num_real": int(len(left)),
        "num_generated": int(len(right)),
        "wasserstein_1": float(np.mean(np.abs(left_q - right_q))),
        "ks": float(np.max(np.abs(left_cdf - right_cdf))),
        "quantiles": {
            f"q{int(level * 100)}": {
                "real": float(np.quantile(left, level)),
                "generated": float(np.quantile(right, level)),
                "absolute_error": float(
                    abs(np.quantile(left, level) - np.quantile(right, level))
                ),
            }
            for level in QUANTILES
        },
    }


def speed_kl_divergence(real: np.ndarray, generated: np.ndarray) -> float:
    """KL(real || generated) on the public 0--50 m/s, 0.5 m/s grid."""
    bins = np.linspace(0.0, 50.0, 101)
    left, right = finite(real), finite(generated)
    if not len(left) or not len(right):
        return float("nan")
    p = np.histogram(left, bins=bins)[0].astype(np.float64) + 1.0e-8
    q = np.histogram(right, bins=bins)[0].astype(np.float64) + 1.0e-8
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def histogram_kl_divergence(real: np.ndarray, generated: np.ndarray, *, bins: int = 100) -> float:
    """KL(real || generated) on a robust shared numeric support.

    Unlike :func:`speed_kl_divergence`, this is appropriate for signed motion
    variables such as curvature and yaw-related quantities.  The outer 0.1%
    is clipped into the edge bins so a single numerical outlier cannot make
    the comparison grid meaningless.
    """
    left, right = finite(real), finite(generated)
    if not len(left) or not len(right):
        return float("nan")
    combined = np.concatenate((left, right))
    lower, upper = np.quantile(combined, (0.001, 0.999))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper - lower < 1.0e-8:
        lower, upper = float(combined.min()) - 0.5, float(combined.max()) + 0.5
    left = np.clip(left, lower, upper)
    right = np.clip(right, lower, upper)
    edges = np.linspace(lower, upper, int(bins) + 1)
    p = np.histogram(left, bins=edges)[0].astype(np.float64) + 1.0e-8
    q = np.histogram(right, bins=edges)[0].astype(np.float64) + 1.0e-8
    p /= p.sum(); q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def _kinematics(states: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    value = np.asarray(states, np.float64)
    mask = np.asarray(valid, bool)
    velocity = value[..., 2:4]
    acceleration = value[..., 4:6]
    speed = np.linalg.norm(velocity, axis=-1)
    acceleration_norm = np.linalg.norm(acceleration, axis=-1)
    jerk = np.zeros_like(acceleration)
    jerk[:, 1:] = np.diff(acceleration, axis=1) / DT_S
    jerk_norm = np.linalg.norm(jerk, axis=-1)
    curvature = (
        velocity[..., 0] * acceleration[..., 1]
        - velocity[..., 1] * acceleration[..., 0]
    ) / np.maximum(speed**3, EPS)
    curvature_valid = mask & (speed >= 1.0)
    return {
        "valid": mask,
        "speed_mps": speed,
        "acceleration_mps2": acceleration_norm,
        "jerk_mps3": jerk_norm,
        "curvature_m_inv": curvature,
        "curvature_valid": curvature_valid,
    }


def trajectory_metrics(
    samples: np.ndarray, target: np.ndarray, valid: np.ndarray
) -> dict[str, Any]:
    """Deterministic, ensemble, and best-of-32 trajectory reconstruction."""
    draws = np.asarray(samples, np.float64)
    reference = np.asarray(target, np.float64)
    mask = np.asarray(valid, bool)
    distance = np.linalg.norm(draws[..., :2] - reference[None, ..., :2], axis=-1)
    weights = mask[None].astype(np.float64)
    episode_denom = weights.sum(axis=(2, 3)).clip(min=1.0)
    ade = (distance * weights).sum(axis=(2, 3)) / episode_denom
    final = mask[:, -1]
    final_denom = final.sum(axis=1).clip(min=1)
    fde = (distance[:, :, -1] * final[None]).sum(axis=2) / final_denom[None]
    result: dict[str, Any] = {
        "ADE_m": float(np.mean(ade[0])),
        "FDE_m": float(np.mean(fde[0])),
        "sample_mean_ADE_m": float(np.mean(ade)),
        "sample_mean_FDE_m": float(np.mean(fde)),
        "minADE_at_32_m": float(np.mean(np.min(ade, axis=0))),
        "minFDE_at_32_m": float(np.mean(np.min(fde, axis=0))),
        "coverage": {
            f"minFDE_le_{threshold:g}m": float(
                np.mean(np.min(fde, axis=0) <= threshold)
            )
            for threshold in (1.0, 2.0, 5.0)
        },
        "per_episode_min_fde_m": np.min(fde, axis=0),
    }
    for second in range(1, min(5, reference.shape[1] // 25) + 1):
        frame = second * 25 - 1
        prefix = mask[:, : frame + 1]
        prefix_count = prefix.sum(axis=(1, 2)).clip(min=1)
        result[f"ADE_{second}s_m"] = float(
            np.mean((distance[0, :, : frame + 1] * prefix).sum(axis=(1, 2)) / prefix_count)
        )
        result[f"FDE_{second}s_m"] = masked_mean(distance[0, :, frame], mask[:, frame])
    return result


def kinematic_reconstruction_metrics(
    predicted: np.ndarray, target: np.ndarray, valid: np.ndarray
) -> dict[str, float]:
    """Pointwise dynamics errors for the deterministic conditional rollout."""
    pred, ref = _kinematics(predicted, valid), _kinematics(target, valid)
    acceleration_error = np.linalg.norm(
        np.asarray(predicted)[..., 4:6] - np.asarray(target)[..., 4:6], axis=-1
    )
    jerk_pred = np.zeros_like(np.asarray(predicted)[..., 4:6])
    jerk_ref = np.zeros_like(np.asarray(target)[..., 4:6])
    jerk_pred[:, 1:] = np.diff(np.asarray(predicted)[..., 4:6], axis=1) / DT_S
    jerk_ref[:, 1:] = np.diff(np.asarray(target)[..., 4:6], axis=1) / DT_S
    jerk_error = np.linalg.norm(jerk_pred - jerk_ref, axis=-1)
    curve_mask = pred["curvature_valid"] & ref["curvature_valid"]
    return {
        "acceleration_vector_mae_mps2": masked_mean(acceleration_error, valid),
        "jerk_vector_mae_mps3": masked_mean(jerk_error[:, 1:], np.asarray(valid)[:, 1:]),
        "curvature_mae_m_inv": masked_mean(
            np.abs(pred["curvature_m_inv"] - ref["curvature_m_inv"]), curve_mask
        ),
    }


def _traffic_states(
    background: np.ndarray, ego: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    bg = np.asarray(background, np.float64)
    ego_value = np.asarray(ego, np.float64)
    bg_valid = np.asarray(valid, bool)
    ego_valid = np.isfinite(ego_value).all(axis=-1)
    return np.concatenate((ego_value[:, :, None], bg), axis=2), np.concatenate(
        (ego_valid[:, :, None], bg_valid), axis=2
    )


def traffic_fields(
    background: np.ndarray, ego: np.ndarray, valid: np.ndarray
) -> dict[str, np.ndarray]:
    """All-vehicle kinematics and directed same-lane following quantities."""
    states, present = _traffic_states(background, ego, valid)
    basic = _kinematics(states, present)
    x, y, vx = states[..., 0], states[..., 1], states[..., 2]
    # axis 2 is rear vehicle; axis 3 is potential front vehicle.
    dx = x[:, :, None, :] - x[:, :, :, None]
    dy = y[:, :, None, :] - y[:, :, :, None]
    closing = vx[:, :, :, None] - vx[:, :, None, :]
    directed = (
        present[:, :, :, None]
        & present[:, :, None, :]
        & (dx > 0.0)
        & (np.abs(dy) < 1.8)
    )
    gap = np.maximum(dx - 4.8, 0.0)
    ttc = np.where(closing > EPS, gap / np.maximum(closing, EPS), 10.0)
    ttc = np.clip(ttc, 0.0, 10.0)
    drac = np.where(
        closing > EPS, np.square(closing) / np.maximum(2.0 * gap, 0.1), 0.0
    )
    upper = np.triu(np.ones((states.shape[2], states.shape[2]), bool), k=1)
    collision = (
        present[:, :, :, None]
        & present[:, :, None, :]
        & upper[None, None]
        & (np.abs(x[:, :, :, None] - x[:, :, None, :]) < 4.8)
        & (np.abs(y[:, :, :, None] - y[:, :, None, :]) < 1.9)
    )
    return {
        **basic,
        "states": states,
        "present": present,
        "following_valid": directed,
        "gap_m": gap,
        "ttc_s": ttc,
        "drac_mps2": drac,
        "relative_speed_mps": closing,
        "collision": collision,
    }


def collision_metrics(fields: dict[str, np.ndarray]) -> dict[str, float]:
    collision = np.asarray(fields["collision"], bool)
    present = np.asarray(fields["present"], bool)
    vehicles = present.shape[-1]
    upper = np.triu(np.ones((vehicles, vehicles), bool), k=1)
    pair_valid = present[:, :, :, None] & present[:, :, None, :] & upper[None, None]
    return {
        "collision_pair_point_rate": float(collision.sum() / max(pair_valid.sum(), 1)),
        "collision_episode_rate": float(np.mean(collision.any(axis=(1, 2, 3)))),
    }


def following_error_metrics(
    predicted: dict[str, np.ndarray], target: dict[str, np.ndarray]
) -> dict[str, float]:
    mask = np.asarray(target["following_valid"], bool)
    return {
        "gap_mae_m": masked_mean(
            np.abs(predicted["gap_m"] - target["gap_m"]), mask
        ),
        "relative_speed_mae_mps": masked_mean(
            np.abs(predicted["relative_speed_mps"] - target["relative_speed_mps"]),
            mask,
        ),
        "ttc_mae_s": masked_mean(
            np.abs(predicted["ttc_s"] - target["ttc_s"]), mask
        ),
        "drac_mae_mps2": masked_mean(
            np.abs(predicted["drac_mps2"] - target["drac_mps2"]), mask
        ),
    }


def distribution_values(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    following = np.asarray(fields["following_valid"], bool)
    basic = np.asarray(fields["valid"], bool)
    return {
        "speed_mps": np.asarray(fields["speed_mps"])[basic],
        "acceleration_mps2": np.asarray(fields["acceleration_mps2"])[basic],
        "jerk_mps3": np.asarray(fields["jerk_mps3"])[basic],
        "curvature_m_inv": np.asarray(fields["curvature_m_inv"])[
            np.asarray(fields["curvature_valid"], bool)
        ],
        "gap_m": np.asarray(fields["gap_m"])[following],
        "ttc_s": np.asarray(fields["ttc_s"])[following],
        "drac_mps2": np.asarray(fields["drac_mps2"])[following],
        "relative_speed_mps": np.asarray(fields["relative_speed_mps"])[following],
    }


def _brake_response(fields: dict[str, np.ndarray]) -> dict[str, Any]:
    states = np.asarray(fields["states"])
    following = np.asarray(fields["following_valid"], bool)
    acceleration = states[..., 4]
    leader = np.broadcast_to(
        acceleration[:, :, None, :], fields["following_valid"].shape
    )
    follower = np.broadcast_to(
        acceleration[:, :, :, None], fields["following_valid"].shape
    )
    bins = np.asarray((-8.0, -3.0, -1.0, 0.0, 1.0, 3.0, 8.0))
    curve = []
    for low, high in zip(bins[:-1], bins[1:]):
        mask = following & (leader >= low) & (leader < high)
        curve.append(masked_mean(follower, mask))
    return {"bin_edges_mps2": bins.tolist(), "mean_follower_acceleration_mps2": curve}


def social_response_metrics(
    predicted: dict[str, np.ndarray], target: dict[str, np.ndarray]
) -> dict[str, Any]:
    pred, ref = _brake_response(predicted), _brake_response(target)
    return {
        "brake_response_real": ref,
        "brake_response_generated": pred,
        "brake_response_curve_mae_mps2": float(
            np.nanmean(
                np.abs(
                    np.asarray(pred["mean_follower_acceleration_mps2"])
                    - np.asarray(ref["mean_follower_acceleration_mps2"])
                )
            )
        ),
    }


def _feature_rows(background: np.ndarray, ego: np.ndarray, valid: np.ndarray) -> np.ndarray:
    fields = traffic_fields(background, ego, valid)
    states = fields["states"][:, :, 1:]
    present = fields["present"][:, :, 1:]
    kinematics = _kinematics(states, present)
    rows: list[list[float]] = []
    for episode in range(states.shape[0]):
        for agent in range(states.shape[2]):
            take = present[episode, :, agent]
            if not take.any():
                continue
            index = np.flatnonzero(take)
            first, last = int(index[0]), int(index[-1])
            speed = kinematics["speed_mps"][episode, take, agent]
            accel = kinematics["acceleration_mps2"][episode, take, agent]
            jerk = kinematics["jerk_mps3"][episode, take, agent]
            curve = kinematics["curvature_m_inv"][episode, take, agent]
            # ``fields`` includes ego at participant zero; ``states`` above
            # intentionally excludes it, so offset every background slot.
            following = fields["following_valid"][episode, :, agent + 1]
            gap = fields["gap_m"][episode, :, agent + 1][following]
            ttc = fields["ttc_s"][episode, :, agent + 1][following]
            drac = fields["drac_mps2"][episode, :, agent + 1][following]
            braking = accel > 0.5
            rows.append([
                float(speed[0]), float(speed[-1]), float(speed[-1] - speed[0]),
                float(accel.mean()), float(accel.min()), float(jerk.mean()),
                float(np.abs(curve).mean()),
                float(states[episode, last, agent, 0] - states[episode, first, agent, 0]),
                float(states[episode, last, agent, 1] - states[episode, first, agent, 1]),
                float(np.flatnonzero(braking)[0] * DT_S) if braking.any() else 5.0,
                float(braking.mean() * len(accel) * DT_S),
                float(gap.min()) if len(gap) else 100.0,
                float(ttc.min()) if len(ttc) else 10.0,
                float(drac.max()) if len(drac) else 0.0,
            ])
    return np.asarray(rows, np.float64).reshape(-1, 14)


def feature_distribution_distance(
    generated_background: np.ndarray,
    generated_ego: np.ndarray,
    generated_valid: np.ndarray,
    real_background: np.ndarray,
    real_ego: np.ndarray,
    real_valid: np.ndarray,
    *,
    seed: int,
    maximum: int = 4096,
) -> dict[str, float]:
    """Traffic-feature Fréchet distance and RBF-MMD in one shared space."""
    left = _feature_rows(generated_background, generated_ego, generated_valid)
    right = _feature_rows(real_background, real_ego, real_valid)
    rng = np.random.default_rng(int(seed))
    if len(left) > maximum:
        left = left[rng.choice(len(left), maximum, replace=False)]
    if len(right) > maximum:
        right = right[rng.choice(len(right), maximum, replace=False)]
    if not len(left) or not len(right):
        return {"traffic_feature_frechet_distance": float("nan"), "mmd_rbf": float("nan")}
    mean = np.concatenate((left, right)).mean(axis=0)
    scale = np.maximum(np.concatenate((left, right)).std(axis=0), 1.0e-4)
    left, right = (left - mean) / scale, (right - mean) / scale
    covariance_left = np.cov(left, rowvar=False) + np.eye(left.shape[1]) * 1.0e-6
    covariance_right = np.cov(right, rowvar=False) + np.eye(right.shape[1]) * 1.0e-6
    eig, vectors = np.linalg.eigh(covariance_left)
    root_left = (vectors * np.sqrt(np.clip(eig, 0.0, None))) @ vectors.T
    middle = root_left @ covariance_right @ root_left
    middle_eig = np.linalg.eigvalsh((middle + middle.T) * 0.5)
    frechet = (
        np.square(left.mean(axis=0) - right.mean(axis=0)).sum()
        + np.trace(covariance_left)
        + np.trace(covariance_right)
        - 2.0 * np.sqrt(np.clip(middle_eig, 0.0, None)).sum()
    )

    def kernel(first: np.ndarray, second: np.ndarray, bandwidth: float) -> float:
        total = count = 0.0
        for start in range(0, len(first), 256):
            squared = ((first[start : start + 256, None] - second[None]) ** 2).sum(axis=-1)
            total += float(np.exp(-squared / (2.0 * bandwidth**2)).sum())
            count += squared.size
        return total / max(count, 1.0)

    probe = np.linalg.norm(left[: min(512, len(left)), None] - right[None, : min(512, len(right))], axis=-1)
    bandwidth = float(np.median(probe).clip(min=1.0e-4))
    mmd = kernel(left, left, bandwidth) + kernel(right, right, bandwidth) - 2.0 * kernel(left, right, bandwidth)
    return {
        "traffic_feature_frechet_distance": float(max(frechet, 0.0)),
        "mmd_rbf": float(mmd),
    }


def event_masks(target: np.ndarray, ego: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    """Deterministic event labels computed only from held-out real traffic."""
    fields = traffic_fields(target, ego, valid)
    present = np.asarray(valid, bool)
    acceleration = np.asarray(target)[..., 4]
    relative = np.asarray(target)[..., 2] - np.asarray(ego)[:, :, None, 2]
    min_gap = np.full(len(target), np.inf)
    min_ttc = np.full(len(target), np.inf)
    max_closing = np.zeros(len(target))
    following = fields["following_valid"]
    for index in range(len(target)):
        take = following[index]
        if take.any():
            min_gap[index] = fields["gap_m"][index][take].min()
            min_ttc[index] = fields["ttc_s"][index][take].min()
            max_closing[index] = fields["relative_speed_mps"][index][take].max()
    high = np.where(present, relative, -np.inf).max(axis=1)
    low = np.where(present, relative, np.inf).min(axis=1)
    delta = np.where(present.any(axis=1), high - low, -np.inf).max(axis=1)
    return {
        "high_risk_following": min_ttc < 3.0,
        "hard_braking": np.where(present, acceleration, np.inf).min(axis=(1, 2)) < -1.5,
        "high_speed_approach": max_closing > 5.0,
        "close_interaction": min_gap < 8.0,
        "strong_relative_speed_change": delta > 3.0,
    }
