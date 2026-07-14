"""Open-loop, closed-loop and physical diagnostics for world-model rollouts."""
from __future__ import annotations

from typing import Any

import numpy as np

from .schema import (
    DEFAULT_EGO_LENGTH_M,
    DEFAULT_EGO_WIDTH_M,
    DEFAULT_OTHER_LENGTH_M,
    DEFAULT_OTHER_WIDTH_M,
    SLOT_NAMES,
)


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(values[mask]))


def action_error_metrics(
    predicted_actions: np.ndarray,
    target_actions: np.ndarray,
    target_valid: np.ndarray,
) -> dict[str, float]:
    pred = np.asarray(predicted_actions, dtype=np.float32)
    target = np.asarray(target_actions, dtype=np.float32)
    valid = np.asarray(target_valid, dtype=bool)
    err = pred - target
    mse_ax = _masked_mean(err[..., 0] ** 2, valid)
    mse_ay = _masked_mean(err[..., 1] ** 2, valid)
    mae_ax = _masked_mean(np.abs(err[..., 0]), valid)
    mae_ay = _masked_mean(np.abs(err[..., 1]), valid)
    return {
        "action_rmse_ax_mps2": float(np.sqrt(mse_ax)) if np.isfinite(mse_ax) else float("nan"),
        "action_rmse_ay_left_mps2": float(np.sqrt(mse_ay)) if np.isfinite(mse_ay) else float("nan"),
        "action_mae_ax_mps2": mae_ax,
        "action_mae_ay_left_mps2": mae_ay,
    }


def trajectory_error_metrics(
    predicted_states: np.ndarray,
    target_states: np.ndarray,
    target_valid: np.ndarray,
) -> dict[str, float]:
    pred = np.asarray(predicted_states, dtype=np.float32)
    target = np.asarray(target_states, dtype=np.float32)
    valid = np.asarray(target_valid, dtype=bool)
    dist = np.linalg.norm(pred[..., :2] - target[..., :2], axis=-1)
    ade = _masked_mean(dist, valid)
    final_valid = valid[:, -1, :]
    fde = _masked_mean(dist[:, -1, :], final_valid)
    out = {
        "ADE_m": ade,
        "FDE_m": fde,
        "longitudinal_mae_m": _masked_mean(np.abs(pred[..., 0] - target[..., 0]), valid),
        "lateral_mae_m": _masked_mean(np.abs(pred[..., 1] - target[..., 1]), valid),
        "velocity_mae_mps": _masked_mean(np.linalg.norm(pred[..., 2:4] - target[..., 2:4], axis=-1), valid),
        "acceleration_mae_mps2": _masked_mean(np.linalg.norm(pred[..., 4:6] - target[..., 4:6], axis=-1), valid),
    }
    for slot_idx, slot_name in enumerate(SLOT_NAMES):
        slot_valid = valid[:, :, slot_idx]
        slot_dist = dist[:, :, slot_idx]
        out[f"slot_{slot_name}_ADE_m"] = _masked_mean(slot_dist, slot_valid)
        out[f"slot_{slot_name}_FDE_m"] = _masked_mean(
            slot_dist[:, -1],
            slot_valid[:, -1],
        )
    return out


def min_sample_trajectory_metrics(
    sampled_states: np.ndarray,
    target_states: np.ndarray,
    target_valid: np.ndarray,
) -> dict[str, float]:
    """Compute minADE/minFDE over sampled futures.

    Args:
        sampled_states: `[S, N, K, 6, state_dim]`.
    """
    samples = np.asarray(sampled_states, dtype=np.float32)
    target = np.asarray(target_states, dtype=np.float32)[None]
    valid = np.asarray(target_valid, dtype=bool)[None]
    dist = np.linalg.norm(samples[..., :2] - target[..., :2], axis=-1)
    denom = valid.sum(axis=(2, 3)).clip(min=1)
    ade_per_sample = (dist * valid).sum(axis=(2, 3)) / denom
    min_ade = np.min(ade_per_sample, axis=0)
    final_valid = np.asarray(target_valid[:, -1, :], dtype=bool)
    fde_per_sample = dist[:, :, -1, :]
    min_fde = np.min(np.where(final_valid[None], fde_per_sample, np.nan), axis=0)
    return {
        "minADE_m": float(np.nanmean(min_ade)),
        "minFDE_m": float(np.nanmean(min_fde)),
    }


def branch_diversity_metrics(sampled_states: np.ndarray, target_valid: np.ndarray) -> dict[str, float]:
    samples = np.asarray(sampled_states, dtype=np.float32)
    valid = np.asarray(target_valid, dtype=bool)
    if samples.shape[0] < 2:
        return {"branch_pairwise_ADE_m": float("nan"), "effective_unique_futures": 1.0}
    pair_values: list[float] = []
    unique_counts: list[int] = []
    for n in range(samples.shape[1]):
        sample_scores = []
        for i in range(samples.shape[0]):
            for j in range(i + 1, samples.shape[0]):
                dist = np.linalg.norm(samples[i, n, ..., :2] - samples[j, n, ..., :2], axis=-1)
                if np.any(valid[n]):
                    pair_values.append(float(np.mean(dist[valid[n]])))
                    sample_scores.append(float(np.mean(samples[i, n, ..., :2][valid[n]])))
        if sample_scores:
            rounded = {round(value, 2) for value in sample_scores}
            unique_counts.append(len(rounded))
    return {
        "branch_pairwise_ADE_m": float(np.mean(pair_values)) if pair_values else float("nan"),
        "effective_unique_futures": float(np.mean(unique_counts)) if unique_counts else float("nan"),
        "mode_collapse_rate": float(np.mean([count <= 1 for count in unique_counts])) if unique_counts else float("nan"),
    }


def physical_diagnostics(
    predicted_states: np.ndarray,
    predicted_valid: np.ndarray,
    *,
    ego_future_states: np.ndarray | None = None,
    actions: np.ndarray | None = None,
    dt: float = 0.04,
) -> dict[str, Any]:
    states = np.asarray(predicted_states, dtype=np.float32)
    valid = np.asarray(predicted_valid, dtype=bool)
    n = int(states.shape[0])
    if ego_future_states is None:
        ego = np.zeros((n, states.shape[1], states.shape[-1]), dtype=np.float32)
    else:
        ego = np.asarray(ego_future_states, dtype=np.float32)

    speed = np.linalg.norm(states[..., 2:4], axis=-1)
    accel = np.linalg.norm(states[..., 4:6], axis=-1)
    speed_bad = valid & ((speed < -1.0e-6) | (speed > 75.0))
    accel_bad = valid & (accel > 12.0)
    jerk_bad = np.zeros_like(valid)
    if actions is not None and actions.shape[1] > 1:
        jerk = np.linalg.norm(np.diff(np.asarray(actions, dtype=np.float32), axis=1), axis=-1) / max(float(dt), 1.0e-6)
        jerk_bad[:, 1:, :] = jerk > 40.0

    semantic_bad = np.zeros_like(valid)
    gap_bad = np.zeros_like(valid)
    overlap_bad = np.zeros_like(valid)
    for slot_idx, slot_name in enumerate(SLOT_NAMES):
        rel_x = states[:, :, slot_idx, 0] - ego[:, :, 0]
        rel_y = states[:, :, slot_idx, 1] - ego[:, :, 1]
        if "front" in slot_name:
            semantic_bad[:, :, slot_idx] |= valid[:, :, slot_idx] & (rel_x <= 0.0)
        if "rear" in slot_name:
            semantic_bad[:, :, slot_idx] |= valid[:, :, slot_idx] & (rel_x >= 0.0)
        if slot_name.startswith("left"):
            semantic_bad[:, :, slot_idx] |= valid[:, :, slot_idx] & (rel_y <= 0.0)
        if slot_name.startswith("right"):
            semantic_bad[:, :, slot_idx] |= valid[:, :, slot_idx] & (rel_y >= 0.0)
        longitudinal_gap = np.abs(rel_x) - 0.5 * (DEFAULT_EGO_LENGTH_M + DEFAULT_OTHER_LENGTH_M)
        gap_bad[:, :, slot_idx] |= valid[:, :, slot_idx] & (longitudinal_gap <= 0.0)
        lateral_overlap = np.abs(rel_y) < 0.5 * (DEFAULT_EGO_WIDTH_M + DEFAULT_OTHER_WIDTH_M)
        longitudinal_overlap = np.abs(rel_x) < 0.5 * (DEFAULT_EGO_LENGTH_M + DEFAULT_OTHER_LENGTH_M)
        overlap_bad[:, :, slot_idx] |= valid[:, :, slot_idx] & lateral_overlap & longitudinal_overlap

    denom = max(int(np.sum(valid)), 1)
    sample_invalid = np.any(speed_bad | accel_bad | jerk_bad | semantic_bad | gap_bad | overlap_bad, axis=(1, 2))
    return {
        "num_samples": n,
        "invalid_rate": float(np.mean(sample_invalid)) if n else float("nan"),
        "speed_out_of_range_rate": float(np.sum(speed_bad) / denom),
        "acceleration_out_of_range_rate": float(np.sum(accel_bad) / denom),
        "jerk_out_of_range_rate": float(np.sum(jerk_bad) / denom),
        "semantic_error_rate": float(np.sum(semantic_bad) / denom),
        "negative_gap_rate": float(np.sum(gap_bad) / denom),
        "overlap_rate": float(np.sum(overlap_bad) / denom),
    }


def interaction_metrics(
    predicted_states: np.ndarray,
    target_states: np.ndarray,
    valid: np.ndarray,
    *,
    ego_future_states: np.ndarray,
) -> dict[str, float]:
    pred = np.asarray(predicted_states, dtype=np.float32)
    target = np.asarray(target_states, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    ego = np.asarray(ego_future_states, dtype=np.float32)
    pred_gap = np.abs(pred[..., 0] - ego[:, :, None, 0])
    target_gap = np.abs(target[..., 0] - ego[:, :, None, 0])
    pred_rel_v = pred[..., 2] - ego[:, :, None, 2]
    target_rel_v = target[..., 2] - ego[:, :, None, 2]
    gap_mae = _masked_mean(np.abs(pred_gap - target_gap), mask)
    rel_v_mae = _masked_mean(np.abs(pred_rel_v - target_rel_v), mask)

    eps = 1.0e-3
    closing = pred_rel_v < -eps
    pred_ttc = np.where(closing, pred_gap / np.maximum(-pred_rel_v, eps), np.inf)
    target_closing = target_rel_v < -eps
    target_ttc = np.where(target_closing, target_gap / np.maximum(-target_rel_v, eps), np.inf)
    finite_ttc_mask = mask & np.isfinite(pred_ttc) & np.isfinite(target_ttc)
    pred_ttc_clip = np.clip(pred_ttc, 0.0, 10.0)
    target_ttc_clip = np.clip(target_ttc, 0.0, 10.0)
    pred_closing_speed = np.maximum(-pred_rel_v, 0.0)
    target_closing_speed = np.maximum(-target_rel_v, 0.0)
    pred_drac = np.where(
        pred_closing_speed > eps,
        pred_closing_speed ** 2 / np.maximum(2.0 * pred_gap, eps),
        0.0,
    )
    target_drac = np.where(
        target_closing_speed > eps,
        target_closing_speed ** 2 / np.maximum(2.0 * target_gap, eps),
        0.0,
    )
    pred_small_ttc = mask & (pred_ttc < 3.0)
    target_small_ttc = mask & (target_ttc < 3.0)
    pred_high_risk = mask & ((pred_ttc_clip < 3.0) | (pred_drac > 2.0))
    target_high_risk = mask & ((target_ttc_clip < 3.0) | (target_drac > 2.0))
    return {
        "gap_mae_m": gap_mae,
        "relative_vx_mae_mps": rel_v_mae,
        "ttc_error_s": _masked_mean(np.abs(pred_ttc_clip - target_ttc_clip), finite_ttc_mask),
        "drac_error_mps2": _masked_mean(np.abs(pred_drac - target_drac), mask),
        "small_ttc_rate_pred": float(np.sum(pred_small_ttc) / max(int(np.sum(mask)), 1)),
        "small_ttc_rate_target": float(np.sum(target_small_ttc) / max(int(np.sum(mask)), 1)),
        "risk_relaxation_rate": float(
            np.sum(target_high_risk & ~pred_high_risk) / max(int(np.sum(target_high_risk)), 1)
        ),
    }
