"""Evaluation metrics for tail c0 normalizing flows."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from .data import split_indices
from .features import (
    DEFAULT_EGO_LENGTH_M,
    DEFAULT_EGO_WIDTH_M,
    DEFAULT_OTHER_LENGTH_M,
    DEFAULT_OTHER_WIDTH_M,
    EGO_FEATURES,
    SLOT_FEATURES,
    SLOT_NAMES,
    TRAJECTORY_FEATURES,
)


def evaluate_conditional_nll(flow, arrays: dict[str, np.ndarray], split: str, device, batch_size: int = 512) -> float:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    idx = split_indices(arrays, split)
    x = torch.from_numpy(arrays["features_normalized"][idx]).float()
    c = torch.from_numpy(arrays["contexts"][idx]).float()
    loader = DataLoader(TensorDataset(x, c), batch_size=int(batch_size), shuffle=False)
    total = 0.0
    total_n = 0
    flow.eval()
    with torch.no_grad():
        for batch_x, batch_c in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_c = batch_c.to(device, non_blocking=True)
            nll = -flow.log_prob(batch_x, context=batch_c)
            total += float(nll.sum().detach().cpu())
            total_n += int(batch_x.shape[0])
    return total / max(total_n, 1)


def conditional_nll_by_group(
    flow,
    arrays: dict[str, np.ndarray],
    *,
    schema: dict[str, Any],
    split: str,
    device,
) -> dict[str, Any]:
    import torch

    idx = split_indices(arrays, split)
    x = torch.from_numpy(arrays["features_normalized"][idx]).float().to(device)
    c = torch.from_numpy(arrays["contexts"][idx]).float().to(device)
    flow.eval()
    with torch.no_grad():
        nll = (-flow.log_prob(x, context=c)).detach().cpu().numpy()

    out: dict[str, Any] = {}
    mask_pattern = arrays["mask_pattern"][idx]
    out["mask_pattern"] = {}
    for pattern in sorted(np.unique(mask_pattern).tolist()):
        mask = mask_pattern == int(pattern)
        out["mask_pattern"][str(int(pattern))] = {
            "count": int(np.sum(mask)),
            "nll": float(np.mean(nll[mask])),
        }
    return out


def feature_valid_from_slot_mask(schema: dict[str, Any], slot_mask: np.ndarray) -> np.ndarray:
    n = slot_mask.shape[0]
    d = len(schema["feature_names"])
    out = np.zeros((n, d), dtype=bool)
    out[:, : len(EGO_FEATURES)] = True
    base = len(EGO_FEATURES)
    width = len(SLOT_FEATURES)
    for slot_idx in range(len(SLOT_NAMES)):
        start = base + slot_idx * width
        out[:, start : start + width] = slot_mask[:, [slot_idx]]
    trajectory_start = base + len(SLOT_NAMES) * width
    trajectory_width = len(TRAJECTORY_FEATURES)
    for slot_idx in range(len(SLOT_NAMES)):
        start = trajectory_start + slot_idx * trajectory_width
        out[:, start : start + trajectory_width] = slot_mask[:, [slot_idx]]
    return out


def distribution_match_metrics(
    real_features: np.ndarray,
    generated_features: np.ndarray,
    real_valid: np.ndarray,
    generated_valid: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    rows: list[dict[str, float | str | int]] = []
    for j, name in enumerate(feature_names):
        real = real_features[real_valid[:, j], j]
        gen = generated_features[generated_valid[:, j], j]
        real = real[np.isfinite(real)]
        gen = gen[np.isfinite(gen)]
        if len(real) < 5 or len(gen) < 5:
            continue
        rows.append(
            {
                "feature": name,
                "real_count": int(len(real)),
                "generated_count": int(len(gen)),
                "ks": float(ks_2samp(real, gen).statistic),
                "wasserstein": float(wasserstein_distance(real, gen)),
                "real_mean": float(np.mean(real)),
                "generated_mean": float(np.mean(gen)),
            }
        )
    if rows:
        avg_ks = float(np.mean([float(row["ks"]) for row in rows]))
        avg_w = float(np.mean([float(row["wasserstein"]) for row in rows]))
    else:
        avg_ks = float("nan")
        avg_w = float("nan")
    return {
        "num_features_compared": int(len(rows)),
        "mean_ks": avg_ks,
        "mean_wasserstein": avg_w,
        "per_feature": rows,
    }


def correlation_error(
    real_normalized: np.ndarray,
    generated_normalized: np.ndarray,
) -> dict[str, float]:
    if len(real_normalized) < 3 or len(generated_normalized) < 3:
        return {"pearson_corr_mae": float("nan")}
    real_corr = np.corrcoef(real_normalized, rowvar=False)
    gen_corr = np.corrcoef(generated_normalized, rowvar=False)
    real_corr = np.nan_to_num(real_corr, nan=0.0)
    gen_corr = np.nan_to_num(gen_corr, nan=0.0)
    mask = ~np.eye(real_corr.shape[0], dtype=bool)
    return {"pearson_corr_mae": float(np.mean(np.abs(real_corr[mask] - gen_corr[mask])))}


def _feature_index(slot_name: str | None, feature: str) -> int:
    if slot_name is None:
        return EGO_FEATURES.index(feature)
    return len(EGO_FEATURES) + SLOT_NAMES.index(slot_name) * len(SLOT_FEATURES) + SLOT_FEATURES.index(feature)


def _trajectory_feature_index(slot_name: str, feature: str) -> int:
    return (
        len(EGO_FEATURES)
        + len(SLOT_NAMES) * len(SLOT_FEATURES)
        + SLOT_NAMES.index(slot_name) * len(TRAJECTORY_FEATURES)
        + TRAJECTORY_FEATURES.index(feature)
    )


def physical_validity_flags(
    features: np.ndarray,
    slot_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, int], dict[str, np.ndarray]]:
    n = int(features.shape[0])
    reason_counts: Counter[str] = Counter()
    invalid_sample = np.zeros(n, dtype=bool)
    overlap_sample = np.zeros(n, dtype=bool)
    negative_gap_sample = np.zeros(n, dtype=bool)
    semantic_sample = np.zeros(n, dtype=bool)

    idx_vx = _feature_index(None, "ego_vx_mps")
    idx_vy = _feature_index(None, "ego_vy_left_mps")
    idx_ax = _feature_index(None, "ego_ax_mps2")
    idx_ay = _feature_index(None, "ego_ay_left_mps2")
    for i in range(n):
        ego_vx = float(features[i, idx_vx])
        ego_vy = float(features[i, idx_vy])
        ego_ax = float(features[i, idx_ax])
        ego_ay = float(features[i, idx_ay])
        ego_len = DEFAULT_EGO_LENGTH_M
        ego_wid = DEFAULT_EGO_WIDTH_M
        if not (-5.0 <= ego_vx <= 70.0 and abs(ego_vy) <= 10.0):
            reason_counts["ego_speed_out_of_range"] += 1
            invalid_sample[i] = True
        if abs(ego_ax) > 10.0 or abs(ego_ay) > 5.0:
            reason_counts["ego_acceleration_out_of_range"] += 1
            invalid_sample[i] = True
        for slot_idx, slot_name in enumerate(SLOT_NAMES):
            if not bool(slot_mask[i, slot_idx]):
                continue
            rel_x = float(features[i, _feature_index(slot_name, "rel_x_m")])
            rel_y = float(features[i, _feature_index(slot_name, "rel_y_left_m")])
            rel_vx = float(features[i, _feature_index(slot_name, "rel_vx_mps")])
            other_ax = float(features[i, _feature_index(slot_name, "other_ax_mps2")])
            other_ay = float(features[i, _feature_index(slot_name, "other_ay_left_mps2")])
            other_len = DEFAULT_OTHER_LENGTH_M
            other_wid = DEFAULT_OTHER_WIDTH_M
            other_vx = ego_vx + rel_vx
            if not (-10.0 <= other_vx <= 75.0):
                reason_counts["slot_speed_out_of_range"] += 1
                invalid_sample[i] = True
            if abs(other_ax) > 10.0 or abs(other_ay) > 5.0:
                reason_counts["slot_acceleration_out_of_range"] += 1
                invalid_sample[i] = True
            if "front" in slot_name and rel_x <= 0.0:
                reason_counts["front_slot_not_ahead"] += 1
                semantic_sample[i] = True
                invalid_sample[i] = True
            if "rear" in slot_name and rel_x >= 0.0:
                reason_counts["rear_slot_not_behind"] += 1
                semantic_sample[i] = True
                invalid_sample[i] = True
            if slot_name.startswith("left") and rel_y <= 0.0:
                reason_counts["left_slot_not_left"] += 1
                semantic_sample[i] = True
                invalid_sample[i] = True
            if slot_name.startswith("right") and rel_y >= 0.0:
                reason_counts["right_slot_not_right"] += 1
                semantic_sample[i] = True
                invalid_sample[i] = True
            longitudinal_gap = abs(rel_x) - 0.5 * (ego_len + other_len)
            if longitudinal_gap <= 0.0:
                reason_counts["negative_longitudinal_gap"] += 1
                negative_gap_sample[i] = True
                invalid_sample[i] = True
            lateral_overlap = abs(rel_y) < 0.5 * (ego_wid + other_wid)
            longitudinal_overlap = abs(rel_x) < 0.5 * (ego_len + other_len)
            if lateral_overlap and longitudinal_overlap:
                reason_counts["bounding_box_overlap"] += 1
                overlap_sample[i] = True
                invalid_sample[i] = True
            delta_vx = float(features[i, _trajectory_feature_index(slot_name, "delta_vx_1s_mps")])
            delta_vy = float(features[i, _trajectory_feature_index(slot_name, "delta_vy_left_1s_mps")])
            mean_ax = float(features[i, _trajectory_feature_index(slot_name, "mean_ax_1s_mps2")])
            min_ax = float(features[i, _trajectory_feature_index(slot_name, "min_ax_1s_mps2")])
            final_ax = float(features[i, _trajectory_feature_index(slot_name, "final_ax_1s_mps2")])
            mean_ay = float(features[i, _trajectory_feature_index(slot_name, "mean_ay_left_1s_mps2")])
            if not (
                -8.0 <= delta_vx <= 5.0
                and abs(delta_vy) <= 3.0
                and -8.0 <= mean_ax <= 5.0
                and -8.0 <= min_ax <= 5.0
                and -8.0 <= final_ax <= 5.0
                and abs(mean_ay) <= 3.0
            ):
                reason_counts["slot_action_summary_out_of_range"] += 1
                invalid_sample[i] = True
            if min_ax > mean_ax + 1.0e-3:
                reason_counts["slot_min_ax_exceeds_mean_ax"] += 1
                invalid_sample[i] = True

    return invalid_sample, {key: int(value) for key, value in sorted(reason_counts.items())}, {
        "overlap": overlap_sample,
        "negative_gap": negative_gap_sample,
        "semantic": semantic_sample,
    }


def physical_validity_metrics(
    features: np.ndarray,
    slot_mask: np.ndarray,
    schema: dict[str, Any],
) -> dict[str, Any]:
    del schema
    invalid_sample, reason_counts, detail = physical_validity_flags(features, slot_mask)
    n = int(features.shape[0])
    return {
        "num_samples": n,
        "invalid_rate": float(np.mean(invalid_sample)) if n else float("nan"),
        "overlap_rate": float(np.mean(detail["overlap"])) if n else float("nan"),
        "negative_gap_rate": float(np.mean(detail["negative_gap"])) if n else float("nan"),
        "semantic_error_rate": float(np.mean(detail["semantic"])) if n else float("nan"),
        "reason_counts": reason_counts,
    }


def occupancy_metrics(real_slot_mask: np.ndarray, generated_slot_mask: np.ndarray) -> dict[str, Any]:
    def counts(mask: np.ndarray) -> dict[str, int]:
        powers = (1 << np.arange(mask.shape[1], dtype=np.int64)).reshape(1, -1)
        pattern = np.sum(mask.astype(np.int64) * powers, axis=1)
        return {str(int(k)): int(v) for k, v in Counter(pattern.tolist()).items()}

    real_counts = counts(real_slot_mask)
    gen_counts = counts(generated_slot_mask)
    keys = sorted(set(real_counts) | set(gen_counts), key=int)
    real_total = max(int(real_slot_mask.shape[0]), 1)
    gen_total = max(int(generated_slot_mask.shape[0]), 1)
    l1 = 0.0
    for key in keys:
        l1 += abs(real_counts.get(key, 0) / real_total - gen_counts.get(key, 0) / gen_total)
    return {
        "mask_pattern_l1": float(l1),
        "real_counts": real_counts,
        "generated_counts": gen_counts,
    }


def nll_table_for_report(
    main_nll: dict[str, float],
    baselines: dict[str, Any],
    *,
    main_model: str = "conditional_maf",
) -> list[dict[str, Any]]:
    rows = [{"model": main_model, **main_nll}]
    for name, payload in baselines.items():
        row = {"model": name}
        row.update(payload.get("nll", {}))
        rows.append(row)
    return rows
