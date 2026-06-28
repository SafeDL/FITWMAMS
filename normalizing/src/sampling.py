"""Sampling and export interfaces for trained highD tail-event flows."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import SPLIT_TO_INDEX, load_tail_dataset
from .features import (
    DEFAULT_EGO_LENGTH_M,
    DEFAULT_EGO_WIDTH_M,
    DEFAULT_LANE_WIDTH_M,
    DEFAULT_OTHER_LENGTH_M,
    DEFAULT_OTHER_WIDTH_M,
    EGO_FEATURES,
    SLOT_FEATURES,
    SLOT_NAMES,
    TRAJECTORY_FEATURES,
)
from .metrics import feature_valid_from_slot_mask, physical_validity_flags
from .model import load_maf_checkpoint
from .transforms import inverse_transform_model_features, transform_features_for_model


def inverse_normalize_features(x_norm: np.ndarray, schema: dict[str, Any]) -> np.ndarray:
    norm = schema["normalization"]
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    model_features = (np.asarray(x_norm, dtype=np.float32) * std + mean).astype(np.float32)
    return inverse_transform_model_features(
        model_features,
        list(schema["feature_names"]),
    )


def normalize_features(raw: np.ndarray, valid: np.ndarray, schema: dict[str, Any]) -> np.ndarray:
    norm = schema["normalization"]
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    model_features = transform_features_for_model(
        np.asarray(raw, dtype=np.float32),
        np.asarray(valid, dtype=bool),
        list(schema["feature_names"]),
    )
    out = np.zeros_like(raw, dtype=np.float32)
    out[valid] = ((model_features - mean) / std)[valid]
    return out


def zero_inactive_slot_features(raw: np.ndarray, slot_mask: np.ndarray) -> np.ndarray:
    out = np.asarray(raw, dtype=np.float32).copy()
    base = len(EGO_FEATURES)
    width = len(SLOT_FEATURES)
    trajectory_base = base + len(SLOT_NAMES) * width
    trajectory_width = len(TRAJECTORY_FEATURES)
    for slot_idx in range(len(SLOT_NAMES)):
        inactive = ~slot_mask[:, slot_idx].astype(bool)
        start = base + slot_idx * width
        out[inactive, start : start + width] = 0.0
        trajectory_start = trajectory_base + slot_idx * trajectory_width
        out[inactive, trajectory_start : trajectory_start + trajectory_width] = 0.0
    return out


def _mask_pattern(slot_mask: np.ndarray) -> np.ndarray:
    powers = (1 << np.arange(slot_mask.shape[1], dtype=np.int64)).reshape(1, -1)
    return np.sum(slot_mask.astype(np.int64) * powers, axis=1).astype(np.int64)


def _slot_mask_from_pattern(mask_pattern: np.ndarray) -> np.ndarray:
    pattern = np.asarray(mask_pattern, dtype=np.int64).reshape(-1)
    powers = (1 << np.arange(len(SLOT_NAMES), dtype=np.int64)).reshape(1, -1)
    return ((pattern.reshape(-1, 1) & powers) > 0)


def _split_candidates(arrays: dict[str, np.ndarray], split: str) -> np.ndarray:
    if str(split).lower() in {"all", "full", "dataset"}:
        return np.arange(len(arrays["features"]), dtype=np.int64)
    split_value = SPLIT_TO_INDEX.get(str(split), SPLIT_TO_INDEX["train"])
    return np.where(arrays["split_index"] == split_value)[0]


def _primary_slot_index(primary_slot: str | int | None) -> int | None:
    if primary_slot is None:
        return None
    if isinstance(primary_slot, (int, np.integer)):
        value = int(primary_slot)
        if value < 0 or value >= len(SLOT_NAMES):
            raise ValueError(f"primary_slot index out of range: {primary_slot}")
        return value
    value = str(primary_slot)
    if value not in SLOT_NAMES:
        raise ValueError(f"Unknown primary_slot={primary_slot!r}; expected one of {SLOT_NAMES}")
    return SLOT_NAMES.index(value)


def _contexts_from_event_structure(
    schema: dict[str, Any],
    slot_mask: np.ndarray,
    primary_slot_index: np.ndarray,
) -> np.ndarray:
    context_names = list(schema["context_names"])
    contexts = np.zeros((len(slot_mask), len(context_names)), dtype=np.float32)
    for slot_idx, slot_name in enumerate(SLOT_NAMES):
        mask_name = f"mask_{slot_name}"
        if mask_name in context_names:
            contexts[:, context_names.index(mask_name)] = slot_mask[:, slot_idx].astype(np.float32)
        primary_name = f"primary_slot_{slot_name}"
        if primary_name in context_names:
            contexts[:, context_names.index(primary_name)] = (
                primary_slot_index.astype(np.int64) == slot_idx
            ).astype(np.float32)
    return contexts


def _event_structure_distribution(
    arrays: dict[str, np.ndarray],
    *,
    split: str = "train",
    mask_pattern: int | None = None,
    primary_slot: str | int | None = None,
) -> dict[str, np.ndarray]:
    candidates = _split_candidates(arrays, split)
    if mask_pattern is not None:
        candidates = candidates[arrays["mask_pattern"][candidates] == int(mask_pattern)]
    primary_idx = _primary_slot_index(primary_slot)
    if primary_idx is not None:
        candidates = candidates[arrays["primary_slot_index"][candidates].astype(np.int64) == primary_idx]
    if len(candidates) == 0:
        raise RuntimeError(
            "No empirical EVT-tail event structures match requested controlled condition: "
            f"split={split} mask_pattern={mask_pattern} primary_slot={primary_slot}"
        )
    keys = np.stack(
        [
            arrays["mask_pattern"][candidates].astype(np.int64),
            arrays["primary_slot_index"][candidates].astype(np.int64),
        ],
        axis=1,
    )
    unique, counts = np.unique(keys, axis=0, return_counts=True)
    probs = counts.astype(np.float64) / float(np.sum(counts))
    return {
        "mask_pattern": unique[:, 0].astype(np.int64),
        "primary_slot_index": unique[:, 1].astype(np.int64),
        "counts": counts.astype(np.int64),
        "probabilities": probs.astype(np.float64),
    }


def sample_event_structures(
    arrays: dict[str, np.ndarray],
    schema: dict[str, Any],
    *,
    num_samples: int,
    seed: int,
    split: str = "train",
    mask_pattern: int | None = None,
    primary_slot: str | int | None = None,
) -> dict[str, np.ndarray]:
    """Sample the discrete event skeleton p(mask, primary_slot | EVT-tail).

    This is part of the default joint-event sampler. `mask_pattern` and
    `primary_slot` are optional controlled-test filters, not required inputs.
    """

    rng = np.random.default_rng(int(seed))
    table = _event_structure_distribution(
        arrays,
        split=split,
        mask_pattern=mask_pattern,
        primary_slot=primary_slot,
    )
    category = rng.choice(
        np.arange(len(table["probabilities"]), dtype=np.int64),
        size=int(num_samples),
        replace=True,
        p=table["probabilities"],
    )
    sampled_pattern = table["mask_pattern"][category].astype(np.int64)
    sampled_primary = table["primary_slot_index"][category].astype(np.int64)
    slot_mask = _slot_mask_from_pattern(sampled_pattern)
    contexts = _contexts_from_event_structure(schema, slot_mask, sampled_primary)
    return {
        "contexts": contexts,
        "event_structure": contexts.copy(),
        "slot_mask": slot_mask.astype(bool),
        "mask_pattern": sampled_pattern,
        "primary_slot_index": sampled_primary,
        "primary_slot_name": np.asarray([SLOT_NAMES[int(idx)] for idx in sampled_primary]),
        "event_structure_id": category.astype(np.int64),
        "event_structure_log_prob": np.log(table["probabilities"][category]).astype(np.float32),
    }


def event_structure_log_prob(
    arrays: dict[str, np.ndarray],
    *,
    mask_pattern: np.ndarray,
    primary_slot_index: np.ndarray,
    split: str = "train",
    smoothing: float = 1.0e-6,
) -> np.ndarray:
    """Evaluate log p(mask, primary_slot | EVT-tail) with empirical smoothing."""

    support_keys = np.stack(
        [
            arrays["mask_pattern"].astype(np.int64),
            arrays["primary_slot_index"].astype(np.int64),
        ],
        axis=1,
    )
    support = np.unique(support_keys, axis=0)
    candidates = _split_candidates(arrays, split)
    train_keys = np.stack(
        [
            arrays["mask_pattern"][candidates].astype(np.int64),
            arrays["primary_slot_index"][candidates].astype(np.int64),
        ],
        axis=1,
    )
    counts = {tuple(key.tolist()): float(smoothing) for key in support}
    for key in train_keys:
        counts[tuple(key.tolist())] = counts.get(tuple(key.tolist()), float(smoothing)) + 1.0
    total = float(sum(counts.values()))
    out = np.empty(len(mask_pattern), dtype=np.float32)
    for idx, key in enumerate(zip(mask_pattern.astype(np.int64), primary_slot_index.astype(np.int64))):
        out[idx] = float(np.log(counts.get(tuple(key), float(smoothing)) / max(total, 1.0e-12)))
    return out


def sample_tail_c0(
    flow,
    arrays: dict[str, np.ndarray],
    schema: dict[str, Any],
    *,
    num_samples: int,
    device,
    seed: int = 42,
    mask_pattern: int | None = None,
    primary_slot: str | int | None = None,
    event_structure_split: str = "train",
    reject_invalid: bool = True,
    max_rounds: int = 10,
) -> dict[str, np.ndarray]:
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    flow.eval()
    accepted: list[dict[str, np.ndarray]] = []
    rejected = 0
    remaining = int(num_samples)
    rounds = 0
    while remaining > 0 and rounds < int(max_rounds):
        rounds += 1
        draw_n = remaining if not reject_invalid else max(remaining * 3, 128)
        event_structure = sample_event_structures(
            arrays,
            schema,
            num_samples=draw_n,
            seed=int(seed) + rounds * 1543,
            split=event_structure_split,
            mask_pattern=mask_pattern,
            primary_slot=primary_slot,
        )
        contexts = event_structure["contexts"].astype(np.float32)
        slot_mask = event_structure["slot_mask"].astype(bool)
        context_t = torch.from_numpy(contexts).float().to(device)
        with torch.no_grad():
            samples = flow.sample(1, context=context_t)
            if samples.ndim == 3:
                samples = samples[:, 0, :]
            x_norm_model = samples.detach().cpu().numpy().astype(np.float32)
        raw = inverse_normalize_features(x_norm_model, schema)
        raw = zero_inactive_slot_features(raw, slot_mask)
        valid = feature_valid_from_slot_mask(schema, slot_mask)
        x_norm = normalize_features(raw, valid, schema)
        finite = np.isfinite(x_norm).all(axis=1) & np.isfinite(raw).all(axis=1)
        conditional_log_prob = np.full(draw_n, -np.inf, dtype=np.float32)
        if np.any(finite):
            with torch.no_grad():
                conditional_log_prob[finite] = flow.log_prob(
                    torch.from_numpy(x_norm[finite]).float().to(device),
                    context=torch.from_numpy(contexts[finite]).float().to(device),
                ).detach().cpu().numpy().astype(np.float32)
        keep = finite.copy()
        if reject_invalid:
            invalid, _reasons, _detail = physical_validity_flags(
                np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0),
                slot_mask,
            )
            too_large = np.max(np.abs(x_norm), axis=1) > float(
                schema.get("sampling_max_abs_normalized", 8.0)
            )
            keep = finite & ~(invalid | too_large)
            rejected += int(np.sum(~keep))
        take = np.where(keep)[0][:remaining]
        if len(take):
            event_log_prob = event_structure["event_structure_log_prob"][take].astype(np.float32)
            accepted.append(
                {
                    "features": raw[take].astype(np.float32),
                    "features_normalized": x_norm[take].astype(np.float32),
                    "feature_valid": valid[take],
                    "contexts": contexts[take],
                    "event_structure": event_structure["event_structure"][take],
                    "slot_mask": slot_mask[take],
                    "mask_pattern": event_structure["mask_pattern"][take].astype(np.int64),
                    "primary_slot_index": event_structure["primary_slot_index"][take].astype(np.int64),
                    "primary_slot_name": event_structure["primary_slot_name"][take],
                    "event_structure_id": event_structure["event_structure_id"][take].astype(np.int64),
                    "event_structure_log_prob": event_log_prob,
                    "conditional_log_prob": conditional_log_prob[take].astype(np.float32),
                    "log_prob": (conditional_log_prob[take] + event_log_prob).astype(np.float32),
                }
            )
            remaining -= len(take)
    if not accepted:
        raise RuntimeError("Flow sampler could not produce any accepted samples")
    out = {
        key: np.concatenate([chunk[key] for chunk in accepted], axis=0)[: int(num_samples)]
        for key in accepted[0]
    }
    out["num_rejected"] = np.asarray([rejected], dtype=np.int64)
    out["rejection_rate"] = np.asarray(
        [rejected / max(rejected + len(out["features"]), 1)],
        dtype=np.float32,
    )
    return out


def feature_frame(features: np.ndarray, schema: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(features, columns=list(schema["feature_names"]))


def samples_to_frame(samples: dict[str, np.ndarray], schema: dict[str, Any]) -> pd.DataFrame:
    frame = feature_frame(samples["features"], schema)
    frame.insert(0, "sample_id", np.arange(len(frame), dtype=np.int64))
    frame.insert(1, "joint_log_prob", samples["log_prob"])
    if "conditional_log_prob" in samples:
        frame.insert(2, "continuous_log_prob", samples["conditional_log_prob"])
        frame.insert(3, "event_structure_log_prob", samples["event_structure_log_prob"])
        frame.insert(4, "mask_pattern", samples["mask_pattern"].astype(np.int64))
        next_insert = 5
    else:
        frame.insert(2, "mask_pattern", samples["mask_pattern"].astype(np.int64))
        next_insert = 3
    if "primary_slot_name" in samples:
        frame.insert(next_insert, "primary_slot", samples["primary_slot_name"].astype(str))
        mask_insert_start = next_insert + 1
    else:
        mask_insert_start = next_insert
    for slot_idx, slot_name in enumerate(SLOT_NAMES):
        frame.insert(mask_insert_start + slot_idx, f"{slot_name}_mask", samples["slot_mask"][:, slot_idx].astype(int))
    return frame


def _ego_dict(row: np.ndarray) -> dict[str, float]:
    values = {name: float(row[EGO_FEATURES.index(name)]) for name in EGO_FEATURES}
    return {
        "slot": "ego",
        "x_m": 0.0,
        "y_left_m": 0.0,
        "speed_mps": float(np.hypot(values["ego_vx_mps"], values["ego_vy_left_mps"])),
        "vx_mps": float(values["ego_vx_mps"]),
        "vy_left_mps": float(values["ego_vy_left_mps"]),
        "ax_mps2": float(values["ego_ax_mps2"]),
        "ay_left_mps2": float(values["ego_ay_left_mps2"]),
        "length_m": DEFAULT_EGO_LENGTH_M,
        "width_m": DEFAULT_EGO_WIDTH_M,
        "lane_width_m": DEFAULT_LANE_WIDTH_M,
    }


def _slot_feature(row: np.ndarray, slot_idx: int, name: str) -> float:
    start = len(EGO_FEATURES) + slot_idx * len(SLOT_FEATURES)
    return float(row[start + SLOT_FEATURES.index(name)])


def _trajectory_feature(row: np.ndarray, slot_idx: int, name: str) -> float:
    start = (
        len(EGO_FEATURES)
        + len(SLOT_NAMES) * len(SLOT_FEATURES)
        + int(slot_idx) * len(TRAJECTORY_FEATURES)
    )
    return float(row[start + TRAJECTORY_FEATURES.index(name)])


def _slot_action_summary(row: np.ndarray, slot_idx: int, slot_name: str) -> dict[str, Any]:
    return {
        "slot": slot_name,
        "horizon_seconds": 1.0,
        "delta_vx_1s_mps": _trajectory_feature(row, slot_idx, "delta_vx_1s_mps"),
        "delta_vy_left_1s_mps": _trajectory_feature(row, slot_idx, "delta_vy_left_1s_mps"),
        "mean_ax_1s_mps2": _trajectory_feature(row, slot_idx, "mean_ax_1s_mps2"),
        "min_ax_1s_mps2": _trajectory_feature(row, slot_idx, "min_ax_1s_mps2"),
        "final_ax_1s_mps2": _trajectory_feature(row, slot_idx, "final_ax_1s_mps2"),
        "mean_ay_left_1s_mps2": _trajectory_feature(row, slot_idx, "mean_ay_left_1s_mps2"),
    }


def _traffic_action_summaries(row: np.ndarray, slot_mask_row: np.ndarray) -> dict[str, Any]:
    return {
        slot_name: (
            None
            if not bool(slot_mask_row[idx])
            else _slot_action_summary(row, idx, slot_name)
        )
        for idx, slot_name in enumerate(SLOT_NAMES)
    }


def _primary_action_summary(
    row: np.ndarray,
    slot_mask_row: np.ndarray,
    primary_slot_name: str | None,
) -> dict[str, Any] | None:
    if primary_slot_name not in SLOT_NAMES:
        return None
    slot_idx = SLOT_NAMES.index(str(primary_slot_name))
    if not bool(slot_mask_row[slot_idx]):
        return None
    out = _slot_action_summary(row, slot_idx, str(primary_slot_name))
    out["primary_slot"] = str(primary_slot_name)
    return out


def to_ads_initialization(
    feature_row: np.ndarray,
    slot_mask_row: np.ndarray,
    *,
    sample_id: int = 0,
) -> dict[str, Any]:
    """Convert one c0 sample to an ego-centric ADS initialization payload."""
    ego = _ego_dict(feature_row)
    vehicles: list[dict[str, Any]] = []
    failures: list[str] = []
    ego_box = (ego["length_m"], ego["width_m"])
    for slot_idx, slot_name in enumerate(SLOT_NAMES):
        if not bool(slot_mask_row[slot_idx]):
            continue
        rel_x = _slot_feature(feature_row, slot_idx, "rel_x_m")
        rel_y = _slot_feature(feature_row, slot_idx, "rel_y_left_m")
        length = DEFAULT_OTHER_LENGTH_M
        width = DEFAULT_OTHER_WIDTH_M
        if abs(rel_x) < 0.5 * (ego_box[0] + length) and abs(rel_y) < 0.5 * (ego_box[1] + width):
            failures.append(f"{slot_name}:bounding_box_overlap")
        vehicles.append(
            {
                "slot": slot_name,
                "x_m": rel_x,
                "y_left_m": ego["y_left_m"] + rel_y,
                "vx_mps": ego["vx_mps"] + _slot_feature(feature_row, slot_idx, "rel_vx_mps"),
                "vy_left_mps": ego["vy_left_mps"] + _slot_feature(feature_row, slot_idx, "rel_vy_left_mps"),
                "ax_mps2": _slot_feature(feature_row, slot_idx, "other_ax_mps2"),
                "ay_left_mps2": _slot_feature(feature_row, slot_idx, "other_ay_left_mps2"),
                "length_m": length,
                "width_m": width,
            }
        )
    return {
        "sample_id": int(sample_id),
        "coordinate_frame": "ego_centric_local_left_positive",
        "ego": ego,
        "background_vehicles": vehicles,
        "slot_mask": {
            slot_name: bool(slot_mask_row[idx])
            for idx, slot_name in enumerate(SLOT_NAMES)
        },
        "valid": len(failures) == 0,
        "failure_reasons": failures,
    }


def to_world_model_start_condition(
    feature_row: np.ndarray,
    slot_mask_row: np.ndarray,
    *,
    sample_id: int = 0,
    fps: float = 25.0,
    primary_slot_name: str | None = None,
) -> dict[str, Any]:
    return {
        "sample_id": int(sample_id),
        "mode": "START",
        "fps": float(fps),
        "t0_only": True,
        "ego": _ego_dict(feature_row),
        "primary_slot": primary_slot_name,
        "primary_interaction_1s_summary": _primary_action_summary(
            feature_row,
            slot_mask_row,
            primary_slot_name,
        ),
        "traffic_action_1s_summary": _traffic_action_summaries(feature_row, slot_mask_row),
        "slots": {
            slot_name: (
                None
                if not bool(slot_mask_row[idx])
                else {
                    "rel_x_m": _slot_feature(feature_row, idx, "rel_x_m"),
                    "rel_y_left_m": _slot_feature(feature_row, idx, "rel_y_left_m"),
                    "rel_vx_mps": _slot_feature(feature_row, idx, "rel_vx_mps"),
                    "rel_vy_left_mps": _slot_feature(feature_row, idx, "rel_vy_left_mps"),
                    "ax_mps2": _slot_feature(feature_row, idx, "other_ax_mps2"),
                    "ay_left_mps2": _slot_feature(feature_row, idx, "other_ay_left_mps2"),
                    "action_1s_summary": _slot_action_summary(feature_row, idx, slot_name),
                    "length_m": DEFAULT_OTHER_LENGTH_M,
                    "width_m": DEFAULT_OTHER_WIDTH_M,
                }
            )
            for idx, slot_name in enumerate(SLOT_NAMES)
        },
    }


def load_checkpoint_and_dataset(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    repo_root: str | Path,
    device,
):
    arrays, schema = load_tail_dataset(output_dir)
    flow, payload = load_maf_checkpoint(checkpoint, repo_root=repo_root, map_location=device)
    return flow, arrays, schema, payload
