"""Sampling and export interfaces for trained highD tail-event flows."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .data import load_tail_dataset, split_indices
from .features import (
    DEFAULT_EGO_LENGTH_M,
    DEFAULT_EGO_WIDTH_M,
    DEFAULT_LANE_WIDTH_M,
    DEFAULT_OTHER_LENGTH_M,
    DEFAULT_OTHER_WIDTH_M,
    EGO_FEATURES,
    SLOT_NAMES,
    feature_valid_from_slot_mask,
    slot_feature_index,
    slot_mask_from_pattern,
    trajectory_feature_index,
    zero_inactive_slot_features,
)
from .metrics import physical_validity_flags
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
        list(schema.get("model_feature_transforms", [])) or None,
    )


def normalize_features(raw: np.ndarray, valid: np.ndarray, schema: dict[str, Any]) -> np.ndarray:
    norm = schema["normalization"]
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    model_features = transform_features_for_model(
        np.asarray(raw, dtype=np.float32),
        np.asarray(valid, dtype=bool),
        list(schema["feature_names"]),
        list(schema.get("model_feature_transforms", [])) or None,
    )
    out = np.zeros_like(raw, dtype=np.float32)
    out[valid] = ((model_features - mean) / std)[valid]
    return out


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
    candidates = split_indices(arrays, split)
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
    slot_mask = slot_mask_from_pattern(sampled_pattern)
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
    candidates = split_indices(arrays, split)
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


def _event_structures_from_keys(
    schema: dict[str, Any],
    *,
    mask_pattern: np.ndarray,
    primary_slot_index: np.ndarray,
    event_structure_id: np.ndarray | None = None,
    event_structure_log_prob: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    sampled_pattern = np.asarray(mask_pattern, dtype=np.int64).reshape(-1)
    sampled_primary = np.asarray(primary_slot_index, dtype=np.int64).reshape(-1)
    slot_mask = slot_mask_from_pattern(sampled_pattern)
    contexts = _contexts_from_event_structure(schema, slot_mask, sampled_primary)
    n = len(sampled_pattern)
    if event_structure_id is None:
        event_structure_id = np.zeros(n, dtype=np.int64)
    if event_structure_log_prob is None:
        event_structure_log_prob = np.zeros(n, dtype=np.float32)
    return {
        "contexts": contexts,
        "event_structure": contexts.copy(),
        "slot_mask": slot_mask.astype(bool),
        "mask_pattern": sampled_pattern,
        "primary_slot_index": sampled_primary,
        "primary_slot_name": np.asarray([SLOT_NAMES[int(idx)] for idx in sampled_primary]),
        "event_structure_id": np.asarray(event_structure_id, dtype=np.int64).reshape(-1),
        "event_structure_log_prob": np.asarray(
            event_structure_log_prob,
            dtype=np.float32,
        ).reshape(-1),
    }


def _allocate_quota_counts(
    probabilities: np.ndarray,
    counts: np.ndarray,
    *,
    num_samples: int,
) -> np.ndarray:
    total_counts = int(np.sum(counts))
    if int(num_samples) == total_counts:
        return np.asarray(counts, dtype=np.int64).copy()
    expected = np.asarray(probabilities, dtype=np.float64) * int(num_samples)
    quota = np.floor(expected).astype(np.int64)
    remaining = int(num_samples) - int(np.sum(quota))
    if remaining > 0:
        residual = expected - quota
        order = np.argsort(-residual, kind="mergesort")
        quota[order[:remaining]] += 1
    elif remaining < 0:
        residual = expected - quota
        order = np.argsort(residual, kind="mergesort")
        for idx in order[: abs(remaining)]:
            if quota[idx] > 0:
                quota[idx] -= 1
    return quota.astype(np.int64)


def _quota_event_structure_plan(
    arrays: dict[str, np.ndarray],
    *,
    split: str,
    num_samples: int,
    mask_pattern: int | None,
    primary_slot: str | int | None,
) -> list[dict[str, Any]]:
    table = _event_structure_distribution(
        arrays,
        split=split,
        mask_pattern=mask_pattern,
        primary_slot=primary_slot,
    )
    quota = _allocate_quota_counts(
        table["probabilities"],
        table["counts"],
        num_samples=int(num_samples),
    )
    plan: list[dict[str, Any]] = []
    for event_id, count in enumerate(quota.tolist()):
        if int(count) <= 0:
            continue
        plan.append(
            {
                "event_structure_id": int(event_id),
                "mask_pattern": int(table["mask_pattern"][event_id]),
                "primary_slot_index": int(table["primary_slot_index"][event_id]),
                "event_structure_log_prob": float(
                    np.log(max(float(table["probabilities"][event_id]), 1.0e-12))
                ),
                "quota": int(count),
            }
        )
    return plan


def _slice_sample_batch(
    batch: dict[str, np.ndarray],
    take: np.ndarray,
) -> dict[str, np.ndarray]:
    event_log_prob = batch["event_structure_log_prob"][take].astype(np.float32)
    return {
        "features": batch["features"][take].astype(np.float32),
        "features_normalized": batch["features_normalized"][take].astype(np.float32),
        "feature_valid": batch["feature_valid"][take],
        "contexts": batch["contexts"][take],
        "event_structure": batch["event_structure"][take],
        "slot_mask": batch["slot_mask"][take],
        "mask_pattern": batch["mask_pattern"][take].astype(np.int64),
        "primary_slot_index": batch["primary_slot_index"][take].astype(np.int64),
        "primary_slot_name": batch["primary_slot_name"][take],
        "event_structure_id": batch["event_structure_id"][take].astype(np.int64),
        "event_structure_log_prob": event_log_prob,
        "conditional_log_prob": batch["conditional_log_prob"][take].astype(np.float32),
        "log_prob": (batch["conditional_log_prob"][take] + event_log_prob).astype(np.float32),
    }


def _concat_sample_chunks(
    chunks: list[dict[str, np.ndarray]],
    *,
    num_samples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    out = {
        key: np.concatenate([chunk[key] for chunk in chunks], axis=0)[: int(num_samples)]
        for key in chunks[0]
    }
    rng = np.random.default_rng(int(seed) + 9901)
    order = rng.permutation(len(out["features"]))
    return {key: value[order] for key, value in out.items()}


def _draw_flow_batch(
    flow,
    event_structure: dict[str, np.ndarray],
    schema: dict[str, Any],
    *,
    device,
    reject_invalid: bool,
    temperature: float = 1.0,
) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    import torch

    contexts = event_structure["contexts"].astype(np.float32)
    slot_mask = event_structure["slot_mask"].astype(bool)
    draw_n = int(len(contexts))
    temp = float(temperature)
    if temp <= 0.0:
        raise ValueError(f"sampling temperature must be positive, got {temperature}")
    context_t = torch.from_numpy(contexts).float().to(device)
    with torch.no_grad():
        embedded_context = flow._embedding_net(context_t)
        if getattr(flow, "_context_used_in_base", False):
            noise = flow._distribution.sample(1, context=embedded_context)
            if noise.ndim == 3:
                noise = noise[:, 0, :]
        else:
            noise = flow._distribution.sample(draw_n)
        if temp != 1.0:
            noise = noise * temp
        samples, _ = flow._transform.inverse(noise, context=embedded_context)
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
    rejected = 0
    if reject_invalid:
        invalid, _reasons, _detail = physical_validity_flags(
            np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0),
            slot_mask,
        )
        too_large = np.max(np.abs(x_norm), axis=1) > float(
            schema.get("sampling_max_abs_normalized", 8.0)
        )
        keep = finite & ~(invalid | too_large)
        rejected = int(np.sum(~keep))
    batch = {
        "features": raw.astype(np.float32),
        "features_normalized": x_norm.astype(np.float32),
        "feature_valid": valid,
        "contexts": contexts,
        "event_structure": event_structure["event_structure"],
        "slot_mask": slot_mask,
        "mask_pattern": event_structure["mask_pattern"].astype(np.int64),
        "primary_slot_index": event_structure["primary_slot_index"].astype(np.int64),
        "primary_slot_name": event_structure["primary_slot_name"],
        "event_structure_id": event_structure["event_structure_id"].astype(np.int64),
        "event_structure_log_prob": event_structure["event_structure_log_prob"].astype(np.float32),
        "conditional_log_prob": conditional_log_prob.astype(np.float32),
    }
    return batch, keep, rejected


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
    event_structure_sampling: str = "multinomial",
    reject_invalid: bool = True,
    max_rounds: int = 10,
    oversample_factor: int = 3,
    min_draw: int | None = None,
    temperature: float = 1.0,
) -> dict[str, np.ndarray]:
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    flow.eval()
    accepted: list[dict[str, np.ndarray]] = []
    rejected = 0
    strategy = str(event_structure_sampling or "multinomial").lower()
    multinomial_min_draw = 128 if min_draw is None else int(min_draw)
    quota_min_draw = 32 if min_draw is None else int(min_draw)
    if strategy == "multinomial":
        remaining = int(num_samples)
        rounds = 0
        while remaining > 0 and rounds < int(max_rounds):
            rounds += 1
            draw_n = remaining if not reject_invalid else max(
                remaining * int(oversample_factor),
                multinomial_min_draw,
            )
            event_structure = sample_event_structures(
                arrays,
                schema,
                num_samples=draw_n,
                seed=int(seed) + rounds * 1543,
                split=event_structure_split,
                mask_pattern=mask_pattern,
                primary_slot=primary_slot,
            )
            batch, keep, num_rejected = _draw_flow_batch(
                flow,
                event_structure,
                schema,
                device=device,
                reject_invalid=reject_invalid,
                temperature=float(temperature),
            )
            rejected += num_rejected
            take = np.where(keep)[0][:remaining]
            if len(take):
                accepted.append(_slice_sample_batch(batch, take))
                remaining -= len(take)
    elif strategy == "quota":
        plan = _quota_event_structure_plan(
            arrays,
            split=event_structure_split,
            num_samples=int(num_samples),
            mask_pattern=mask_pattern,
            primary_slot=primary_slot,
        )
        for item in plan:
            remaining = int(item["quota"])
            rounds = 0
            while remaining > 0 and rounds < int(max_rounds):
                rounds += 1
                draw_n = remaining if not reject_invalid else max(
                    remaining * int(oversample_factor),
                    quota_min_draw,
                )
                event_structure = _event_structures_from_keys(
                    schema,
                    mask_pattern=np.full(draw_n, int(item["mask_pattern"]), dtype=np.int64),
                    primary_slot_index=np.full(
                        draw_n,
                        int(item["primary_slot_index"]),
                        dtype=np.int64,
                    ),
                    event_structure_id=np.full(
                        draw_n,
                        int(item["event_structure_id"]),
                        dtype=np.int64,
                    ),
                    event_structure_log_prob=np.full(
                        draw_n,
                        float(item["event_structure_log_prob"]),
                        dtype=np.float32,
                    ),
                )
                batch, keep, num_rejected = _draw_flow_batch(
                    flow,
                    event_structure,
                    schema,
                    device=device,
                    reject_invalid=reject_invalid,
                    temperature=float(temperature),
                )
                rejected += num_rejected
                take = np.where(keep)[0][:remaining]
                if len(take):
                    accepted.append(_slice_sample_batch(batch, take))
                    remaining -= len(take)
            if remaining > 0:
                raise RuntimeError(
                    "Quota sampler could not fill event structure "
                    f"mask_pattern={item['mask_pattern']} "
                    f"primary_slot_index={item['primary_slot_index']} "
                    f"remaining={remaining}"
                )
    else:
        raise ValueError(
            "Unsupported event_structure_sampling="
            f"{event_structure_sampling!r}; expected 'multinomial' or 'quota'"
        )
    if not accepted:
        raise RuntimeError("Flow sampler could not produce any accepted samples")
    out = _concat_sample_chunks(accepted, num_samples=int(num_samples), seed=int(seed))
    out["num_rejected"] = np.asarray([rejected], dtype=np.int64)
    out["event_structure_sampling"] = np.asarray([strategy])
    out["sampling_temperature"] = np.asarray([float(temperature)], dtype=np.float32)
    out["rejection_rate"] = np.asarray(
        [rejected / max(rejected + len(out["features"]), 1)],
        dtype=np.float32,
    )
    return out


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
    return float(row[slot_feature_index(SLOT_NAMES[int(slot_idx)], name)])


def _trajectory_feature(row: np.ndarray, slot_idx: int, name: str) -> float:
    return float(row[trajectory_feature_index(SLOT_NAMES[int(slot_idx)], name)])


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
