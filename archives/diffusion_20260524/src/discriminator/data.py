"""Dataset construction and loading for the Stage 2 naturalness discriminator."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from diffusion.src.data import SPLIT_TO_INDEX
from diffusion.src.utils import load_json, save_json

from .features import (
    SUMMARY_FEATURE_KEYS,
    build_future_features_numpy,
    normalize_numpy,
    selected_future_feature_keys,
)
from .negatives import (
    RANDOM_PERTURB_SOURCE,
    RULE_BRAKE_SOURCE,
    generate_random_perturb_negatives,
    generate_rule_brake_negatives,
    load_external_negatives,
)

logger = logging.getLogger(__name__)

POSITIVE_SOURCE = "highd_real"
RSS_SOURCE = "rss_over_guided"
HIGHWAY_ENV_SOURCE = "highway_env_hard_negative"


@dataclass(frozen=True)
class DiscriminatorPaths:
    natural_dataset_dir: Path
    output_dir: Path


def _resolve_paths(config: dict, config_dir: str | Path | None = None) -> DiscriminatorPaths:
    base = Path(config_dir).resolve() if config_dir is not None else Path.cwd()
    paths = config.get("paths", {})
    natural_dir = (base / paths.get("natural_dataset_dir", "../../../data/diffusion_natural/following")).resolve()
    output_dir = (base / paths.get("output_dir", "../../../data/diffusion_natural/following/discriminator")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return DiscriminatorPaths(natural_dataset_dir=natural_dir, output_dir=output_dir)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _limit_indices(split_index: np.ndarray, max_samples_per_split: int, seed: int) -> np.ndarray:
    all_idx = np.arange(len(split_index), dtype=np.int64)
    if max_samples_per_split <= 0:
        return all_idx
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    for split_id in sorted(SPLIT_TO_INDEX.values()):
        idx = all_idx[split_index == split_id]
        if len(idx) > max_samples_per_split:
            idx = rng.choice(idx, size=int(max_samples_per_split), replace=False)
            idx.sort()
        keep.append(idx)
    return np.concatenate(keep, axis=0)


def _fit_normalizer(x: np.ndarray, axis: tuple[int, ...]) -> dict[str, list[float]]:
    mean = np.mean(x, axis=axis).astype(np.float32)
    std = np.std(x, axis=axis).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return {"mean": mean.tolist(), "std": std.tolist()}


def _source_counts(source_type: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(source_type.astype(str), return_counts=True)
    return {str(k): int(v) for k, v in zip(values, counts)}


def _repeat_meta(raw: dict[str, np.ndarray], idx: np.ndarray, repeats: int) -> dict[str, np.ndarray]:
    return {key: np.repeat(raw[key][idx], repeats, axis=0) for key in ("split_index", "recording_id", "event_id", "anchor_frame", "ego_length", "adv_length")}


def _append_block(
    blocks: dict[str, list[np.ndarray]],
    *,
    norm: dict[str, np.ndarray],
    raw: dict[str, np.ndarray],
    idx: np.ndarray,
    actions: np.ndarray,
    source_type: str,
    label: float,
    sample_weight: float,
    schema: dict,
    config: dict,
    repeat_from_index: bool = False,
) -> None:
    repeats = int(actions.shape[0] // max(len(idx), 1)) if repeat_from_index else 1
    if repeat_from_index and repeats * len(idx) != actions.shape[0]:
        raise ValueError(f"Cannot map {actions.shape[0]} generated actions back to {len(idx)} source samples")
    meta_idx = np.repeat(idx, repeats) if repeat_from_index else idx
    context_states_norm = norm["context_states"][meta_idx]
    context_features_norm = norm["context_features"][meta_idx]
    relative_history_norm = norm["relative_history"][meta_idx]
    context_states_raw = raw["context_states"][meta_idx]
    ego_length = raw["ego_length"][meta_idx]
    adv_length = raw["adv_length"][meta_idx]
    future_features, summary_features = build_future_features_numpy(
        actions,
        context_states_raw,
        raw.get("relative_history", None)[meta_idx] if raw.get("relative_history", None) is not None else None,
        ego_length,
        adv_length,
        schema,
        config,
    )
    n = int(actions.shape[0])
    blocks["context_states"].append(context_states_norm.astype(np.float32))
    blocks["context_features"].append(context_features_norm.astype(np.float32))
    blocks["relative_history"].append(relative_history_norm.astype(np.float32))
    blocks["future_action_features"].append(future_features.astype(np.float32))
    blocks["summary_features"].append(summary_features.astype(np.float32))
    blocks["labels"].append(np.full((n,), float(label), dtype=np.float32))
    blocks["sample_weights"].append(np.full((n,), float(sample_weight), dtype=np.float32))
    blocks["source_type"].append(np.asarray([source_type] * n, dtype=object))
    blocks["split_index"].append(raw["split_index"][meta_idx].astype(np.int8))
    blocks["recording_id"].append(raw["recording_id"][meta_idx])
    blocks["event_id"].append(raw["event_id"][meta_idx])
    blocks["anchor_frame"].append(raw["anchor_frame"][meta_idx])
    blocks["ego_length"].append(ego_length.astype(np.float32))
    blocks["adv_length"].append(adv_length.astype(np.float32))
    blocks["lane_width"].append(np.full((n,), float(config.get("data", {}).get("lane_width", 3.5)), dtype=np.float32))


def _negative_copy_counts(config: dict) -> dict[str, int]:
    data_cfg = config.get("data", {})
    neg_cfg = data_cfg.get("negatives", {})
    total = max(0, int(data_cfg.get("negatives_per_positive", 3)))
    enabled = [
        source
        for source in (RANDOM_PERTURB_SOURCE, RULE_BRAKE_SOURCE)
        if bool(neg_cfg.get(source, source in {RANDOM_PERTURB_SOURCE, RULE_BRAKE_SOURCE}))
    ]
    if total == 0 or not enabled:
        return {source: 0 for source in (RANDOM_PERTURB_SOURCE, RULE_BRAKE_SOURCE)}
    counts = {source: total // len(enabled) for source in enabled}
    for source in enabled[: total % len(enabled)]:
        counts[source] += 1
    return {source: counts.get(source, 0) for source in (RANDOM_PERTURB_SOURCE, RULE_BRAKE_SOURCE)}


def _append_external_negatives(
    blocks: dict[str, list[np.ndarray]],
    *,
    external: dict[str, Any] | None,
    source_type: str,
    norm: dict[str, np.ndarray],
    raw: dict[str, np.ndarray],
    schema: dict,
    config: dict,
) -> None:
    if external is None:
        return
    actions = np.asarray(external["actions"], dtype=np.float32)
    if "sample_index" in external:
        idx = np.asarray(external["sample_index"], dtype=np.int64)
    else:
        idx = np.arange(actions.shape[0], dtype=np.int64)
    weight = float(external.get("sample_weight", config.get("data", {}).get("external_negative_weight", 1.0)))
    _append_block(
        blocks,
        norm=norm,
        raw=raw,
        idx=idx,
        actions=actions,
        source_type=source_type,
        label=0.0,
        sample_weight=weight,
        schema=schema,
        config=config,
    )


def build_discriminator_dataset(
    config: dict,
    *,
    config_dir: str | Path | None = None,
    max_samples_per_split: int = 0,
) -> dict[str, Any]:
    paths = _resolve_paths(config, config_dir)
    raw = _load_npz(paths.natural_dataset_dir / "dataset.npz")
    norm = _load_npz(paths.natural_dataset_dir / "dataset_normalized.npz")
    schema = load_json(paths.natural_dataset_dir / "feature_schema.json")
    seed = int(config.get("training", {}).get("seed", 42))
    rng = np.random.default_rng(seed)
    idx = _limit_indices(raw["split_index"], int(max_samples_per_split), seed)
    logger.info("Building discriminator dataset from %d Stage 1 samples", len(idx))

    blocks: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "context_states",
            "context_features",
            "relative_history",
            "future_action_features",
            "summary_features",
            "labels",
            "sample_weights",
            "source_type",
            "split_index",
            "recording_id",
            "event_id",
            "anchor_frame",
            "ego_length",
            "adv_length",
            "lane_width",
        )
    }
    _append_block(
        blocks,
        norm=norm,
        raw=raw,
        idx=idx,
        actions=raw["actions"][idx],
        source_type=POSITIVE_SOURCE,
        label=1.0,
        sample_weight=1.0,
        schema=schema,
        config=config,
    )

    counts = _negative_copy_counts(config)
    if counts[RANDOM_PERTURB_SOURCE] > 0:
        actions = generate_random_perturb_negatives(
            raw["actions"][idx],
            rng=rng,
            config=config,
            copies_per_positive=counts[RANDOM_PERTURB_SOURCE],
        )
        _append_block(
            blocks,
            norm=norm,
            raw=raw,
            idx=idx,
            actions=actions,
            source_type=RANDOM_PERTURB_SOURCE,
            label=0.0,
            sample_weight=1.0,
            schema=schema,
            config=config,
            repeat_from_index=True,
        )
    if counts[RULE_BRAKE_SOURCE] > 0:
        actions = generate_rule_brake_negatives(
            raw["actions"][idx],
            raw["context_states"][idx],
            rng=rng,
            schema=schema,
            config=config,
            copies_per_positive=counts[RULE_BRAKE_SOURCE],
        )
        _append_block(
            blocks,
            norm=norm,
            raw=raw,
            idx=idx,
            actions=actions,
            source_type=RULE_BRAKE_SOURCE,
            label=0.0,
            sample_weight=1.0,
            schema=schema,
            config=config,
            repeat_from_index=True,
        )

    neg_cfg = config.get("data", {}).get("negatives", {})
    if bool(neg_cfg.get(RSS_SOURCE, False)):
        _append_external_negatives(
            blocks,
            external=load_external_negatives(config.get("paths", {}).get("rss_over_guided_negatives")),
            source_type=RSS_SOURCE,
            norm=norm,
            raw=raw,
            schema=schema,
            config=config,
        )
    if bool(neg_cfg.get(HIGHWAY_ENV_SOURCE, False)):
        _append_external_negatives(
            blocks,
            external=load_external_negatives(config.get("paths", {}).get("highway_env_hard_negatives")),
            source_type=HIGHWAY_ENV_SOURCE,
            norm=norm,
            raw=raw,
            schema=schema,
            config=config,
        )

    arrays = {key: np.concatenate(values, axis=0) for key, values in blocks.items()}
    train_mask = arrays["split_index"] == SPLIT_TO_INDEX["train"]
    if not np.any(train_mask):
        train_mask = np.ones_like(arrays["split_index"], dtype=bool)
    future_norm = _fit_normalizer(arrays["future_action_features"][train_mask], axis=(0, 1))
    summary_norm = _fit_normalizer(arrays["summary_features"][train_mask], axis=(0,))
    arrays["future_action_features"] = normalize_numpy(
        arrays["future_action_features"],
        future_norm["mean"],
        future_norm["std"],
    )
    arrays["summary_features"] = normalize_numpy(
        arrays["summary_features"],
        summary_norm["mean"],
        summary_norm["std"],
    )

    stats = {
        "future_action_features": future_norm,
        "summary_features": summary_norm,
        "stage1_normalization_stats": str(paths.natural_dataset_dir / "normalization_stats.json"),
    }
    discriminator_schema = {
        "event_type": schema.get("event_type", "following"),
        "history_steps": int(schema["history_steps"]),
        "horizon_steps": int(schema["horizon_steps"]),
        "num_actors": int(schema["num_actors"]),
        "state_features": list(schema["state_features"]),
        "context_keys": list(schema["context_keys"]),
        "relative_history_keys": list(schema.get("relative_history_keys", [])),
        "action_representation": schema.get("action_representation", config.get("action", {}).get("representation", "jerk")),
        "action_keys": list(schema.get("action_keys", ["jx"])),
        "future_feature_keys": list(selected_future_feature_keys(config)),
        "summary_feature_keys": list(SUMMARY_FEATURE_KEYS),
        "dt": float(schema.get("dt", 0.04)),
        "source_types": sorted(_source_counts(arrays["source_type"]).keys()),
        "num_samples": int(arrays["labels"].shape[0]),
    }
    np.savez_compressed(paths.output_dir / "discriminator_dataset.npz", **arrays)
    save_json(discriminator_schema, paths.output_dir / "discriminator_schema.json")
    save_json(stats, paths.output_dir / "discriminator_stats.json")
    summary = {
        "natural_dataset_dir": str(paths.natural_dataset_dir),
        "output_dir": str(paths.output_dir),
        "max_samples_per_split": int(max_samples_per_split),
        "num_samples": int(arrays["labels"].shape[0]),
        "source_counts": _source_counts(arrays["source_type"]),
        "split_counts": {
            name: int(np.sum(arrays["split_index"] == split_id))
            for name, split_id in SPLIT_TO_INDEX.items()
        },
    }
    save_json(summary, paths.output_dir / "discriminator_build_summary.json")
    logger.info("Wrote discriminator dataset to %s", paths.output_dir)
    return {"arrays": arrays, "schema": discriminator_schema, "stats": stats, "output_dir": paths.output_dir}


def load_discriminator_dataset(dataset_dir: str | Path) -> dict[str, np.ndarray]:
    path = Path(dataset_dir) / "discriminator_dataset.npz"
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}
