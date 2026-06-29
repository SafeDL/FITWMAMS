"""Dataset preparation for highD EVT-tail c0 normalizing flows."""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from process_highD.src.io_utils import load_config as load_highd_config
from process_highD.src.io_utils import resolve_data_path
from process_highD.src.loader import HighDRecording, load_recording
from process_highD.src.natural_evt_pipeline import select_natural_tail_contexts
from process_highD.src.preprocess import (
    filter_abnormal_tracks,
    normalize_driving_direction,
    resample_recording,
)

from .features import (
    SLOT_NAMES,
    build_feature_schema,
    extract_c0_features_for_segment,
    mask_pattern_from_slot_mask,
)
from .transforms import (
    feature_transform_kinds,
    transform_features_for_model,
)
from .utils import ensure_dir, load_json, resolve_path, save_json


logger = logging.getLogger(__name__)

SPLIT_TO_INDEX = {"train": 0, "val": 1, "test": 2}


def expected_context_names() -> tuple[str, ...]:
    return tuple(f"mask_{slot}" for slot in SLOT_NAMES) + (
        tuple(f"primary_slot_{slot}" for slot in SLOT_NAMES)
    )


def output_dir_from_config(config: dict[str, Any], config_dir: str | Path) -> Path:
    return ensure_dir(resolve_path(config["paths"]["output_dir"], base=config_dir))


def dataset_npz_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "dataset.npz"


def schema_json_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "dataset_schema.json"


def dataset_schema_is_current(output_dir: str | Path) -> bool:
    schema_path = schema_json_path(output_dir)
    if not schema_path.exists():
        return False
    try:
        schema = load_json(schema_path)
    except Exception:  # noqa: BLE001 - corrupted schema should trigger rebuild.
        return False
    feature_schema = build_feature_schema()
    expected_transforms = feature_transform_kinds(feature_schema.feature_names)
    return (
        list(schema.get("feature_names", [])) == list(feature_schema.feature_names)
        and list(schema.get("context_names", [])) == list(expected_context_names())
        and list(schema.get("model_feature_transforms", [])) == list(expected_transforms)
    )


def _resolve_highd_evt_config(config: dict[str, Any], config_dir: Path) -> Path:
    return resolve_path(config["paths"]["highd_evt_config"], base=config_dir)


def _resolve_tail_context_csv(config: dict[str, Any], config_dir: Path) -> Path:
    raw = config["paths"].get("tail_context_csv")
    if raw:
        return resolve_path(raw, base=config_dir)
    highd_cfg_path = _resolve_highd_evt_config(config, config_dir)
    highd_cfg = load_highd_config(highd_cfg_path)
    out_dir = resolve_data_path(highd_cfg["paths"]["output_dir"], highd_cfg_path)
    return out_dir / "natural_tail_contexts.csv"


def _resolve_raw_dir(config: dict[str, Any], config_dir: Path) -> Path:
    raw = config["paths"].get("raw_dir")
    if raw:
        return resolve_path(raw, base=config_dir)
    highd_cfg_path = _resolve_highd_evt_config(config, config_dir)
    highd_cfg = load_highd_config(highd_cfg_path)
    return resolve_data_path(highd_cfg["paths"]["raw_dir"], highd_cfg_path)


def ensure_tail_context_csv(config: dict[str, Any], config_dir: Path) -> Path:
    path = _resolve_tail_context_csv(config, config_dir)
    if path.exists():
        return path
    highd_cfg_path = _resolve_highd_evt_config(config, config_dir)
    logger.info("Tail context CSV is missing; selecting EVT tail contexts first")
    select_natural_tail_contexts(
        config_path=highd_cfg_path,
        output_csv=path,
    )
    return path


def prepare_recording(
    raw_dir: str | Path,
    recording_id: int,
    config: dict[str, Any],
) -> HighDRecording:
    rec = load_recording(str(raw_dir), int(recording_id))
    rec = normalize_driving_direction(rec)
    rec = filter_abnormal_tracks(rec, config)
    target_fps = int(config.get("sampling", {}).get("target_fps", 25))
    rec = resample_recording(rec, target_fps)
    return rec


def _group_values(meta: list[dict[str, Any]], mode: str) -> np.ndarray:
    mode = str(mode).lower()
    out: list[str] = []
    for item in meta:
        if mode == "recording":
            out.append(str(item["recording_id"]))
        elif mode in {"recording_ego", "vehicle", "ego"}:
            out.append(f"{item['recording_id']}:{item['ego_id']}")
        elif mode in {"segment", "none"}:
            out.append(str(item["segment_id"]))
        else:
            raise ValueError(f"Unsupported split group mode: {mode}")
    return np.asarray(out, dtype="U64")


def split_indices_by_group(
    metadata: list[dict[str, Any]],
    split_cfg: dict[str, Any],
) -> np.ndarray:
    groups = _group_values(metadata, str(split_cfg.get("group_by", "recording")))
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(int(split_cfg.get("seed", 42)))
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    ratios = split_cfg.get("ratios", [0.70, 0.15, 0.15])
    if len(ratios) != 3:
        raise ValueError("split.ratios must contain train/val/test ratios")
    ratios = np.asarray(ratios, dtype=np.float64)
    ratios = ratios / np.maximum(np.sum(ratios), 1.0e-12)
    n = len(shuffled)
    if n < 3:
        raise RuntimeError("Need at least three split groups for train/val/test")
    n_train = max(1, int(round(n * ratios[0])))
    n_val = max(1, int(round(n * ratios[1])))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    train_groups = set(shuffled[:n_train])
    val_groups = set(shuffled[n_train : n_train + n_val])
    split_index = np.full(len(groups), SPLIT_TO_INDEX["test"], dtype=np.int64)
    for idx, group in enumerate(groups):
        if group in train_groups:
            split_index[idx] = SPLIT_TO_INDEX["train"]
        elif group in val_groups:
            split_index[idx] = SPLIT_TO_INDEX["val"]
    return split_index


def build_contexts(
    *,
    slot_mask: np.ndarray,
    metadata: list[dict[str, Any]],
) -> tuple[np.ndarray, tuple[str, ...]]:
    context_names = expected_context_names()
    names: list[str] = []
    parts: list[np.ndarray] = []

    slot_bits = slot_mask.astype(np.float32)
    names.extend(context_names[: len(SLOT_NAMES)])
    parts.append(slot_bits)

    primary_one_hot = np.zeros((len(slot_mask), len(SLOT_NAMES)), dtype=np.float32)
    primary_idx = np.asarray(
        [int(item["primary_slot_index"]) for item in metadata],
        dtype=np.int64,
    )
    primary_one_hot[np.arange(len(slot_mask)), primary_idx] = 1.0
    names.extend(context_names[len(SLOT_NAMES) :])
    parts.append(primary_one_hot)

    return np.concatenate(parts, axis=1).astype(np.float32), tuple(names)


def fit_feature_normalizer(
    raw_features: np.ndarray,
    feature_valid: np.ndarray,
    split_index: np.ndarray,
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    transform_kinds = feature_transform_kinds(feature_names)
    model_features = transform_features_for_model(
        raw_features,
        feature_valid,
        feature_names,
        transform_kinds,
    )
    train = split_index == SPLIT_TO_INDEX["train"]
    mean = np.zeros(raw_features.shape[1], dtype=np.float64)
    std = np.ones(raw_features.shape[1], dtype=np.float64)
    count = np.zeros(raw_features.shape[1], dtype=np.int64)
    for j in range(raw_features.shape[1]):
        valid = train & feature_valid[:, j] & np.isfinite(model_features[:, j])
        count[j] = int(np.sum(valid))
        if count[j] == 0:
            continue
        values = model_features[valid, j].astype(np.float64)
        mean[j] = float(np.mean(values))
        value_std = float(np.std(values))
        std[j] = value_std if value_std > 1.0e-6 else 1.0
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "valid_train_count": count,
    }


def apply_feature_normalizer(
    raw_features: np.ndarray,
    feature_valid: np.ndarray,
    normalizer: dict[str, Any],
    feature_names: tuple[str, ...],
) -> np.ndarray:
    mean = np.asarray(normalizer["mean"], dtype=np.float32)
    std = np.asarray(normalizer["std"], dtype=np.float32)
    transform_kinds = feature_transform_kinds(feature_names)
    model_features = transform_features_for_model(
        raw_features,
        feature_valid,
        feature_names,
        transform_kinds,
    )
    out = np.zeros_like(raw_features, dtype=np.float32)
    valid = feature_valid & np.isfinite(model_features)
    out[valid] = ((model_features - mean) / std)[valid]
    return out


def _metadata_arrays(metadata: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    keys = (
        "segment_id",
        "recording_id",
        "ego_id",
        "anchor_frame",
        "event_risk",
        "primary_slot_name",
        "primary_slot_index",
    )
    arrays: dict[str, np.ndarray] = {}
    for key in keys:
        values = [item.get(key, "") for item in metadata]
        if key in {"segment_id", "peak_slot_name", "primary_slot_name"}:
            arrays[key] = np.asarray(values, dtype="U64")
        elif key == "event_risk":
            arrays[key] = np.asarray(values, dtype=np.float32)
        else:
            arrays[key] = np.asarray(values, dtype=np.int64)
    return arrays


def build_tail_flow_dataset(
    config: dict[str, Any],
    *,
    config_dir: str | Path,
    rebuild_tail_contexts: bool = False,
) -> dict[str, Any]:
    config_dir = Path(config_dir).resolve()
    output_dir = output_dir_from_config(config, config_dir)
    tail_context_path = _resolve_tail_context_csv(config, config_dir)
    if rebuild_tail_contexts and tail_context_path.exists():
        tail_context_path.unlink()
    tail_context_path = ensure_tail_context_csv(config, config_dir)
    raw_dir = _resolve_raw_dir(config, config_dir)
    highd_cfg = load_highd_config(_resolve_highd_evt_config(config, config_dir))

    tail_contexts = pd.read_csv(tail_context_path)
    if tail_contexts.empty:
        raise RuntimeError(f"Tail context CSV is empty: {tail_context_path}")
    schema = build_feature_schema()
    features: list[np.ndarray] = []
    valids: list[np.ndarray] = []
    slot_masks: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    reject: Counter[str] = Counter()

    logger.info("Extracting c0 features from %d tail contexts", len(tail_contexts))
    for recording_id, frame in tail_contexts.groupby("recording_id", sort=True):
        rec = prepare_recording(raw_dir, int(recording_id), highd_cfg)
        logger.info("Recording %02d: %d tail contexts", int(recording_id), len(frame))
        for _, row in frame.iterrows():
            try:
                feat, valid, mask, meta = extract_c0_features_for_segment(
                    rec,
                    row,
                    schema=schema,
                )
            except Exception as exc:  # noqa: BLE001 - keep per-segment audit.
                reject[type(exc).__name__] += 1
                logger.warning(
                    "Skipping segment %s: %s",
                    row.get("segment_id", "<unknown>"),
                    exc,
                )
                continue
            features.append(feat)
            valids.append(valid)
            slot_masks.append(mask)
            metadata.append(meta)

    if not features:
        raise RuntimeError("No c0 features were extracted from tail contexts")

    raw_features = np.stack(features).astype(np.float32)
    feature_valid = np.stack(valids).astype(bool)
    slot_mask = np.stack(slot_masks).astype(bool)
    split_index = split_indices_by_group(metadata, dict(config.get("split", {})))
    contexts, context_names = build_contexts(
        slot_mask=slot_mask,
        metadata=metadata,
    )
    normalizer = fit_feature_normalizer(
        raw_features,
        feature_valid,
        split_index,
        schema.feature_names,
    )
    features_normalized = apply_feature_normalizer(
        raw_features,
        feature_valid,
        normalizer,
        schema.feature_names,
    )
    mask_pattern = mask_pattern_from_slot_mask(slot_mask)
    meta_arrays = _metadata_arrays(metadata)

    arrays = {
        "features": raw_features,
        "features_normalized": features_normalized,
        "feature_valid": feature_valid,
        "contexts": contexts,
        "slot_mask": slot_mask,
        "mask_pattern": mask_pattern,
        "split_index": split_index,
        **meta_arrays,
    }
    np.savez_compressed(dataset_npz_path(output_dir), **arrays)

    split_summary = {
        split: int(np.sum(split_index == idx))
        for split, idx in SPLIT_TO_INDEX.items()
    }
    mask_summary = {
        str(int(pattern)): int(count)
        for pattern, count in Counter(mask_pattern.tolist()).most_common()
    }
    schema_payload = {
        "dataset_npz": str(dataset_npz_path(output_dir)),
        "tail_context_csv": str(tail_context_path),
        "raw_dir": str(raw_dir),
        "num_samples": int(raw_features.shape[0]),
        "feature_names": list(schema.feature_names),
        "ego_features": list(schema.ego_features),
        "slot_features": list(schema.slot_features),
        "trajectory_features": list(schema.trajectory_features),
        "model_feature_transforms": list(
            feature_transform_kinds(schema.feature_names)
        ),
        "slot_names": list(SLOT_NAMES),
        "context_names": list(context_names),
        "split_index": SPLIT_TO_INDEX,
        "split_summary": split_summary,
        "mask_pattern_summary": mask_summary,
        "normalization": {
            "mean": np.asarray(normalizer["mean"], dtype=float).tolist(),
            "std": np.asarray(normalizer["std"], dtype=float).tolist(),
            "valid_train_count": np.asarray(
                normalizer["valid_train_count"],
                dtype=int,
            ).tolist(),
            "fit_split": "train",
            "inactive_slot_policy": "zero_placeholder_not_a_vehicle",
            "coordinate_note": (
                "Mean/std normalization is fitted in model coordinates after "
                "the positive mean-minus-min-ax transform. Raw feature values "
                "remain stored in dataset.npz."
            ),
        },
        "reject_counts": {key: int(value) for key, value in sorted(reject.items())},
    }
    save_json(schema_payload, schema_json_path(output_dir))
    logger.info("Wrote normalizing dataset: %s", dataset_npz_path(output_dir))
    logger.info("Wrote normalizing schema: %s", schema_json_path(output_dir))
    return schema_payload


def load_tail_dataset(output_dir: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    output_dir = Path(output_dir)
    npz_path = dataset_npz_path(output_dir)
    schema_path = schema_json_path(output_dir)
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing dataset NPZ: {npz_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing dataset schema: {schema_path}")
    data = np.load(npz_path, allow_pickle=False)
    arrays = {key: data[key] for key in data.files}
    return arrays, load_json(schema_path)


def split_indices(arrays: dict[str, np.ndarray], split: str) -> np.ndarray:
    split_name = str(split).lower()
    if split_name in {"all", "full", "dataset"}:
        return np.arange(len(arrays["split_index"]), dtype=np.int64)
    if split_name not in SPLIT_TO_INDEX:
        raise KeyError(
            f"Unknown split={split!r}; expected one of {sorted(SPLIT_TO_INDEX)} or 'all'"
        )
    return np.where(arrays["split_index"] == SPLIT_TO_INDEX[split_name])[0]
