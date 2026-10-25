"""Shared highD data for the EVT-tail scenario Flow."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from process_highD.src.io_utils import load_config as load_highd_config
from process_highD.src.io_utils import resolve_data_path
from process_highD.src.natural_segments import (
    validate_natural_segment_contract,
)
from process_highD.src.preprocess import prepare_recording

from .constraints import (
    KNOT_FEATURE_NAMES,
    KNOT_INDICES,
    KNOT_TIMES_S,
    constraint_to_model_coordinates,
    extract_constraint_for_segment,
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
INACTIVE_REFERENCE_NOISE_SEED = 20260811
INACTIVE_REFERENCE_NOISE_DISTRIBUTION = "standard_normal"
METADATA_ARRAY_NAMES = (
    "segment_id",
    "recording_id",
    "ego_id",
    "anchor_frame",
    "event_risk",
    "is_evt_tail",
    "primary_slot_index",
)


def output_dir_from_config(config: dict[str, Any], config_dir: str | Path) -> Path:
    """Return the directory that stores trained Flow artifacts."""
    return ensure_dir(resolve_path(config["paths"]["output_dir"], base=config_dir))


def dataset_dir_from_config(config: dict[str, Any], config_dir: str | Path) -> Path:
    """Return the shared full-natural-driving dataset directory."""
    return ensure_dir(resolve_path(config["paths"]["dataset_dir"], base=config_dir))


def dataset_npz_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "dataset.npz"


def schema_json_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "dataset_schema.json"


def _resolve_highd_preprocessing_config(
    config: dict[str, Any], config_dir: Path
) -> Path:
    return resolve_path(config["paths"]["highd_preprocessing_config"], base=config_dir)


def _resolve_natural_segments_csv(config: dict[str, Any], config_dir: Path) -> Path:
    highd_cfg_path = _resolve_highd_preprocessing_config(config, config_dir)
    highd_cfg = load_highd_config(highd_cfg_path)
    out_dir = resolve_data_path(highd_cfg["paths"]["output_dir"], highd_cfg_path)
    return out_dir / "natural_segments.csv"


def _resolve_raw_dir(config: dict[str, Any], config_dir: Path) -> Path:
    highd_cfg_path = _resolve_highd_preprocessing_config(config, config_dir)
    highd_cfg = load_highd_config(highd_cfg_path)
    return resolve_data_path(highd_cfg["paths"]["raw_dir"], highd_cfg_path)


def _load_evt_reference(config: dict[str, Any], config_dir: Path) -> tuple[Path, float]:
    evt_cfg_path = _resolve_highd_preprocessing_config(config, config_dir)
    evt_cfg = load_highd_config(evt_cfg_path)
    out_dir = resolve_data_path(evt_cfg["paths"]["output_dir"], evt_cfg_path)
    summary_path = out_dir / "evt_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing current EVT summary: {summary_path}")
    summary = load_json(summary_path)
    return summary_path, float(summary["u"])


def split_indices_by_recording(
    metadata: list[dict[str, Any]],
    split_cfg: dict[str, Any],
) -> np.ndarray:
    groups = np.asarray([item["recording_id"] for item in metadata], np.int64)
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(int(split_cfg["seed"]))
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    ratios = np.asarray(split_cfg["ratios"], dtype=np.float64)
    ratios = ratios / np.maximum(np.sum(ratios), 1.0e-12)
    n = len(shuffled)
    if n < 3:
        raise RuntimeError("Need at least three split groups for train/val/test")
    n_train = max(1, round(n * ratios[0]))
    n_val = max(1, round(n * ratios[1]))
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


def fit_feature_normalizer(
    raw_features: np.ndarray,
    feature_valid: np.ndarray,
    fit_rows: np.ndarray,
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    transform_kinds = feature_transform_kinds(feature_names)
    model_features = transform_features_for_model(
        raw_features,
        feature_valid,
        transform_kinds,
    )
    selected_rows = np.zeros(len(raw_features), dtype=bool)
    selected_rows[np.asarray(fit_rows, dtype=np.int64)] = True
    mean = np.zeros(raw_features.shape[1], dtype=np.float64)
    std = np.ones(raw_features.shape[1], dtype=np.float64)
    count = np.zeros(raw_features.shape[1], dtype=np.int64)
    for j in range(raw_features.shape[1]):
        valid = selected_rows & feature_valid[:, j] & np.isfinite(model_features[:, j])
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
        transform_kinds,
    )
    out = np.zeros_like(raw_features, dtype=np.float32)
    valid = feature_valid & np.isfinite(model_features)
    out[valid] = ((model_features - mean) / std)[valid]
    return out


def add_inactive_reference_noise(
    features_normalized: np.ndarray,
    feature_valid: np.ndarray,
    *,
    seed: int = INACTIVE_REFERENCE_NOISE_SEED,
) -> np.ndarray:
    """Make the continuous density well-defined when a neighbour is absent.

    A missing slot is a *discrete* event-structure outcome, not an observation
    at the continuous point zero.  The mask factor models that discrete
    structure.  For coordinates that are undefined under a mask we
    use a fixed, independent N(0, I) reference measure during flow fitting.
    These values are never interpreted as vehicle states: sampling clears all
    inactive coordinates before a physical state is returned.
    """
    out = np.asarray(features_normalized, dtype=np.float32).copy()
    invalid = ~np.asarray(feature_valid, dtype=bool)
    rng = np.random.default_rng(int(seed))
    out[invalid] = rng.standard_normal(int(np.sum(invalid))).astype(np.float32)
    return out


def _normalize_masked_array(
    values: np.ndarray,
    valid: np.ndarray,
    fit_rows: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize valid coordinates and regularize undefined coordinates."""
    raw = np.asarray(values, np.float32)
    present = np.asarray(valid, bool)
    selected_rows = np.zeros(len(raw), dtype=bool)
    selected_rows[np.asarray(fit_rows, dtype=np.int64)] = True
    mean = np.zeros(raw.shape[1:], np.float32)
    std = np.ones(raw.shape[1:], np.float32)
    count = np.zeros(raw.shape[1:], np.int64)
    for agent in range(raw.shape[1]):
        for feature in range(raw.shape[2]):
            selected = selected_rows & present[:, agent, feature]
            count[agent, feature] = int(selected.sum())
            if not selected.any():
                continue
            item = raw[selected, agent, feature].astype(np.float64)
            mean[agent, feature] = float(item.mean())
            value_std = float(item.std())
            std[agent, feature] = value_std if value_std > 1.0e-6 else 1.0
    normalized = np.zeros_like(raw)
    normalized[present] = ((raw - mean) / std)[present]
    rng = np.random.default_rng(INACTIVE_REFERENCE_NOISE_SEED + 1)
    normalized[~present] = rng.standard_normal(int((~present).sum())).astype(np.float32)
    return normalized, {
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "valid_train_count": count.astype(int).tolist(),
        "undefined_coordinate_policy": "independent_standard_normal_reference_noise",
    }


def _metadata_arrays(metadata: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key in METADATA_ARRAY_NAMES:
        values = [item[key] for item in metadata]
        if key == "segment_id":
            arrays[key] = np.asarray(values, dtype="U64")
        elif key == "event_risk":
            arrays[key] = np.asarray(values, dtype=np.float32)
        elif key == "is_evt_tail":
            arrays[key] = np.asarray(values, dtype=bool)
        else:
            arrays[key] = np.asarray(values, dtype=np.int64)
    return arrays


def build_scenario_dataset(
    config: dict[str, Any], *, config_dir: str | Path
) -> dict[str, Any]:
    config_dir = Path(config_dir).resolve()
    output_dir = dataset_dir_from_config(config, config_dir)
    segment_path = _resolve_natural_segments_csv(config, config_dir)
    if not segment_path.exists():
        raise FileNotFoundError(f"Missing current natural segments: {segment_path}")
    raw_dir = _resolve_raw_dir(config, config_dir)
    highd_cfg = load_highd_config(
        _resolve_highd_preprocessing_config(config, config_dir)
    )
    evt_summary_path, evt_u = _load_evt_reference(config, config_dir)

    natural_segments = pd.read_csv(segment_path)
    if natural_segments.empty:
        raise RuntimeError(f"Natural segment CSV is empty: {segment_path}")
    validate_natural_segment_contract(
        natural_segments,
        source=str(segment_path),
    )
    schema = build_feature_schema()
    features: list[np.ndarray] = []
    valids: list[np.ndarray] = []
    slot_masks: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    trajectory_constraints: list[np.ndarray] = []
    trajectory_constraint_valids: list[np.ndarray] = []
    vehicle_dimensions: list[np.ndarray] = []
    logger.info(
        "Extracting C0 features from %d current natural segments", len(natural_segments)
    )
    for recording_id, frame in natural_segments.groupby("recording_id", sort=True):
        rec = prepare_recording(raw_dir, int(recording_id), highd_cfg)
        logger.info(
            "Recording %02d: %d natural segments", int(recording_id), len(frame)
        )
        for _, row in frame.iterrows():
            feat, valid, mask, meta = extract_c0_features_for_segment(
                rec,
                row,
                schema=schema,
            )
            meta["is_evt_tail"] = bool(float(row["event_risk"]) > evt_u)
            constraint, constraint_valid = extract_constraint_for_segment(rec, row)
            features.append(feat)
            valids.append(valid)
            slot_masks.append(mask)
            metadata.append(meta)
            trajectory_constraints.append(constraint)
            trajectory_constraint_valids.append(constraint_valid)
            ids = [int(row["ego_id"]), *[int(row[f"{name}_id"]) for name in SLOT_NAMES]]
            dimensions = np.zeros((7, 2), np.float32)
            for agent, vehicle_id in enumerate(ids):
                if vehicle_id >= 0 and (agent == 0 or mask[agent - 1]):
                    dimensions[agent] = rec.tracks_meta.loc[
                        vehicle_id, ["width", "height"]
                    ]
            vehicle_dimensions.append(dimensions)

    raw_features = np.stack(features).astype(np.float32)
    feature_valid = np.stack(valids).astype(bool)
    slot_mask = np.stack(slot_masks).astype(bool)
    split_index = split_indices_by_recording(metadata, dict(config["dataset_split"]))
    dimensions_array = np.stack(vehicle_dimensions).astype(np.float32)
    present = np.column_stack((np.ones(len(slot_mask), bool), slot_mask))
    recorded_dimensions = dimensions_array[present]
    if not np.isfinite(recorded_dimensions).all() or (recorded_dimensions <= 0).any():
        raise ValueError("recorded vehicle dimensions must be finite and positive")
    trajectory_constraint_array = np.stack(trajectory_constraints).astype(np.float32)
    trajectory_constraint_valid_array = np.stack(trajectory_constraint_valids).astype(
        bool
    )
    tail_rows = np.flatnonzero(
        np.asarray([item["is_evt_tail"] for item in metadata], dtype=bool)
    )
    constraint_model_array = constraint_to_model_coordinates(
        trajectory_constraint_array, raw_features
    )
    constraint_normalized, constraint_normalization = _normalize_masked_array(
        constraint_model_array,
        trajectory_constraint_valid_array,
        tail_rows,
    )
    normalizer = fit_feature_normalizer(
        raw_features,
        feature_valid,
        tail_rows,
        schema.feature_names,
    )
    features_normalized = apply_feature_normalizer(
        raw_features,
        feature_valid,
        normalizer,
        schema.feature_names,
    )
    features_normalized = add_inactive_reference_noise(
        features_normalized,
        feature_valid,
    )
    mask_pattern = mask_pattern_from_slot_mask(slot_mask)
    meta_arrays = _metadata_arrays(metadata)

    arrays = {
        "features": raw_features,
        "features_normalized": features_normalized,
        "feature_valid": feature_valid,
        "slot_mask": slot_mask,
        "mask_pattern": mask_pattern,
        "trajectory_constraint": trajectory_constraint_array,
        "trajectory_constraint_normalized": constraint_normalized,
        "trajectory_constraint_valid": trajectory_constraint_valid_array,
        "split_index": split_index,
        "vehicle_dimensions": dimensions_array,
        **meta_arrays,
    }
    np.savez_compressed(dataset_npz_path(output_dir), **arrays)

    split_summary = {
        split: int(np.sum(split_index == idx)) for split, idx in SPLIT_TO_INDEX.items()
    }
    mask_summary = {
        str(int(pattern)): int(count)
        for pattern, count in Counter(mask_pattern.tolist()).most_common()
    }
    schema_payload = {
        "dataset_npz": str(dataset_npz_path(output_dir)),
        "dataset_scope": "full_clean_natural_driving",
        "source_segments_csv": str(segment_path),
        "evt_reference": {
            "summary_path": str(evt_summary_path),
            "pot_threshold_u": float(evt_u),
            "tail_label": "event_risk > u",
            "num_evt_tail": int(np.sum(meta_arrays["is_evt_tail"])),
            "num_non_tail": int(np.sum(~meta_arrays["is_evt_tail"])),
        },
        "raw_dir": str(raw_dir),
        "num_samples": int(raw_features.shape[0]),
        "metadata_arrays": list(METADATA_ARRAY_NAMES),
        "lateral_event_integrity_required": True,
        "background_slot_stability_required": True,
        "feature_names": list(schema.feature_names),
        "feature_mode": schema.feature_mode,
        "c0_initial_observation_only": True,
        "ego_features": list(schema.ego_features),
        "slot_features": list(schema.slot_features),
        "model_feature_transforms": list(feature_transform_kinds(schema.feature_names)),
        "slot_names": list(SLOT_NAMES),
        "probability_factorization": "p(M) p(C0|M) p(K|C0,M)",
        "vehicle_dimensions": {
            "shape_per_sequence": [7, 2],
            "feature_names": ["length_m", "width_m"],
            "units": "m",
        },
        "long_horizon_constraint": {
            "shape_per_sequence": [6, len(KNOT_FEATURE_NAMES)],
            "feature_names": list(KNOT_FEATURE_NAMES),
            "knot_frames": list(KNOT_INDICES),
            "knot_times_s": list(KNOT_TIMES_S),
            "coordinate_frame": "ego_forward_positive_x_left_positive_y",
            "model_coordinates": (
                "longitudinal interval-mean-speed changes and lateral positions"
            ),
            "normalization": constraint_normalization,
            "generation": "single_conditional_rq_spline_maf",
        },
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
            "fit_population": "all_evt_tail",
            "inactive_slot_policy": (
                "discrete_mask_with_independent_standard_normal_reference_noise"
            ),
            "inactive_reference_noise": {
                "distribution": INACTIVE_REFERENCE_NOISE_DISTRIBUTION,
                "seed": INACTIVE_REFERENCE_NOISE_SEED,
                "physical_output_policy": "zero_inactive_slot_features",
            },
            "coordinate_note": (
                "Mean/std normalization is fitted after the declared identity "
                "or signed-log transform. Raw values remain in dataset.npz."
            ),
        },
    }
    save_json(schema_payload, schema_json_path(output_dir))
    logger.info("Wrote normalizing dataset: %s", dataset_npz_path(output_dir))
    logger.info("Wrote normalizing schema: %s", schema_json_path(output_dir))
    return schema_payload


def load_scenario_dataset(
    output_dir: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
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


def evt_tail_indices(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Return every EVT-tail row used by the released Flow."""
    return np.flatnonzero(np.asarray(arrays["is_evt_tail"], bool))
