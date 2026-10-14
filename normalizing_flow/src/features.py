"""Feature extraction for full natural-driving initial conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from process_highD.src.natural_segments import (
    SLOT_NAMES,
    _lateral_sign,
)

EGO_FEATURES: tuple[str, ...] = (
    "ego_vx_mps",
    "ego_vy_left_mps",
)

SLOT_FEATURES: tuple[str, ...] = (
    "rel_x_m",
    "rel_y_left_m",
    "rel_vx_mps",
    "rel_vy_left_mps",
)

DEFAULT_EGO_LENGTH_M = 4.8
DEFAULT_EGO_WIDTH_M = 1.9
DEFAULT_OTHER_LENGTH_M = 4.8
DEFAULT_OTHER_WIDTH_M = 1.9
DEFAULT_LANE_WIDTH_M = 3.6


@dataclass(frozen=True)
class C0FeatureSchema:
    feature_names: tuple[str, ...]
    ego_features: tuple[str, ...] = EGO_FEATURES
    slot_features: tuple[str, ...] = SLOT_FEATURES
    slot_names: tuple[str, ...] = SLOT_NAMES
    feature_mode: str = "initial_observation"

    @property
    def num_features(self) -> int:
        return len(self.feature_names)


def build_feature_schema() -> C0FeatureSchema:
    """Build the fixed initial-observation feature schema."""
    names: list[str] = [*EGO_FEATURES]
    for slot_name in SLOT_NAMES:
        names.extend(f"{slot_name}_{feature}" for feature in SLOT_FEATURES)
    return C0FeatureSchema(
        feature_names=tuple(names),
        feature_mode="initial_observation",
    )


def mask_pattern_from_slot_mask(slot_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(slot_mask, dtype=bool)
    if mask.ndim == 1:
        mask = mask.reshape(1, -1)
    powers = (1 << np.arange(mask.shape[1], dtype=np.int64)).reshape(1, -1)
    return np.sum(mask.astype(np.int64) * powers, axis=1).astype(np.int64)


def slot_feature_index(slot_name: str, feature: str) -> int:
    return (
        len(EGO_FEATURES)
        + SLOT_NAMES.index(slot_name) * len(SLOT_FEATURES)
        + SLOT_FEATURES.index(feature)
    )


def feature_index(slot_name: str | None, feature: str) -> int:
    if slot_name is None:
        return EGO_FEATURES.index(feature)
    return slot_feature_index(slot_name, feature)


def slot_state_slice(slot_idx: int) -> slice:
    start = len(EGO_FEATURES) + int(slot_idx) * len(SLOT_FEATURES)
    return slice(start, start + len(SLOT_FEATURES))


def feature_valid_from_slot_mask(
    schema: dict[str, Any], slot_mask: np.ndarray
) -> np.ndarray:
    mask = np.asarray(slot_mask, dtype=bool)
    if mask.ndim == 1:
        mask = mask.reshape(1, -1)
    out = np.zeros((mask.shape[0], len(schema["feature_names"])), dtype=bool)
    out[:, : len(EGO_FEATURES)] = True
    for slot_idx in range(len(SLOT_NAMES)):
        out[:, slot_state_slice(slot_idx)] = mask[:, [slot_idx]]
    return out


def _series_at_frame(track: pd.DataFrame, frame: int) -> pd.Series | None:
    try:
        row = track.loc[int(frame)]
    except KeyError:
        return None
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    if bool(row.get("_abnormal", False)):
        return None
    return row


def _vehicle_id_for_slot(segment_row: pd.Series, slot_name: str) -> int:
    return int(segment_row[f"{slot_name}_id"])


def _primary_slot_index(segment_row: pd.Series, slot_mask: np.ndarray) -> int:
    name = str(segment_row["peak_slot_name"])
    if name in SLOT_NAMES:
        return SLOT_NAMES.index(name)
    active = np.flatnonzero(slot_mask)
    return int(active[0]) if active.size else 0


def extract_c0_features_for_segment(
    recording: Any,
    segment_row: pd.Series,
    *,
    schema: C0FeatureSchema,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract fixed-width C0 features for one natural-driving segment.

    Inactive slots are encoded as zeros and marked invalid in ``feature_valid``;
    downstream checks and exports must consult ``slot_mask`` rather than reading
    placeholder values as real vehicles.
    """
    feature = np.zeros(schema.num_features, dtype=np.float32)
    feature_valid = np.zeros(schema.num_features, dtype=bool)
    slot_mask = np.zeros(len(SLOT_NAMES), dtype=bool)

    anchor_frame = int(segment_row["anchor_frame"])
    ego_id = int(segment_row["ego_id"])
    ego_track = recording.get_vehicle_track(ego_id)
    ego = _series_at_frame(ego_track, anchor_frame)
    if ego is None:
        raise ValueError(
            f"missing ego state for segment={segment_row['segment_id']} "
            f"ego={ego_id} frame={anchor_frame}"
        )
    ego_direction = int(segment_row["ego_driving_direction"])
    lat_sign = _lateral_sign(ego_direction)
    ego_vy_left = lat_sign * float(ego["yVelocity"])
    ego_values = {
        "ego_vx_mps": float(ego["xVelocity"]),
        "ego_vy_left_mps": ego_vy_left,
    }
    for idx, name in enumerate(EGO_FEATURES):
        value = float(ego_values[name])
        feature[idx] = value if np.isfinite(value) else 0.0
        feature_valid[idx] = np.isfinite(value)

    offset = len(EGO_FEATURES)
    for slot_idx, slot_name in enumerate(SLOT_NAMES):
        vehicle_id = _vehicle_id_for_slot(segment_row, slot_name)
        slot_start = offset + slot_idx * len(SLOT_FEATURES)
        if vehicle_id < 0:
            continue
        other_track = recording.get_vehicle_track(vehicle_id)
        other = _series_at_frame(other_track, anchor_frame)
        if other is None:
            raise ValueError(
                f"missing {slot_name} state for segment={segment_row['segment_id']} "
                f"vehicle={vehicle_id} frame={anchor_frame}"
            )
        rel_values = {
            "rel_x_m": float(other["x"]) - float(ego["x"]),
            "rel_y_left_m": lat_sign * (float(other["y"]) - float(ego["y"])),
            "rel_vx_mps": float(other["xVelocity"]) - float(ego["xVelocity"]),
            "rel_vy_left_mps": lat_sign
            * (float(other["yVelocity"]) - float(ego["yVelocity"])),
        }
        finite = all(np.isfinite(value) for value in rel_values.values())
        if not finite:
            raise ValueError(
                f"non-finite {slot_name} state for segment={segment_row['segment_id']}"
            )
        slot_mask[slot_idx] = True
        for local_idx, name in enumerate(SLOT_FEATURES):
            feature[slot_start + local_idx] = float(rel_values[name])
            feature_valid[slot_start + local_idx] = True

    metadata = {
        "segment_id": str(segment_row["segment_id"]),
        "recording_id": int(segment_row["recording_id"]),
        "ego_id": ego_id,
        "anchor_frame": anchor_frame,
        "event_risk": float(segment_row["event_risk"]),
        "primary_slot_index": _primary_slot_index(segment_row, slot_mask),
    }
    return feature, feature_valid, slot_mask, metadata
