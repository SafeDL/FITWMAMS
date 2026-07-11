"""Feature extraction for EVT tail initial-state conditions."""
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
    "ego_ax_mps2",
    "ego_ay_left_mps2",
)

SLOT_FEATURES: tuple[str, ...] = (
    "rel_x_m",
    "rel_y_left_m",
    "rel_vx_mps",
    "rel_vy_left_mps",
    "other_ax_mps2",
    "other_ay_left_mps2",
)

TRAJECTORY_FEATURES: tuple[str, ...] = (
    "delta_vx_1s_mps",
    "delta_vy_left_1s_mps",
    "mean_ax_1s_mps2",
    "min_ax_1s_mps2",
    "final_ax_1s_mps2",
    "mean_ay_left_1s_mps2",
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
    trajectory_features: tuple[str, ...] = TRAJECTORY_FEATURES
    slot_names: tuple[str, ...] = SLOT_NAMES

    @property
    def num_features(self) -> int:
        return len(self.feature_names)


def build_feature_schema() -> C0FeatureSchema:
    names: list[str] = [*EGO_FEATURES]
    for slot_name in SLOT_NAMES:
        names.extend(f"{slot_name}_{feature}" for feature in SLOT_FEATURES)
    for slot_name in SLOT_NAMES:
        names.extend(f"{slot_name}_{feature}" for feature in TRAJECTORY_FEATURES)
    return C0FeatureSchema(feature_names=tuple(names))


def mask_pattern_from_slot_mask(slot_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(slot_mask, dtype=bool)
    if mask.ndim == 1:
        mask = mask.reshape(1, -1)
    powers = (1 << np.arange(mask.shape[1], dtype=np.int64)).reshape(1, -1)
    return np.sum(mask.astype(np.int64) * powers, axis=1).astype(np.int64)


def slot_mask_from_pattern(mask_pattern: np.ndarray) -> np.ndarray:
    pattern = np.asarray(mask_pattern, dtype=np.int64).reshape(-1)
    powers = (1 << np.arange(len(SLOT_NAMES), dtype=np.int64)).reshape(1, -1)
    return (pattern.reshape(-1, 1) & powers) > 0


def slot_feature_index(slot_name: str, feature: str) -> int:
    return (
        len(EGO_FEATURES)
        + SLOT_NAMES.index(slot_name) * len(SLOT_FEATURES)
        + SLOT_FEATURES.index(feature)
    )


def trajectory_feature_index(slot_name: str, feature: str) -> int:
    return (
        len(EGO_FEATURES)
        + len(SLOT_NAMES) * len(SLOT_FEATURES)
        + SLOT_NAMES.index(slot_name) * len(TRAJECTORY_FEATURES)
        + TRAJECTORY_FEATURES.index(feature)
    )


def feature_index(slot_name: str | None, feature: str) -> int:
    if slot_name is None:
        return EGO_FEATURES.index(feature)
    return slot_feature_index(slot_name, feature)


def slot_state_slice(slot_idx: int) -> slice:
    start = len(EGO_FEATURES) + int(slot_idx) * len(SLOT_FEATURES)
    return slice(start, start + len(SLOT_FEATURES))


def slot_trajectory_slice(slot_idx: int) -> slice:
    start = (
        len(EGO_FEATURES)
        + len(SLOT_NAMES) * len(SLOT_FEATURES)
        + int(slot_idx) * len(TRAJECTORY_FEATURES)
    )
    return slice(start, start + len(TRAJECTORY_FEATURES))


def feature_valid_from_slot_mask(schema: dict[str, Any], slot_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(slot_mask, dtype=bool)
    if mask.ndim == 1:
        mask = mask.reshape(1, -1)
    out = np.zeros((mask.shape[0], len(schema["feature_names"])), dtype=bool)
    out[:, : len(EGO_FEATURES)] = True
    for slot_idx in range(len(SLOT_NAMES)):
        out[:, slot_state_slice(slot_idx)] = mask[:, [slot_idx]]
        out[:, slot_trajectory_slice(slot_idx)] = mask[:, [slot_idx]]
    return out


def zero_inactive_slot_features(raw: np.ndarray, slot_mask: np.ndarray) -> np.ndarray:
    out = np.asarray(raw, dtype=np.float32).copy()
    mask = np.asarray(slot_mask, dtype=bool)
    if mask.ndim == 1:
        mask = mask.reshape(1, -1)
    for slot_idx in range(len(SLOT_NAMES)):
        inactive = ~mask[:, slot_idx]
        out[inactive, slot_state_slice(slot_idx)] = 0.0
        out[inactive, slot_trajectory_slice(slot_idx)] = 0.0
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
    return int(segment_row.get(f"{slot_name}_id", -1))


def _choose_primary_slot(segment_row: pd.Series, slot_mask: np.ndarray) -> tuple[int, str]:
    peak_slot = str(segment_row.get("peak_slot_name", "none"))
    if peak_slot in SLOT_NAMES:
        peak_idx = SLOT_NAMES.index(peak_slot)
        peak_vehicle_id = _vehicle_id_for_slot(segment_row, peak_slot)
        if bool(slot_mask[peak_idx]) and peak_vehicle_id >= 0:
            return peak_idx, peak_slot

    if bool(slot_mask[SLOT_NAMES.index("same_front")]):
        slot_name = "same_front"
        return SLOT_NAMES.index(slot_name), slot_name

    active = np.where(slot_mask)[0]
    if len(active) == 0:
        raise ValueError("tail context has no active neighbor slot")
    slot_idx = int(active[0])
    slot_name = SLOT_NAMES[slot_idx]
    return slot_idx, slot_name


def _extract_slot_action_features(
    *,
    target_track: pd.DataFrame,
    target_anchor: pd.Series,
    anchor_frame: int,
    horizon_steps: int,
    lat_sign: int,
) -> dict[str, float]:
    final_frame = int(anchor_frame) + int(horizon_steps)
    target_final = _series_at_frame(target_track, final_frame)
    if target_final is None:
        raise ValueError(
            f"missing slot future state at frame={final_frame} for 1s action summary"
        )

    window = target_track.loc[
        (target_track.index >= int(anchor_frame))
        & (target_track.index <= final_frame)
    ]
    if window.empty:
        raise ValueError("missing slot future window for 1s action summary")
    if "_abnormal" in window.columns and bool(window["_abnormal"].any()):
        raise ValueError("slot future window contains abnormal frames")

    ax = window["xAcceleration"].astype(np.float32).to_numpy()
    if "yAcceleration" in window.columns:
        ay = window["yAcceleration"].astype(np.float32).to_numpy()
    else:
        ay = np.zeros(len(window), dtype=np.float32)
    ay_left = lat_sign * ay
    return {
        "delta_vx_1s_mps": float(target_final["xVelocity"] - target_anchor["xVelocity"]),
        "delta_vy_left_1s_mps": float(
            lat_sign
            * (
                float(target_final.get("yVelocity", 0.0))
                - float(target_anchor.get("yVelocity", 0.0))
            )
        ),
        "mean_ax_1s_mps2": float(np.mean(ax)),
        "min_ax_1s_mps2": float(np.min(ax)),
        "final_ax_1s_mps2": float(target_final.get("xAcceleration", 0.0)),
        "mean_ay_left_1s_mps2": float(np.mean(ay_left)),
    }


def extract_c0_features_for_segment(
    recording: Any,
    segment_row: pd.Series,
    *,
    schema: C0FeatureSchema | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract fixed-width c0 features for one selected tail segment.

    Inactive slots are encoded as zeros and marked invalid in ``feature_valid``;
    downstream checks and exports must consult ``slot_mask`` rather than reading
    placeholder values as real vehicles.
    """
    schema = schema or build_feature_schema()
    feature = np.zeros(schema.num_features, dtype=np.float32)
    feature_valid = np.zeros(schema.num_features, dtype=bool)
    slot_mask = np.zeros(len(SLOT_NAMES), dtype=bool)

    anchor_frame = int(segment_row["anchor_frame"])
    ego_id = int(segment_row["ego_id"])
    ego_track = recording.get_vehicle_track(ego_id)
    ego = _series_at_frame(ego_track, anchor_frame)
    if ego is None:
        raise ValueError(
            f"missing ego state for segment={segment_row.get('segment_id')} "
            f"ego={ego_id} frame={anchor_frame}"
        )
    ego_meta = recording.tracks_meta.loc[ego_id]
    ego_direction = int(segment_row.get("ego_driving_direction", ego_meta["drivingDirection"]))
    lat_sign = _lateral_sign(ego_direction)
    ego_vy_left = lat_sign * float(ego.get("yVelocity", 0.0))
    ego_ay_left = lat_sign * float(ego.get("yAcceleration", 0.0))
    ego_values = {
        "ego_vx_mps": float(ego["xVelocity"]),
        "ego_vy_left_mps": ego_vy_left,
        "ego_ax_mps2": float(ego.get("xAcceleration", 0.0)),
        "ego_ay_left_mps2": ego_ay_left,
    }
    for idx, name in enumerate(EGO_FEATURES):
        value = float(ego_values[name])
        feature[idx] = value if np.isfinite(value) else 0.0
        feature_valid[idx] = np.isfinite(value)

    offset = len(EGO_FEATURES)
    slot_rows: dict[str, pd.Series] = {}
    slot_tracks: dict[str, pd.DataFrame] = {}
    for slot_idx, slot_name in enumerate(SLOT_NAMES):
        vehicle_id = _vehicle_id_for_slot(segment_row, slot_name)
        slot_start = offset + slot_idx * len(SLOT_FEATURES)
        if vehicle_id < 0:
            continue
        try:
            other_track = recording.get_vehicle_track(vehicle_id)
        except KeyError:
            continue
        other = _series_at_frame(other_track, anchor_frame)
        if other is None:
            continue
        rel_values = {
            "rel_x_m": float(other["x"]) - float(ego["x"]),
            "rel_y_left_m": lat_sign * (float(other["y"]) - float(ego["y"])),
            "rel_vx_mps": float(other["xVelocity"]) - float(ego["xVelocity"]),
            "rel_vy_left_mps": lat_sign
            * (float(other.get("yVelocity", 0.0)) - float(ego.get("yVelocity", 0.0))),
            "other_ax_mps2": float(other.get("xAcceleration", 0.0)),
            "other_ay_left_mps2": lat_sign * float(other.get("yAcceleration", 0.0)),
        }
        finite = all(np.isfinite(value) for value in rel_values.values())
        if not finite:
            continue
        slot_rows[slot_name] = other
        slot_tracks[slot_name] = other_track
        slot_mask[slot_idx] = True
        for local_idx, name in enumerate(SLOT_FEATURES):
            feature[slot_start + local_idx] = float(rel_values[name])
            feature_valid[slot_start + local_idx] = True

    primary_slot_idx, primary_slot_name = _choose_primary_slot(
        segment_row,
        slot_mask,
    )
    if primary_slot_name not in slot_rows:
        raise ValueError(f"primary slot {primary_slot_name!r} has no valid anchor row")
    fps = float(recording.recording_meta.get("frameRate", 25.0))
    horizon_steps = int(round(fps))
    trajectory_offset = len(EGO_FEATURES) + len(SLOT_NAMES) * len(SLOT_FEATURES)
    for slot_idx, slot_name in enumerate(SLOT_NAMES):
        if not bool(slot_mask[slot_idx]):
            continue
        action_values = _extract_slot_action_features(
            target_track=slot_tracks[slot_name],
            target_anchor=slot_rows[slot_name],
            anchor_frame=anchor_frame,
            horizon_steps=horizon_steps,
            lat_sign=lat_sign,
        )
        slot_action_start = trajectory_offset + slot_idx * len(TRAJECTORY_FEATURES)
        for local_idx, name in enumerate(TRAJECTORY_FEATURES):
            value = float(action_values[name])
            feature[slot_action_start + local_idx] = value if np.isfinite(value) else 0.0
            feature_valid[slot_action_start + local_idx] = np.isfinite(value)

    metadata = {
        "segment_id": str(segment_row["segment_id"]),
        "recording_id": int(segment_row["recording_id"]),
        "ego_id": ego_id,
        "anchor_frame": anchor_frame,
        "event_risk": float(segment_row["event_risk"]),
        "primary_slot_name": primary_slot_name,
        "primary_slot_index": int(primary_slot_idx),
    }
    return feature, feature_valid, slot_mask, metadata
