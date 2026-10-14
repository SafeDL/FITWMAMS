"""Sparse long-horizon background-state constraints."""

from __future__ import annotations

from typing import Any

import numpy as np

from process_highD.src.natural_segments import SLOT_NAMES, _lateral_sign

from .features import feature_index

KNOT_INDICES = (50, 100, 149)
KNOT_TIMES_S = (2.0, 4.0, 5.96)
KNOT_FEATURE_NAMES = tuple(
    f"{field}_{time_name}"
    for time_name in ("2s", "4s", "end")
    for field in ("dx_m", "dy_left_m")
)


def constraint_to_model_coordinates(
    values: np.ndarray,
    c0: np.ndarray,
) -> np.ndarray:
    """Represent longitudinal knots by changes in interval-mean speed."""
    physical = np.asarray(values, np.float32)
    output = physical.copy()
    initial_speed = np.stack(
        [
            c0[:, feature_index(None, "ego_vx_mps")]
            + c0[:, feature_index(name, "rel_vx_mps")]
            for name in SLOT_NAMES
        ],
        axis=1,
    )
    position = np.zeros_like(initial_speed)
    previous_speed = initial_speed
    previous_time = 0.0
    for knot, time in enumerate(KNOT_TIMES_S):
        index = 2 * knot
        duration = float(time - previous_time)
        speed = (physical[..., index] - position) / duration
        output[..., index] = speed - previous_speed
        position = physical[..., index]
        previous_speed = speed
        previous_time = float(time)
    return output


def constraint_from_model_coordinates(
    values: np.ndarray,
    c0: np.ndarray,
) -> np.ndarray:
    """Restore cumulative longitudinal positions from interval-speed changes."""
    model = np.asarray(values, np.float32)
    output = model.copy()
    speed = np.stack(
        [
            c0[:, feature_index(None, "ego_vx_mps")]
            + c0[:, feature_index(name, "rel_vx_mps")]
            for name in SLOT_NAMES
        ],
        axis=1,
    )
    position = np.zeros_like(speed)
    previous_time = 0.0
    for knot, time in enumerate(KNOT_TIMES_S):
        index = 2 * knot
        duration = float(time - previous_time)
        speed += model[..., index]
        position += speed * duration
        output[..., index] = position
        previous_time = float(time)
    return output


def extract_constraint_for_segment(
    recording: Any,
    segment_row: Any,
    *,
    frames: int = 150,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the physical `[6,6]` position-knot condition from one clean window."""
    anchor = int(segment_row["anchor_frame"])
    sign = _lateral_sign(int(segment_row["ego_driving_direction"]))
    frame_index = np.arange(anchor, anchor + frames, dtype=np.int64)
    values = np.zeros((6, len(KNOT_FEATURE_NAMES)), np.float32)
    valid = np.zeros_like(values, dtype=bool)
    for slot_index, slot_name in enumerate(SLOT_NAMES):
        vehicle_id = int(segment_row[f"{slot_name}_id"])
        if vehicle_id < 0:
            continue
        track = recording.get_vehicle_track(vehicle_id).reindex(frame_index)
        required = ("x", "y", "xVelocity")
        if any(name not in track or track[name].isna().any() for name in required):
            raise ValueError(f"incomplete constraint track for {slot_name}")
        x = track["x"].to_numpy(np.float32)
        y = sign * track["y"].to_numpy(np.float32)
        for knot, index in enumerate(KNOT_INDICES):
            offset = 2 * knot
            values[slot_index, offset : offset + 2] = (
                x[index] - x[0],
                y[index] - y[0],
            )
        valid[slot_index] = True
    return values, valid


def derived_modes(constraint: np.ndarray, slot_mask: np.ndarray) -> np.ndarray:
    """Derive coarse longitudinal/lateral labels solely for diagnostics."""
    values = np.asarray(constraint, np.float32)
    slots = np.asarray(slot_mask, bool)
    modes = np.zeros((*slots.shape, 2), np.int64)
    early_speed = values[..., 0] / float(KNOT_TIMES_S[0])
    late_speed = (values[..., 4] - values[..., 2]) / float(
        KNOT_TIMES_S[2] - KNOT_TIMES_S[1]
    )
    speed_change = late_speed - early_speed
    modes[..., 0] = np.where(speed_change < -0.5, 0, np.where(speed_change > 0.5, 2, 1))
    end_dy = values[..., 5]
    modes[..., 1] = np.where(end_dy > 1.8, 1, np.where(end_dy < -1.8, 2, 0))
    modes[~slots] = 0
    return modes
