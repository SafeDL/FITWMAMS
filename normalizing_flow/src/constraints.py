"""Sparse long-horizon background-state constraints."""

from __future__ import annotations

from typing import Any

import numpy as np

from process_highD.src.natural_segments import SLOT_NAMES, _lateral_sign

KNOT_INDICES = (50, 100, 149)
KNOT_TIMES_S = (2.0, 4.0, 5.96)
KNOT_FEATURE_NAMES = tuple(
    f"{field}_{time_name}"
    for time_name in ("2s", "4s", "end")
    for field in ("dx_m", "dy_left_m", "dvx_mps", "dvy_left_mps")
)


def extract_constraint_for_segment(
    recording: Any,
    segment_row: Any,
    *,
    frames: int = 150,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the physical `[6,12]` diffusion condition from one clean window."""
    anchor = int(segment_row["anchor_frame"])
    sign = _lateral_sign(int(segment_row["ego_driving_direction"]))
    frame_index = np.arange(anchor, anchor + frames, dtype=np.int64)
    values = np.zeros((6, len(KNOT_FEATURE_NAMES)), np.float32)
    valid = np.zeros_like(values, dtype=bool)
    for slot_index, slot_name in enumerate(SLOT_NAMES):
        vehicle_id = int(segment_row.get(f"{slot_name}_id", -1))
        if vehicle_id < 0:
            continue
        track = recording.get_vehicle_track(vehicle_id).reindex(frame_index)
        required = ("x", "y", "xVelocity")
        if any(name not in track or track[name].isna().any() for name in required):
            raise ValueError(f"incomplete constraint track for {slot_name}")
        x = track["x"].to_numpy(np.float32)
        y = sign * track["y"].to_numpy(np.float32)
        vx = track["xVelocity"].to_numpy(np.float32)
        vy = (
            sign * track["yVelocity"].to_numpy(np.float32)
            if "yVelocity" in track
            else np.zeros(frames, np.float32)
        )
        for knot, index in enumerate(KNOT_INDICES):
            offset = 4 * knot
            values[slot_index, offset : offset + 4] = (
                x[index] - x[0],
                y[index] - y[0],
                vx[index] - vx[0],
                vy[index] - vy[0],
            )
        valid[slot_index] = True
    return values, valid


def derived_modes(constraint: np.ndarray, slot_mask: np.ndarray) -> np.ndarray:
    """Derive coarse longitudinal/lateral labels solely for diagnostics."""
    values = np.asarray(constraint, np.float32)
    slots = np.asarray(slot_mask, bool)
    modes = np.zeros((*slots.shape, 2), np.int64)
    end_dvx = values[..., 10]
    modes[..., 0] = np.where(end_dvx < -0.5, 0, np.where(end_dvx > 0.5, 2, 1))
    end_dy = values[..., 9]
    modes[..., 1] = np.where(end_dy > 1.8, 1, np.where(end_dy < -1.8, 2, 0))
    modes[~slots] = 0
    return modes
