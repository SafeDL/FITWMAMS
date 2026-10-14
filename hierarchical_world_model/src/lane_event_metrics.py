"""Pooled lane-change event detector shared by log and world replays."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d


def lane_change_events(
    states: np.ndarray,
    valid: np.ndarray,
    maps: np.ndarray,
    map_valid: np.ndarray,
    *,
    dt_s: float = 0.04,
) -> list[dict[str, float | int]]:
    """Detect complete lateral-motion segments ending in an adjacent lane crossing.

    ``states`` and ``valid`` use the standard seven-vehicle scene layout. Left
    and right changes are deliberately pooled; direction is not a model split.
    """
    centers = maps[:, :, 0, 1]
    lane_present = map_valid.any(axis=-1)
    y = states[:, :, 1:, 1]
    distance = np.abs(y[..., None] - centers[:, None, None, :])
    distance = np.where(lane_present[:, None, None, :], distance, np.inf)
    lane_index = distance.argmin(axis=-1)
    lateral_speed = np.abs(states[:, :, 1:, 3])
    smooth_speed = uniform_filter1d(
        lateral_speed, size=5, axis=1, mode="constant", cval=0.0
    )
    moving = (smooth_speed > 0.15) & valid[:, :, 1:]
    events: list[dict[str, float | int]] = []

    for scene in range(len(states)):
        for slot in range(6):
            active = moving[scene, :, slot]
            edges = np.diff(np.r_[False, active, False].astype(np.int8))
            begins = np.flatnonzero(edges == 1)
            ends = np.flatnonzero(edges == -1) - 1
            crossings = (
                np.flatnonzero(
                    lane_index[scene, 1:, slot] != lane_index[scene, :-1, slot]
                )
                + 1
            )
            for begin, end in zip(begins, ends, strict=True):
                duration = (end - begin + 1) * dt_s
                if begin == 0 or end == len(active) - 1 or not 2.0 <= duration <= 10.0:
                    continue
                if not valid[scene, begin : end + 1, slot + 1].all():
                    continue
                peak = float(lateral_speed[scene, begin : end + 1, slot].max())
                if not 0.4 <= peak <= 3.0:
                    continue
                nearby = crossings[(crossings >= begin - 10) & (crossings <= end + 10)]
                if not len(nearby):
                    continue
                crossing = int(nearby[0])
                source = lane_index[scene, crossing - 1, slot]
                target = lane_index[scene, crossing, slot]
                width = abs(float(centers[scene, source] - centers[scene, target]))
                if not 1.8 < width < 5.4:
                    continue
                speed = np.linalg.norm(states[scene, begin, slot + 1, 2:4])
                events.append(
                    {
                        "scene": scene,
                        "slot": slot,
                        "start_frame": int(begin),
                        "speed_mps": float(speed),
                        "peak_lateral_speed_mps": peak,
                        "duration_s": float(duration),
                    }
                )
    return events


def summarize_lane_change_events(
    events: list[dict[str, float | int]],
) -> dict[str, object]:
    """Summarize pooled event count and the shared detector's core behavior features."""
    result: dict[str, object] = {"count": len(events)}
    for source, target in (
        ("duration_s", "duration_q05_q50_q95_s"),
        ("peak_lateral_speed_mps", "peak_lateral_speed_q05_q50_q95_mps"),
        ("speed_mps", "entry_speed_q05_q50_q95_mps"),
    ):
        values = [float(event[source]) for event in events]
        result[target] = (
            np.quantile(values, [0.05, 0.5, 0.95]).tolist() if values else None
        )
    return result
