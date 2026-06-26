"""Exposure calibration helpers for highD following-event mileage estimates."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd


METER_PER_MILE = 1609.344
KM_PER_MILE = 1.609344


def union_inclusive_intervals(
    intervals: Iterable[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return sorted inclusive intervals with overlaps and adjacency merged."""
    normalized: list[tuple[int, int]] = []
    for start, end in intervals:
        start_i = int(start)
        end_i = int(end)
        if end_i < start_i:
            raise ValueError(f"Invalid interval with end < start: {(start, end)}")
        normalized.append((start_i, end_i))
    if not normalized:
        return []

    normalized.sort(key=lambda item: (item[0], item[1]))
    merged = [normalized[0]]
    for start, end in normalized[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def track_interval_exposure(
    track: pd.DataFrame,
    intervals: Sequence[tuple[int, int]],
    *,
    fps: float,
    x_column: str = "x",
) -> dict[str, float]:
    """Compute distance and time for a single ego track over unioned intervals."""
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    if x_column not in track.columns:
        raise KeyError(f"track is missing required x column: {x_column}")
    if track.empty or not intervals:
        return {"distance_m": 0.0, "hours": 0.0}

    total_distance_m = 0.0
    total_seconds = 0.0
    frame_index = np.asarray(track.index, dtype=np.int64)
    x_values = track[x_column].astype(float).to_numpy()
    for start, end in intervals:
        mask = (frame_index >= int(start)) & (frame_index <= int(end))
        if int(np.count_nonzero(mask)) < 2:
            continue
        x = x_values[mask]
        total_distance_m += float(np.sum(np.abs(np.diff(x))))
        total_seconds += float(int(np.count_nonzero(mask)) - 1) / float(fps)
    return {
        "distance_m": float(total_distance_m),
        "hours": float(total_seconds / 3600.0),
    }


def following_exposure_for_recording(
    events: pd.DataFrame,
    *,
    recording_id: int,
    get_track: Any,
    fps: float,
) -> dict[str, Any]:
    """Compute following ego exposure for one recording without double-counting."""
    if events.empty:
        return {
            "recording_id": int(recording_id),
            "num_following_events": 0,
            "num_ego_vehicles": 0,
            "num_union_intervals": 0,
            "following_ego_distance_m": 0.0,
            "following_ego_miles": 0.0,
            "following_ego_hours": 0.0,
        }

    required = {"ego_id", "start_frame", "end_frame"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise KeyError(f"events is missing required columns: {missing}")

    total_distance_m = 0.0
    total_hours = 0.0
    total_intervals = 0
    for ego_id, group in events.groupby("ego_id", sort=True):
        intervals = union_inclusive_intervals(
            (
                int(row["start_frame"]),
                int(row["end_frame"]),
            )
            for _, row in group.iterrows()
        )
        total_intervals += len(intervals)
        exposure = track_interval_exposure(
            get_track(int(ego_id)),
            intervals,
            fps=fps,
        )
        total_distance_m += float(exposure["distance_m"])
        total_hours += float(exposure["hours"])

    return {
        "recording_id": int(recording_id),
        "num_following_events": int(len(events)),
        "num_ego_vehicles": int(events["ego_id"].nunique()),
        "num_union_intervals": int(total_intervals),
        "following_ego_distance_m": float(total_distance_m),
        "following_ego_miles": float(total_distance_m / METER_PER_MILE),
        "following_ego_hours": float(total_hours),
    }


def all_vehicle_exposure_for_recording(
    *,
    recording_id: int,
    vehicle_ids: Sequence[int],
    get_track: Any,
    fps: float,
) -> dict[str, Any]:
    """Compute full highD vehicle exposure for one recording."""
    if fps <= 0.0:
        raise ValueError("fps must be positive")

    total_distance_m = 0.0
    total_hours = 0.0
    num_tracks = 0
    for vehicle_id in vehicle_ids:
        track = get_track(int(vehicle_id))
        exposure = track_interval_exposure(
            track,
            [
                (
                    int(track.index.min()),
                    int(track.index.max()),
                )
            ],
            fps=fps,
        )
        total_distance_m += float(exposure["distance_m"])
        total_hours += float(exposure["hours"])
        num_tracks += 1

    return {
        "recording_id": int(recording_id),
        "num_all_vehicles": int(num_tracks),
        "all_vehicle_distance_m": float(total_distance_m),
        "all_vehicle_miles": float(total_distance_m / METER_PER_MILE),
        "all_vehicle_hours": float(total_hours),
    }


def extract_independent_peaks(
    events: pd.DataFrame,
    *,
    run_length_seconds: float,
    fps: float,
    group_keys: Sequence[str] = ("recording_id", "ego_id"),
    threshold_u: float | None = None,
    score_column: str = "y_long",
) -> list[dict[str, Any]]:
    """Decluster event rows and return one representative peak per cluster."""
    score_column = str(score_column)
    required = {
        *group_keys,
        "event_id",
        "target_id",
        "start_frame",
        "end_frame",
        "anchor_frame",
        score_column,
    }
    missing = sorted(set(required) - set(events.columns))
    if missing:
        raise KeyError(f"events is missing required columns: {missing}")
    if fps <= 0.0:
        raise ValueError("fps must be positive")

    score_values = pd.to_numeric(events[score_column], errors="coerce")
    mask = np.isfinite(score_values)
    if threshold_u is not None:
        mask = mask & (score_values > float(threshold_u))
    finite = events[mask].copy()
    if finite.empty:
        return []
    finite["_peak_score_numeric"] = pd.to_numeric(
        finite[score_column],
        errors="coerce",
    )
    run_length_frames = max(0, int(np.ceil(float(run_length_seconds) * float(fps))))
    peaks: list[dict[str, Any]] = []
    for _, group in finite.groupby(list(group_keys), sort=True):
        group = group.sort_values(["anchor_frame", "event_id"], kind="mergesort")
        cluster: list[pd.Series] = []
        last_anchor: int | None = None
        for _, row in group.iterrows():
            anchor = int(row["anchor_frame"])
            if last_anchor is None or anchor - last_anchor <= run_length_frames:
                cluster.append(row)
            else:
                peaks.append(
                    _cluster_peak_row(
                        cluster,
                        len(peaks),
                        score_column=score_column,
                    )
                )
                cluster = [row]
            last_anchor = anchor
        if cluster:
            peaks.append(
                _cluster_peak_row(
                    cluster,
                    len(peaks),
                    score_column=score_column,
                )
            )
    return peaks


def peak_rate_summary(
    *,
    total_exposure_miles: float,
    total_exposure_hours: float,
    num_independent_tail_peaks: int,
) -> dict[str, float]:
    """Return independent tail-peak rates for mileage conversion."""
    peaks = int(num_independent_tail_peaks)
    miles = float(total_exposure_miles)
    hours = float(total_exposure_hours)
    return {
        "tail_peak_rate_per_mile": float(peaks / miles) if miles > 0.0 else 0.0,
        "tail_peak_rate_per_hour": float(peaks / hours) if hours > 0.0 else 0.0,
    }


def collision_distance_summary(
    *,
    tail_peak_rate_per_mile: float,
    tail_peak_rate_per_hour: float,
    tail_conditional_probability_above_collision_level: float,
) -> dict[str, float]:
    """Return critical-event intensity from a POT conditional tail probability."""
    survival = max(float(tail_conditional_probability_above_collision_level), 0.0)
    rate_mile = max(float(tail_peak_rate_per_mile), 0.0)
    rate_hour = max(float(tail_peak_rate_per_hour), 0.0)
    intensity_mile = float(rate_mile * survival)
    intensity_hour = float(rate_hour * survival)
    return {
        "tail_conditional_probability_above_collision_level": survival,
        "tail_conditional_probability_above_safety_critical_level": survival,
        "highd_safety_critical_intensity_per_mile": intensity_mile,
        "highd_safety_critical_return_period_miles": (
            float(1.0 / intensity_mile)
            if intensity_mile > 0.0
            else float("inf")
        ),
        "highd_safety_critical_intensity_per_km": float(intensity_mile / KM_PER_MILE),
        "highd_safety_critical_return_period_km": (
            float(KM_PER_MILE / intensity_mile)
            if intensity_mile > 0.0
            else float("inf")
        ),
        "highd_safety_critical_intensity_per_hour": intensity_hour,
        "highd_safety_critical_return_period_hours": (
            float(1.0 / intensity_hour)
            if intensity_hour > 0.0
            else float("inf")
        ),
        "highd_collision_intensity_per_mile": intensity_mile,
        "highd_collision_return_period_miles": (
            float(1.0 / intensity_mile)
            if intensity_mile > 0.0
            else float("inf")
        ),
        "highd_collision_intensity_per_km": float(intensity_mile / KM_PER_MILE),
        "highd_collision_return_period_km": (
            float(KM_PER_MILE / intensity_mile)
            if intensity_mile > 0.0
            else float("inf")
        ),
        "highd_collision_intensity_per_hour": intensity_hour,
        "highd_collision_return_period_hours": (
            float(1.0 / intensity_hour)
            if intensity_hour > 0.0
            else float("inf")
        ),
    }


def _cluster_peak_row(
    cluster: Sequence[pd.Series],
    peak_index: int,
    *,
    score_column: str = "y_long",
) -> dict[str, Any]:
    frame = pd.DataFrame(cluster)
    rep_idx = frame["_peak_score_numeric"].astype(float).idxmax()
    rep = frame.loc[rep_idx]
    score_value = float(rep["_peak_score_numeric"])
    out = {
        "peak_id": f"peak_{peak_index:06d}",
        "recording_id": int(rep["recording_id"]),
        "ego_id": int(rep["ego_id"]),
        "target_id": int(rep["target_id"]),
        "cluster_start_frame": int(frame["start_frame"].astype(int).min()),
        "cluster_end_frame": int(frame["end_frame"].astype(int).max()),
        "representative_event_id": str(rep["event_id"]),
        "representative_anchor_frame": int(rep["anchor_frame"]),
        "num_events_in_cluster": int(len(frame)),
    }
    out[f"{score_column}_max"] = score_value
    return out
