"""Natural highD local segments and scenario-agnostic risk traces.

This module builds fixed-length ego-centric highD snippets for EVT
calibration.  It deliberately does not classify snippets as following,
cut-in, lane-change, etc.; every valid anchor uses the same six semantic
neighbor slots assigned at the anchor frame.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .lane_utils import are_adjacent_lanes, parse_lane_markings
from .safety_envelope_risk import (
    SEI_COMPONENT_NAMES,
    SafetyEnvelopeRiskOptions,
    pairwise_safety_envelope_intrusion,
    smoothmax_axis,
    trajectory_safety_envelope_risk_trace,
)

SLOT_NAMES = (
    "same_front",
    "same_rear",
    "left_front",
    "left_rear",
    "right_front",
    "right_rear",
)

RISK_COMPONENT_NAMES = SEI_COMPONENT_NAMES


@dataclass(frozen=True)
class NaturalSegmentOptions:
    fps: float = 25.0
    window_steps: int = 150
    anchor_stride_steps: int = 150
    min_ego_speed_mps: float = 5.0
    require_passenger_car_ego: bool = True
    require_ego_no_abnormal: bool = True
    require_valid_ego_lane_full_window: bool = True
    max_slot_longitudinal_distance_m: float = 150.0
    require_stable_background_slots: bool = True
    require_complete_lateral_events: bool = True
    lateral_pre_cross_steps: int = 25
    lateral_post_cross_steps: int = 50
    lateral_stable_steps: int = 5
    excluded_risk_slots: tuple[str, ...] = ()
    sei: SafetyEnvelopeRiskOptions = field(default_factory=SafetyEnvelopeRiskOptions)

    def __post_init__(self) -> None:
        if not self.require_stable_background_slots:
            raise ValueError(
                "highD natural-segment cleaning requires C0-active background "
                "slots to remain valid through the complete window; "
                "require_stable_background_slots cannot be disabled"
            )
        if not self.require_complete_lateral_events:
            raise ValueError(
                "highD natural-segment cleaning requires complete lateral "
                "events; require_complete_lateral_events cannot be disabled"
            )
        unknown = set(self.excluded_risk_slots) - set(SLOT_NAMES)
        if unknown:
            raise ValueError(f"unknown excluded risk slots: {sorted(unknown)}")

    @property
    def total_steps(self) -> int:
        return int(self.window_steps)


def options_from_config(config: dict[str, Any]) -> NaturalSegmentOptions:
    sampling = dict(config.get("sampling", {}))
    segment = dict(config.get("segments", {}))
    risk = dict(config.get("risk", {}))
    fps = float(sampling.get("target_fps", 25))
    window_seconds = float(segment.get("window_seconds", 6.0))
    window_steps = int(round(window_seconds * fps))
    anchor_stride_steps = int(
        round(float(segment.get("anchor_stride_seconds", window_seconds)) * fps)
    )
    return NaturalSegmentOptions(
        fps=fps,
        window_steps=max(window_steps, 1),
        anchor_stride_steps=max(anchor_stride_steps, 1),
        min_ego_speed_mps=float(segment.get("min_ego_speed_mps", 5.0)),
        require_passenger_car_ego=bool(segment.get("require_passenger_car_ego", True)),
        require_ego_no_abnormal=bool(segment.get("require_ego_no_abnormal", True)),
        require_valid_ego_lane_full_window=bool(
            segment.get("require_valid_ego_lane_full_window", True)
        ),
        max_slot_longitudinal_distance_m=float(
            segment.get("max_slot_longitudinal_distance_m", 150.0)
        ),
        require_stable_background_slots=bool(
            segment.get("require_stable_background_slots", True)
        ),
        require_complete_lateral_events=bool(
            segment.get("require_complete_lateral_events", True)
        ),
        lateral_pre_cross_steps=max(
            1,
            int(round(float(segment.get("lateral_pre_cross_seconds", 1.0)) * fps)),
        ),
        lateral_post_cross_steps=max(
            1,
            int(round(float(segment.get("lateral_post_cross_seconds", 2.0)) * fps)),
        ),
        lateral_stable_steps=max(1, int(segment.get("lateral_stable_steps", 5))),
        excluded_risk_slots=tuple(
            str(value) for value in segment.get("excluded_risk_slots", ())
        ),
        sei=SafetyEnvelopeRiskOptions.from_config(risk, fps=fps),
    )


def validate_lateral_integrity(
    rows: pd.DataFrame,
    *,
    required: bool,
    source: str,
) -> None:
    """Reject stale downstream data built before lateral-event cleaning."""
    if not required:
        return
    column = "lateral_event_complete"
    if column not in rows.columns:
        raise RuntimeError(
            f"{source} predates lateral-event integrity screening; rebuild "
            "natural segments and all downstream EVT/Flow/world-model data"
        )
    invalid = pd.to_numeric(rows[column], errors="coerce").fillna(0).ne(1)
    if bool(invalid.any()):
        raise RuntimeError(
            f"{source} contains {int(invalid.sum())} incomplete lateral events"
        )


def validate_stable_background_slots(
    rows: pd.DataFrame,
    *,
    required: bool,
    source: str,
) -> None:
    """Reject downstream data not cleaned for full-horizon slot stability."""
    if not required:
        return
    column = "background_slots_stable"
    if column not in rows.columns:
        raise RuntimeError(
            f"{source} predates background-slot stability screening; rebuild "
            "natural segments and all downstream EVT/Flow/world-model data"
        )
    invalid = pd.to_numeric(rows[column], errors="coerce").fillna(0).ne(1)
    if bool(invalid.any()):
        raise RuntimeError(
            f"{source} contains {int(invalid.sum())} unstable background-slot windows"
        )


def validate_natural_segment_contract(
    rows: pd.DataFrame,
    *,
    options: NaturalSegmentOptions,
    source: str,
) -> None:
    """Require every downstream consumer to use the one cleaned cohort."""
    validate_lateral_integrity(
        rows,
        required=options.require_complete_lateral_events,
        source=source,
    )
    validate_stable_background_slots(
        rows,
        required=options.require_stable_background_slots,
        source=source,
    )


def _is_passenger_car(meta_row: pd.Series) -> bool:
    return str(meta_row.get("class", "")).strip().lower() == "car"


def _lateral_sign(driving_direction: int) -> float:
    # highD image y increases downward.  Direction 1 drives toward negative x,
    # so physical left is increasing y; direction 2 physical left is decreasing y.
    return 1.0 if int(driving_direction) == 1 else -1.0


def _lane_left_coordinate(
    lane_id: int, lane_info: dict[str, Any], direction: int
) -> float:
    lane = lane_info.get("lanes", {}).get(int(lane_id))
    if lane is None:
        return float("nan")
    return _lateral_sign(direction) * float(lane["center"])


def _adjacent_lanes_ego_left(
    lane_id: int,
    lane_info: dict[str, Any],
    direction: int,
) -> tuple[int | None, int | None]:
    lanes_key = "direction_1_lanes" if int(direction) == 1 else "direction_2_lanes"
    lane_ids = [int(value) for value in lane_info.get(lanes_key, [])]
    if int(lane_id) not in lane_ids:
        return None, None

    current = _lane_left_coordinate(int(lane_id), lane_info, int(direction))
    if not np.isfinite(current):
        return None, None
    lane_coords = [
        (other_lane, _lane_left_coordinate(other_lane, lane_info, int(direction)))
        for other_lane in lane_ids
    ]
    lane_coords = [(lid, coord) for lid, coord in lane_coords if np.isfinite(coord)]
    left_candidates = [(lid, coord) for lid, coord in lane_coords if coord > current]
    right_candidates = [(lid, coord) for lid, coord in lane_coords if coord < current]
    left_lane = (
        min(left_candidates, key=lambda item: item[1] - current)[0]
        if left_candidates
        else None
    )
    right_lane = (
        min(right_candidates, key=lambda item: current - item[1])[0]
        if right_candidates
        else None
    )
    return left_lane, right_lane


def _build_vehicle_cache(recording: Any) -> dict[int, dict[str, Any]]:
    cache: dict[int, dict[str, Any]] = {}
    meta = recording.tracks_meta
    for vehicle_id in meta.index:
        vid = int(vehicle_id)
        track = recording.get_vehicle_track(vid)
        frames = np.asarray(track.index, dtype=np.int64)
        continuous = bool(frames.size <= 1 or np.all(np.diff(frames) == 1))
        frame_to_pos = None
        if not continuous:
            frame_to_pos = {int(frame): idx for idx, frame in enumerate(frames)}
        row = meta.loc[vid]
        direction = int(row.get("drivingDirection", 0))
        lat_sign = _lateral_sign(direction)
        zeros = np.zeros(frames.size, dtype=np.float32)
        cache[vid] = {
            "id": vid,
            "frames": frames,
            "initial": int(frames[0]) if frames.size else 0,
            "final": int(frames[-1]) if frames.size else -1,
            "continuous": continuous,
            "frame_to_pos": frame_to_pos,
            "x": track["x"].to_numpy(dtype=np.float32, copy=True),
            "y": track["y"].to_numpy(dtype=np.float32, copy=True),
            "y_left": (lat_sign * track["y"].to_numpy(dtype=np.float32, copy=True)),
            "vx": track["xVelocity"].to_numpy(dtype=np.float32, copy=True),
            "vy": track.get("yVelocity", pd.Series(zeros, index=track.index)).to_numpy(
                dtype=np.float32,
                copy=True,
            ),
            "vy_left": (
                lat_sign
                * track.get("yVelocity", pd.Series(zeros, index=track.index)).to_numpy(
                    dtype=np.float32,
                    copy=True,
                )
            ),
            "ax": track.get(
                "xAcceleration",
                pd.Series(zeros, index=track.index),
            ).to_numpy(dtype=np.float32, copy=True),
            "ay_left": (
                lat_sign
                * track.get(
                    "yAcceleration",
                    pd.Series(zeros, index=track.index),
                ).to_numpy(dtype=np.float32, copy=True)
            ),
            "lane": track["laneId"].to_numpy(dtype=np.int16, copy=True),
            "abnormal": track.get(
                "_abnormal",
                pd.Series(False, index=track.index),
            ).to_numpy(dtype=bool, copy=True),
            "length": float(row.get("width", np.nan)),
            "lat_width": float(row.get("height", np.nan)),
            "direction": direction,
            "class": str(row.get("class", "")),
        }
    return cache


def _build_frame_index(recording: Any) -> dict[int, np.ndarray]:
    index = recording.tracks.index
    frames = index.get_level_values("frame").to_numpy(dtype=np.int64)
    ids = index.get_level_values("id").to_numpy(dtype=np.int64)
    order = np.argsort(frames, kind="mergesort")
    frames = frames[order]
    ids = ids[order]
    unique_frames, starts = np.unique(frames, return_index=True)
    frame_index: dict[int, np.ndarray] = {}
    for idx, frame in enumerate(unique_frames):
        start = int(starts[idx])
        end = int(starts[idx + 1]) if idx + 1 < len(starts) else len(frames)
        frame_index[int(frame)] = ids[start:end].astype(np.int64, copy=True)
    return frame_index


def _position_at(vehicle: dict[str, Any], frame: int) -> int | None:
    frame_i = int(frame)
    if vehicle["continuous"]:
        if frame_i < vehicle["initial"] or frame_i > vehicle["final"]:
            return None
        return frame_i - int(vehicle["initial"])
    return vehicle["frame_to_pos"].get(frame_i)


def _has_full_range(vehicle: dict[str, Any], start_frame: int, steps: int) -> bool:
    if steps <= 0:
        return False
    end_frame = int(start_frame) + int(steps) - 1
    if vehicle["continuous"]:
        return int(start_frame) >= vehicle["initial"] and end_frame <= vehicle["final"]
    frame_to_pos = vehicle["frame_to_pos"]
    return all(
        int(frame) in frame_to_pos for frame in range(int(start_frame), end_frame + 1)
    )


def _slice_range(
    vehicle: dict[str, Any],
    start_frame: int,
    steps: int,
) -> tuple[Any, np.ndarray]:
    end_frame = int(start_frame) + int(steps) - 1
    if vehicle["continuous"]:
        offset = int(start_frame) - int(vehicle["initial"])
        if offset < 0 or end_frame > int(vehicle["final"]):
            valid = np.zeros(int(steps), dtype=bool)
            positions = np.full(int(steps), -1, dtype=np.int64)
            if end_frame < vehicle["initial"] or int(start_frame) > vehicle["final"]:
                return positions, valid
            valid_start_frame = max(int(start_frame), int(vehicle["initial"]))
            valid_end_frame = min(end_frame, int(vehicle["final"]))
            src = np.arange(
                valid_start_frame - int(vehicle["initial"]),
                valid_end_frame - int(vehicle["initial"]) + 1,
                dtype=np.int64,
            )
            dst_start = valid_start_frame - int(start_frame)
            positions[dst_start : dst_start + src.size] = src
            valid[dst_start : dst_start + src.size] = True
            return positions, valid
        return slice(offset, offset + int(steps)), np.ones(int(steps), dtype=bool)

    positions = np.full(int(steps), -1, dtype=np.int64)
    valid = np.zeros(int(steps), dtype=bool)
    frame_to_pos = vehicle["frame_to_pos"]
    for idx, frame in enumerate(range(int(start_frame), end_frame + 1)):
        pos = frame_to_pos.get(int(frame))
        if pos is not None:
            positions[idx] = int(pos)
            valid[idx] = True
    return positions, valid


def _values(
    vehicle: dict[str, Any],
    key: str,
    selector: Any,
    valid: np.ndarray,
    *,
    fill_value: float = np.nan,
) -> np.ndarray:
    arr = np.asarray(vehicle[key])
    if isinstance(selector, slice) and bool(np.all(valid)):
        return arr[selector].astype(np.float32, copy=False)
    out = np.full(valid.shape, fill_value, dtype=np.float32)
    if np.any(valid):
        out[valid] = arr[np.asarray(selector)[valid]]
    return out


def _select_slots_at_anchor(
    *,
    ego_id: int,
    anchor_frame: int,
    vehicles: dict[int, dict[str, Any]],
    frame_index: dict[int, np.ndarray],
    lane_info: dict[str, Any],
    options: NaturalSegmentOptions,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    ego = vehicles[int(ego_id)]
    ego_pos = _position_at(ego, int(anchor_frame))
    if ego_pos is None:
        return (
            np.full(len(SLOT_NAMES), -1, dtype=np.int64),
            np.zeros(len(SLOT_NAMES), dtype=bool),
            {
                "slot_anchor_missing": 1,
            },
        )

    ego_lane = int(ego["lane"][ego_pos])
    ego_direction = int(ego["direction"])
    left_lane, right_lane = _adjacent_lanes_ego_left(
        ego_lane,
        lane_info,
        ego_direction,
    )
    lane_to_side = {ego_lane: "same"}
    if left_lane is not None:
        lane_to_side[int(left_lane)] = "left"
    if right_lane is not None:
        lane_to_side[int(right_lane)] = "right"

    slot_ids = np.full(len(SLOT_NAMES), -1, dtype=np.int64)
    slot_dist = np.full(len(SLOT_NAMES), np.inf, dtype=np.float64)
    ego_x = float(ego["x"][ego_pos])
    counts: Counter[str] = Counter()
    for other_id in frame_index.get(int(anchor_frame), np.asarray([], dtype=np.int64)):
        oid = int(other_id)
        if oid == int(ego_id) or oid not in vehicles:
            continue
        other = vehicles[oid]
        if int(other["direction"]) != ego_direction:
            continue
        other_pos = _position_at(other, int(anchor_frame))
        if other_pos is None:
            continue
        if bool(other["abnormal"][other_pos]):
            counts["slot_candidate_abnormal_at_anchor"] += 1
            continue
        other_lane = int(other["lane"][other_pos])
        side = lane_to_side.get(other_lane)
        if side is None:
            continue
        dx = float(other["x"][other_pos]) - ego_x
        if not np.isfinite(dx) or abs(dx) > options.max_slot_longitudinal_distance_m:
            continue
        front_rear = "front" if dx >= 0.0 else "rear"
        slot_name = f"{side}_{front_rear}"
        slot_idx = SLOT_NAMES.index(slot_name)
        dist = abs(dx)
        if dist < slot_dist[slot_idx]:
            slot_ids[slot_idx] = oid
            slot_dist[slot_idx] = dist

    return slot_ids, slot_ids >= 0, dict(counts)


def _background_slots_stable(
    *,
    slot_ids: np.ndarray,
    start_frame: int,
    window_steps: int,
    vehicles: dict[int, dict[str, Any]],
) -> tuple[bool, str]:
    """Require every C0-active background slot through the full data window."""
    for vehicle_id in np.asarray(slot_ids, dtype=np.int64):
        if int(vehicle_id) < 0:
            continue
        vehicle = vehicles.get(int(vehicle_id))
        if vehicle is None:
            return False, "background_slot_not_found"
        selector, present = _slice_range(vehicle, start_frame, window_steps)
        if not bool(np.all(present)):
            return False, "background_slot_exits_window"
        if isinstance(selector, slice):
            abnormal = np.asarray(vehicle["abnormal"][selector], dtype=bool)
        else:
            abnormal = np.asarray(vehicle["abnormal"])[np.asarray(selector)]
        if bool(np.any(abnormal)):
            return False, "background_slot_abnormal_window"
    return True, "complete"


def compute_segment_risk(
    *,
    ego_id: int,
    anchor_frame: int,
    slot_ids: np.ndarray,
    vehicles: dict[int, dict[str, Any]],
    options: NaturalSegmentOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    window_steps = int(options.total_steps)
    ego = vehicles[int(ego_id)]
    ego_selector, ego_valid = _slice_range(ego, int(anchor_frame), window_steps)
    if not bool(np.all(ego_valid)):
        raise ValueError("ego must be present for the complete risk window")

    ego_ax = _values(ego, "ax", ego_selector, ego_valid)
    components = np.zeros((window_steps, len(RISK_COMPONENT_NAMES)), dtype=np.float32)

    ego_x = _values(ego, "x", ego_selector, ego_valid)
    ego_y_left = _values(ego, "y_left", ego_selector, ego_valid)
    ego_vx = _values(ego, "vx", ego_selector, ego_valid)
    ego_vy_left = _values(ego, "vy_left", ego_selector, ego_valid)
    ego_ay_left = _values(ego, "ay_left", ego_selector, ego_valid)
    pair_sei_scores: list[np.ndarray] = []
    slot_pair_matrix = np.zeros((window_steps, len(SLOT_NAMES)), dtype=np.float32)
    slot_time_mask = np.zeros((window_steps, len(SLOT_NAMES)), dtype=bool)
    for slot_idx, vehicle_id in enumerate(np.asarray(slot_ids, dtype=np.int64)):
        if SLOT_NAMES[slot_idx] in options.excluded_risk_slots:
            continue
        if int(vehicle_id) < 0:
            continue
        other = vehicles.get(int(vehicle_id))
        if other is None:
            continue
        other_selector, other_present = _slice_range(
            other,
            int(anchor_frame),
            window_steps,
        )
        if isinstance(other_selector, slice) and bool(np.all(other_present)):
            other_abnormal = other["abnormal"][other_selector]
        else:
            other_abnormal = np.zeros(window_steps, dtype=bool)
            if np.any(other_present):
                other_abnormal[other_present] = other["abnormal"][
                    np.asarray(other_selector)[other_present]
                ]
        valid = other_present & ~other_abnormal
        slot_time_mask[:, slot_idx] = valid
        other_x = _values(other, "x", other_selector, valid)
        other_y_left = _values(other, "y_left", other_selector, valid)
        other_vx = _values(other, "vx", other_selector, valid)
        other_vy_left = _values(other, "vy_left", other_selector, valid)
        other_ax = _values(other, "ax", other_selector, valid)
        other_ay_left = _values(other, "ay_left", other_selector, valid)
        pair_sei, pair_sei_components = pairwise_safety_envelope_intrusion(
            ego_x=ego_x,
            ego_y=ego_y_left,
            ego_vx=ego_vx,
            ego_vy=ego_vy_left,
            ego_ax=ego_ax,
            ego_ay=ego_ay_left,
            other_x=other_x,
            other_y=other_y_left,
            other_vx=other_vx,
            other_vy=other_vy_left,
            other_ax=other_ax,
            other_ay=other_ay_left,
            ego_length=float(ego["length"]),
            ego_width=float(ego["lat_width"]),
            other_length=float(other["length"]),
            other_width=float(other["lat_width"]),
            valid=valid,
            options=options.sei,
        )
        pair_sei_scores.append(pair_sei)
        slot_pair_matrix[:, slot_idx] = pair_sei
        components[:, : len(SEI_COMPONENT_NAMES)] = np.maximum(
            components[:, : len(SEI_COMPONENT_NAMES)],
            pair_sei_components,
        )

    if pair_sei_scores:
        pair_matrix = np.stack(pair_sei_scores, axis=1)
        instant_sei = smoothmax_axis(
            pair_matrix,
            float(options.sei.pair_smooth_beta),
            axis=1,
            empty_value=0.0,
        )
    else:
        instant_sei = np.zeros(window_steps, dtype=np.float32)
    components[:, RISK_COMPONENT_NAMES.index("sei_instant")] = instant_sei

    risk_trace = trajectory_safety_envelope_risk_trace(
        instant_sei,
        options=options.sei,
        dt_seconds=1.0 / float(options.fps),
    )

    peak_instant_offset = int(np.argmax(instant_sei)) if instant_sei.size else 0
    if slot_pair_matrix.size:
        peak_slot_idx = int(np.argmax(slot_pair_matrix[peak_instant_offset, :]))
        peak_pair_risk = float(slot_pair_matrix[peak_instant_offset, peak_slot_idx])
    else:
        peak_slot_idx = -1
        peak_pair_risk = 0.0
    if peak_pair_risk <= 0.0:
        peak_slot_idx = -1
    risk_info = {
        "peak_slot_name": (
            str(SLOT_NAMES[peak_slot_idx]) if peak_slot_idx >= 0 else "none"
        ),
        "peak_neighbor_id": (
            int(np.asarray(slot_ids, dtype=np.int64)[peak_slot_idx])
            if peak_slot_idx >= 0
            else -1
        ),
        "peak_pair_risk": peak_pair_risk,
        "peak_instant_risk": (
            float(instant_sei[peak_instant_offset]) if instant_sei.size else 0.0
        ),
        "peak_instant_frame": int(anchor_frame) + peak_instant_offset,
        "peak_instant_offset": peak_instant_offset,
    }
    return (
        risk_trace.astype(np.float32),
        components.astype(np.float32),
        slot_time_mask,
        risk_info,
    )


def _collect_untracked_candidate_ids(
    *,
    ego_id: int,
    anchor_frame: int,
    slot_ids: np.ndarray,
    vehicles: dict[int, dict[str, Any]],
    frame_index: dict[int, np.ndarray],
    lane_info: dict[str, Any],
    options: NaturalSegmentOptions,
) -> list[int]:
    ego = vehicles[int(ego_id)]
    slot_vehicle_ids = {
        int(vehicle_id)
        for vehicle_id in np.asarray(slot_ids, dtype=np.int64)
        if int(vehicle_id) >= 0
    }
    candidate_ids: set[int] = set()
    for frame in range(int(anchor_frame), int(anchor_frame) + int(options.total_steps)):
        ego_pos = _position_at(ego, frame)
        if ego_pos is None:
            continue
        ego_lane = int(ego["lane"][ego_pos])
        ego_direction = int(ego["direction"])
        left_lane, right_lane = _adjacent_lanes_ego_left(
            ego_lane,
            lane_info,
            ego_direction,
        )
        candidate_lanes = {ego_lane}
        if left_lane is not None:
            candidate_lanes.add(int(left_lane))
        if right_lane is not None:
            candidate_lanes.add(int(right_lane))
        ego_x = float(ego["x"][ego_pos])
        for other_id in frame_index.get(frame, np.asarray([], dtype=np.int64)):
            oid = int(other_id)
            if oid == int(ego_id) or oid in slot_vehicle_ids or oid not in vehicles:
                continue
            other = vehicles[oid]
            if int(other["direction"]) != ego_direction:
                continue
            other_pos = _position_at(other, frame)
            if other_pos is None or bool(other["abnormal"][other_pos]):
                continue
            if int(other["lane"][other_pos]) not in candidate_lanes:
                continue
            dx = float(other["x"][other_pos]) - ego_x
            if (
                not np.isfinite(dx)
                or abs(dx) > options.max_slot_longitudinal_distance_m
            ):
                continue
            candidate_ids.add(oid)
    return sorted(candidate_ids)


def _vehicle_between(
    *,
    ego_id: int,
    target_id: int,
    frame: int,
    lane_id: int,
    vehicles: dict[int, dict[str, Any]],
    frame_index: dict[int, np.ndarray],
) -> bool:
    ego_pos = _position_at(vehicles[ego_id], frame)
    target_pos = _position_at(vehicles[target_id], frame)
    if ego_pos is None or target_pos is None:
        return True
    ego_x = float(vehicles[ego_id]["x"][ego_pos])
    target_x = float(vehicles[target_id]["x"][target_pos])
    if target_x <= ego_x:
        return True
    for vehicle_id in frame_index.get(frame, np.asarray([], dtype=np.int64)):
        other_id = int(vehicle_id)
        if other_id in {ego_id, target_id}:
            continue
        other = vehicles.get(other_id)
        if other is None:
            continue
        position = _position_at(other, frame)
        if position is None or bool(other["abnormal"][position]):
            continue
        if int(other["lane"][position]) != int(lane_id):
            continue
        if ego_x < float(other["x"][position]) < target_x:
            return True
    return False


def _strict_cutin_at_crossing(
    *,
    ego_id: int,
    target_id: int,
    crossing_frame: int,
    target_lane: int,
    vehicles: dict[int, dict[str, Any]],
    frame_index: dict[int, np.ndarray],
    post_steps: int,
) -> bool:
    ego = vehicles[ego_id]
    target = vehicles[target_id]
    if (
        str(ego["class"]).strip().lower() != "car"
        or str(target["class"]).strip().lower() != "car"
    ):
        return False
    same_lane = 0
    common = 0
    for frame in range(crossing_frame, crossing_frame + post_steps):
        ego_pos = _position_at(ego, frame)
        target_pos = _position_at(target, frame)
        if ego_pos is None or target_pos is None:
            return False
        if bool(ego["abnormal"][ego_pos]) or bool(target["abnormal"][target_pos]):
            return False
        if float(target["x"][target_pos]) <= float(ego["x"][ego_pos]):
            return False
        common += 1
        if int(ego["lane"][ego_pos]) == int(target_lane) and int(
            target["lane"][target_pos]
        ) == int(target_lane):
            same_lane += 1
        if _vehicle_between(
            ego_id=ego_id,
            target_id=target_id,
            frame=frame,
            lane_id=target_lane,
            vehicles=vehicles,
            frame_index=frame_index,
        ):
            return False
    return common >= post_steps and same_lane / max(common, 1) >= 0.7


def _audit_lateral_events(
    *,
    ego_id: int,
    vehicle_ids: list[int],
    start_frame: int,
    options: NaturalSegmentOptions,
    vehicles: dict[int, dict[str, Any]],
    frame_index: dict[int, np.ndarray],
    lane_info: dict[str, Any],
) -> dict[str, Any]:
    """Require every modeled lateral event touching a window to be complete."""
    end_frame = start_frame + int(options.total_steps) - 1
    events: list[tuple[int, int, int, int]] = []
    rejection = ""
    for vehicle_id in sorted(set(vehicle_ids)):
        vehicle = vehicles.get(int(vehicle_id))
        if vehicle is None:
            continue
        selector, present = _slice_range(vehicle, start_frame, int(options.total_steps))
        lane = _values(vehicle, "lane", selector, present, fill_value=-1).astype(
            np.int16
        )
        stable = int(options.lateral_stable_steps)
        changes = np.flatnonzero(present[1:] & present[:-1] & (lane[1:] != lane[:-1]))
        for index in changes:
            crossing_offset = int(index) + 1
            crossing_frame = start_frame + crossing_offset
            source_lane = int(lane[index])
            target_lane = int(lane[index + 1])
            if not are_adjacent_lanes(source_lane, target_lane, lane_info):
                rejection = "non_adjacent_lane_change"
                break
            if crossing_offset < int(options.lateral_pre_cross_steps):
                rejection = "insufficient_pre_cross_context"
                break
            if end_frame - crossing_frame + 1 < int(options.lateral_post_cross_steps):
                rejection = "insufficient_post_cross_context"
                break
            before = lane[crossing_offset - stable : crossing_offset]
            after = lane[crossing_offset : crossing_offset + stable]
            if not (np.all(before == source_lane) and np.all(after == target_lane)):
                rejection = "unstable_lane_transition"
                break
            events.append((int(vehicle_id), crossing_frame, source_lane, target_lane))
        if rejection:
            break

    strict_cutins = [
        event
        for event in events
        if event[0] != ego_id
        and _strict_cutin_at_crossing(
            ego_id=ego_id,
            target_id=event[0],
            crossing_frame=event[1],
            target_lane=event[3],
            vehicles=vehicles,
            frame_index=frame_index,
            post_steps=int(options.lateral_post_cross_steps),
        )
    ]
    primary = strict_cutins[0] if strict_cutins else (-1, -1, -1, -1)
    return {
        "complete": not rejection,
        "reject_reason": rejection,
        "num_lane_changes": len(events),
        "num_strict_cutins": len(strict_cutins),
        "primary_cutin_target_id": primary[0],
        "primary_cutin_cross_frame": primary[1],
    }


def compute_untracked_audit(
    *,
    ego_id: int,
    anchor_frame: int,
    slot_ids: np.ndarray,
    vehicles: dict[int, dict[str, Any]],
    frame_index: dict[int, np.ndarray],
    lane_info: dict[str, Any],
    options: NaturalSegmentOptions,
    tracked_peak_pair_risk: float,
) -> dict[str, Any]:
    window_steps = int(options.total_steps)
    ego = vehicles[int(ego_id)]
    ego_selector, ego_valid = _slice_range(ego, int(anchor_frame), window_steps)
    if not bool(np.all(ego_valid)):
        raise ValueError("ego must be present for the complete untracked audit window")

    candidate_ids = _collect_untracked_candidate_ids(
        ego_id=int(ego_id),
        anchor_frame=int(anchor_frame),
        slot_ids=slot_ids,
        vehicles=vehicles,
        frame_index=frame_index,
        lane_info=lane_info,
        options=options,
    )
    if not candidate_ids:
        return {
            "num_untracked_candidates": 0,
            "max_untracked_pair_risk": 0.0,
            "max_untracked_neighbor_id": -1,
            "max_untracked_risk_frame": -1,
            "untracked_risk_exceeds_tracked_peak": 0,
        }

    ego_x = _values(ego, "x", ego_selector, ego_valid)
    ego_y_left = _values(ego, "y_left", ego_selector, ego_valid)
    ego_vx = _values(ego, "vx", ego_selector, ego_valid)
    ego_vy_left = _values(ego, "vy_left", ego_selector, ego_valid)
    ego_ax = _values(ego, "ax", ego_selector, ego_valid)
    ego_ay_left = _values(ego, "ay_left", ego_selector, ego_valid)

    best_risk = 0.0
    best_id = -1
    best_offset = -1
    for other_id in candidate_ids:
        other = vehicles.get(int(other_id))
        if other is None:
            continue
        if not (np.isfinite(other["length"]) and np.isfinite(other["lat_width"])):
            continue
        other_selector, other_present = _slice_range(
            other,
            int(anchor_frame),
            window_steps,
        )
        if isinstance(other_selector, slice) and bool(np.all(other_present)):
            other_abnormal = other["abnormal"][other_selector]
        else:
            other_abnormal = np.zeros(window_steps, dtype=bool)
            if np.any(other_present):
                other_abnormal[other_present] = other["abnormal"][
                    np.asarray(other_selector)[other_present]
                ]
        valid = other_present & ~other_abnormal
        if not np.any(valid):
            continue
        pair_sei, _components = pairwise_safety_envelope_intrusion(
            ego_x=ego_x,
            ego_y=ego_y_left,
            ego_vx=ego_vx,
            ego_vy=ego_vy_left,
            ego_ax=ego_ax,
            ego_ay=ego_ay_left,
            other_x=_values(other, "x", other_selector, valid),
            other_y=_values(other, "y_left", other_selector, valid),
            other_vx=_values(other, "vx", other_selector, valid),
            other_vy=_values(other, "vy_left", other_selector, valid),
            other_ax=_values(other, "ax", other_selector, valid),
            other_ay=_values(other, "ay_left", other_selector, valid),
            ego_length=float(ego["length"]),
            ego_width=float(ego["lat_width"]),
            other_length=float(other["length"]),
            other_width=float(other["lat_width"]),
            valid=valid,
            options=options.sei,
        )
        offset = int(np.argmax(pair_sei))
        risk = float(pair_sei[offset])
        if risk > best_risk:
            best_risk = risk
            best_id = int(other_id)
            best_offset = offset

    return {
        "num_untracked_candidates": int(len(candidate_ids)),
        "max_untracked_pair_risk": best_risk,
        "max_untracked_neighbor_id": best_id,
        "max_untracked_risk_frame": (
            int(anchor_frame) + best_offset if best_offset >= 0 else -1
        ),
        "untracked_risk_exceeds_tracked_peak": int(
            best_risk > float(tracked_peak_pair_risk) + 1.0e-9
        ),
    }


def _valid_lanes_for_direction(
    lane_values: np.ndarray,
    lane_info: dict[str, Any],
    direction: int,
) -> bool:
    lanes_key = "direction_1_lanes" if int(direction) == 1 else "direction_2_lanes"
    allowed = {int(value) for value in lane_info.get(lanes_key, [])}
    if not allowed:
        return False
    return all(int(value) in allowed for value in np.asarray(lane_values).ravel())


def build_natural_segments_for_recording(
    recording: Any,
    options: NaturalSegmentOptions,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build fixed-length natural segment metadata and risk traces."""
    lane_info = parse_lane_markings(recording.recording_meta)
    vehicles = _build_vehicle_cache(recording)
    frame_index = _build_frame_index(recording)

    rows: list[dict[str, Any]] = []
    risk_traces: list[np.ndarray] = []
    slot_masks: list[np.ndarray] = []
    reject: Counter[str] = Counter()

    for ego_id in recording.tracks_meta.index:
        ego_id = int(ego_id)
        ego_meta = recording.tracks_meta.loc[ego_id]
        if options.require_passenger_car_ego and not _is_passenger_car(ego_meta):
            reject["ego_not_passenger_car"] += 1
            continue
        ego = vehicles[ego_id]
        if not ego["continuous"]:
            reject["ego_discontinuous_track"] += 1
            continue
        if not (np.isfinite(ego["length"]) and np.isfinite(ego["lat_width"])):
            reject["ego_bad_dimensions"] += 1
            continue

        first_anchor = int(ego["initial"])
        last_anchor = int(ego["final"]) - int(options.total_steps) + 1
        if last_anchor < first_anchor:
            reject["ego_track_too_short_for_window"] += 1
            continue

        for anchor_frame in range(
            first_anchor,
            last_anchor + 1,
            int(options.anchor_stride_steps),
        ):
            start_frame = int(anchor_frame)
            end_frame = start_frame + int(options.total_steps) - 1
            total_selector, total_valid = _slice_range(
                ego,
                start_frame,
                int(options.total_steps),
            )
            if not bool(np.all(total_valid)):
                reject["ego_missing_total_window"] += 1
                continue
            if isinstance(total_selector, slice):
                total_abnormal = ego["abnormal"][total_selector]
                total_lane = ego["lane"][total_selector]
                anchor_pos = int(start_frame) - int(ego["initial"])
            else:
                positions = np.asarray(total_selector)
                total_abnormal = ego["abnormal"][positions]
                total_lane = ego["lane"][positions]
                anchor_pos = positions[0]

            if options.require_ego_no_abnormal and bool(np.any(total_abnormal)):
                reject["ego_abnormal_total_window"] += 1
                continue
            if (
                options.require_valid_ego_lane_full_window
                and not _valid_lanes_for_direction(
                    total_lane,
                    lane_info,
                    int(ego["direction"]),
                )
            ):
                reject["ego_invalid_lane_total_window"] += 1
                continue

            ego_speed = float(
                np.hypot(float(ego["vx"][anchor_pos]), float(ego["vy"][anchor_pos]))
            )
            if ego_speed < float(options.min_ego_speed_mps):
                reject["ego_speed_below_minimum"] += 1
                continue

            slot_ids, slot_present, slot_counts = _select_slots_at_anchor(
                ego_id=ego_id,
                anchor_frame=int(anchor_frame),
                vehicles=vehicles,
                frame_index=frame_index,
                lane_info=lane_info,
                options=options,
            )
            reject.update(slot_counts)
            slots_stable, slot_reject_reason = _background_slots_stable(
                slot_ids=slot_ids,
                start_frame=start_frame,
                window_steps=int(options.total_steps),
                vehicles=vehicles,
            )
            if options.require_stable_background_slots and not slots_stable:
                reject[slot_reject_reason] += 1
                continue
            untracked_ids = _collect_untracked_candidate_ids(
                ego_id=ego_id,
                anchor_frame=int(anchor_frame),
                slot_ids=slot_ids,
                vehicles=vehicles,
                frame_index=frame_index,
                lane_info=lane_info,
                options=options,
            )
            interaction_ids = [
                ego_id,
                *[int(vehicle_id) for vehicle_id in slot_ids if int(vehicle_id) >= 0],
                *untracked_ids,
            ]
            lateral_audit = _audit_lateral_events(
                ego_id=ego_id,
                vehicle_ids=interaction_ids,
                start_frame=start_frame,
                options=options,
                vehicles=vehicles,
                frame_index=frame_index,
                lane_info=lane_info,
            )
            if options.require_complete_lateral_events and not bool(
                lateral_audit["complete"]
            ):
                reject[f"incomplete_{lateral_audit['reject_reason']}"] += 1
                continue
            risk_trace, components, slot_time_mask, risk_info = compute_segment_risk(
                ego_id=ego_id,
                anchor_frame=int(anchor_frame),
                slot_ids=slot_ids,
                vehicles=vehicles,
                options=options,
            )
            untracked_audit = compute_untracked_audit(
                ego_id=ego_id,
                anchor_frame=int(anchor_frame),
                slot_ids=slot_ids,
                vehicles=vehicles,
                frame_index=frame_index,
                lane_info=lane_info,
                options=options,
                tracked_peak_pair_risk=float(risk_info["peak_pair_risk"]),
            )
            event_risk = float(np.max(risk_trace))
            peak_offset = int(np.argmax(risk_trace))
            max_components = np.max(components, axis=0)

            segment_index = len(rows)
            segment_id = (
                f"nat_{int(recording.recording_id):02d}_{ego_id:06d}_"
                f"{int(anchor_frame):06d}"
            )
            row: dict[str, Any] = {
                "segment_id": segment_id,
                "recording_id": int(recording.recording_id),
                "ego_id": ego_id,
                "ego_driving_direction": int(ego["direction"]),
                "window_start_frame": start_frame,
                "window_end_frame": end_frame,
                "anchor_frame": int(anchor_frame),
                "ego_anchor_lane_id": int(ego["lane"][anchor_pos]),
                "ego_anchor_speed_mps": ego_speed,
                "num_slots_present_at_anchor": int(np.sum(slot_present)),
                "background_slots_stable": int(slots_stable),
                "lateral_event_complete": int(lateral_audit["complete"]),
                "lateral_event_reject_reason": str(lateral_audit["reject_reason"]),
                "num_lane_changes": int(lateral_audit["num_lane_changes"]),
                "num_strict_cutins": int(lateral_audit["num_strict_cutins"]),
                "primary_cutin_target_id": int(
                    lateral_audit["primary_cutin_target_id"]
                ),
                "primary_cutin_cross_frame": int(
                    lateral_audit["primary_cutin_cross_frame"]
                ),
                "event_risk": event_risk,
                "peak_risk_frame": int(anchor_frame) + peak_offset,
                "peak_risk_offset": peak_offset,
                "peak_slot_name": str(risk_info["peak_slot_name"]),
                "peak_neighbor_id": int(risk_info["peak_neighbor_id"]),
                "peak_pair_risk": float(risk_info["peak_pair_risk"]),
                "peak_instant_risk": float(risk_info["peak_instant_risk"]),
                "peak_instant_frame": int(risk_info["peak_instant_frame"]),
                "peak_instant_offset": int(risk_info["peak_instant_offset"]),
                "num_untracked_candidates": int(
                    untracked_audit["num_untracked_candidates"]
                ),
                "max_untracked_pair_risk": float(
                    untracked_audit["max_untracked_pair_risk"]
                ),
                "max_untracked_neighbor_id": int(
                    untracked_audit["max_untracked_neighbor_id"]
                ),
                "max_untracked_risk_frame": int(
                    untracked_audit["max_untracked_risk_frame"]
                ),
                "untracked_risk_exceeds_tracked_peak": int(
                    untracked_audit["untracked_risk_exceeds_tracked_peak"]
                ),
                "risk_trace_row": segment_index,
            }
            for slot_idx, slot_name in enumerate(SLOT_NAMES):
                row[f"{slot_name}_id"] = int(slot_ids[slot_idx])
                presence_fraction = float(np.mean(slot_time_mask[:, slot_idx]))
                row[f"{slot_name}_window_presence_fraction"] = presence_fraction
            for comp_idx, comp_name in enumerate(RISK_COMPONENT_NAMES):
                row[f"max_{comp_name}"] = float(max_components[comp_idx])

            rows.append(row)
            risk_traces.append(risk_trace)
            slot_masks.append(slot_time_mask)

    if risk_traces:
        risk_array = np.stack(risk_traces, axis=0).astype(np.float32)
        slot_mask_array = np.stack(slot_masks, axis=0)
    else:
        risk_array = np.zeros((0, int(options.total_steps)), dtype=np.float32)
        slot_mask_array = np.zeros(
            (0, int(options.total_steps), len(SLOT_NAMES)),
            dtype=bool,
        )
    summary = {
        "recording_id": int(recording.recording_id),
        "num_segments": int(len(rows)),
        "reject_counts": {key: int(value) for key, value in sorted(reject.items())},
    }
    return pd.DataFrame(rows), risk_array, slot_mask_array, summary
