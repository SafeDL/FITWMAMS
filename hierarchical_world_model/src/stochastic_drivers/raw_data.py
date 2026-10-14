"""Physical driver observations from complete, screened original highD recordings."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import uniform_filter1d

from process_highD.src.io_utils import load_config, resolve_data_path
from process_highD.src.lane_utils import parse_lane_markings
from process_highD.src.preprocess import prepare_recording

from .idm_population import idm_acceleration
from .driver_policy import physical_leader_observation
from ..npc_experiments.lateral import (
    EVENT_FIELDS,
    lane_events,
    lateral_protocol,
    start_neighbors,
)

ROOT = Path(__file__).resolve().parents[3]
DRIVER_DATA_DIR = ROOT / "results/highd_shared_training_data/driver_behavior"
SPLITS = ("train", "val", "test")
BEHAVIOR_PROTOCOL = "causal_maneuver_and_swept_body_leaders"
LABEL_PROTOCOL = {
    "choice": "keep / lower-y adjacent change / higher-y adjacent change",
    "onset_horizon_seconds": 1.0,
    "confirmation_extra_frames": 4,
    "onset": "five directed vy>0.15 m/s frames and displacement>0.1 m",
    "motion_episode": "uniform_filter1d(abs(vy),5)>0.15 on contiguous non-abnormal observations",
    "episode_cross_association_frames": 10,
    "cross_confirmation_horizon_seconds": 10.0,
    "confirmation": "first adjacent original laneId crossing after anchor, same canonical direction and same motion episode",
    "non_crossing_episode": "observed episode closure plus10 post frames is keep; a later unrelated crossing cannot confirm the cue",
    "opposite_first_crossing": "keep; later return cannot confirm original cue",
    "censoring": "unfinished/truncated episode, gap before evidence, nonadjacent or unmapped first transition, ambiguous onset",
    "future_use": "labels only; NPC inputs contain current state and observed history",
}
LABEL_AUDIT_NAMES = (
    "onset_cues",
    "cross_confirmed",
    "no_cross",
    "truncated_future",
    "observation_gap",
    "opposite_cross",
    "multiple_crosses",
    "nonadjacent_transition",
    "ambiguous_onset",
    "onset_unobserved",
    "horizon_unresolved",
)
EVENT_COLUMNS = (
    "event_id",
    "split",
    "recording_id",
    "vehicle_id",
    "cross_frame",
    "start_frame",
    "end_frame",
    "source_y_m",
    "target_y_m",
    "direction",
    "speed_mps",
    "peak_lateral_speed_mps",
    "duration_s",
    "length_m",
    "width_m",
    "trajectory_index",
    "class",
    "driving_direction",
)


def recording_splits():
    report = json.loads(
        (ROOT / "results/hierarchical_world_model/model_assets/mobil.json").read_text()
    )
    return report["split_recordings"]


def canonical_lateral(recording, geometry):
    """Use the same direction-dependent lateral coordinates as the world model."""
    flipped = recording.tracks.index.get_level_values("id").isin(
        recording.tracks_meta.index[recording.tracks_meta.drivingDirection == 2]
    )
    recording.tracks.loc[flipped, ["y", "yVelocity", "yAcceleration"]] *= -1
    for lane in geometry["direction_2_lanes"]:
        geometry["lanes"][lane]["center"] *= -1
    recording._vehicle_cache.clear()
    recording._frame_cache.clear()


def motion_episodes(track):
    """Reuse the complete-maneuver motion threshold without bridging observation gaps."""
    frames = track.index.to_numpy()
    valid = ~track._abnormal.to_numpy()
    breaks = np.flatnonzero((np.diff(frames) != 1) | ~valid[:-1] | ~valid[1:]) + 1
    episode = np.full(len(track), -1, np.int32)
    starts, stops, part_ends = [], [], []
    for indices in np.split(np.arange(len(track)), breaks):
        if not valid[indices].all():
            continue
        moving = (
            uniform_filter1d(
                np.abs(track.yVelocity.to_numpy()[indices]), size=5, mode="constant"
            )
            > 0.15
        )
        edges = np.diff(np.r_[False, moving, False].astype(np.int8))
        for start, stop in zip(
            np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
        ):
            episode[indices[start : stop + 1]] = len(starts)
            starts.append(indices[start])
            stops.append(indices[stop])
            part_ends.append(indices[-1])
    return episode, np.asarray(starts), np.asarray(stops), np.asarray(part_ends)


def lane_labels(track, lane_centers):
    """One-second starts are confirmed by an adjacent crossing in the same motion episode."""
    count = len(track)
    label = np.zeros(count, dtype=np.uint8)
    observed = np.zeros(count, dtype=bool)
    audit = {name: np.zeros(count, bool) for name in LABEL_AUDIT_NAMES}
    audit["onset_unobserved"][:] = True
    if count < 30:
        return label, observed, audit
    y = track.y.to_numpy()
    vy = track.yVelocity.to_numpy()
    abnormal = track._abnormal.to_numpy()
    frames = track.index.to_numpy()
    future_y = np.lib.stride_tricks.sliding_window_view(y[1:], 29)
    future_vy = np.lib.stride_tricks.sliding_window_view(vy[1:], 29)
    future_abnormal = np.lib.stride_tricks.sliding_window_view(abnormal[1:], 29)
    future_frames = np.lib.stride_tricks.sliding_window_view(frames[1:], 29)
    moving = future_vy[..., None] * np.array([-1, 1]) > 0.15
    sustained = np.lib.stride_tricks.sliding_window_view(moving, 5, axis=1).all(-1)
    displaced = (future_y[:, 4:, None] - future_y[:, :1, None]) * np.array(
        [-1, 1]
    ) > 0.1
    signals = sustained & displaced
    sides = signals.any(1)
    valid = ~future_abnormal.any(1) & (np.diff(future_frames, axis=1) == 1).all(1)
    valid &= (future_frames[:, 0] == frames[: len(valid)] + 1) & ~abnormal[: len(valid)]
    ambiguous = sides.all(1)
    observed[: len(valid)] = valid & ~ambiguous
    audit["onset_unobserved"][: len(valid)] = ~valid
    audit["ambiguous_onset"][: len(valid)] = valid & ambiguous
    cues = np.flatnonzero(valid & sides.any(1) & ~ambiguous)
    audit["onset_cues"][cues] = True
    if not len(cues):
        return label, observed, audit
    side = np.where(sides[cues, 0], -1, 1)
    cue_start = cues + 1 + signals[cues, :, (side > 0).astype(int)].argmax(1)
    episode, starts, stops, part_ends = motion_episodes(track)
    episode_id = episode[cue_start + 2]
    episode_start, episode_stop, part_end = (
        value[episode_id] for value in (starts, stops, part_ends)
    )
    lanes = track.laneId.to_numpy()
    crosses = np.flatnonzero(np.diff(lanes) != 0) + 1
    first_position = np.searchsorted(crosses, cues, side="right")
    first = np.r_[crosses, count][first_position]
    horizon = frames[cues] + 250
    within_horizon = (first < count) & (frames[first.clip(max=count - 1)] <= horizon)
    source = lanes[cues]
    target = lanes[first.clip(max=count - 1)]
    mapped = np.isin(source, list(lane_centers)) & np.isin(target, list(lane_centers))
    adjacent = mapped & (np.abs(target - source) == 1)
    cross_direction = np.array(
        [
            np.sign(
                lane_centers.get(int(to), np.nan) - lane_centers.get(int(frm), np.nan)
            )
            for frm, to in zip(source, target)
        ]
    )
    invalid = np.unique(
        np.r_[np.flatnonzero(np.diff(frames) != 1) + 1, np.flatnonzero(abnormal)]
    )
    next_invalid = np.r_[invalid, count][np.searchsorted(invalid, cues, side="right")]
    clear_cross = within_horizon & (first < next_invalid)
    same_episode = (first >= episode_start - 10) & (first <= episode_stop + 10)
    confirmed = clear_cross & adjacent & same_episode & (cross_direction == side)
    opposite = clear_cross & adjacent & (cross_direction != side)
    nonadjacent = clear_cross & ~adjacent
    # Completed centering episodes are keep even if a later real change has the same side.
    closed = (episode_stop + 10 <= part_end) & (
        frames[(episode_stop + 10).clip(max=count - 1)] <= horizon
    )
    closed &= episode_stop + 10 < next_invalid
    keep = ~confirmed & ~nonadjacent & (opposite | closed)
    unresolved = ~confirmed & ~keep
    invalid_frame = frames[next_invalid.clip(max=count - 1)].copy()
    gap_before_row = (next_invalid > 0) & (next_invalid < count)
    gap_before_row &= invalid_frame - frames[(next_invalid - 1).clip(min=0)] > 1
    invalid_frame[gap_before_row] = frames[next_invalid[gap_before_row] - 1] + 1
    gap = (
        unresolved & ~nonadjacent & (next_invalid < count) & (invalid_frame <= horizon)
    )
    truncated = (frames[-1] < horizon) | (episode_stop + 10 > part_end)
    audit["cross_confirmed"][cues[confirmed]] = True
    audit["opposite_cross"][cues[opposite]] = True
    audit["nonadjacent_transition"][cues[nonadjacent]] = True
    audit["no_cross"][cues[keep & ~opposite]] = True
    audit["observation_gap"][cues[gap]] = True
    audit["truncated_future"][cues[unresolved & ~gap & ~nonadjacent & truncated]] = True
    audit["horizon_unresolved"][
        cues[unresolved & ~gap & ~nonadjacent & ~truncated]
    ] = True
    observed[cues] = confirmed | keep
    label[cues[confirmed]] = np.where(side[confirmed] < 0, 1, 2)
    visible_end = np.minimum(horizon, frames[(next_invalid - 1).clip(min=0)])
    remaining = (
        np.searchsorted(frames[crosses], visible_end, side="right") - first_position
    )
    audit["multiple_crosses"][cues[remaining > 1]] = True
    return label, observed, audit


def extract_complete_events(recording, split, geometry):
    """Retain contiguous 2–10 s adjacent changes with observed pre/post settling."""
    events, trajectories, rejection = [], [], Counter()
    for vehicle in recording.vehicle_ids():
        track = recording.get_vehicle_track(vehicle)
        meta = recording.tracks_meta.loc[vehicle]
        direction = int(meta.drivingDirection)
        centers = np.sort(
            [
                geometry["lanes"][lane]["center"]
                for lane in geometry[f"direction_{direction}_lanes"]
            ]
        )
        frame = track.index.to_numpy()
        columns = ["y", "xVelocity", "yVelocity", "yAcceleration"]
        values = track[columns].to_numpy()
        valid = ~track._abnormal.to_numpy() & np.isfinite(values).all(1)
        breaks = np.flatnonzero((np.diff(frame) != 1) | ~valid[:-1] | ~valid[1:]) + 1
        for indices in np.split(np.arange(len(track)), breaks):
            if len(indices) < 75 or not valid[indices].all():
                continue
            data, frames = values[indices], frame[indices]
            lane_y = centers[np.abs(data[:, :1] - centers).argmin(1)]
            active = (
                uniform_filter1d(np.abs(data[:, 2]), size=5, mode="constant") > 0.15
            )
            edges = np.diff(np.r_[False, active, False].astype(np.int8))
            starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
            crossings = np.flatnonzero(np.abs(np.diff(lane_y)) > 1.8) + 1
            for start, stop in zip(starts, stops):
                nearby = crossings[(crossings >= start - 10) & (crossings <= stop + 10)]
                if len(nearby) != 1:
                    rejection["no_single_adjacent_crossing"] += 1
                    continue
                cross = int(nearby[0])
                duration = (stop - start + 1) * 0.04
                peak = float(np.abs(data[start : stop + 1, 2]).max())
                if start < 25 or stop + 25 >= len(data):
                    rejection["insufficient_observed_pre_or_post_history"] += 1
                    continue
                source, target = lane_y[cross - 1], lane_y[cross]
                if not (
                    2 <= duration <= 10
                    and 0.4 <= peak <= 3
                    and 1.8 < abs(target - source) < 5.4
                ):
                    rejection["duration_peak_or_lane_width_outside_scope"] += 1
                    continue
                if (
                    abs(data[start, 0] - source) > 0.7
                    or abs(data[stop, 0] - target) > 0.7
                ):
                    rejection["motion_does_not_start_and_finish_at_lane_centres"] += 1
                    continue
                trajectory = data[start - 25 : stop + 26].copy()
                trajectory[:, 0] -= source
                trajectories.append(trajectory.astype(np.float32))
                events.append(
                    {
                        "event_id": f"lane_{recording.recording_id:02d}_{vehicle}_{frames[cross]}",
                        "split": split,
                        "recording_id": recording.recording_id,
                        "vehicle_id": int(vehicle),
                        "cross_frame": int(frames[cross]),
                        "start_frame": int(frames[start]),
                        "end_frame": int(frames[stop]),
                        "source_y_m": float(source),
                        "target_y_m": float(target),
                        "direction": int(np.sign(target - source)),
                        "speed_mps": float(np.hypot(data[start, 1], data[start, 2])),
                        "peak_lateral_speed_mps": peak,
                        "duration_s": duration,
                        "length_m": float(meta.width),
                        "width_m": float(meta.height),
                        "trajectory_index": len(trajectories) - 1,
                        "class": 0 if meta["class"] == "Car" else 1,
                        "driving_direction": direction,
                    }
                )
    states = (
        np.concatenate(trajectories) if trajectories else np.empty((0, 4), np.float32)
    )
    offsets = np.r_[0, np.cumsum([len(item) for item in trajectories])]
    return pd.DataFrame(events, columns=EVENT_COLUMNS), states, offsets, dict(rejection)


def nearest_neighbors(states, centers_by_direction):
    """Physical same/adjacent-lane neighbors from every observed vehicle at a frame."""
    count = len(states)
    current_front = np.full(count, -1, np.int32)
    current_rear = np.full(count, -1, np.int32)
    fronts = np.full((count, 2), -1, np.int32)
    rears = np.full((count, 2), -1, np.int32)
    available = np.zeros((count, 2), bool)
    order = np.lexsort((states.x, states.lane, states.frame, states.direction))
    sorted_frame, sorted_dir, sorted_lane = (
        states[name].to_numpy()[order] for name in ("frame", "direction", "lane")
    )
    boundaries = np.r_[
        0,
        np.flatnonzero(
            (np.diff(sorted_frame) != 0)
            | (np.diff(sorted_dir) != 0)
            | (np.diff(sorted_lane) != 0)
        )
        + 1,
        count,
    ]
    groups = {}
    x = states.x.to_numpy()
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        indices = order[begin:end]
        current_front[indices[:-1]] = indices[1:]
        current_rear[indices[1:]] = indices[:-1]
        groups[
            (int(sorted_dir[begin]), int(sorted_frame[begin]), int(sorted_lane[begin]))
        ] = indices
    for (direction, frame, lane), indices in groups.items():
        for side, target in enumerate((lane - 1, lane + 1)):
            if not 0 <= target < len(centers_by_direction[direction]):
                continue
            available[indices, side] = True
            other = groups.get((direction, frame, target))
            if other is None:
                continue
            rank = np.searchsorted(x[other], x[indices], side="right")
            has_front = rank < len(other)
            has_rear = rank > 0
            fronts[indices[has_front], side] = other[rank[has_front]]
            rears[indices[has_rear], side] = other[rank[has_rear] - 1]
    return current_front, current_rear, fronts, rears, available


def swept_leaders(states):
    """Batch the exact shared body-corridor observation over physical frames."""
    order = np.lexsort((states.frame, states.direction))
    direction, frame = (
        states[name].to_numpy()[order] for name in ("direction", "frame")
    )
    boundaries = np.r_[
        0,
        np.flatnonzero((np.diff(direction) != 0) | (np.diff(frame) != 0)) + 1,
        len(order),
    ]
    leader = np.full(len(states), -1, np.int32)
    gap = np.full(len(states), 1e6, np.float32)
    front_speed = np.hypot(states.xVelocity, states.yVelocity).to_numpy(np.float32)
    motion = states[
        ["x", "y", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]
    ].to_numpy(np.float32)
    sizes = states[["length", "width"]].to_numpy(np.float32)
    for begin in range(0, len(boundaries) - 1, 256):
        groups = [
            order[lo:hi]
            for lo, hi in zip(
                boundaries[begin : begin + 256], boundaries[begin + 1 : begin + 257]
            )
        ]
        width = max(map(len, groups))
        packed = np.zeros((len(groups), width, 6), np.float32)
        dimensions = np.ones((len(groups), width, 2), np.float32)
        valid = np.zeros((len(groups), width), bool)
        for local, indices in enumerate(groups):
            packed[local, : len(indices)] = motion[indices]
            dimensions[local, : len(indices)] = sizes[indices]
            valid[local, : len(indices)] = ~states._abnormal.to_numpy()[indices]
        observed = physical_leader_observation(
            torch.from_numpy(packed),
            torch.from_numpy(valid),
            torch.from_numpy(dimensions),
        )
        gaps, _, velocities, neighbors = [value.numpy() for value in observed]
        for local, indices in enumerate(groups):
            choice = neighbors[local, : len(indices)]
            leader[indices] = np.where(choice >= 0, indices[choice.clip(0)], -1)
            gap[indices] = gaps[local, : len(indices)]
            front_speed[indices] = velocities[local, : len(indices)]
    return leader, gap, front_speed


def extract_recording(recording_id, split, raw_dir, cfg, output):
    torch.set_num_threads(1)
    start = time.perf_counter()
    recording = prepare_recording(raw_dir, recording_id, cfg)
    geometry = parse_lane_markings(recording.recording_meta)
    canonical_lateral(recording, geometry)
    centers = {
        direction: np.sort(
            [
                geometry["lanes"][lane]["center"]
                for lane in geometry[f"direction_{direction}_lanes"]
            ]
        )
        for direction in (1, 2)
    }
    blocks = []
    observed_events = []
    for vehicle in recording.vehicle_ids():
        track = recording.get_vehicle_track(vehicle)
        meta = recording.tracks_meta.loc[vehicle]
        selected = track.index.to_numpy() % 5 == 0
        data = track.loc[
            selected,
            [
                "x",
                "y",
                "xVelocity",
                "yVelocity",
                "xAcceleration",
                "yAcceleration",
                "_abnormal",
                "precedingId",
                "frontSightDistance",
            ],
        ].copy()
        data = data.reset_index()
        data["vehicle"] = vehicle
        data["direction"] = int(meta.drivingDirection)
        data["class"] = 0 if meta["class"] == "Car" else 1
        data["length"] = float(meta.width)
        data["width"] = float(meta.height)
        data["lane"] = np.abs(
            data.y.to_numpy()[:, None] - centers[int(meta.drivingDirection)]
        ).argmin(1)
        track_centers = centers[int(meta.drivingDirection)]
        center = track_centers[
            np.abs(track.y.to_numpy()[:, None] - track_centers).argmin(1)
        ]
        settled = (np.abs(track.yVelocity.to_numpy()) < 0.15) & (
            np.abs(track.y.to_numpy() - center) < 0.6
        )
        previous = np.maximum.accumulate(np.where(settled, np.arange(len(track)), 0))
        source_y = center[previous]
        offset = abs(track.y.to_numpy() - source_y)
        width = (
            float(np.median(np.diff(track_centers))) if len(track_centers) > 1 else 4.0
        )
        data["changing"] = (
            (abs(track.yVelocity.to_numpy()) >= 0.15) | (offset >= 0.6)
        )[selected]
        data["progress"] = (offset / width).clip(0, 1)[selected]
        data["lateral_speed"] = abs(track.yVelocity.to_numpy())[selected]
        trajectory = track[
            ["x", "y", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]
        ].to_numpy(np.float32)
        native_frames = track.index.to_numpy()
        for part in np.split(
            np.arange(len(track)), np.flatnonzero(np.diff(native_frames) != 1) + 1
        ):
            rows, _, _ = lane_events(
                trajectory[part, None],
                ~track._abnormal.to_numpy()[part, None],
                track_centers,
                np.array([[float(meta.width), float(meta.height)]]),
                recording=recording_id,
                vehicle_ids=[vehicle],
                frame_origin=int(native_frames[part[0]]),
            )
            observed_events.extend(rows)
        lane_centers = {
            lane: geometry["lanes"][lane]["center"]
            for lane in geometry[f"direction_{int(meta.drivingDirection)}_lanes"]
        }
        label, observed, label_audit = lane_labels(track, lane_centers)
        data["label"] = label[selected]
        data["label_observed"] = observed[selected]
        for name, mask in label_audit.items():
            data[f"label_audit__{name}"] = mask[selected]
        blocks.append(data)
    states = pd.concat(blocks, ignore_index=True)
    front, rear, target_front, target_rear, available = nearest_neighbors(
        states, centers
    )
    x, vx, vy, ax = (
        states[name].to_numpy()
        for name in ("x", "xVelocity", "yVelocity", "xAcceleration")
    )
    speed = np.hypot(vx, vy)
    length = states.length.to_numpy()
    abnormal = states._abnormal.to_numpy()

    def neighbor_values(index, is_front):
        own_x, own_length = (
            value[:, None] if index.ndim == 2 else value for value in (x, length)
        )
        valid = index >= 0
        other = index.clip(0)
        gap = (x[other] - own_x if is_front else own_x - x[other]) - 0.5 * (
            own_length + length[other]
        )
        own_speed = speed[:, None] if index.ndim == 2 else speed
        return np.where(valid, gap, 1e6), np.where(valid, speed[other], own_speed)

    gap, leader_speed = neighbor_values(front, True)
    rear_gap, rear_speed = neighbor_values(rear, False)
    target_gap, target_speed = neighbor_values(target_front, True)
    target_rear_gap, target_rear_speed = neighbor_values(target_rear, False)
    road_length = {
        direction: float(
            states.loc[states.direction == direction, "x"].max()
            - states.loc[states.direction == direction, "x"].min()
        )
        for direction in (1, 2)
    }
    free = (front < 0) & (states.frontSightDistance.to_numpy() >= 120)
    valid = (
        ~abnormal & (speed > 5) & (speed < 60) & (np.abs(vy) < 0.3) & (np.abs(ax) < 8)
    )
    following = (front >= 0) & (gap > 2) & (gap < 150)
    following &= ~abnormal[front.clip(0)] & (np.abs(vy[front.clip(0)]) < 0.3)
    keep = valid & (following | free)
    density = (
        states.groupby(["direction", "frame"])["vehicle"].transform("count").to_numpy()
    )
    density = (
        density
        / np.array(
            [
                road_length[int(direction)] * len(centers[int(direction)])
                for direction in states.direction
            ]
        )
        * 1000
    )
    identities = {
        "recording": np.full(len(states), recording_id, np.int16),
        "vehicle": states.vehicle.to_numpy(np.int32),
        "frame": states.frame.to_numpy(np.int32),
        "direction": states.direction.to_numpy(np.uint8),
        "class": states["class"].to_numpy(np.uint8),
        "density": density.astype(np.float32),
    }
    payload = {f"follow__{key}": value[keep] for key, value in identities.items()}
    payload.update(
        {
            f"follow__{key}": value[keep].astype(np.float32)
            for key, value in {
                "gap": np.where(free, 1e6, gap),
                "speed": speed,
                "leader_speed": leader_speed,
                "acceleration": ax,
            }.items()
        }
    )
    payload["follow__free"] = free[keep]
    payload["follow__leader_vehicle"] = np.where(
        front >= 0, identities["vehicle"][front.clip(0)], -1
    )[keep].astype(np.int32)
    physical_front, physical_gap, physical_speed = swept_leaders(states)
    physical_free = (physical_front < 0) & (states.frontSightDistance.to_numpy() >= 120)
    response_keep = ~abnormal & (speed > 5) & (speed < 60) & (abs(ax) < 8)
    response_keep &= physical_free | (
        (physical_front >= 0) & (physical_gap > 2) & (physical_gap < 150)
    )
    for key, value in identities.items():
        payload[f"response__{key}"] = value[response_keep]
    for key, value in {
        "gap": physical_gap,
        "speed": speed,
        "leader_speed": physical_speed,
        "acceleration": (vx * ax + vy * states.yAcceleration.to_numpy())
        / speed.clip(0.5),
        "changing": states.changing.to_numpy(),
        "progress": states.progress.to_numpy(),
        "lateral_speed": states.lateral_speed.to_numpy(),
        "free": physical_free,
        "leader_vehicle": np.where(
            physical_front >= 0, identities["vehicle"][physical_front.clip(0)], -1
        ),
    }.items():
        payload[f"response__{key}"] = value[response_keep].astype(np.float32)
    center_y = np.array(
        [
            centers[int(direction)][int(lane)]
            for direction, lane in zip(states.direction, states.lane)
        ]
    )
    stable = (
        valid
        & (np.abs(vy) < 0.15)
        & (np.abs(states.y.to_numpy() - center_y) < 0.6)
        & available.any(1)
    )
    usable = stable & states.label_observed.to_numpy()
    label = states.label.to_numpy()
    usable &= (
        (label == 0)
        | ((label == 1) & available[:, 0])
        | ((label == 2) & available[:, 1])
    )
    usable &= available.any(1)
    features = np.stack(
        (
            np.broadcast_to(speed[:, None], target_gap.shape),
            np.broadcast_to(ax[:, None], target_gap.shape),
            np.broadcast_to(gap[:, None].clip(0, 120), target_gap.shape),
            np.broadcast_to((speed - leader_speed)[:, None], target_gap.shape),
            target_gap.clip(0, 120),
            target_speed - leader_speed[:, None],
            target_rear_gap.clip(0, 120),
            target_rear_speed - speed[:, None],
        ),
        -1,
    )
    payload.update({f"lane__{key}": value[usable] for key, value in identities.items()})
    for key, value in {
        "features": features,
        "own_length": length,
        "own_speed": speed,
        "own_acceleration": ax,
        "current_front_gap": gap,
        "current_front_speed": leader_speed,
        "current_rear_gap": rear_gap,
        "current_rear_speed": rear_speed,
        "target_front_gap": target_gap,
        "target_front_speed": target_speed,
        "target_rear_gap": target_rear_gap,
        "target_rear_speed": target_rear_speed,
    }.items():
        payload[f"lane__{key}"] = value[usable].astype(np.float32)
    payload["lane__label"] = label[usable].astype(np.uint8)
    payload["lane__side_label"] = label[usable, None] == np.array([1, 2])
    payload["lane__available"] = available[usable]
    for key, index in (
        ("current_front_vehicle", front),
        ("current_rear_vehicle", rear),
        ("target_front_vehicle", target_front),
        ("target_rear_vehicle", target_rear),
    ):
        physical_id = np.where(index >= 0, identities["vehicle"][index.clip(0)], -1)
        payload[f"lane__{key}"] = physical_id[usable].astype(np.int32)
    events, event_states, offsets, rejects = extract_complete_events(
        recording, split, geometry
    )
    for event in observed_events:
        frame = recording.get_frame(event["start_frame"]).droplevel("frame")
        direction = int(recording.tracks_meta.loc[event["vehicle"], "drivingDirection"])
        frame = frame[
            frame.index.isin(
                recording.tracks_meta.index[
                    recording.tracks_meta.drivingDirection == direction
                ]
            )
        ]
        ids = frame.index.to_numpy()
        actor = int(np.flatnonzero(ids == event["vehicle"])[0])
        motion = frame[
            ["x", "y", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]
        ].to_numpy()
        dimensions = recording.tracks_meta.loc[ids, ["width", "height"]].to_numpy()
        source_lane = int(np.abs(centers[direction] - motion[actor, 1]).argmin())
        event.update(
            start_neighbors(
                motion,
                ~frame._abnormal.to_numpy(),
                dimensions,
                centers[direction],
                actor,
                int(
                    np.clip(
                        source_lane + event["direction"], 0, len(centers[direction]) - 1
                    )
                ),
            )
        )
        event["split"] = split
    directory = Path(output) / "recordings"
    directory.mkdir(parents=True, exist_ok=True)
    pending = directory / f"{recording_id:02d}.pending.npz"
    np.savez_compressed(pending, **payload)
    pending.replace(directory / f"{recording_id:02d}.npz")
    events.to_csv(directory / f"{recording_id:02d}_events.csv", index=False)
    pd.DataFrame(observed_events, columns=(*EVENT_FIELDS, "split")).to_csv(
        directory / f"{recording_id:02d}_behavior_events.csv", index=False
    )
    np.savez_compressed(
        directory / f"{recording_id:02d}_observations.npz",
        states=event_states,
        offsets=offsets,
    )
    real_leader = np.where(front >= 0, identities["vehicle"][front.clip(0)], -1)
    both = (front >= 0) & (states.precedingId.to_numpy() >= 0)
    report = {
        "recording": int(recording_id),
        "split": split,
        "physical_vehicles": int(len(recording.vehicle_ids())),
        "sampled_states": int(len(states)),
        "following_states": int((keep & following).sum()),
        "free_states": int((keep & free).sum()),
        "response_states": int(response_keep.sum()),
        "response_maneuver_states": int(
            (response_keep & states.changing.to_numpy()).sum()
        ),
        "observed_lane_events": len(observed_events),
        "lane_opportunities": int(usable.sum()),
        "lane_side_candidates": int(available[usable].sum()),
        "lane_start_labels": int((label[usable] != 0).sum()),
        "lane_label_audit": {
            name: int((stable & states[f"label_audit__{name}"].to_numpy()).sum())
            for name in LABEL_AUDIT_NAMES
        },
        "complete_lane_events": int(len(events)),
        "leader_identity_compared": int(both.sum()),
        "leader_identity_agreement": (
            float(np.mean(real_leader[both] == states.precedingId.to_numpy()[both]))
            if both.any()
            else None
        ),
        "event_rejections": rejects,
        "raw_rows": len(recording.tracks),
        "abnormal_rows": int(recording.tracks._abnormal.sum()),
        "elapsed_seconds": time.perf_counter() - start,
    }
    print(
        f"raw highD {recording_id:02d} {split}: following={report['following_states']:,}, free={report['free_states']:,}, opportunities={report['lane_opportunities']:,}, complete changes={len(events)}",
        flush=True,
    )
    return report


def load_group(output=DRIVER_DATA_DIR, group="follow", splits=SPLITS):
    manifest = json.loads((Path(output) / "manifest.json").read_text())
    result = {}
    for split in splits:
        blocks = []
        for recording in manifest["split_recordings"][split]:
            with np.load(
                Path(output) / "recordings" / f"{recording:02d}.npz"
            ) as arrays:
                blocks.append(
                    {
                        name.split("__", 1)[1]: arrays[name]
                        for name in arrays.files
                        if name.startswith(f"{group}__")
                    }
                )
        result[split] = {
            key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]
        }
    return result


def load_following(output=DRIVER_DATA_DIR, splits=SPLITS, *, group="follow"):
    groups = load_group(output, group, splits)
    for data in groups.values():
        data["free"] = data["free"].astype(bool)
        order = np.lexsort((data["frame"], data["vehicle"], data["recording"]))
        prior = np.roll(order, 1)
        contiguous = (
            (data["recording"][order] == data["recording"][prior])
            & (data["vehicle"][order] == data["vehicle"][prior])
            & (data["frame"][order] - data["frame"][prior] == 5)
        )
        same_leader = (
            contiguous
            & ~data["free"][order]
            & ~data["free"][prior]
            & (data["leader_vehicle"][order] == data["leader_vehicle"][prior])
        )
        data["leader_acceleration"] = np.zeros(len(order), np.float32)
        data["leader_history_available"] = np.zeros(len(order), bool)
        data["leader_acceleration"][order[same_leader]] = (
            data["leader_speed"][order[same_leader]]
            - data["leader_speed"][prior[same_leader]]
        ) / 0.2
        data["leader_history_available"][order] = same_leader
    return groups


def load_lane_opportunities(output=DRIVER_DATA_DIR, splits=SPLITS):
    return load_group(output, "lane", splits)


def lane_side_data(data, theta, safe_braking_mps2=4):
    """Recompute exact physical-neighbor MOBIL incentives after each IDM refit."""
    speed = data["own_speed"]
    length = data["own_length"]
    current = idm_acceleration(
        data["current_front_gap"], speed, data["current_front_speed"], theta
    )
    own_after = idm_acceleration(
        data["target_front_gap"], speed[:, None], data["target_front_speed"], theta
    )
    rear_after = idm_acceleration(
        data["target_rear_gap"], data["target_rear_speed"], speed[:, None], theta
    )
    rear_before = idm_acceleration(
        data["target_rear_gap"] + length[:, None] + data["target_front_gap"],
        data["target_rear_speed"],
        data["target_front_speed"],
        theta,
    )
    target_rear_exists = data["target_rear_gap"] < 1e6
    old_before = idm_acceleration(
        data["current_rear_gap"], data["current_rear_speed"], speed, theta
    )
    old_after = idm_acceleration(
        data["current_rear_gap"] + length + data["current_front_gap"],
        data["current_rear_speed"],
        data["current_front_speed"],
        theta,
    )
    old_delta = np.where(data["current_rear_gap"] < 1e6, old_after - old_before, 0)
    own = own_after - current[:, None]
    social = (
        own
        + np.where(target_rear_exists, rear_after - rear_before, 0)
        + old_delta[:, None]
    )
    safe = data["available"] & (
        ~target_rear_exists | (rear_after >= -safe_braking_mps2)
    )
    keep = data["available"]
    result = {
        key: np.broadcast_to(data[key][:, None], keep.shape)[keep]
        for key in ("recording", "vehicle", "frame")
    }
    result.update(
        {
            "features": data["features"][keep],
            "label": data["side_label"][keep],
            "mobil_own": own[keep].astype(np.float32),
            "mobil_social": social[keep].astype(np.float32),
            "mobil_safe": safe[keep],
            "side": np.broadcast_to(np.array([-1, 1]), keep.shape)[keep],
        }
    )
    return result


def load_lane_events(output=DRIVER_DATA_DIR, splits=SPLITS):
    result = {}
    for split in splits:
        events = pd.read_csv(Path(output) / f"{split}_events.csv")
        with np.load(Path(output) / f"{split}_observations.npz") as arrays:
            observations = {key: arrays[key] for key in arrays.files}
        result[split] = events, observations
    return result


def prepare_raw_data(output=DRIVER_DATA_DIR, workers=4):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg_path = ROOT / "process_highD/config.yaml"
    cfg = load_config(cfg_path)
    raw_dir = resolve_data_path(cfg["paths"]["raw_dir"], cfg_path)
    splits = recording_splits()
    manifest = {
        "source": str(raw_dir),
        "preprocessing_config": str(cfg_path),
        "split_recordings": splits,
        "status": "extracting",
        "fps": 25,
        "state_sampling_hz": 5,
        "identity": "recording, physical vehicle, original frame",
        "class_encoding": {"0": "Car", "1": "Truck"},
        "lane_side_order": ["lower_y", "higher_y"],
        "lane_label_encoding": {"0": "keep", "1": "lower_y", "2": "higher_y"},
        "lateral_coordinates": "world-model convention: direction1 y,vy,ay unchanged; direction2 mirrored; sides in increasing canonical y",
        "free_driving_front_sight_m": 120,
        "following_selection": "gap 2–150 m; speed 5–60 m/s; own and leader |vy|<0.3 m/s; |ax|<8 m/s2; both non-abnormal",
        "complete_lane_selection": "contiguous raw track; 2–10 s adjacent change; peak |vy| 0.4–3 m/s; start/end within0.7 m of source/target centre; observed1 s pre/post history",
        "trajectory_fields": ["y_relative_to_source_m", "vx_mps", "vy_mps", "ay_mps2"],
        "label_horizon_s": 1,
        "confirmation_extra_frames": 4,
        "label_protocol": LABEL_PROTOCOL,
        "behavior_protocol": BEHAVIOR_PROTOCOL,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    started = time.perf_counter()
    reports = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(extract_recording, recording, split, raw_dir, cfg, output)
            for split in SPLITS
            for recording in splits[split]
        ]
        for future in as_completed(futures):
            reports.append(future.result())
            (output / "coverage.json").write_text(
                json.dumps(
                    {
                        "status": "extracting",
                        "recordings": sorted(
                            reports, key=lambda item: item["recording"]
                        ),
                    },
                    indent=2,
                )
            )
    for split in SPLITS:
        frames, state_blocks, lengths = [], [], []
        trajectory_count = 0
        for recording in splits[split]:
            events = pd.read_csv(output / "recordings" / f"{recording:02d}_events.csv")
            events.trajectory_index += trajectory_count
            frames.append(events)
            with np.load(
                output / "recordings" / f"{recording:02d}_observations.npz"
            ) as arrays:
                state_blocks.append(arrays["states"])
                lengths.extend(np.diff(arrays["offsets"]))
            trajectory_count += len(events)
        pd.concat(frames, ignore_index=True).to_csv(
            output / f"{split}_events.csv", index=False
        )
        np.savez_compressed(
            output / f"{split}_observations.npz",
            states=np.concatenate(state_blocks),
            offsets=np.r_[0, np.cumsum(lengths)],
        )
        pd.concat(
            [
                pd.read_csv(
                    output / "recordings" / f"{recording:02d}_behavior_events.csv"
                )
                for recording in splits[split]
            ],
            ignore_index=True,
        ).to_csv(output / f"{split}_behavior_events.csv", index=False)
    protocol = lateral_protocol(pd.read_csv(output / "train_behavior_events.csv"))
    (output / "lateral_protocol.json").write_text(json.dumps(protocol, indent=2))
    fields = (
        "physical_vehicles",
        "sampled_states",
        "following_states",
        "free_states",
        "response_states",
        "response_maneuver_states",
        "observed_lane_events",
        "lane_opportunities",
        "lane_side_candidates",
        "lane_start_labels",
        "complete_lane_events",
        "raw_rows",
        "abnormal_rows",
    )
    coverage = {
        "status": "complete",
        "elapsed_seconds": time.perf_counter() - started,
        "recordings": sorted(reports, key=lambda item: item["recording"]),
        "splits": {
            split: {
                key: int(sum(item[key] for item in reports if item["split"] == split))
                for key in fields
            }
            for split in SPLITS
        },
        "total": {key: int(sum(item[key] for item in reports)) for key in fields},
    }
    for split in SPLITS:
        coverage["splits"][split]["lane_label_audit"] = {
            name: int(
                sum(
                    item["lane_label_audit"][name]
                    for item in reports
                    if item["split"] == split
                )
            )
            for name in LABEL_AUDIT_NAMES
        }
    coverage["total"]["lane_label_audit"] = {
        name: int(sum(item["lane_label_audit"][name] for item in reports))
        for name in LABEL_AUDIT_NAMES
    }
    (output / "coverage.json").write_text(json.dumps(coverage, indent=2))
    manifest["status"] = "complete"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return coverage


if __name__ == "__main__":
    result = prepare_raw_data()
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "elapsed_seconds", "splits", "total")
            },
            indent=2,
        ),
        flush=True,
    )
