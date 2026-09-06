"""Independent highD leader–follower evidence for reaction policies."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from diffusion.src.data import ANCHOR_INDEX


SLOT_NAMES = (
    "same_front", "same_rear", "left_front", "left_rear",
    "right_front", "right_rear",
)
FEATURE_NAMES = (
    "acceleration_mps2", "abs_jerk_mps3", "speed_mps",
    "gap_m", "closing_mps", "ttc_s",
)
BRAKE_EDGES = (0.5, 2.0, 4.0, 6.0, 8.0001)
TTC_EDGES = (0.0, 2.0, 4.0, np.inf)
PRE_EVENT_FRAMES = 25
EVALUATION_FRAMES = 25
RECOVERY_FRAMES = 75


def support_cell(brake_mps2: float, ttc_s: float) -> int:
    brake_index = int(np.searchsorted(BRAKE_EDGES, brake_mps2, side="right") - 1)
    ttc_index = int(np.searchsorted(TTC_EDGES, max(ttc_s, 0.0), side="right") - 1)
    if brake_index not in range(len(BRAKE_EDGES) - 1) or ttc_index not in range(len(TTC_EDGES) - 1):
        return -1
    return brake_index * (len(TTC_EDGES) - 1) + ttc_index


def cell_label(cell: int) -> str:
    if cell < 0:
        return "unsupported"
    brake, ttc = divmod(cell, len(TTC_EDGES) - 1)
    upper = "inf" if np.isinf(TTC_EDGES[ttc + 1]) else f"{TTC_EDGES[ttc + 1]:g}"
    return f"brake_{BRAKE_EDGES[brake]:g}_{BRAKE_EDGES[brake + 1]:g}_ttc_{TTC_EDGES[ttc]:g}_{upper}"


@dataclass(frozen=True)
class ReactionEvents:
    """Deduplicated physical events with one trajectory per vehicle pair."""

    row_index: np.ndarray
    recording_id: np.ndarray
    leader_id: np.ndarray
    follower_id: np.ndarray
    absolute_onset_frame: np.ndarray
    local_onset_frame: np.ndarray
    leader_slot: np.ndarray
    follower_slot: np.ndarray
    cell: np.ndarray
    initial_conditions: np.ndarray
    trajectory: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.row_index)
        fields = (
            self.recording_id, self.leader_id, self.follower_id,
            self.absolute_onset_frame, self.local_onset_frame,
            self.leader_slot, self.follower_slot, self.cell,
            self.initial_conditions, self.trajectory,
        )
        if any(len(field) != count for field in fields):
            raise ValueError("reaction event fields must have equal length")
        keys = np.stack((self.recording_id, self.leader_id, self.follower_id, self.absolute_onset_frame), axis=1)
        if len(np.unique(keys, axis=0)) != count:
            raise ValueError("reaction event keys must be unique")

    def indices(self, cells: tuple[int, ...] | None = None) -> np.ndarray:
        if cells is None:
            return np.arange(len(self.row_index), dtype=np.int64)
        return np.flatnonzero(np.isin(self.cell, np.asarray(cells, np.int16)))


@dataclass(frozen=True)
class ReactionEventReference:
    """One split's event evidence and train-defined support table."""

    split: str
    events: ReactionEvents
    supported_cells: tuple[int, ...]
    event_counts: dict[int, int]
    recording_counts: dict[int, int]
    minimum_events: int = 100
    minimum_recordings: int = 5

    def is_supported(self, cell: int) -> bool:
        return int(cell) in self.supported_cells

    def save(self, directory: str | Path) -> None:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(root / "reaction_events.npz", **self.events.__dict__)
        manifest = {
            "schema_name": "reaction_event_reference",
            "schema_version": 1,
            "split": self.split,
            "feature_names": FEATURE_NAMES,
            "initial_condition_names": (
                "leader_brake_mps2", "gap_m", "closing_mps", "ttc_s",
                "follower_acceleration_mps2",
            ),
            "pre_event_frames": PRE_EVENT_FRAMES,
            "evaluation_frames": EVALUATION_FRAMES,
            "recovery_frames": RECOVERY_FRAMES,
            "minimum_events": self.minimum_events,
            "minimum_recordings": self.minimum_recordings,
            "event_counts": {str(int(cell)): int(count) for cell, count in self.event_counts.items()},
            "recording_counts": {str(int(cell)): int(count) for cell, count in self.recording_counts.items()},
            "supported_cells": [int(cell) for cell in self.supported_cells],
            "supported_cell_labels": [cell_label(cell) for cell in self.supported_cells],
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    @classmethod
    def load(cls, directory: str | Path) -> "ReactionEventReference":
        root = Path(directory)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("schema_name") != "reaction_event_reference" or manifest.get("schema_version") != 1:
            raise ValueError("unsupported reaction event reference")
        with np.load(root / "reaction_events.npz") as values:
            events = ReactionEvents(**{name: values[name].copy() for name in ReactionEvents.__dataclass_fields__})
        return cls(
            split=str(manifest["split"]), events=events,
            supported_cells=tuple(int(cell) for cell in manifest["supported_cells"]),
            event_counts={int(cell): int(count) for cell, count in manifest["event_counts"].items()},
            recording_counts={int(cell): int(count) for cell, count in manifest["recording_counts"].items()},
            minimum_events=int(manifest["minimum_events"]),
            minimum_recordings=int(manifest["minimum_recordings"]),
        )


def _forward_acceleration(states: np.ndarray) -> np.ndarray:
    return (states[1:, :, 2] - states[:-1, :, 2]) / 0.04


def _pair_features(states: np.ndarray, leader: int, follower: int, start: int, stop: int) -> np.ndarray:
    acceleration = _forward_acceleration(states)
    follower_acceleration = acceleration[start:stop, follower]
    previous = acceleration[start - 1:stop - 1, follower]
    leader_state = states[start + 1:stop + 1, leader]
    follower_state = states[start + 1:stop + 1, follower]
    gap = leader_state[:, 0] - follower_state[:, 0] - 4.8
    closing = follower_state[:, 2] - leader_state[:, 2]
    ttc = np.where(closing > 1.0e-4, gap / np.maximum(closing, 1.0e-4), 10.0)
    return np.stack((
        follower_acceleration,
        np.abs(follower_acceleration - previous) / 0.04,
        follower_state[:, 2], gap, closing, np.clip(ttc, 0.0, 10.0),
    ), axis=-1).astype(np.float32)


def build_reaction_event_reference(
    arrays: dict[str, np.ndarray], *, split: str,
    minimum_events: int = 100, minimum_recordings: int = 5,
    brake_threshold_mps2: float = 0.5, minimum_brake_frames: int = 3,
    merge_gap_frames: int = 25, lane_half_width_m: float = 1.8,
    supported_cells: tuple[int, ...] | None = None,
) -> ReactionEventReference:
    """Mine sustained same-lane braking responses without future event selection."""
    states_all = np.asarray(arrays["agent_states"], np.float32)
    valid_all = np.asarray(arrays["agent_valid"], bool)
    agent_ids = np.asarray(arrays["agent_ids"], np.int64)
    rows = np.asarray(arrays.get("row_index", np.arange(len(states_all))), np.int64)
    recordings = np.asarray(arrays["recording_id"], np.int64)
    anchors = np.asarray(arrays["anchor_frame"], np.int64)
    if agent_ids.shape != (len(states_all), 7):
        raise ValueError("agent_ids must align with [ego, six stable background slots]")

    retained: dict[tuple[int, int, int, int], tuple] = {}
    start_limit = max(ANCHOR_INDEX + PRE_EVENT_FRAMES, 1)
    stop_limit = states_all.shape[1] - RECOVERY_FRAMES - 1
    for local_row, (states, valid) in enumerate(zip(states_all, valid_all)):
        acceleration = _forward_acceleration(states)
        for leader in range(7):
            if agent_ids[local_row, leader] < 0:
                continue
            for onset in range(start_limit, stop_limit):
                if not valid[onset - 1:onset + minimum_brake_frames + 1, leader].all():
                    continue
                current = acceleration[onset, leader]
                previous = acceleration[onset - 1, leader]
                if previous <= -brake_threshold_mps2 or current > -brake_threshold_mps2:
                    continue
                if not np.all(acceleration[onset:onset + minimum_brake_frames, leader] <= -brake_threshold_mps2):
                    continue
                dx = states[onset, :, 0] - states[onset, leader, 0]
                dy = np.abs(states[onset, :, 1] - states[onset, leader, 1])
                followers = np.flatnonzero(
                    valid[onset] & (agent_ids[local_row] >= 0) & (dx < 0.0) & (dy < lane_half_width_m)
                )
                if not len(followers):
                    continue
                follower = int(followers[np.argmax(dx[followers])])
                if not valid[onset - PRE_EVENT_FRAMES:onset + RECOVERY_FRAMES + 1, (leader, follower)].all():
                    continue
                recording = int(recordings[local_row])
                leader_id = int(agent_ids[local_row, leader])
                follower_id = int(agent_ids[local_row, follower])
                absolute_onset = int(anchors[local_row]) + onset - ANCHOR_INDEX
                pair = (recording, leader_id, follower_id)
                gap = float(states[onset, leader, 0] - states[onset, follower, 0] - 4.8)
                closing = float(states[onset, follower, 2] - states[onset, leader, 2])
                ttc = gap / closing if gap > 0.0 and closing > 1.0e-4 else 10.0
                cell = support_cell(float(-current), float(ttc))
                if cell < 0:
                    continue
                trajectory = _pair_features(
                    states, leader, follower,
                    onset - PRE_EVENT_FRAMES, onset + RECOVERY_FRAMES,
                )
                follower_previous = float(acceleration[onset - 1, follower])
                initial = np.asarray(
                    (-current, gap, closing, min(max(ttc, 0.0), 10.0), follower_previous),
                    np.float32,
                )
                key = (*pair, absolute_onset)
                retained.setdefault(key, (
                    int(rows[local_row]), recording, leader_id, follower_id,
                    absolute_onset, onset, leader, follower, cell, initial, trajectory,
                ))

    values = []
    last_onset: dict[tuple[int, int, int], int] = {}
    for candidate in sorted(retained.values(), key=lambda item: item[1:5]):
        pair, onset = tuple(candidate[1:4]), int(candidate[4])
        if onset - last_onset.get(pair, -10**9) >= merge_gap_frames:
            values.append(candidate)
            last_onset[pair] = onset
    if values:
        columns = list(zip(*values))
        events = ReactionEvents(
            row_index=np.asarray(columns[0], np.int64),
            recording_id=np.asarray(columns[1], np.int64),
            leader_id=np.asarray(columns[2], np.int64),
            follower_id=np.asarray(columns[3], np.int64),
            absolute_onset_frame=np.asarray(columns[4], np.int64),
            local_onset_frame=np.asarray(columns[5], np.int16),
            leader_slot=np.asarray(columns[6], np.int8),
            follower_slot=np.asarray(columns[7], np.int8),
            cell=np.asarray(columns[8], np.int16),
            initial_conditions=np.stack(columns[9]).astype(np.float32),
            trajectory=np.stack(columns[10]).astype(np.float32),
        )
    else:
        events = ReactionEvents(
            *(np.empty(0, dtype=dtype) for dtype in (
                np.int64, np.int64, np.int64, np.int64, np.int64,
                np.int16, np.int8, np.int8, np.int16,
            )),
            initial_conditions=np.empty((0, 5), np.float32),
            trajectory=np.empty((0, PRE_EVENT_FRAMES + RECOVERY_FRAMES, len(FEATURE_NAMES)), np.float32),
        )
    event_counts = {cell: int((events.cell == cell).sum()) for cell in np.unique(events.cell)}
    recording_counts = {
        cell: int(len(np.unique(events.recording_id[events.cell == cell])))
        for cell in event_counts
    }
    supported = (
        tuple(sorted(
            cell for cell in event_counts
            if event_counts[cell] >= minimum_events and recording_counts[cell] >= minimum_recordings
        ))
        if supported_cells is None
        else tuple(sorted(cell for cell in supported_cells if event_counts.get(cell, 0) > 0))
    )
    return ReactionEventReference(
        split=split, events=events, supported_cells=supported,
        event_counts=event_counts, recording_counts=recording_counts,
        minimum_events=minimum_events, minimum_recordings=minimum_recordings,
    )


def assert_split_isolation(*references: ReactionEventReference) -> None:
    recordings: dict[int, str] = {}
    keys: dict[tuple[int, int, int, int], str] = {}
    for reference in references:
        for index in range(len(reference.events.row_index)):
            recording = int(reference.events.recording_id[index])
            key = (
                recording, int(reference.events.leader_id[index]),
                int(reference.events.follower_id[index]),
                int(reference.events.absolute_onset_frame[index]),
            )
            if recording in recordings and recordings[recording] != reference.split:
                raise ValueError("recording appears in more than one split")
            if key in keys and keys[key] != reference.split:
                raise ValueError("reaction event appears in more than one split")
            recordings[recording] = reference.split
            keys[key] = reference.split


def event_window(events: ReactionEvents, *, recovery: bool = False) -> np.ndarray:
    stop = PRE_EVENT_FRAMES + (RECOVERY_FRAMES if recovery else EVALUATION_FRAMES)
    return events.trajectory[:, PRE_EVENT_FRAMES:stop]


def energy_score(futures: np.ndarray, observed: np.ndarray) -> float:
    samples = np.asarray(futures, np.float64).reshape(len(futures), -1)
    target = np.asarray(observed, np.float64).reshape(1, -1)
    if len(samples) < 2:
        raise ValueError("Energy Score requires at least two futures")
    return float(
        np.linalg.norm(samples - target, axis=1).mean()
        - 0.5 * np.linalg.norm(samples[:, None] - samples[None, :], axis=-1).mean()
    )


def wasserstein_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.sort(np.asarray(left, np.float64).reshape(-1))
    right = np.sort(np.asarray(right, np.float64).reshape(-1))
    if not len(left) or not len(right):
        return float("inf")
    quantiles = np.linspace(0.0, 1.0, min(len(left), len(right)))
    return float(np.abs(np.quantile(left, quantiles) - np.quantile(right, quantiles)).mean())


def recording_cluster_bootstrap(
    delta: np.ndarray, recording_id: np.ndarray, *, draws: int = 2000, seed: int = 17,
) -> dict[str, float | int]:
    values = np.asarray(delta, np.float64)
    records = np.asarray(recording_id)
    if values.ndim != 1 or len(values) != len(records) or not len(values):
        raise ValueError("event deltas and recording ids must align")
    groups = [values[records == record] for record in np.unique(records)]
    rng = np.random.default_rng(seed)
    samples = np.asarray([
        np.concatenate([groups[index] for index in rng.integers(len(groups), size=len(groups))]).mean()
        for _ in range(draws)
    ])
    return {
        "mean": float(values.mean()), "lcb95": float(np.quantile(samples, 0.05)),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "recordings": len(groups), "events": len(values),
    }
