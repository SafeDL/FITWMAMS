"""Mine deduplicated logged interaction events from the strict causal queue."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .causal_cache import load_causal_cache
from .manifest import load_benchmark_config

PRE_FRAMES = 25
POST_FRAMES = 75
SEARCH_START = PRE_FRAMES - 1
SEARCH_STOP = 174 - POST_FRAMES - 1
STABLE = 5


def _lane_assignments(
    states: np.ndarray, polylines: np.ndarray, poly_valid: np.ndarray
) -> np.ndarray:
    lane_valid = poly_valid.any(-1)
    lines = polylines[lane_valid]
    mask = poly_valid[lane_valid]
    centers = (lines[..., 1] * mask).sum(-1) / mask.sum(-1).clip(1)
    return np.abs(states[..., 1, None] - centers).argmin(-1).astype(np.int16)


def _front_relations(
    states: np.ndarray, lanes: np.ndarray, valid: np.ndarray, lengths: np.ndarray
) -> np.ndarray:
    time, agents = valid.shape
    x = states[..., 0]
    dx = x[:, :, None] - x[:, None, :]
    same = lanes[:, :, None] == lanes[:, None, :]
    pair = same & valid[:, :, None] & valid[:, None, :] & (dx > 0)
    gap = dx - (lengths[None, :, None] + lengths[None, None, :]) * 0.5
    gap[~pair] = np.inf
    choice = gap.argmin(1)
    minimum = np.take_along_axis(gap, choice[:, None], axis=1)[:, 0]
    leaders = choice.astype(np.int16)
    leaders[~np.isfinite(minimum)] = -1
    return leaders


def _stable(values: np.ndarray, start: int, stop: int, expected: int) -> bool:
    return (
        start >= 0
        and stop <= len(values)
        and bool(np.all(values[start:stop] == expected))
    )


def mine_event_manifest(config_path: str | Path) -> dict[str, Any]:
    config, _ = load_benchmark_config(config_path)
    arrays, metadata, _ = load_causal_cache(config_path)
    candidates: dict[tuple, dict[str, Any]] = {}
    for row in range(len(metadata["sequence_id"])):
        states = np.asarray(arrays["agent_states"][row])
        valid = np.asarray(arrays["agent_valid"][row])
        ids = metadata["agent_ids"][row]
        active = ids >= 0
        lanes = _lane_assignments(
            states,
            np.asarray(arrays["map_polylines"][row]),
            np.asarray(arrays["map_polyline_valid"][row]),
        )
        relations = _front_relations(states, lanes, valid, metadata["lengths_m"][row])
        recording = int(metadata["recording_id"][row])
        split = int(metadata["split_index"][row])
        start_frame = int(metadata["history_start_frame"][row])

        def retain(
            event_type: str,
            onset: int,
            stimulus: int,
            response: int,
            direction: str = "",
        ):
            if (
                not (SEARCH_START <= onset <= SEARCH_STOP)
                or stimulus < 0
                or response < 0
            ):
                return
            stimulus_id, response_id = int(ids[stimulus]), int(ids[response])
            if stimulus_id < 0 or response_id < 0:
                return
            absolute = start_frame + onset
            key = (recording, event_type, stimulus_id, response_id, absolute)
            value = {
                "event_id": f"{recording:02d}:{event_type}:{stimulus_id}:{response_id}:{absolute}",
                "event_type": event_type,
                "direction": direction,
                "recording_id": recording,
                "split_index": split,
                "scenario_row": row,
                "scenario_id": str(metadata["sequence_id"][row]),
                "local_onset_frame": onset,
                "absolute_onset_frame": absolute,
                "stimulus_agent_index": stimulus,
                "response_agent_index": response,
                "stimulus_agent_id": stimulus_id,
                "response_agent_id": response_id,
                "active_agents": int(active.sum()),
            }
            if (
                key not in candidates
                or value["active_agents"] > candidates[key]["active_agents"]
            ):
                candidates[key] = value

        # Sustained braking onset with the nearest same-lane follower.
        acceleration = states[..., 4]
        sustained_brake = (
            (acceleration[:-2] <= -0.5)
            & (acceleration[1:-1] <= -0.5)
            & (acceleration[2:] <= -0.5)
        )
        for leader in np.flatnonzero(active):
            starts = np.arange(SEARCH_START, SEARCH_STOP + 1)
            onset_mask = acceleration[starts - 1, leader] > -0.5
            onset_mask &= sustained_brake[starts, leader]
            for onset in starts[onset_mask]:
                followers = np.flatnonzero(relations[onset] == leader)
                if len(followers):
                    follower = int(
                        followers[
                            np.argmin(
                                states[onset, leader, 0] - states[onset, followers, 0]
                            )
                        ]
                    )
                    retain("following_brake", onset, leader, follower)

        # Completed adjacent lane changes by persistent identity.
        for agent in np.flatnonzero(active):
            series = lanes[:, agent]
            changed = np.flatnonzero(series[1:] != series[:-1]) + 1
            for onset in changed[(changed >= SEARCH_START) & (changed <= SEARCH_STOP)]:
                before, after = int(series[onset - 1]), int(series[onset])
                if before == after or abs(after - before) != 1:
                    continue
                if _stable(series, onset - STABLE, onset, before) and _stable(
                    series, onset, onset + STABLE, after
                ):
                    direction = (
                        "left"
                        if states[onset, agent, 1] > states[onset - 1, agent, 1]
                        else "right"
                    )
                    retain("lane_change", onset, agent, agent, direction)

        # Relation changes, separated into entry (cut-in) and departure (cut-out).
        for follower in np.flatnonzero(active):
            series = relations[:, follower]
            changed = np.flatnonzero(series[1:] != series[:-1]) + 1
            for onset in changed[(changed >= SEARCH_START) & (changed <= SEARCH_STOP)]:
                old, new = int(series[onset - 1]), int(series[onset])
                if old == new:
                    continue
                if new >= 0 and _stable(series, onset, onset + STABLE, new):
                    was_adjacent = (
                        lanes[onset - STABLE, new] != lanes[onset - STABLE, follower]
                    )
                    if was_adjacent and lanes[onset, new] == lanes[onset, follower]:
                        retain("cut_in", onset, new, follower)
                if old >= 0 and _stable(series, onset - STABLE, onset, old):
                    left_lane = (
                        lanes[onset + STABLE - 1, old]
                        != lanes[onset + STABLE - 1, follower]
                    )
                    if left_lane:
                        retain("cut_out", onset, old, follower)

    frame = (
        pd.DataFrame(candidates.values())
        .sort_values(
            ["split_index", "recording_id", "absolute_onset_frame", "event_type"]
        )
        .reset_index(drop=True)
    )
    frame["event_family"] = frame["event_type"]
    output = Path(config["paths"]["output_dir"])
    frame.to_csv(output / "event_manifest.csv", index=False)
    parquet_written = False
    try:
        frame.to_parquet(output / "event_manifest.parquet", index=False)
        parquet_written = True
    except (ImportError, ModuleNotFoundError):
        pass
    names = {0: "train", 1: "validation", 2: "test"}
    counts = {
        names[split]: {
            event: int(
                ((frame["split_index"] == split) & (frame["event_type"] == event)).sum()
            )
            for event in sorted(frame["event_type"].unique())
        }
        for split in names
    }
    report = {
        "event_manifest_version": "npc_interaction_d1_fixed_population_v1",
        "events": int(len(frame)),
        "counts": counts,
        "pre_event_frames": PRE_FRAMES,
        "post_event_frames": POST_FRAMES,
        "deduplication_key": [
            "recording_id",
            "event_type",
            "stimulus_agent_id",
            "response_agent_id",
            "absolute_onset_frame",
        ],
        "parquet_written": parquet_written,
        "scope": "all qualifying events represented in the strict D0 fixed-population windows",
    }
    (output / "event_manifest_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report
