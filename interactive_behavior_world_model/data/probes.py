"""Build deterministic D3 longitudinal stimulus anchors from causal prefixes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .causal_cache import load_causal_cache
from .manifest import load_benchmark_config


def _nearest_lanes(
    states: np.ndarray, polylines: np.ndarray, poly_valid: np.ndarray
) -> np.ndarray:
    lane_valid = poly_valid.any(-1)
    lines = polylines[lane_valid]
    mask = poly_valid[lane_valid]
    centers = (lines[..., 1] * mask).sum(-1) / mask.sum(-1).clip(1)
    return np.abs(states[:, 1, None] - centers[None]).argmin(-1).astype(np.int16)


def _ordered_lane_centers(polylines: np.ndarray, poly_valid: np.ndarray) -> np.ndarray:
    lane_valid = np.asarray(poly_valid, bool).any(-1)
    lines = np.asarray(polylines)[lane_valid]
    mask = np.asarray(poly_valid, bool)[lane_valid]
    centers = (lines[..., 1] * mask).sum(-1) / mask.sum(-1).clip(1)
    return np.unique(np.round(centers.astype(np.float64), 4))


def mine_probe_manifest(config_path: str | Path) -> dict[str, Any]:
    """Select one closest valid leader/follower pair per strict D0 prefix.

    The selection only reads the common 25-frame prefix.  Keeping one pair per
    scene avoids treating several correlated relations as independent probes.
    """
    config, _ = load_benchmark_config(config_path)
    arrays, metadata, _ = load_causal_cache(config_path)
    records: list[dict[str, Any]] = []
    exclusion = {"no_valid_npc_follower": 0, "no_qualifying_pair": 0}
    for row in range(len(metadata["sequence_id"])):
        state = np.asarray(arrays["agent_states"][row, 24])
        prefix_valid = np.asarray(arrays["agent_valid"][row, :25]).all(0)
        ids = metadata["agent_ids"][row]
        active = prefix_valid & (ids >= 0)
        if not active[1:].any():
            exclusion["no_valid_npc_follower"] += 1
            continue
        lanes = _nearest_lanes(
            state,
            np.asarray(arrays["map_polylines"][row]),
            np.asarray(arrays["map_polyline_valid"][row]),
        )
        candidates = []
        for response in np.flatnonzero(active & (np.arange(7) > 0)):
            ahead = (
                active & (lanes == lanes[response]) & (state[:, 0] > state[response, 0])
            )
            for stimulus in np.flatnonzero(ahead):
                gap = (
                    state[stimulus, 0]
                    - state[response, 0]
                    - 0.5
                    * (
                        metadata["lengths_m"][row, stimulus]
                        + metadata["lengths_m"][row, response]
                    )
                )
                response_speed = float(np.linalg.norm(state[response, 2:4]))
                stimulus_speed = float(np.linalg.norm(state[stimulus, 2:4]))
                if (
                    5.0 <= gap <= 80.0
                    and response_speed >= 5.0
                    and stimulus_speed >= 3.0
                ):
                    candidates.append(
                        (
                            float(gap),
                            int(stimulus),
                            int(response),
                            response_speed,
                            stimulus_speed,
                        )
                    )
        if not candidates:
            exclusion["no_qualifying_pair"] += 1
            continue
        gap, stimulus, response, response_speed, stimulus_speed = min(candidates)
        recording = int(metadata["recording_id"][row])
        scenario_id = str(metadata["sequence_id"][row])
        records.append(
            {
                "probe_id": f"{scenario_id}:longitudinal:{int(ids[stimulus])}:{int(ids[response])}",
                "scenario_row": row,
                "scenario_id": scenario_id,
                "recording_id": recording,
                "split_index": int(metadata["split_index"][row]),
                "stimulus_agent_index": stimulus,
                "response_agent_index": response,
                "stimulus_agent_id": int(ids[stimulus]),
                "response_agent_id": int(ids[response]),
                "initial_gap_m": gap,
                "stimulus_speed_mps": stimulus_speed,
                "response_speed_mps": response_speed,
                "selection_rank": "closest_qualifying_pair",
            }
        )
    frame = (
        pd.DataFrame(records)
        .sort_values(["split_index", "recording_id", "scenario_id"])
        .reset_index(drop=True)
    )
    output = Path(config["paths"]["output_dir"])
    frame.to_csv(output / "probe_manifest.csv", index=False)
    report = {
        "probe_manifest_version": "npc_interaction_d3_longitudinal_v1",
        "anchors": int(len(frame)),
        "counts": {
            name: int((frame.split_index == split).sum())
            for split, name in enumerate(("train", "validation", "test"))
        },
        "selection": "one closest same-lane leader/NPC-follower pair per prefix",
        "prefix_only": True,
        "gap_bounds_m": [5.0, 80.0],
        "minimum_response_speed_mps": 5.0,
        "minimum_stimulus_speed_mps": 3.0,
        "exclusions": exclusion,
        "stimuli": {
            "brake_mps2": [1.5, 3.0, 5.0],
            "speed_recovery_mps2": [0.5, 1.0, 2.0],
        },
        "pulse_duration_s": 1.0,
        "horizon_s": 3.0,
    }
    (output / "probe_manifest_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def mine_lateral_probe_manifest(config_path: str | Path) -> dict[str, Any]:
    """Mine prefix-only merge and cut-out anchors for D3 lateral probes."""
    config, _ = load_benchmark_config(config_path)
    arrays, metadata, _ = load_causal_cache(config_path)
    records: list[dict[str, Any]] = []
    exclusion = {"fewer_than_two_lanes": 0, "no_merge_pair": 0, "no_cut_out_pair": 0}
    for row in range(len(metadata["sequence_id"])):
        state = np.asarray(arrays["agent_states"][row, 24])
        active = np.asarray(arrays["agent_valid"][row, :25]).all(0) & (
            metadata["agent_ids"][row] >= 0
        )
        centers = _ordered_lane_centers(
            np.asarray(arrays["map_polylines"][row]),
            np.asarray(arrays["map_polyline_valid"][row]),
        )
        if len(centers) < 2:
            exclusion["fewer_than_two_lanes"] += 1
            continue
        lanes = np.abs(state[:, 1, None] - centers[None]).argmin(-1)
        ids = metadata["agent_ids"][row]
        merge_candidates = []
        cut_out_candidates = []
        for response in np.flatnonzero(active & (np.arange(7) > 0)):
            response_speed = float(np.linalg.norm(state[response, 2:4]))
            if response_speed < 5.0:
                continue
            for stimulus in np.flatnonzero(active & (np.arange(7) != response)):
                stimulus_speed = float(np.linalg.norm(state[stimulus, 2:4]))
                if stimulus_speed < 3.0:
                    continue
                gap = (
                    state[stimulus, 0]
                    - state[response, 0]
                    - 0.5
                    * (
                        metadata["lengths_m"][row, stimulus]
                        + metadata["lengths_m"][row, response]
                    )
                )
                lane_difference = int(lanes[stimulus]) - int(lanes[response])
                merge_shift = abs(
                    float(centers[int(lanes[response])] - state[stimulus, 1])
                )
                if (
                    abs(lane_difference) == 1
                    and 2.0 <= merge_shift <= 5.0
                    and 2.0 <= gap <= 50.0
                ):
                    merge_candidates.append(
                        (
                            abs(float(gap) - 15.0),
                            float(gap),
                            stimulus,
                            response,
                            int(lanes[response]),
                        )
                    )
                if lane_difference == 0 and 5.0 <= gap <= 50.0:
                    for target_lane in (
                        int(lanes[stimulus]) - 1,
                        int(lanes[stimulus]) + 1,
                    ):
                        if not 0 <= target_lane < len(centers):
                            continue
                        target_shift = abs(
                            float(centers[target_lane] - state[stimulus, 1])
                        )
                        if not 2.0 <= target_shift <= 5.0:
                            continue
                        occupants = active & (lanes == target_lane)
                        clearance = np.abs(state[:, 0] - state[stimulus, 0])
                        clearance[~occupants] = np.inf
                        minimum_clearance = (
                            float(clearance.min()) if occupants.any() else 100.0
                        )
                        if minimum_clearance >= 12.0:
                            cut_out_candidates.append(
                                (
                                    abs(float(gap) - 15.0),
                                    -minimum_clearance,
                                    float(gap),
                                    stimulus,
                                    response,
                                    target_lane,
                                )
                            )
        scenario_id = str(metadata["sequence_id"][row])
        common = {
            "scenario_row": row,
            "scenario_id": scenario_id,
            "recording_id": int(metadata["recording_id"][row]),
            "split_index": int(metadata["split_index"][row]),
        }
        if merge_candidates:
            _, gap, stimulus, response, target_lane = min(merge_candidates)
            records.append(
                {
                    **common,
                    "probe_id": f"{scenario_id}:merge:{int(ids[stimulus])}:{int(ids[response])}",
                    "stimulus_family": "merge",
                    "stimulus_agent_index": stimulus,
                    "response_agent_index": response,
                    "stimulus_agent_id": int(ids[stimulus]),
                    "response_agent_id": int(ids[response]),
                    "initial_gap_m": gap,
                    "source_lane_index": int(lanes[stimulus]),
                    "target_lane_index": target_lane,
                    "target_lane_center_y_m": float(centers[target_lane]),
                    "lateral_direction": (
                        "left" if centers[target_lane] > state[stimulus, 1] else "right"
                    ),
                    "selection_rank": "gap_nearest_15m",
                }
            )
        else:
            exclusion["no_merge_pair"] += 1
        if cut_out_candidates:
            _, negative_clearance, gap, stimulus, response, target_lane = min(
                cut_out_candidates
            )
            records.append(
                {
                    **common,
                    "probe_id": f"{scenario_id}:cut_out:{int(ids[stimulus])}:{int(ids[response])}",
                    "stimulus_family": "cut_out",
                    "stimulus_agent_index": stimulus,
                    "response_agent_index": response,
                    "stimulus_agent_id": int(ids[stimulus]),
                    "response_agent_id": int(ids[response]),
                    "initial_gap_m": gap,
                    "source_lane_index": int(lanes[stimulus]),
                    "target_lane_index": target_lane,
                    "target_lane_center_y_m": float(centers[target_lane]),
                    "lateral_direction": (
                        "left" if centers[target_lane] > state[stimulus, 1] else "right"
                    ),
                    "target_lane_initial_clearance_m": -negative_clearance,
                    "selection_rank": "gap_nearest_15m_then_clearance",
                }
            )
        else:
            exclusion["no_cut_out_pair"] += 1
    frame = (
        pd.DataFrame(records)
        .sort_values(["split_index", "recording_id", "scenario_id", "stimulus_family"])
        .reset_index(drop=True)
    )
    # Keep reasonable interaction probes in D3; trajectories that force the
    # evaluator-owned stimulus into another constant-velocity vehicle belong
    # to the adversarial T4 queue instead.
    from interactive_behavior_world_model.evaluation.geometry import collision_matrix
    from interactive_behavior_world_model.evaluation.stimuli import lateral_stimulus_trajectories

    feasible = np.ones(len(frame), bool)
    for begin in range(0, len(frame), 512):
        batch = frame.iloc[begin : begin + 512]
        rows = batch.scenario_row.to_numpy(np.int64)
        initial = np.asarray(arrays["agent_states"][rows, 24])
        valid = np.asarray(arrays["agent_valid"][rows, 24])
        stimulus = batch.stimulus_agent_index.to_numpy(np.int64)
        _, changed = lateral_stimulus_trajectories(
            initial,
            valid,
            stimulus,
            batch.target_lane_center_y_m.to_numpy(np.float32),
            np.full(len(batch), 1.2, np.float32),
        )
        future_valid = np.broadcast_to(valid[:, None], changed.shape[:-1])
        lengths = metadata["lengths_m"][rows]
        widths = metadata["widths_m"][rows]
        initial_collision = collision_matrix(initial, valid, lengths, widths)
        collision = collision_matrix(
            changed, future_valid, lengths[:, None], widths[:, None]
        )
        new_collision = collision & ~initial_collision[:, None]
        for local, stimulus_index in enumerate(stimulus):
            feasible[begin + local] = not bool(
                new_collision[local, :, stimulus_index].any()
            )
    # D3 measures response to a feasible stimulus. A stimulus that already
    # strikes a constant-velocity neighbour is excluded from its denominator.
    collision_excluded = (
        frame.loc[~feasible].groupby("stimulus_family").size().to_dict()
    )
    frame = frame.loc[feasible].reset_index(drop=True)
    output = Path(config["paths"]["output_dir"])
    frame.to_csv(output / "lateral_probe_manifest.csv", index=False)
    report = {
        "probe_manifest_version": "npc_interaction_d3_lateral_v1",
        "anchors": int(len(frame)),
        "counts": {
            name: int((frame.split_index == split).sum())
            for split, name in enumerate(("train", "validation", "test"))
        },
        "family_counts": {
            str(key): int(value)
            for key, value in frame.groupby(["split_index", "stimulus_family"])
            .size()
            .items()
        },
        "selection": "at_most_one_prefix_only_anchor_per_scene_and_family",
        "prefix_only": True,
        "merge_gap_bounds_m": [2.0, 50.0],
        "cut_out_gap_bounds_m": [5.0, 50.0],
        "target_lateral_shift_bounds_m": [2.0, 5.0],
        "cut_out_target_lane_clearance_m": 12.0,
        "exclusions": exclusion,
        "strong_dose_constant_velocity_collision_exclusions": {
            str(key): int(value) for key, value in collision_excluded.items()
        },
        "horizon_s": 3.0,
    }
    (output / "lateral_probe_manifest_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report
