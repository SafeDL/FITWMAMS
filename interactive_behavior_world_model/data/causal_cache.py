"""Materialize real 25-frame histories and 149-frame futures from raw highD."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from process_highD.src.io_utils import load_config
from process_highD.src.natural_segments import _build_vehicle_cache
from process_highD.src.preprocess import prepare_recording
from world_model.src.core.data import _action_window, _state_window
from world_model.src.traffic_graph.highd_adapter import HighDGraphAdapter

from .manifest import SLOT_NAMES, load_benchmark_config

ARRAY_SPECS = {
    "agent_states": (np.float32, (174, 7, 6)),
    "agent_valid": (bool, (174, 7)),
    "target_actions_highd": (np.float32, (149, 6, 2)),
    "target_controls": (np.float32, (149, 6, 2)),
    "map_polylines": (np.float32, (8, 8, 6)),
    "map_polyline_valid": (bool, (8, 8)),
    "lane_graph_edges": (np.int64, (14, 3)),
}


def _universal_controls(actions: np.ndarray, states: np.ndarray) -> np.ndarray:
    speed = np.linalg.norm(states[..., 2:4], axis=-1).clip(0.5)
    heading = np.arctan2(
        states[..., 3],
        np.where(np.abs(states[..., 2]) < 1.0e-4, 1.0e-4, states[..., 2]),
    )
    ax, ay = actions[..., 0], actions[..., 1]
    longitudinal = ax * np.cos(heading) + ay * np.sin(heading)
    lateral = -ax * np.sin(heading) + ay * np.cos(heading)
    return np.stack((longitudinal, lateral / speed), axis=-1).astype(np.float32)


def _open_arrays(root: Path, count: int) -> dict[str, np.memmap]:
    root.mkdir(parents=True, exist_ok=False)
    arrays = {}
    for name, (dtype, tail) in ARRAY_SPECS.items():
        fill = -1 if name == "lane_graph_edges" else 0
        array = np.lib.format.open_memmap(
            root / f"{name}.npy", mode="w+", dtype=dtype, shape=(count, *tail)
        )
        array[...] = fill
        arrays[name] = array
    return arrays


def build_causal_cache(config_path: str | Path) -> dict[str, Any]:
    config, _ = load_benchmark_config(config_path)
    output = Path(config["paths"]["output_dir"])
    scenario_path = output / "scenario_manifest.npz"
    dataset_report_path = output / "dataset_manifest.json"
    if not scenario_path.exists() or not dataset_report_path.exists():
        raise FileNotFoundError(
            "run interactive_behavior_world_model.scripts.prepare before building "
            "the causal cache"
        )
    final_root = output / "data" / "causal_cache"
    final_manifest = final_root / "manifest.json"
    if final_manifest.exists():
        return json.loads(final_manifest.read_text())
    building_root = output / "data" / "causal_cache.building"
    if building_root.exists():
        raise RuntimeError(
            f"partial cache exists at {building_root}; inspect it before retrying"
        )

    with np.load(scenario_path, allow_pickle=False) as manifest_npz:
        scenario = {name: np.asarray(manifest_npz[name]) for name in manifest_npz.files}
    count = len(scenario["source_row"])
    arrays = _open_arrays(building_root, count)
    cache = Path(config["paths"]["canonical_cache"])
    legacy_evt = np.load(cache / "is_evt_tail.npy", mmap_mode="r", allow_pickle=False)
    source = pd.read_csv(config["paths"]["natural_segments_csv"])
    source["segment_id"] = source["segment_id"].astype(str)
    source = source.set_index("segment_id", drop=False)
    raw_dir = Path(config["paths"]["raw_highd_dir"])
    highd_config_path = (
        Path(__file__).resolve().parents[2]
        / "process_highD/scripts/configs/highd_natural_evt.yaml"
    )
    highd_config = load_config(str(highd_config_path))
    adapter = HighDGraphAdapter()
    kept: list[int] = []
    rejected_abnormal: list[str] = []
    cursor = 0

    for recording_id in sorted(np.unique(scenario["recording_id"]).tolist()):
        recording = prepare_recording(raw_dir, int(recording_id), highd_config)
        vehicles = _build_vehicle_cache(recording)
        local_rows = np.flatnonzero(scenario["recording_id"] == int(recording_id))
        for manifest_row in local_rows:
            metadata = source.loc[str(scenario["sequence_id"][manifest_row])]
            slot_ids = np.asarray(
                [int(metadata[f"{slot}_id"]) for slot in SLOT_NAMES], np.int64
            )
            start = int(scenario["history_start_frame"][manifest_row])
            decision = int(scenario["decision_frame"][manifest_row])
            state_window = _state_window(
                vehicles,
                ego_id=int(metadata["ego_id"]),
                slot_ids=slot_ids,
                start_frame=start,
                steps=174,
                origin_frame=decision,
            )
            if state_window is None:
                rejected_abnormal.append(str(metadata["segment_id"]))
                continue
            states, valid = state_window
            active = scenario["agent_ids"][manifest_row] >= 0
            if not np.all(valid[:, active]):
                rejected_abnormal.append(str(metadata["segment_id"]))
                continue
            actions, action_valid = _action_window(
                vehicles,
                slot_ids=slot_ids,
                start_frame=decision,
                steps=149,
            )
            if not np.all(action_valid[:, active[1:]]):
                rejected_abnormal.append(str(metadata["segment_id"]))
                continue
            ego = vehicles[int(metadata["ego_id"])]
            ego_position = (
                decision - int(ego["initial"])
                if ego["continuous"]
                else ego["frame_to_pos"][decision]
            )
            lateral_sign = 1.0 if int(ego["direction"]) == 1 else -1.0
            polylines, poly_valid, lane_edges = adapter.map_from_recording_metadata(
                recording.recording_meta,
                ego_global_y_m=float(ego["y_left"][ego_position]),
                lateral_sign=lateral_sign,
            )
            arrays["agent_states"][cursor] = states
            arrays["agent_valid"][cursor] = valid
            arrays["target_actions_highd"][cursor] = actions
            arrays["target_controls"][cursor] = _universal_controls(
                actions, states[24:-1, 1:]
            )
            lane_count = min(8, len(polylines))
            arrays["map_polylines"][cursor, :lane_count] = polylines[:lane_count]
            arrays["map_polyline_valid"][cursor, :lane_count] = poly_valid[:lane_count]
            edge_count = min(14, len(lane_edges))
            arrays["lane_graph_edges"][cursor, :edge_count] = lane_edges[:edge_count]
            kept.append(int(manifest_row))
            cursor += 1

    for array in arrays.values():
        array.flush()
    kept_index = np.asarray(kept, np.int64)
    aligned = {name: values[kept_index] for name, values in scenario.items()}
    aligned["is_evt_tail"] = np.asarray(legacy_evt)[aligned["source_row"]]
    np.savez_compressed(building_root / "scenario_metadata.npz", **aligned)
    report = {
        "cache_format": "npc_interaction_causal174_v1",
        "allocated_rows": int(count),
        "num_scenarios": int(cursor),
        "unused_trailing_rows": int(count - cursor),
        "history_frames": 25,
        "horizon_frames": 149,
        "decision_index": 24,
        "rejected_abnormal_or_incomplete": len(rejected_abnormal),
        "rejected_scenario_ids": rejected_abnormal,
        "arrays": {
            name: {"dtype": np.dtype(dtype).name, "shape": [int(count), *tail]}
            for name, (dtype, tail) in ARRAY_SPECS.items()
        },
        "coordinate_system": "ego-centered at decision frame; longitudinal forward; lateral left",
        "state_features": [
            "x_m",
            "y_left_m",
            "vx_mps",
            "vy_left_mps",
            "ax_mps2",
            "ay_left_mps2",
        ],
        "control_features": ["acceleration_mps2", "yaw_rate_rps"],
    }
    (building_root / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    final_root.parent.mkdir(parents=True, exist_ok=True)
    building_root.rename(final_root)
    return report


def load_causal_cache(
    config_path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    config, _ = load_benchmark_config(config_path)
    root = Path(config["paths"]["output_dir"]) / "data" / "causal_cache"
    report = json.loads((root / "manifest.json").read_text())
    count = int(report["num_scenarios"])
    arrays = {
        name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)[:count]
        for name in ARRAY_SPECS
    }
    with np.load(root / "scenario_metadata.npz", allow_pickle=False) as data:
        metadata = {name: np.asarray(data[name]) for name in data.files}
    return arrays, metadata, report
