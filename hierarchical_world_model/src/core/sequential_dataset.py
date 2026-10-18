"""Shared highD cache: 150 recorded states, real road geometry and vehicle sizes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from normalizing_flow.src.features import SLOT_NAMES
from process_highD.src.lane_utils import parse_lane_markings
from process_highD.src.natural_segments import (
    _build_vehicle_cache,
    _position_at,
    _slice_range,
    _values,
)
from process_highD.src.preprocess import prepare_recording

from .schema import AGENT_NAMES, AGENT_STATE_FEATURES
from .utils import ensure_dir, load_json, save_json

logger = logging.getLogger(__name__)
SPLIT_TO_INDEX = {"train": 0, "val": 1, "test": 2}


def cohort_manifest(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Summarize the recording-level split shared by all learned components."""
    split = np.asarray(arrays["split_index"], np.int64)
    return {
        "cohort": "canonical_cleaned_full_horizon_background_slots",
        "canonical_sequences": int(len(arrays["agent_valid"])),
        "background_slot_stability_required": True,
        "splits": {
            name: {"sequences": int(np.sum(split == index))}
            for name, index in SPLIT_TO_INDEX.items()
        },
    }


def recording_lane_map(
    recording_meta: dict,
    *,
    ego_global_y_m: float,
    lateral_sign: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Express measured lane geometry in the scene's initial ego coordinates."""
    lane_info = parse_lane_markings(recording_meta)
    directions = (lane_info["direction_1_lanes"], lane_info["direction_2_lanes"])
    lane_ids = (directions[0] + directions[1])[:8]
    polylines = np.zeros((len(lane_ids), 8, 6), dtype=np.float32)
    valid = np.ones((len(lane_ids), 8), dtype=bool)
    x = np.linspace(-120.0, 120.0, 8)
    index_by_id = {lane_id: index for index, lane_id in enumerate(lane_ids)}
    for index, lane_id in enumerate(lane_ids):
        item = lane_info["lanes"][lane_id]
        polylines[index, :, 0] = x
        polylines[index, :, 1] = lateral_sign * item["center"] - ego_global_y_m
        polylines[index, :, 2] = 1.0
        polylines[index, :, 4] = item["width"]
        polylines[index, :, 5] = 1.0
    edges = []
    for direction in directions:
        visible = [lane for lane in direction if lane in index_by_id]
        for left, right in zip(visible, visible[1:]):
            a, b = index_by_id[left], index_by_id[right]
            edges.extend(((a, b, 1), (b, a, 1)))
    return polylines, valid, np.asarray(edges, np.int64).reshape(-1, 3)


def _segment_slot_ids(row: pd.Series) -> np.ndarray:
    return np.asarray(
        [int(row.get(f"{slot}_id", -1)) for slot in SLOT_NAMES], dtype=np.int64
    )


def _vehicle_sequence(
    vehicle: dict[str, Any],
    *,
    start_frame: int,
    steps: int,
    origin_x: float,
    origin_y_left: float,
) -> tuple[np.ndarray, np.ndarray]:
    selector, present = _slice_range(vehicle, int(start_frame), int(steps))
    if isinstance(selector, slice) and bool(np.all(present)):
        abnormal = np.asarray(vehicle["abnormal"][selector], dtype=bool)
    else:
        abnormal = np.zeros(int(steps), dtype=bool)
        if np.any(present):
            abnormal[present] = np.asarray(vehicle["abnormal"])[
                np.asarray(selector)[present]
            ]
    valid = np.asarray(present, dtype=bool) & ~abnormal
    state = np.zeros((int(steps), len(AGENT_STATE_FEATURES)), dtype=np.float32)
    if not np.any(valid):
        return state, valid
    state[:, 0] = _values(vehicle, "x", selector, valid) - float(origin_x)
    state[:, 1] = _values(vehicle, "y_left", selector, valid) - float(origin_y_left)
    state[:, 2] = _values(vehicle, "vx", selector, valid)
    state[:, 3] = _values(vehicle, "vy_left", selector, valid)
    state[:, 4] = _values(vehicle, "ax", selector, valid)
    state[:, 5] = _values(vehicle, "ay_left", selector, valid)
    state[~valid] = 0.0
    return state.astype(np.float32), valid


def _origin_for(
    vehicles: dict[int, dict[str, Any]], ego_id: int, frame: int
) -> tuple[float, float] | None:
    ego = vehicles.get(int(ego_id))
    if ego is None:
        return None
    pos = _position_at(ego, int(frame))
    if pos is None or bool(ego["abnormal"][pos]):
        return None
    return float(ego["x"][pos]), float(ego["y_left"][pos])


def _state_window(
    vehicles: dict[int, dict[str, Any]],
    *,
    ego_id: int,
    slot_ids: np.ndarray,
    start_frame: int,
    steps: int,
    origin_frame: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    origin = _origin_for(vehicles, int(ego_id), int(origin_frame))
    if origin is None:
        return None
    origin_x, origin_y_left = origin
    states = np.zeros(
        (int(steps), len(AGENT_NAMES), len(AGENT_STATE_FEATURES)),
        dtype=np.float32,
    )
    valid = np.zeros((int(steps), len(AGENT_NAMES)), dtype=bool)
    ids = [int(ego_id), *[int(value) for value in np.asarray(slot_ids, dtype=np.int64)]]
    for agent_idx, vehicle_id in enumerate(ids):
        if vehicle_id < 0 or vehicle_id not in vehicles:
            continue
        seq, seq_valid = _vehicle_sequence(
            vehicles[vehicle_id],
            start_frame=int(start_frame),
            steps=int(steps),
            origin_x=origin_x,
            origin_y_left=origin_y_left,
        )
        states[:, agent_idx, :] = seq
        valid[:, agent_idx] = seq_valid
    return states, valid


def _action_window(
    vehicles: dict[int, dict[str, Any]],
    *,
    slot_ids: np.ndarray,
    start_frame: int,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    actions = np.zeros((int(steps), len(SLOT_NAMES), 2), dtype=np.float32)
    valid = np.zeros((int(steps), len(SLOT_NAMES)), dtype=bool)
    for slot_idx, vehicle_id in enumerate(np.asarray(slot_ids, dtype=np.int64)):
        if int(vehicle_id) < 0 or int(vehicle_id) not in vehicles:
            continue
        vehicle = vehicles[int(vehicle_id)]
        selector, present = _slice_range(vehicle, int(start_frame), int(steps))
        if isinstance(selector, slice) and bool(np.all(present)):
            abnormal = np.asarray(vehicle["abnormal"][selector], dtype=bool)
        else:
            abnormal = np.zeros(int(steps), dtype=bool)
            if np.any(present):
                abnormal[present] = np.asarray(vehicle["abnormal"])[
                    np.asarray(selector)[present]
                ]
        action_valid = np.asarray(present, dtype=bool) & ~abnormal
        valid[:, slot_idx] = action_valid
        actions[:, slot_idx, 0] = _values(vehicle, "ax", selector, action_valid)
        actions[:, slot_idx, 1] = _values(vehicle, "ay_left", selector, action_valid)
        actions[~action_valid, slot_idx, :] = 0.0
    return actions, valid


# QR's canonical cache retains all 150 *recorded* points S0..S149 rather than
# synthesising an S150 endpoint.  The baseline signature is only retained for
# the independently maintained baseline models that still consume that cache.
CANONICAL_SEQUENCE_CACHE_FORMAT = "highd_canonical_raw150"
CANONICAL_SEQUENCE_PROTOCOL = "fixed_horizon_5p96"
SEQUENCE_ARRAYS = (
    "sequence_id",
    "agent_states",
    "agent_valid",
    "ego_index",
    "map_polylines",
    "map_polyline_valid",
    "lane_graph_edges",
    "vehicle_dimensions",
    "actions_highd",
    "split_index",
    "is_evt_tail",
)
# The first 24 cache positions are an invalid compatibility prefix.  Position
# 24 is C0/S0; the following 149 entries are the remaining recorded states.
HISTORY_PADDING_FRAMES = 24
HISTORY_FRAMES = HISTORY_PADDING_FRAMES + 1
RAW_WINDOW_STATE_FRAMES = 150
FUTURE_TRANSITION_FRAMES = RAW_WINDOW_STATE_FRAMES - 1
START_RECONSTRUCTION_FRAMES = 25
ROLL_TRANSITION_FRAMES = FUTURE_TRANSITION_FRAMES - START_RECONSTRUCTION_FRAMES
SEQUENCE_FRAMES = HISTORY_PADDING_FRAMES + RAW_WINDOW_STATE_FRAMES


def sequence_cache_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "sequence_cache"


def sequence_manifest_path(output_dir: str | Path) -> Path:
    return sequence_cache_dir(output_dir) / "manifest.json"


def sequence_cache_owner_dir(config: dict[str, Any], *, config_dir: Path) -> Path:
    """Resolve the directory that owns a reusable sequence cache.

    Fine-tuning experiments can write checkpoints to a new output directory
    while reusing an immutable, already audited sequence cache.  This avoids
    rebuilding data and prevents a trial run from overwriting the cache used by
    an earlier checkpoint.
    """
    paths = config["paths"]
    value = paths.get("sequence_cache_dir") or paths["output_dir"]
    owner = Path(value)
    return owner if owner.is_absolute() else (config_dir / owner).resolve()


def sequence_cache_available(output_dir: str | Path) -> bool:
    root = sequence_cache_dir(output_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists() or not all(
        (root / f"{key}.npy").exists() for key in SEQUENCE_ARRAYS
    ):
        return False
    try:
        manifest = load_json(manifest_path)
        return manifest.get("cache_format") == CANONICAL_SEQUENCE_CACHE_FORMAT
    except (OSError, ValueError):
        return False


def is_canonical_sequence_manifest(manifest: dict[str, Any]) -> bool:
    """Whether a manifest fulfils the canonical non-fabricated contract."""
    return (
        manifest.get("cache_format") == CANONICAL_SEQUENCE_CACHE_FORMAT
        and int(manifest.get("future_transition_frames", -1))
        == FUTURE_TRANSITION_FRAMES
        and int(manifest.get("start_reconstruction_frames", -1))
        == START_RECONSTRUCTION_FRAMES
        and int(manifest.get("roll_transition_frames", -1)) == ROLL_TRANSITION_FRAMES
        and bool(manifest.get("lateral_event_integrity_required", False))
        and bool(manifest.get("background_slot_stability_required", False))
    )


def _require_stable_background_slots(valid: np.ndarray) -> None:
    """Assert that the canonical cache has no C0-active slot exits."""
    values = np.asarray(valid, bool)
    active = values[:, HISTORY_PADDING_FRAMES, 1:]
    full = values[:, HISTORY_PADDING_FRAMES:, 1:].all(axis=1)
    stable = (~active | full).all(axis=1)
    unstable = np.flatnonzero(~stable)
    if len(unstable):
        raise RuntimeError(
            f"sequence cache contains {len(unstable)} C0-active background-slot exits"
        )


def prepare_sequential_dataset(
    config: dict[str, Any],
    *,
    config_dir: Path,
    rebuild: bool = False,
    max_sequences: int | None = None,
) -> dict[str, Any]:
    """Materialize one raw sequence for every shared natural-driving row."""
    if config["dataset"]["sequence_protocol"] != CANONICAL_SEQUENCE_PROTOCOL:
        raise ValueError(
            f"dataset.sequence_protocol must be {CANONICAL_SEQUENCE_PROTOCOL!r}"
        )
    paths = config["paths"]
    flow_schema_path = Path(paths["flow_schema"])
    if not flow_schema_path.is_absolute():
        flow_schema_path = (config_dir / flow_schema_path).resolve()
    flow_schema = load_json(flow_schema_path)
    flow_dataset = Path(flow_schema["dataset_npz"])
    if not flow_dataset.exists():
        raise FileNotFoundError(f"canonical Flow dataset is missing: {flow_dataset}")
    output_dir = sequence_cache_owner_dir(config, config_dir=config_dir)
    root = sequence_cache_dir(output_dir)
    if sequence_cache_available(output_dir) and not rebuild:
        manifest = load_json(sequence_manifest_path(output_dir))
        if is_canonical_sequence_manifest(manifest):
            return manifest
        raise RuntimeError(
            f"canonical sequences require cache format {CANONICAL_SEQUENCE_CACHE_FORMAT}, but {root} contains "
            f"{manifest.get('cache_format')!r}; rebuild it explicitly."
        )
    if root.exists():
        if not rebuild:
            raise RuntimeError(
                f"Outdated or partial sequence cache at {root}; rebuild it explicitly "
                "before tracking/NPC training/evaluation."
            )
        import shutil

        shutil.rmtree(root)
    ensure_dir(root)
    with np.load(flow_dataset, allow_pickle=False) as source:
        arrays = {
            name: np.asarray(source[name]).copy()
            for name in (
                "segment_id",
                "recording_id",
                "ego_id",
                "anchor_frame",
                "split_index",
                "is_evt_tail",
            )
        }
    dataset_cfg = config.get("dataset", {})
    selected_max = int(
        max_sequences
        if max_sequences is not None
        else dataset_cfg.get("max_sequences", 0) or 0
    )
    rows = np.arange(len(arrays["segment_id"]), dtype=np.int64)
    if selected_max > 0:
        rng = np.random.default_rng(int(config.get("split", {}).get("seed", 42)))
        rng.shuffle(rows)
        rows = rows[:selected_max]
    if len(rows) == 0:
        raise RuntimeError("No source sequences selected")
    recording_cache: dict[int, tuple[Any, dict[int, dict[str, Any]]]] = {}
    highd_cfg: dict[str, Any] | None = None
    raw_dir = Path(flow_schema.get("raw_dir", ""))
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"sequence construction requires raw highD data: {raw_dir}"
        )
    from process_highD.src.io_utils import load_config as load_highd_config

    highd_config_value = paths.get("highd_preprocessing_config")
    if not highd_config_value:
        raise KeyError(
            "paths.highd_preprocessing_config is required to construct "
            "the raw START+ROLL sequence cache"
        )
    highd_config_path = Path(highd_config_value)
    if not highd_config_path.is_absolute():
        highd_config_path = (config_dir / highd_config_path).resolve()
    highd_cfg = load_highd_config(str(highd_config_path))
    natural_csv = Path(str(flow_schema.get("source_segments_csv", "")))
    if not natural_csv.exists():
        raise FileNotFoundError(
            f"sequence construction requires natural segment metadata: {natural_csv}"
        )
    natural_rows = pd.read_csv(natural_csv).set_index("segment_id", drop=False)

    def _recording_data(recording_id: int) -> tuple[Any, dict[int, dict[str, Any]]]:
        cached = recording_cache.get(int(recording_id))
        if cached is None:
            recording = prepare_recording(raw_dir, int(recording_id), highd_cfg)
            cached = (recording, _build_vehicle_cache(recording))
            recording_cache[int(recording_id)] = cached
        return cached

    def _recording_map(recording_id: int, ego_id: int, anchor_frame: int):
        recording, vehicles = _recording_data(int(recording_id))
        vehicle = vehicles[int(ego_id)]
        position = _position_at(vehicle, int(anchor_frame))
        if position is None:
            raise ValueError(
                f"missing ego anchor: {recording_id}/{ego_id}/{anchor_frame}"
            )
        lateral_sign = 1.0 if int(vehicle.get("direction", 0)) == 1 else -1.0
        return recording_lane_map(
            recording.recording_meta,
            ego_global_y_m=float(vehicle["y_left"][position]),
            lateral_sign=lateral_sign,
        )

    s = len(rows)
    # History padding is invalid; S0..S149 and all 149 transitions are recorded.
    t, n, m, p, action_t = (
        SEQUENCE_FRAMES,
        7,
        8,
        8,
        FUTURE_TRANSITION_FRAMES,
    )
    output: dict[str, np.ndarray] = {
        "sequence_id": np.empty(s, dtype="U96"),
        "agent_states": np.zeros((s, t, n, 6), np.float32),
        "agent_valid": np.zeros((s, t, n), bool),
        "ego_index": np.zeros(s, np.int64),
        "map_polylines": np.zeros((s, m, p, 6), np.float32),
        "map_polyline_valid": np.zeros((s, m, p), bool),
        "lane_graph_edges": np.full((s, max(1, 2 * (m - 1)), 3), -1, np.int64),
        "vehicle_dimensions": np.broadcast_to(
            np.asarray((4.8, 1.8), np.float32), (s, n, 2)
        ).copy(),
        "actions_highd": np.zeros((s, action_t, n - 1, 2), np.float32),
        "split_index": np.zeros(s, np.int64),
        "is_evt_tail": np.zeros(s, bool),
    }
    for out_i, row_index in enumerate(rows):
        start = int(row_index)
        segment_id = str(arrays["segment_id"][start])
        if segment_id not in natural_rows.index:
            raise KeyError(
                f"Natural metadata is missing sequence segment_id={segment_id}"
            )
        natural = natural_rows.loc[segment_id]
        if isinstance(natural, pd.DataFrame):
            natural = natural.iloc[0]
        recording_id = int(arrays["recording_id"][start])
        ego_id = int(arrays["ego_id"][start])
        anchor_frame = int(arrays["anchor_frame"][start])
        recording, vehicles = _recording_data(recording_id)
        slot_ids = _segment_slot_ids(natural)
        vehicle_ids = np.concatenate((np.asarray((ego_id,), np.int64), slot_ids))
        for slot_index, vehicle_id in enumerate(vehicle_ids):
            if int(vehicle_id) < 0:
                continue
            vehicle = vehicles.get(int(vehicle_id))
            if vehicle is None:
                raise KeyError(
                    f"vehicle {vehicle_id} for {segment_id} is absent from recording metadata"
                )
            length = float(vehicle["length"])
            width = float(vehicle["lat_width"])
            if (
                not np.isfinite(length)
                or not np.isfinite(width)
                or min(length, width) <= 0.0
            ):
                raise ValueError(
                    f"invalid dimensions for vehicle {vehicle_id} in {segment_id}"
                )
            output["vehicle_dimensions"][out_i, slot_index] = (length, width)
        raw_window = _state_window(
            vehicles,
            ego_id=ego_id,
            slot_ids=slot_ids,
            start_frame=anchor_frame,
            steps=RAW_WINDOW_STATE_FRAMES,
            origin_frame=anchor_frame,
        )
        if raw_window is None:
            raise RuntimeError(f"Could not construct raw highD window for {segment_id}")
        raw_states, raw_valid = raw_window
        highd_actions, action_valid = _action_window(
            vehicles,
            slot_ids=slot_ids,
            start_frame=anchor_frame,
            steps=FUTURE_TRANSITION_FRAMES,
        )
        # A control is supervised only when its next background state exists.
        highd_actions[~(action_valid & raw_valid[1:, 1:])] = 0.0
        states = np.zeros((t, n, 6), dtype=np.float32)
        valid = np.zeros((t, n), dtype=bool)
        states[HISTORY_PADDING_FRAMES:] = raw_states
        valid[HISTORY_PADDING_FRAMES:] = raw_valid
        polylines, map_valid, lane_edges = _recording_map(
            recording_id, ego_id, anchor_frame
        )
        output["sequence_id"][out_i] = segment_id
        output["agent_states"][out_i] = states
        output["agent_valid"][out_i] = valid
        lm = min(m, len(polylines))
        output["map_polylines"][out_i, :lm] = polylines[:lm]
        output["map_polyline_valid"][out_i, :lm] = map_valid[:lm]
        le = min(output["lane_graph_edges"].shape[1], len(lane_edges))
        if le:
            output["lane_graph_edges"][out_i, :le] = lane_edges[:le]
        output["actions_highd"][out_i] = highd_actions
        output["split_index"][out_i] = int(arrays["split_index"][start])
        output["is_evt_tail"][out_i] = bool(arrays["is_evt_tail"][start])
        if (out_i + 1) % 1000 == 0 or out_i + 1 == s:
            logger.info("Prepared highD sequence %d/%d", out_i + 1, s)
    _require_stable_background_slots(output["agent_valid"])
    for key, value in output.items():
        np.save(root / f"{key}.npy", value, allow_pickle=False)
    manifest = {
        "cache_format": CANONICAL_SEQUENCE_CACHE_FORMAT,
        "num_sequences": int(s),
        "frames": t,
        "history_frames": HISTORY_FRAMES,
        "raw_window_state_frames": RAW_WINDOW_STATE_FRAMES,
        "future_transition_frames": FUTURE_TRANSITION_FRAMES,
        "start_reconstruction_frames": START_RECONSTRUCTION_FRAMES,
        "roll_transition_frames": ROLL_TRANSITION_FRAMES,
        "start_reconstruction_seconds": START_RECONSTRUCTION_FRAMES / 25.0,
        "roll_seconds": ROLL_TRANSITION_FRAMES / 25.0,
        "total_rollout_seconds": FUTURE_TRANSITION_FRAMES / 25.0,
        "start_semantics": "segment_start_behavior_reconstruction_not_risk_event_onset",
        "fps": 25.0,
        "source_dataset": str(natural_csv),
        "flow_dataset": str(flow_dataset),
        "flow_schema": str(flow_schema_path),
        "adapter": "highd_straight_lane",
        "lateral_event_integrity_required": bool(
            flow_schema.get("lateral_event_integrity_required", False)
        ),
        "background_slot_stability_required": bool(
            flow_schema.get("background_slot_stability_required", False)
        ),
        "uses_recording_lane_metadata": True,
        "arrays": list(SEQUENCE_ARRAYS),
        "split_summary": {
            name: int(np.sum(output["split_index"] == value))
            for name, value in SPLIT_TO_INDEX.items()
        },
        "evt_tail_sequences": int(output["is_evt_tail"].sum()),
        "bounded_development_cache": bool(selected_max > 0),
    }
    save_json(manifest, sequence_manifest_path(output_dir))
    return manifest


def load_sequential_dataset(
    output_dir: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not sequence_cache_available(output_dir):
        raise FileNotFoundError(
            f"Missing sequence cache under {sequence_cache_dir(output_dir)}; run preparation first"
        )
    root = sequence_cache_dir(output_dir)
    manifest = load_json(root / "manifest.json")
    arrays = {
        key: np.load(root / f"{key}.npy", mmap_mode="r", allow_pickle=False)
        for key in SEQUENCE_ARRAYS
    }
    if bool(manifest.get("background_slot_stability_required", False)):
        _require_stable_background_slots(arrays["agent_valid"])
    return arrays, manifest
