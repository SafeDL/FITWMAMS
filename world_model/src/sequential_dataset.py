"""Six-second sequence cache for semi-Markov relational world-model training."""
from __future__ import annotations

import logging
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .initial_behavior_anchor import FrozenLegacyFlowSchema, summarize_first_second_states

from .adapters.highd_adapter import HighDGraphAdapter
from .data import SPLIT_TO_INDEX, aligned_multichunk_indices, load_world_model_dataset
from .utils import ensure_dir, load_json, save_json

logger = logging.getLogger(__name__)

# v3 is the audited highD cache already present in existing experiments.  v4
# adds optional conflict-zone tensors used by rounD without invalidating the
# immutable v3 cache that full highD training reuses.
SEQUENCE_CACHE_VERSION = "semi_markov_sequence_v3_continuous_coordinates_recording_map"
GENERIC_SEQUENCE_CACHE_VERSION = "semi_markov_sequence_v4_dynamic_graph_conflicts"
SEQUENCE_ARRAYS = (
    "sequence_id", "recording_id", "ego_id", "timestamps", "agent_ids", "agent_states", "agent_valid",
    "ego_index", "primary_agent_index", "map_polylines", "map_polyline_valid", "lane_graph_edges",
    "agent_lane_candidates", "actions_highd", "split_index", "is_evt_tail",
)
OPTIONAL_SEQUENCE_ARRAYS = ("conflict_zone_features", "conflict_zone_valid")
FLOW_ANCHOR_ARRAYS = ("behavior_anchor_raw", "behavior_anchor_std", "behavior_anchor_valid")
FLOW_ANCHOR_CACHE_VERSION = "frozen_flow_behavior_anchor_v1"


def sequence_cache_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "sequence_cache"


def sequence_manifest_path(output_dir: str | Path) -> Path:
    return sequence_cache_dir(output_dir) / "manifest.json"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _flow_anchor_cache_root(output_dir: str | Path, schema: FrozenLegacyFlowSchema) -> Path:
    """Return the immutable sidecar location for one Flow/schema pairing."""
    source_manifest = sequence_manifest_path(output_dir)
    if not source_manifest.exists():
        raise FileNotFoundError(f"missing sequence-cache manifest: {source_manifest}")
    source_hash = _sha256_bytes(source_manifest.read_bytes())
    return sequence_cache_dir(output_dir) / "frozen_flow_behavior_anchors" / f"{schema.schema_sha256}_{source_hash}"


def ensure_frozen_flow_behavior_anchor_cache(
    output_dir: str | Path,
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
    schema: FrozenLegacyFlowSchema,
) -> dict[str, np.ndarray]:
    """Materialize exact logged B0 once, then serve it as read-only tensors.

    This deliberately caches summaries of every six-second logged sequence,
    rather than joining the much smaller Flow-tail dataset.  A Flow-tail row
    is suitable for Flow sampling, but is not a condition for an arbitrary
    logged future trajectory.  The sidecar is content-addressed by both the
    immutable sequence-cache manifest and frozen Flow schema.
    """
    if int(np.asarray(arrays["agent_states"]).shape[2]) != 7:
        raise ValueError("frozen 76-D Flow anchors require ego plus exactly six ordered background slots")
    root = _flow_anchor_cache_root(output_dir, schema)
    metadata_path = root / "manifest.json"
    expected = {
        "cache_version": FLOW_ANCHOR_CACHE_VERSION,
        "flow_schema_sha256": schema.schema_sha256,
        "source_sequence_manifest_sha256": _sha256_bytes(sequence_manifest_path(output_dir).read_bytes()),
        "num_sequences": int(np.asarray(arrays["agent_states"]).shape[0]),
    }
    if metadata_path.exists() and all((root / f"{name}.npy").exists() for name in FLOW_ANCHOR_ARRAYS):
        stored = load_json(metadata_path)
        if all(stored.get(key) == value for key, value in expected.items()):
            return {name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in FLOW_ANCHOR_ARRAYS}
        raise RuntimeError(f"Flow-anchor sidecar contract mismatch at {root}; keep the immutable cache and use its matching schema")
    if root.exists():
        raise RuntimeError(f"partial Flow-anchor sidecar at {root}; remove only this generated sidecar before rebuilding")

    ensure_dir(root)
    count = expected["num_sequences"]
    raw_path, std_path, valid_path = (root / f"{name}.npy" for name in FLOW_ANCHOR_ARRAYS)
    raw_out = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float32, shape=(count, 6, 6))
    std_out = np.lib.format.open_memmap(std_path, mode="w+", dtype=np.float32, shape=(count, 6, 6))
    valid_out = np.lib.format.open_memmap(valid_path, mode="w+", dtype=bool, shape=(count, 6))
    import torch
    chunk = 1024
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        # The source cache is memory-mapped read-only; a small one-time copy
        # avoids handing PyTorch a non-writable view.
        states = torch.from_numpy(np.asarray(arrays["agent_states"][start:stop, 24:50, 1:], np.float32).copy())
        valid = torch.from_numpy(np.asarray(arrays["agent_valid"][start:stop, 24:50, 1:], bool).copy())
        raw, anchor_valid = summarize_first_second_states(states, valid)
        standardized = schema.standardize(raw, anchor_valid)
        raw_out[start:stop] = raw.numpy()
        std_out[start:stop] = standardized.numpy()
        valid_out[start:stop] = anchor_valid.numpy()
    raw_out.flush(); std_out.flush(); valid_out.flush()
    expected.update({"arrays": list(FLOW_ANCHOR_ARRAYS), "summary": "exact_26_state_points", "source_cache_version": manifest.get("cache_version")})
    save_json(expected, metadata_path)
    logger.info("Wrote frozen Flow behavior-anchor sidecar: %s", root)
    return {name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in FLOW_ANCHOR_ARRAYS}


def sequence_cache_owner_dir(config: dict[str, Any], *, config_dir: Path) -> Path:
    """Resolve the directory that owns a reusable sequence cache.

    Fine-tuning experiments can write checkpoints to a new output directory
    while reusing an immutable, already audited sequence cache.  This avoids
    rebuilding data and prevents a trial run from overwriting the cache used by
    an earlier checkpoint.
    """
    paths = config["paths"]
    value = paths.get("sequence_cache_dir", paths["output_dir"])
    owner = Path(value)
    return owner if owner.is_absolute() else (config_dir / owner).resolve()


def sequence_cache_available(output_dir: str | Path) -> bool:
    root = sequence_cache_dir(output_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists() or not all((root / f"{key}.npy").exists() for key in SEQUENCE_ARRAYS):
        return False
    try:
        return load_json(manifest_path).get("cache_version") in {
            SEQUENCE_CACHE_VERSION, GENERIC_SEQUENCE_CACHE_VERSION,
        }
    except (OSError, ValueError):
        return False


def _split_name(index: int) -> str:
    return {value: name for name, value in SPLIT_TO_INDEX.items()}.get(int(index), "train")


def _sequence_rows(arrays: dict[str, np.ndarray], *, max_sequences: int, seed: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    for split in ("train", "val", "test"):
        current = aligned_multichunk_indices(arrays, split, horizon_steps=25, max_chunks=5)
        rows.extend(current)
    if not rows:
        raise RuntimeError("No aligned START + five ROLL chunks available for sequence cache")
    index = np.asarray(rows, dtype=np.int64)
    # Stable random order lets a bounded development cache retain all splits.
    rng = np.random.default_rng(int(seed))
    rng.shuffle(index)
    if max_sequences > 0:
        index = index[:int(max_sequences)]
    return index


def _unnormalize_history(history: np.ndarray, history_valid: np.ndarray, schema: dict[str, Any]) -> np.ndarray:
    norm = schema["normalization"]["state"]
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    raw = np.asarray(history, np.float32) * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
    return raw * np.asarray(history_valid, bool)[..., None]


def prepare_sequential_dataset(
    config: dict[str, Any],
    *,
    config_dir: Path,
    rebuild: bool = False,
    max_sequences: int | None = None,
) -> dict[str, Any]:
    """Materialize sequence-level arrays from the frozen dense highD cache.

    The source cache is used only as a migration source.  The output holds one
    six-second sequence per natural segment (or a documented deterministic
    bounded subset for development), rather than 21 independent START/ROLL
    samples per segment.
    """
    adapter_name = str(config.get("dataset", {}).get("adapter", "highd")).lower()
    if adapter_name in {"round", "roundd", "round_adapter"}:
        return prepare_round_sequential_dataset(
            config, config_dir=config_dir, rebuild=rebuild, max_sequences=max_sequences,
        )
    if adapter_name in {"joint", "highd_round_joint", "multi_dataset"}:
        return prepare_joint_sequential_dataset(
            config, config_dir=config_dir, rebuild=rebuild, max_sequences=max_sequences,
        )
    if adapter_name not in {"highd", "highd_adapter"}:
        raise ValueError(f"unsupported sequence dataset adapter: {adapter_name}")
    paths = config["paths"]
    source_dir = Path(paths["legacy_dataset_dir"])
    if not source_dir.is_absolute():
        source_dir = (config_dir / source_dir).resolve()
    output_dir = sequence_cache_owner_dir(config, config_dir=config_dir)
    root = sequence_cache_dir(output_dir)
    if sequence_cache_available(output_dir) and not rebuild:
        return load_json(sequence_manifest_path(output_dir))
    if root.exists() and rebuild:
        import shutil
        shutil.rmtree(root)
    ensure_dir(root)
    arrays, schema = load_world_model_dataset(source_dir)
    dataset_cfg = config.get("dataset", {})
    graph_cfg = dict(config.get("graph", {}))
    selected_max = int(max_sequences if max_sequences is not None else dataset_cfg.get("max_sequences", 0) or 0)
    rows = _sequence_rows(arrays, max_sequences=selected_max, seed=int(config.get("split", {}).get("seed", 42)))
    if len(rows) == 0:
        raise RuntimeError("No source sequences selected")
    adapter = HighDGraphAdapter(
        lane_width_m=float(graph_cfg.get("lane_width_m", 3.6)),
        top_r_lanes=int(graph_cfg.get("top_r_lanes", 3)),
    )
    recording_cache: dict[int, tuple[Any, dict[int, dict[str, Any]]]] = {}
    use_recording_lane_metadata = bool(graph_cfg.get("use_recording_lane_metadata", True))
    highd_cfg: dict[str, Any] | None = None
    raw_dir = Path(schema.get("raw_dir", ""))
    if use_recording_lane_metadata:
        from process_highD.src.io_utils import load_config as load_highd_config
        from process_highD.src.natural_segments import _build_vehicle_cache, _position_at
        from process_highD.src.preprocess import prepare_recording

        highd_config_value = paths.get("highd_evt_config")
        if not highd_config_value:
            raise KeyError("paths.highd_evt_config is required when graph.use_recording_lane_metadata=true")
        highd_config_path = Path(highd_config_value)
        if not highd_config_path.is_absolute():
            highd_config_path = (config_dir / highd_config_path).resolve()
        highd_cfg = load_highd_config(str(highd_config_path))

        def _recording_map(recording_id: int, ego_id: int, anchor_frame: int):
            cached = recording_cache.get(int(recording_id))
            if cached is None:
                recording = prepare_recording(raw_dir, int(recording_id), highd_cfg)
                cached = (recording, _build_vehicle_cache(recording))
                recording_cache[int(recording_id)] = cached
            recording, vehicles = cached
            vehicle = vehicles.get(int(ego_id))
            position = None if vehicle is None else _position_at(vehicle, int(anchor_frame))
            if vehicle is None or position is None:
                return None
            lateral_sign = 1.0 if int(vehicle.get("direction", 0)) == 1 else -1.0
            return adapter.map_from_recording_metadata(
                recording.recording_meta, ego_global_y_m=float(vehicle["y_left"][position]), lateral_sign=lateral_sign,
            )
    else:
        _recording_map = None
    s = len(rows)
    # Fixed padding capacities; graph validity preserves actual variable N/M.
    t, n, m, p, r, action_t = 150, 7, 8, 8, int(config.get("graph", {}).get("top_r_lanes", 3)), 125
    output: dict[str, np.ndarray] = {
        "sequence_id": np.empty(s, dtype="U96"), "recording_id": np.empty(s, dtype="U24"), "ego_id": np.empty(s, dtype="U24"),
        "timestamps": np.zeros((s, t), np.float32), "agent_ids": np.tile(np.arange(n, dtype=np.int64), (s, 1)),
        "agent_states": np.zeros((s, t, n, 6), np.float32), "agent_valid": np.zeros((s, t, n), bool),
        "ego_index": np.zeros(s, np.int64), "primary_agent_index": np.full(s, -1, np.int64),
        "map_polylines": np.zeros((s, m, p, 6), np.float32), "map_polyline_valid": np.zeros((s, m, p), bool),
        "lane_graph_edges": np.full((s, max(1, 2 * (m - 1)), 3), -1, np.int64),
        "agent_lane_candidates": np.full((s, t, n, r), -1, np.int64),
        "actions_highd": np.zeros((s, action_t, n - 1, 2), np.float32), "split_index": np.zeros(s, np.int64), "is_evt_tail": np.zeros(s, bool),
    }
    for out_i, row_indices in enumerate(rows):
        start = int(row_indices[0])
        hist_valid = np.asarray(arrays["history_valid"][start], bool)
        history = _unnormalize_history(arrays["history_states_normalized"][start], hist_valid, schema)
        future_states: list[np.ndarray] = []
        future_valid: list[np.ndarray] = []
        future_actions: list[np.ndarray] = []
        # Every legacy ROLL row is expressed in a fresh local frame whose
        # origin is the ego position at that row's target frame.  A semi-Markov
        # sequence must instead remain in one continuous local frame.  Carry
        # the already reconstructed ego origin forward before concatenating
        # each 1-second row; velocities and accelerations are translation
        # invariant, so only x/y are translated.
        origin_in_sequence = np.asarray(history[-1, 0, :2], dtype=np.float32).copy()
        for row in row_indices:
            row = int(row)
            bg = np.asarray(arrays["target_states"][row], np.float32)
            bg_valid = np.asarray(arrays["target_valid"][row], bool)
            ego = np.asarray(arrays["ego_future_states"][row], np.float32)
            ego_valid = np.asarray(arrays["ego_future_valid"][row], bool)
            combined = np.zeros((25, n, 6), np.float32)
            combined[:, 0] = ego
            combined[:, 1:] = bg
            combined_valid = np.zeros((25, n), bool)
            combined_valid[:, 0] = ego_valid
            combined_valid[:, 1:] = bg_valid
            combined[:, :, :2] += origin_in_sequence.reshape(1, 1, 2)
            future_states.append(combined)
            future_valid.append(combined_valid)
            future_actions.append(np.asarray(arrays["target_actions"][row], np.float32))
            origin_in_sequence = combined[-1, 0, :2].copy()
        states = np.concatenate((history, *future_states), axis=0)
        valid = np.concatenate((hist_valid, *future_valid), axis=0)
        primary = int(arrays["primary_slot_index"][start])
        map_override = _recording_map(
            int(arrays["recording_id"][start]), int(arrays["ego_id"][start]), int(arrays["anchor_frame"][start])
        ) if _recording_map is not None else None
        seq = adapter.adapt(
            sequence_id=str(arrays["segment_id"][start]), recording_id=str(arrays["recording_id"][start]),
            ego_id=str(arrays["ego_id"][start]), timestamps=np.arange(-24, 126, dtype=np.float32) / 25.0,
            agent_states=states, agent_valid=valid, primary_agent_index=primary,
            split=_split_name(int(arrays["split_index"][start])), is_evt_tail=bool(arrays["is_evt_tail"][start]),
            map_override=map_override,
        )
        output["sequence_id"][out_i] = seq.sequence_id
        output["recording_id"][out_i] = seq.recording_id
        output["ego_id"][out_i] = seq.ego_id
        output["timestamps"][out_i] = seq.timestamps
        output["agent_states"][out_i] = seq.agent_states
        output["agent_valid"][out_i] = seq.agent_valid
        output["ego_index"][out_i] = seq.ego_index
        output["primary_agent_index"][out_i] = seq.primary_agent_index
        lm = min(m, seq.map_polylines.shape[0])
        output["map_polylines"][out_i, :lm] = seq.map_polylines[:lm]
        output["map_polyline_valid"][out_i, :lm] = seq.map_polyline_valid[:lm]
        le = min(output["lane_graph_edges"].shape[1], len(seq.lane_graph_edges))
        if le:
            output["lane_graph_edges"][out_i, :le] = seq.lane_graph_edges[:le]
        output["agent_lane_candidates"][out_i] = seq.agent_lane_candidates
        output["actions_highd"][out_i] = np.concatenate(future_actions, axis=0)
        output["split_index"][out_i] = int(arrays["split_index"][start])
        output["is_evt_tail"][out_i] = bool(arrays["is_evt_tail"][start])
        if (out_i + 1) % 1000 == 0 or out_i + 1 == s:
            logger.info("Prepared semi-Markov sequence %d/%d", out_i + 1, s)
    for key, value in output.items():
        np.save(root / f"{key}.npy", value, allow_pickle=False)
    manifest = {
        "cache_version": SEQUENCE_CACHE_VERSION, "num_sequences": int(s), "frames": t,
        "history_frames": 25, "future_frames": 125, "fps": 25.0,
        "source_dataset": str(source_dir), "adapter": adapter.version,
        "uses_recording_lane_metadata": use_recording_lane_metadata,
        "top_r_lanes": r, "arrays": list(SEQUENCE_ARRAYS),
        "split_summary": {name: int(np.sum(output["split_index"] == value)) for name, value in SPLIT_TO_INDEX.items()},
        "evt_tail_sequences": int(output["is_evt_tail"].sum()), "bounded_development_cache": bool(selected_max > 0),
    }
    save_json(manifest, sequence_manifest_path(output_dir))
    return manifest


def write_dynamic_sequence_cache(
    sequences: list[DynamicTrafficSequence],
    *,
    output_dir: str | Path,
    source_dataset: str,
    adapter: str,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Persist dataset-neutral sequences with padding only at cache time.

    The model always receives dense batches, but each source sequence keeps
    its own valid-agent mask, map size and conflict-zone count.  This writer is
    shared by rounD and future non-highway adapters; highD retains its audited
    v3 migration cache unchanged.
    """
    if not sequences:
        raise ValueError("at least one dynamic traffic sequence is required")
    owner = Path(output_dir)
    root = sequence_cache_dir(owner)
    if sequence_cache_available(owner) and not rebuild:
        return load_json(sequence_manifest_path(owner))
    if root.exists():
        if not rebuild:
            raise FileExistsError(f"partial sequence cache exists at {root}; pass rebuild=True to replace it")
        import shutil
        shutil.rmtree(root)
    ensure_dir(root)
    frames = {int(np.asarray(item.agent_states).shape[0]) for item in sequences}
    if frames != {150}:
        raise ValueError("all dynamic sequences must have exactly 150 frames (one-second history plus five-second future)")
    if any(int(item.ego_index) != 0 for item in sequences):
        raise ValueError("cache requires the externally observed ego at agent index zero")
    s = len(sequences)
    n = max(int(item.agent_states.shape[1]) for item in sequences)
    m = max(1, max(int(item.map_polylines.shape[0]) for item in sequences))
    p = max(1, max(int(item.map_polylines.shape[1]) for item in sequences))
    r = max(1, max(int(item.agent_lane_candidates.shape[-1]) for item in sequences))
    e = max(1, max(int(np.asarray(item.lane_graph_edges).reshape(-1, 3).shape[0]) for item in sequences))
    c = max(int(item.conflict_zone_features.shape[0]) for item in sequences)
    arrays: dict[str, np.ndarray] = {
        "sequence_id": np.empty(s, dtype="U128"), "recording_id": np.empty(s, dtype="U64"), "ego_id": np.empty(s, dtype="U64"),
        "timestamps": np.zeros((s, 150), np.float32), "agent_ids": np.full((s, n), -1, np.int64),
        "agent_states": np.zeros((s, 150, n, 6), np.float32), "agent_valid": np.zeros((s, 150, n), bool),
        "ego_index": np.zeros(s, np.int64), "primary_agent_index": np.full(s, -1, np.int64),
        "map_polylines": np.zeros((s, m, p, 6), np.float32), "map_polyline_valid": np.zeros((s, m, p), bool),
        "lane_graph_edges": np.full((s, e, 3), -1, np.int64),
        "agent_lane_candidates": np.full((s, 150, n, r), -1, np.int64),
        "actions_highd": np.zeros((s, 125, max(n - 1, 0), 2), np.float32),
        "split_index": np.zeros(s, np.int64), "is_evt_tail": np.zeros(s, bool),
        "conflict_zone_features": np.zeros((s, c, 4), np.float32), "conflict_zone_valid": np.zeros((s, c), bool),
    }
    for index, sequence in enumerate(sequences):
        agent_count = int(sequence.agent_states.shape[1])
        map_count, point_count = sequence.map_polylines.shape[:2]
        lane_count = len(np.asarray(sequence.lane_graph_edges).reshape(-1, 3))
        candidate_count = int(sequence.agent_lane_candidates.shape[-1])
        zone_count = int(sequence.conflict_zone_features.shape[0])
        arrays["sequence_id"][index] = str(sequence.sequence_id)
        arrays["recording_id"][index] = str(sequence.recording_id)
        arrays["ego_id"][index] = str(sequence.ego_id)
        arrays["timestamps"][index] = np.asarray(sequence.timestamps, np.float32)
        arrays["agent_ids"][index, :agent_count] = np.asarray(sequence.agent_ids, np.int64)
        arrays["agent_states"][index, :, :agent_count] = np.asarray(sequence.agent_states, np.float32)
        arrays["agent_valid"][index, :, :agent_count] = np.asarray(sequence.agent_valid, bool)
        arrays["primary_agent_index"][index] = int(sequence.primary_agent_index)
        arrays["map_polylines"][index, :map_count, :point_count] = np.asarray(sequence.map_polylines, np.float32)
        arrays["map_polyline_valid"][index, :map_count, :point_count] = np.asarray(sequence.map_polyline_valid, bool)
        if lane_count:
            arrays["lane_graph_edges"][index, :lane_count] = np.asarray(sequence.lane_graph_edges, np.int64).reshape(-1, 3)
        arrays["agent_lane_candidates"][index, :, :agent_count, :candidate_count] = np.asarray(sequence.agent_lane_candidates, np.int64)
        arrays["actions_highd"][index, :, : max(agent_count - 1, 0)] = np.asarray(sequence.agent_states, np.float32)[25:, 1:, 4:6]
        arrays["split_index"][index] = int(SPLIT_TO_INDEX[str(sequence.split)])
        arrays["is_evt_tail"][index] = bool(sequence.is_evt_tail)
        if zone_count:
            arrays["conflict_zone_features"][index, :zone_count] = np.asarray(sequence.conflict_zone_features, np.float32)
            arrays["conflict_zone_valid"][index, :zone_count] = np.asarray(sequence.conflict_zone_valid, bool)
    for key, value in arrays.items():
        np.save(root / f"{key}.npy", value, allow_pickle=False)
    manifest = {
        "cache_version": GENERIC_SEQUENCE_CACHE_VERSION, "num_sequences": int(s), "frames": 150,
        "history_frames": 25, "future_frames": 125, "fps": 25.0,
        "source_dataset": str(source_dataset), "adapter": str(adapter), "top_r_lanes": int(r),
        "arrays": [*SEQUENCE_ARRAYS, *OPTIONAL_SEQUENCE_ARRAYS],
        "split_summary": {name: int(np.sum(arrays["split_index"] == value)) for name, value in SPLIT_TO_INDEX.items()},
        "evt_tail_sequences": int(arrays["is_evt_tail"].sum()), "bounded_development_cache": False,
        "variable_agent_capacity": int(n), "variable_map_capacity": int(m), "conflict_zone_capacity": int(c),
    }
    save_json(manifest, sequence_manifest_path(owner))
    return manifest


def _resolve_config_path(value: str | Path, config_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_dir / path).resolve()


def _round_group_split(recording_id: str, ego_id: int, *, seed: int) -> str:
    """Stable 70/15/15 split that never separates a recording--ego group."""
    digest = hashlib.blake2b(f"{seed}:{recording_id}:{ego_id}".encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little") % 10_000
    return "train" if value < 7000 else "val" if value < 8500 else "test"


def _round_sequence_from_table(
    *,
    adapter: Any,
    tracks: Any,
    columns: dict[str, str | None],
    map_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None],
    ego_id: int,
    start_frame: int,
    num_frames: int,
    recording_id: str,
    split: str,
    frame_rate_hz: float,
) -> DynamicTrafficSequence:
    """Build one rounD sequence from an already-loaded recording table."""
    id_col, frame_col = str(columns["id"]), str(columns["frame"])
    frame_values = np.arange(int(start_frame), int(start_frame) + int(num_frames), dtype=np.int64)
    subset = tracks[tracks[frame_col].isin(frame_values)]
    observed_ids = sorted(int(value) for value in subset[id_col].unique())
    if int(ego_id) not in observed_ids:
        raise ValueError("ego disappeared from a preselected rounD sequence")
    agent_ids = np.asarray([int(ego_id), *[item for item in observed_ids if item != int(ego_id)]], np.int64)
    index_by_id = {int(value): index for index, value in enumerate(agent_ids)}
    states = np.zeros((len(frame_values), len(agent_ids), 6), np.float32)
    valid = np.zeros((len(frame_values), len(agent_ids)), bool)
    frame_index = {int(value): index for index, value in enumerate(frame_values)}
    x_col, y_col, vx_col, vy_col = (str(columns[key]) for key in ("x", "y", "vx", "vy"))
    ax_col, ay_col = columns["ax"], columns["ay"]
    for row in subset.itertuples(index=False):
        item = row._asdict()
        frame, track_id = int(item[frame_col]), int(item[id_col])
        values = (
            item[x_col], item[y_col], item[vx_col], item[vy_col],
            0.0 if ax_col is None else item[str(ax_col)],
            0.0 if ay_col is None else item[str(ay_col)],
        )
        if not np.isfinite(np.asarray(values, np.float32)).all():
            continue
        states[frame_index[frame], index_by_id[track_id]] = values
        valid[frame_index[frame], index_by_id[track_id]] = True
    if not valid[:, 0].all():
        raise ValueError("rounD sequence ego must be observed at every one of its 150 frames")
    polylines, polyline_valid, lane_edges, zones, zone_valid = map_data
    return adapter.adapt(
        sequence_id=f"round-{recording_id}-{int(ego_id)}-{int(start_frame)}",
        recording_id=str(recording_id), ego_id=str(int(ego_id)),
        timestamps=(frame_values - frame_values[0]).astype(np.float32) / float(frame_rate_hz),
        agent_ids=agent_ids, agent_states=states, agent_valid=valid, ego_index=0,
        primary_agent_index=-1, map_polylines=polylines, map_polyline_valid=polyline_valid,
        lane_graph_edges=lane_edges, split=str(split), conflict_zone_features=zones,
        conflict_zone_valid=zone_valid,
    )


def prepare_round_sequential_dataset(
    config: dict[str, Any],
    *,
    config_dir: Path,
    rebuild: bool = False,
    max_sequences: int | None = None,
) -> dict[str, Any]:
    """Materialize rounD six-second dynamic-graph sequences from real files.

    ``dataset.recordings`` accepts one or more mappings with ``tracks_csv``,
    ``vector_map`` and an optional ``recording_id``.  A single-recording form
    using ``paths.round_tracks_csv`` and ``paths.round_vector_map`` is also
    supported.  Split assignment is by recording--ego, never per window.
    """
    from .adapters.round_adapter import RoundGraphAdapter

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - dependency error is actionable
        raise RuntimeError("rounD preparation requires pandas") from exc
    paths, dataset_cfg = config["paths"], config.get("dataset", {})
    recording_specs = list(dataset_cfg.get("recordings", []))
    if not recording_specs:
        tracks_value, map_value = paths.get("round_tracks_csv"), paths.get("round_vector_map")
        if not tracks_value or not map_value:
            raise KeyError("rounD preparation requires dataset.recordings or paths.round_tracks_csv and paths.round_vector_map")
        recording_specs = [{"tracks_csv": tracks_value, "vector_map": map_value, "recording_id": dataset_cfg.get("recording_id", "round")}] 
    graph_cfg = config.get("graph", {})
    adapter = RoundGraphAdapter(
        top_r_lanes=int(graph_cfg.get("top_r_lanes", 3)), lane_width_m=float(graph_cfg.get("lane_width_m", 3.6)),
    )
    num_frames = int(dataset_cfg.get("sequence_frames", 150))
    if num_frames != 150:
        raise ValueError("the semi-Markov rounD cache uses the fixed 150-frame six-second protocol")
    stride = max(1, int(dataset_cfg.get("sequence_stride_frames", 25)))
    frame_rate = float(dataset_cfg.get("frame_rate_hz", 25.0))
    seed = int(config.get("split", {}).get("seed", 42))
    candidates: list[tuple[Any, dict[str, str | None], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None], str, int, int, str]] = []
    for spec in recording_specs:
        tracks_path = _resolve_config_path(spec["tracks_csv"], config_dir)
        map_path = _resolve_config_path(spec["vector_map"], config_dir)
        if not tracks_path.exists() or not map_path.exists():
            raise FileNotFoundError(f"missing rounD recording or vector map: {tracks_path}, {map_path}")
        tracks = pd.read_csv(tracks_path)
        columns = {
            "id": adapter._column(tracks, "trackId", "track_id", "id"),
            "frame": adapter._column(tracks, "frame", "frame_id"),
            "x": adapter._column(tracks, "xCenter", "x", "x_center"),
            "y": adapter._column(tracks, "yCenter", "y", "y_center"),
            "vx": adapter._column(tracks, "xVelocity", "vx", "x_velocity"),
            "vy": adapter._column(tracks, "yVelocity", "vy", "y_velocity"),
            "ax": adapter._column(tracks, "xAcceleration", "ax", "x_acceleration", required=False),
            "ay": adapter._column(tracks, "yAcceleration", "ay", "y_acceleration", required=False),
        }
        map_data = adapter.load_vector_map(map_path, lane_width_m=adapter.builder.cfg.lane_width_m)
        recording_id = str(spec.get("recording_id", tracks_path.stem))
        id_col, frame_col = str(columns["id"]), str(columns["frame"])
        for ego_id, rows in tracks.groupby(id_col, sort=True):
            frames = np.unique(rows[frame_col].to_numpy(dtype=np.int64))
            if not len(frames):
                continue
            run_start, previous = int(frames[0]), int(frames[0])
            runs: list[tuple[int, int]] = []
            for frame in frames[1:]:
                frame = int(frame)
                if frame != previous + 1:
                    runs.append((run_start, previous))
                    run_start = frame
                previous = frame
            runs.append((run_start, previous))
            for begin, end in runs:
                for start in range(begin, end - num_frames + 2, stride):
                    split = _round_group_split(recording_id, int(ego_id), seed=seed)
                    candidates.append((tracks, columns, map_data, recording_id, int(ego_id), int(start), split))
    if not candidates:
        raise RuntimeError("no rounD ego track contains a contiguous 150-frame sequence")
    selected_max = int(max_sequences if max_sequences is not None else dataset_cfg.get("max_sequences", 0) or 0)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates))
    if selected_max > 0:
        order = order[:selected_max]
    sequences = [
        _round_sequence_from_table(
            adapter=adapter, tracks=candidates[int(index)][0], columns=candidates[int(index)][1], map_data=candidates[int(index)][2],
            recording_id=candidates[int(index)][3], ego_id=candidates[int(index)][4], start_frame=candidates[int(index)][5],
            num_frames=num_frames, split=candidates[int(index)][6], frame_rate_hz=frame_rate,
        )
        for index in order
    ]
    manifest = write_dynamic_sequence_cache(
        sequences, output_dir=sequence_cache_owner_dir(config, config_dir=config_dir),
        source_dataset="rounD", adapter=adapter.version, rebuild=rebuild,
    )
    manifest["bounded_development_cache"] = bool(selected_max > 0)
    manifest["source_recordings"] = [str(spec.get("recording_id", Path(spec["tracks_csv"]).stem)) for spec in recording_specs]
    save_json(manifest, sequence_manifest_path(sequence_cache_owner_dir(config, config_dir=config_dir)))
    return manifest


def combine_sequence_caches(
    cache_dirs: list[str | Path],
    *,
    output_dir: str | Path,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Create a disk-backed joint cache without forcing a shared slot count.

    Each source cache may have a different agent/map/conflict capacity.  The
    combined cache pads only at this final batching boundary and preserves
    source split indices, enabling highD+rounD shared-encoder training.
    """
    owners = [Path(value).resolve() for value in cache_dirs]
    if len(owners) < 2:
        raise ValueError("joint training requires at least two prepared sequence caches")
    sources = [load_sequential_dataset(owner) for owner in owners]
    total = sum(len(arrays["sequence_id"]) for arrays, _ in sources)
    if total == 0:
        raise RuntimeError("joint sequence caches contain no sequences")
    owner, root = Path(output_dir), sequence_cache_dir(output_dir)
    if sequence_cache_available(owner) and not rebuild:
        return load_json(sequence_manifest_path(owner))
    if root.exists():
        if not rebuild:
            raise FileExistsError(f"partial joint cache exists at {root}; pass rebuild=True to replace it")
        import shutil
        shutil.rmtree(root)
    ensure_dir(root)
    max_agents = max(int(arrays["agent_states"].shape[2]) for arrays, _ in sources)
    max_lanes = max(int(arrays["map_polylines"].shape[1]) for arrays, _ in sources)
    max_points = max(int(arrays["map_polylines"].shape[2]) for arrays, _ in sources)
    max_edges = max(int(arrays["lane_graph_edges"].shape[1]) for arrays, _ in sources)
    max_candidates = max(int(arrays["agent_lane_candidates"].shape[-1]) for arrays, _ in sources)
    max_conflicts = max(int(arrays.get("conflict_zone_features", np.zeros((0, 0, 4))).shape[1]) for arrays, _ in sources)
    string_lengths = {
        key: max(int(arrays[key].dtype.itemsize // np.dtype("U1").itemsize) for arrays, _ in sources)
        for key in ("sequence_id", "recording_id", "ego_id")
    }
    shapes: dict[str, tuple[int, ...]] = {
        "sequence_id": (total,), "recording_id": (total,), "ego_id": (total,), "timestamps": (total, 150),
        "agent_ids": (total, max_agents), "agent_states": (total, 150, max_agents, 6), "agent_valid": (total, 150, max_agents),
        "ego_index": (total,), "primary_agent_index": (total,), "map_polylines": (total, max_lanes, max_points, 6),
        "map_polyline_valid": (total, max_lanes, max_points), "lane_graph_edges": (total, max_edges, 3),
        "agent_lane_candidates": (total, 150, max_agents, max_candidates), "actions_highd": (total, 125, max_agents - 1, 2),
        "split_index": (total,), "is_evt_tail": (total,),
        "conflict_zone_features": (total, max_conflicts, 4), "conflict_zone_valid": (total, max_conflicts),
    }
    dtypes: dict[str, Any] = {
        "sequence_id": np.dtype(f"U{string_lengths['sequence_id']}"), "recording_id": np.dtype(f"U{string_lengths['recording_id']}"),
        "ego_id": np.dtype(f"U{string_lengths['ego_id']}"), "timestamps": np.float32, "agent_ids": np.int64,
        "agent_states": np.float32, "agent_valid": bool, "ego_index": np.int64, "primary_agent_index": np.int64,
        "map_polylines": np.float32, "map_polyline_valid": bool, "lane_graph_edges": np.int64,
        "agent_lane_candidates": np.int64, "actions_highd": np.float32, "split_index": np.int64, "is_evt_tail": bool,
        "conflict_zone_features": np.float32, "conflict_zone_valid": bool,
    }
    negative = {"agent_ids", "primary_agent_index", "lane_graph_edges", "agent_lane_candidates"}
    for key, shape in shapes.items():
        target = np.lib.format.open_memmap(root / f"{key}.npy", mode="w+", dtype=dtypes[key], shape=shape)
        target[...] = -1 if key in negative else 0
        offset = 0
        for arrays, _manifest in sources:
            count = len(arrays["sequence_id"])
            source = arrays.get(key)
            if source is not None:
                source = np.asarray(source)
                slices = (slice(offset, offset + count), *[slice(0, size) for size in source.shape[1:]])
                target[slices] = source
            offset += count
        del target
    manifests = [manifest for _arrays, manifest in sources]
    split_index = np.load(root / "split_index.npy", mmap_mode="r", allow_pickle=False)
    manifest = {
        "cache_version": GENERIC_SEQUENCE_CACHE_VERSION, "num_sequences": int(total), "frames": 150,
        "history_frames": 25, "future_frames": 125, "fps": 25.0, "adapter": "joint_dynamic_graph_v1",
        "source_sequence_caches": [str(path) for path in owners], "arrays": [*SEQUENCE_ARRAYS, *OPTIONAL_SEQUENCE_ARRAYS],
        "split_summary": {name: int(np.sum(split_index == value)) for name, value in SPLIT_TO_INDEX.items()},
        "evt_tail_sequences": int(np.load(root / "is_evt_tail.npy", mmap_mode="r", allow_pickle=False).sum()),
        "bounded_development_cache": bool(any(item.get("bounded_development_cache", True) for item in manifests)),
        "variable_agent_capacity": int(max_agents), "variable_map_capacity": int(max_lanes), "conflict_zone_capacity": int(max_conflicts),
    }
    save_json(manifest, sequence_manifest_path(owner))
    return manifest


def prepare_joint_sequential_dataset(
    config: dict[str, Any], *, config_dir: Path, rebuild: bool = False, max_sequences: int | None = None,
) -> dict[str, Any]:
    if max_sequences not in {None, 0}:
        raise ValueError("joint cache selection must be performed by preparing bounded source caches explicitly")
    paths = config["paths"]
    values = paths.get("joint_sequence_cache_dirs", [])
    if not values:
        values = [paths.get("highd_sequence_cache_dir"), paths.get("round_sequence_cache_dir")]
    if any(not value for value in values):
        raise KeyError("joint preparation requires paths.joint_sequence_cache_dirs or highd_sequence_cache_dir + round_sequence_cache_dir")
    sources = [_resolve_config_path(value, config_dir) for value in values]
    return combine_sequence_caches(
        sources, output_dir=sequence_cache_owner_dir(config, config_dir=config_dir), rebuild=rebuild,
    )


def load_sequential_dataset(output_dir: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not sequence_cache_available(output_dir):
        raise FileNotFoundError(f"Missing sequence cache under {sequence_cache_dir(output_dir)}; run preparation first")
    root = sequence_cache_dir(output_dir)
    manifest = load_json(root / "manifest.json")
    names = tuple(manifest.get("arrays", SEQUENCE_ARRAYS))
    arrays = {key: np.load(root / f"{key}.npy", mmap_mode="r", allow_pickle=False) for key in names}
    # A v3 cache may predate optional graph tensors; retain only fields that
    # are actually materialized so batch loaders can remain backward-compatible.
    for key in OPTIONAL_SEQUENCE_ARRAYS:
        path = root / f"{key}.npy"
        if key not in arrays and path.exists():
            arrays[key] = np.load(path, mmap_mode="r", allow_pickle=False)
    return arrays, manifest
