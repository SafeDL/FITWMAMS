"""Six-second sequence cache for semi-Markov relational world-model training."""
from __future__ import annotations

import logging
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .initial_behavior_anchor import FrozenLegacyFlowSchema, behavior_anchor_from_flow_feature, summarize_first_second_states

from world_model.src.traffic_graph.highd_adapter import HighDGraphAdapter
from .data import SPLIT_TO_INDEX, aligned_multichunk_indices, load_world_model_dataset
from .utils import ensure_dir, load_json, save_json

logger = logging.getLogger(__name__)

# Audited full highD sequence cache.
SEQUENCE_CACHE_VERSION = "semi_markov_sequence_v4_compact_continuous_coordinates"
SEQUENCE_ARRAYS = (
    "sequence_id", "agent_states", "agent_valid", "ego_index", "map_polylines",
    "map_polyline_valid", "lane_graph_edges", "actions_highd", "split_index", "is_evt_tail",
)
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


def _validate_flow_tail_alignment(
    arrays: dict[str, np.ndarray], schema: FrozenLegacyFlowSchema, raw: np.ndarray, valid: np.ndarray,
) -> dict[str, float | int]:
    """Assert that every overlapping Flow-tail row has the exact cached B0."""
    dataset = schema.source_path.parent / "dataset.npz"
    if not dataset.exists():
        raise FileNotFoundError(f"frozen Flow dataset required for anchor alignment is missing: {dataset}")
    with np.load(dataset, allow_pickle=False) as flow:
        flow_ids = np.asarray(flow["segment_id"]).astype(str)
        cache_index = {str(value): index for index, value in enumerate(np.asarray(arrays["sequence_id"]).astype(str))}
        matched = np.asarray([cache_index.get(value, -1) for value in flow_ids], np.int64)
        take = np.flatnonzero(matched >= 0)
        if not len(take):
            return {"matched_flow_tail_sequences": 0, "flow_tail_max_abs_error": 0.0}
        expected = np.stack([
            behavior_anchor_from_flow_feature(feature, mask)[0]
            for feature, mask in zip(np.asarray(flow["features"])[take], np.asarray(flow["slot_mask"])[take])
        ])
        expected_valid = np.asarray(flow["slot_mask"])[take].astype(bool)
    observed = np.asarray(raw[matched[take]], np.float32)
    observed_valid = np.asarray(valid[matched[take]], bool)
    if not np.array_equal(observed_valid, expected_valid):
        raise ValueError("cached 26-state Flow-anchor validity differs from frozen Flow slot_mask")
    difference = np.abs(observed - expected) * expected_valid[..., None]
    maximum = float(difference.max(initial=0.0))
    if maximum > 5.0e-5:
        raise ValueError(f"cached 26-state behavior anchor differs from frozen Flow features (max_abs_error={maximum:.3g})")
    return {"matched_flow_tail_sequences": int(len(take)), "flow_tail_max_abs_error": maximum}


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
            cached = {name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in FLOW_ANCHOR_ARRAYS}
            alignment = _validate_flow_tail_alignment(arrays, schema, cached["behavior_anchor_raw"], cached["behavior_anchor_valid"])
            if any(stored.get(key) != value for key, value in alignment.items()):
                stored.update(alignment)
                save_json(stored, metadata_path)
            return cached
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
    expected.update({
        "arrays": list(FLOW_ANCHOR_ARRAYS), "summary": "exact_26_state_points", "source_cache_version": manifest.get("cache_version"),
        **_validate_flow_tail_alignment(arrays, schema, raw_out, valid_out),
    })
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
        return load_json(manifest_path).get("cache_version") == SEQUENCE_CACHE_VERSION
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
    if adapter_name not in {"highd", "highd_adapter"}:
        raise ValueError("only the highD sequence adapter is retained")
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
        "sequence_id": np.empty(s, dtype="U96"),
        "agent_states": np.zeros((s, t, n, 6), np.float32), "agent_valid": np.zeros((s, t, n), bool),
        "ego_index": np.zeros(s, np.int64),
        "map_polylines": np.zeros((s, m, p, 6), np.float32), "map_polyline_valid": np.zeros((s, m, p), bool),
        "lane_graph_edges": np.full((s, max(1, 2 * (m - 1)), 3), -1, np.int64),
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
        output["agent_states"][out_i] = seq.agent_states
        output["agent_valid"][out_i] = seq.agent_valid
        output["ego_index"][out_i] = seq.ego_index
        lm = min(m, seq.map_polylines.shape[0])
        output["map_polylines"][out_i, :lm] = seq.map_polylines[:lm]
        output["map_polyline_valid"][out_i, :lm] = seq.map_polyline_valid[:lm]
        le = min(output["lane_graph_edges"].shape[1], len(seq.lane_graph_edges))
        if le:
            output["lane_graph_edges"][out_i, :le] = seq.lane_graph_edges[:le]
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


def load_sequential_dataset(output_dir: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not sequence_cache_available(output_dir):
        raise FileNotFoundError(f"Missing sequence cache under {sequence_cache_dir(output_dir)}; run preparation first")
    root = sequence_cache_dir(output_dir)
    manifest = load_json(root / "manifest.json")
    arrays = {key: np.load(root / f"{key}.npy", mmap_mode="r", allow_pickle=False) for key in SEQUENCE_ARRAYS}
    return arrays, manifest
