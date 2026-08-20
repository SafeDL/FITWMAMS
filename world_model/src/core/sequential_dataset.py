"""START-plus-ROLL sequence cache for relational world-model training.

Each highD source window contributes its 150 recorded state points.  The first
25 physical transitions are the Flow-conditioned START reconstruction and the
remaining 124 transitions are the free ROLL continuation.  At 25 Hz this is
``1.00 s START + 4.96 s ROLL = 5.96 s``; no terminal state is fabricated.
"""
from __future__ import annotations

import logging
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .initial_behavior_anchor import (
    FlowC0Schema,
    behavior_anchor_from_flow_feature,
    summarize_first_second_states,
)

from world_model.src.traffic_graph.highd_adapter import HighDGraphAdapter
from .data import (
    SPLIT_TO_INDEX,
    _action_window,
    _segment_slot_ids,
    _state_window,
    aligned_multichunk_indices,
    load_world_model_dataset,
)
from .utils import ensure_dir, load_json, save_json
from process_highD.src.natural_segments import _build_vehicle_cache, _position_at
from process_highD.src.preprocess import prepare_recording

logger = logging.getLogger(__name__)

# QR's canonical cache retains all 150 *recorded* points S0..S149 rather than
# synthesising an S150 endpoint.  The baseline signature is only retained for
# the independently maintained baseline models that still consume that cache.
BASELINE_SEQUENCE_CACHE_SIGNATURE = "semi_markov_sequence_v4_compact_continuous_coordinates"
CANONICAL_SEQUENCE_CACHE_FORMAT = "highd_canonical_raw150"
CANONICAL_SEQUENCE_PROTOCOL = "fixed_horizon_5p96"
SEQUENCE_ARRAYS = (
    "sequence_id", "agent_states", "agent_valid", "ego_index", "map_polylines",
    "map_polyline_valid", "lane_graph_edges", "actions_highd", "split_index", "is_evt_tail",
)
FLOW_ANCHOR_ARRAYS = ("behavior_anchor_raw", "behavior_anchor_std", "behavior_anchor_valid")
FLOW_ANCHOR_CACHE_VERSION = "frozen_flow_behavior_anchor_v1"

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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _flow_anchor_cache_root(output_dir: str | Path, schema: FlowC0Schema) -> Path:
    """Return the immutable sidecar location for one Flow/schema pairing."""
    source_manifest = sequence_manifest_path(output_dir)
    if not source_manifest.exists():
        raise FileNotFoundError(f"missing sequence-cache manifest: {source_manifest}")
    source_hash = _sha256_bytes(source_manifest.read_bytes())
    return sequence_cache_dir(output_dir) / "frozen_flow_behavior_anchors" / f"{schema.schema_sha256}_{source_hash}"


def _validate_flow_tail_alignment(
    arrays: dict[str, np.ndarray], schema: FlowC0Schema, raw: np.ndarray, valid: np.ndarray,
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
    schema: FlowC0Schema,
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
        states = torch.from_numpy(np.asarray(
            arrays["agent_states"][start:stop, HISTORY_PADDING_FRAMES:HISTORY_PADDING_FRAMES + START_RECONSTRUCTION_FRAMES + 1, 1:],
            np.float32,
        ).copy())
        valid = torch.from_numpy(np.asarray(
            arrays["agent_valid"][start:stop, HISTORY_PADDING_FRAMES:HISTORY_PADDING_FRAMES + START_RECONSTRUCTION_FRAMES + 1, 1:],
            bool,
        ).copy())
        raw, anchor_valid = summarize_first_second_states(states, valid)
        standardized = schema.standardize(raw, anchor_valid)
        raw_out[start:stop] = raw.numpy()
        std_out[start:stop] = standardized.numpy()
        valid_out[start:stop] = anchor_valid.numpy()
    raw_out.flush(); std_out.flush(); valid_out.flush()
    expected.update({
        "arrays": list(FLOW_ANCHOR_ARRAYS), "summary": "exact_26_state_points_from_S0_through_S25",
        "source_sequence_cache_format": manifest.get("cache_format", manifest.get("cache_version")),
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
    value = paths.get("sequence_cache_dir") or paths["output_dir"]
    owner = Path(value)
    return owner if owner.is_absolute() else (config_dir / owner).resolve()


def sequence_cache_available(output_dir: str | Path) -> bool:
    root = sequence_cache_dir(output_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists() or not all((root / f"{key}.npy").exists() for key in SEQUENCE_ARRAYS):
        return False
    try:
        manifest = load_json(manifest_path)
        return (
            manifest.get("cache_format") == CANONICAL_SEQUENCE_CACHE_FORMAT
            or manifest.get("cache_version") == BASELINE_SEQUENCE_CACHE_SIGNATURE
        )
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


def is_canonical_sequence_manifest(manifest: dict[str, Any]) -> bool:
    """Whether a manifest fulfils the canonical non-fabricated contract."""
    return (
        manifest.get("cache_format") == CANONICAL_SEQUENCE_CACHE_FORMAT
        and int(manifest.get("future_transition_frames", -1)) == FUTURE_TRANSITION_FRAMES
        and int(manifest.get("start_reconstruction_frames", -1)) == START_RECONSTRUCTION_FRAMES
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


def _uses_canonical_sequence_protocol(config: dict[str, Any]) -> bool:
    return (
        str(config.get("dataset", {}).get("sequence_protocol", ""))
        == CANONICAL_SEQUENCE_PROTOCOL
    )


def _unnormalize_history(history: np.ndarray, history_valid: np.ndarray, schema: dict[str, Any]) -> np.ndarray:
    norm = schema["normalization"]["state"]
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    raw = np.asarray(history, np.float32) * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
    return raw * np.asarray(history_valid, bool)[..., None]


def prepare_sequential_dataset(
    config: dict[str, Any], *, config_dir: Path, rebuild: bool = False, max_sequences: int | None = None,
) -> dict[str, Any]:
    """Prepare the explicitly selected cache protocol.

    HiQR and diffusion opt into the canonical 5.96-second representation.
    """
    if _uses_canonical_sequence_protocol(config):
        return _prepare_canonical_sequence_dataset(
            config, config_dir=config_dir, rebuild=rebuild, max_sequences=max_sequences,
        )
    return _prepare_baseline_sequence_dataset(
        config, config_dir=config_dir, rebuild=rebuild, max_sequences=max_sequences,
    )


def _prepare_canonical_sequence_dataset(
    config: dict[str, Any],
    *,
    config_dir: Path,
    rebuild: bool = False,
    max_sequences: int | None = None,
) -> dict[str, Any]:
    """Materialize one raw sequence for every canonical Flow/natural row."""
    adapter_name = str(config.get("dataset", {}).get("adapter", "highd")).lower()
    if adapter_name not in {"highd", "highd_adapter"}:
        raise ValueError("only the highD sequence adapter is retained")
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
            f"{manifest.get('cache_format', manifest.get('cache_version'))!r}; rebuild it explicitly."
        )
    if root.exists():
        if not rebuild:
            raise RuntimeError(
                f"Outdated or partial sequence cache at {root}; rebuild it explicitly "
                "before HiQR training/evaluation."
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
                "primary_slot_index",
                "split_index",
                "is_evt_tail",
            )
        }
    dataset_cfg = config.get("dataset", {})
    graph_cfg = dict(config.get("graph", {}))
    selected_max = int(max_sequences if max_sequences is not None else dataset_cfg.get("max_sequences", 0) or 0)
    rows = np.arange(len(arrays["segment_id"]), dtype=np.int64)
    if selected_max > 0:
        rng = np.random.default_rng(int(config.get("split", {}).get("seed", 42)))
        rng.shuffle(rows)
        rows = rows[:selected_max]
    if len(rows) == 0:
        raise RuntimeError("No source sequences selected")
    adapter = HighDGraphAdapter(
        lane_width_m=float(graph_cfg.get("lane_width_m", 3.6)),
        top_r_lanes=int(graph_cfg.get("top_r_lanes", 3)),
    )
    recording_cache: dict[int, tuple[Any, dict[int, dict[str, Any]]]] = {}
    use_recording_lane_metadata = bool(graph_cfg.get("use_recording_lane_metadata", True))
    highd_cfg: dict[str, Any] | None = None
    raw_dir = Path(flow_schema.get("raw_dir", ""))
    if not raw_dir.exists():
        raise FileNotFoundError(f"sequence construction requires raw highD data: {raw_dir}")
    from process_highD.src.io_utils import load_config as load_highd_config

    highd_config_value = paths.get("highd_evt_config")
    if not highd_config_value:
        raise KeyError("paths.highd_evt_config is required to construct the raw START+ROLL sequence cache")
    highd_config_path = Path(highd_config_value)
    if not highd_config_path.is_absolute():
        highd_config_path = (config_dir / highd_config_path).resolve()
    highd_cfg = load_highd_config(str(highd_config_path))
    natural_csv = Path(str(flow_schema.get("source_segments_csv", "")))
    if not natural_csv.exists():
        raise FileNotFoundError(f"sequence construction requires natural segment metadata: {natural_csv}")
    natural_rows = pd.read_csv(natural_csv).set_index("segment_id", drop=False)

    def _recording_data(recording_id: int) -> tuple[Any, dict[int, dict[str, Any]]]:
        cached = recording_cache.get(int(recording_id))
        if cached is None:
            recording = prepare_recording(raw_dir, int(recording_id), highd_cfg)
            cached = (recording, _build_vehicle_cache(recording))
            recording_cache[int(recording_id)] = cached
        return cached

    def _recording_map(recording_id: int, ego_id: int, anchor_frame: int):
        if not use_recording_lane_metadata:
            return None
        recording, vehicles = _recording_data(int(recording_id))
        vehicle = vehicles.get(int(ego_id))
        position = None if vehicle is None else _position_at(vehicle, int(anchor_frame))
        if vehicle is None or position is None:
            return None
        lateral_sign = 1.0 if int(vehicle.get("direction", 0)) == 1 else -1.0
        return adapter.map_from_recording_metadata(
            recording.recording_meta, ego_global_y_m=float(vehicle["y_left"][position]), lateral_sign=lateral_sign,
        )
    s = len(rows)
    # The graph keeps an invalid 24-frame compatibility prefix, then every
    # physically recorded highD state S0..S149.  Actions label S0->S1 through
    # S148->S149, so the final 4-tick response is real rather than padded.
    t, n, m, p, r, action_t = (
        SEQUENCE_FRAMES,
        7,
        8,
        8,
        int(config.get("graph", {}).get("top_r_lanes", 3)),
        FUTURE_TRANSITION_FRAMES,
    )
    output: dict[str, np.ndarray] = {
        "sequence_id": np.empty(s, dtype="U96"),
        "agent_states": np.zeros((s, t, n, 6), np.float32), "agent_valid": np.zeros((s, t, n), bool),
        "ego_index": np.zeros(s, np.int64),
        "map_polylines": np.zeros((s, m, p, 6), np.float32), "map_polyline_valid": np.zeros((s, m, p), bool),
        "lane_graph_edges": np.full((s, max(1, 2 * (m - 1)), 3), -1, np.int64),
        "actions_highd": np.zeros((s, action_t, n - 1, 2), np.float32), "split_index": np.zeros(s, np.int64), "is_evt_tail": np.zeros(s, bool),
    }
    for out_i, row_index in enumerate(rows):
        start = int(row_index)
        segment_id = str(arrays["segment_id"][start])
        if segment_id not in natural_rows.index:
            raise KeyError(f"Natural metadata is missing sequence segment_id={segment_id}")
        natural = natural_rows.loc[segment_id]
        if isinstance(natural, pd.DataFrame):
            natural = natural.iloc[0]
        recording_id = int(arrays["recording_id"][start])
        ego_id = int(arrays["ego_id"][start])
        anchor_frame = int(arrays["anchor_frame"][start])
        recording, vehicles = _recording_data(recording_id)
        slot_ids = _segment_slot_ids(natural)
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
        primary = int(arrays["primary_slot_index"][start])
        map_override = _recording_map(
            recording_id, ego_id, anchor_frame
        )
        seq = adapter.adapt(
            sequence_id=segment_id, recording_id=str(recording_id),
            ego_id=str(ego_id), timestamps=np.arange(-HISTORY_PADDING_FRAMES, RAW_WINDOW_STATE_FRAMES, dtype=np.float32) / 25.0,
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
        output["actions_highd"][out_i] = highd_actions
        output["split_index"][out_i] = int(arrays["split_index"][start])
        output["is_evt_tail"][out_i] = bool(arrays["is_evt_tail"][start])
        if (out_i + 1) % 1000 == 0 or out_i + 1 == s:
            logger.info("Prepared QR START+ROLL sequence %d/%d", out_i + 1, s)
    _require_stable_background_slots(output["agent_valid"])
    for key, value in output.items():
        np.save(root / f"{key}.npy", value, allow_pickle=False)
    manifest = {
        "cache_format": CANONICAL_SEQUENCE_CACHE_FORMAT, "num_sequences": int(s), "frames": t,
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
        "adapter": adapter.version,
        "lateral_event_integrity_required": bool(
            flow_schema.get("lateral_event_integrity_required", False)
        ),
        "background_slot_stability_required": bool(
            flow_schema.get("background_slot_stability_required", False)
        ),
        "uses_recording_lane_metadata": use_recording_lane_metadata,
        "top_r_lanes": r, "arrays": list(SEQUENCE_ARRAYS),
        "split_summary": {name: int(np.sum(output["split_index"] == value)) for name, value in SPLIT_TO_INDEX.items()},
        "evt_tail_sequences": int(output["is_evt_tail"].sum()), "bounded_development_cache": bool(selected_max > 0),
    }
    save_json(manifest, sequence_manifest_path(output_dir))
    return manifest


def _prepare_baseline_sequence_dataset(
    config: dict[str, Any], *, config_dir: Path, rebuild: bool, max_sequences: int | None,
) -> dict[str, Any]:
    """Prepare the separate cache format used by non-QR world models.

    This is deliberately retained rather than reinterpreting old ``S0..S125``
    supervision as the QR protocol.  The two layouts have different temporal
    meanings even though both use highD data at 25 Hz.
    """
    adapter_name = str(config.get("dataset", {}).get("adapter", "highd")).lower()
    if adapter_name not in {"highd", "highd_adapter"}:
        raise ValueError("only the highD sequence adapter is retained")
    paths = config["paths"]
    source_dir = Path(paths["source_dataset_dir"])
    if not source_dir.is_absolute():
        source_dir = (config_dir / source_dir).resolve()
    output_dir = sequence_cache_owner_dir(config, config_dir=config_dir)
    root = sequence_cache_dir(output_dir)
    if sequence_cache_available(output_dir) and not rebuild:
        manifest = load_json(sequence_manifest_path(output_dir))
        if manifest.get("cache_version") == BASELINE_SEQUENCE_CACHE_SIGNATURE:
            return manifest
        raise RuntimeError(
            f"baseline world-model cache requires {BASELINE_SEQUENCE_CACHE_SIGNATURE}, but {root} contains "
            f"{manifest.get('cache_version')!r}; use a separate cache directory."
        )
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
    use_recording_lane_metadata = bool(graph_cfg.get("use_recording_lane_metadata", True))
    raw_dir = Path(schema.get("raw_dir", ""))
    if use_recording_lane_metadata:
        from process_highD.src.io_utils import load_config as load_highd_config

        highd_config_value = paths.get("highd_evt_config")
        if not highd_config_value:
            raise KeyError("paths.highd_evt_config is required when graph.use_recording_lane_metadata=true")
        highd_config_path = Path(highd_config_value)
        if not highd_config_path.is_absolute():
            highd_config_path = (config_dir / highd_config_path).resolve()
        highd_cfg = load_highd_config(str(highd_config_path))
        recording_cache: dict[int, tuple[Any, dict[int, dict[str, Any]]]] = {}

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
    s, t, n, m, p = len(rows), 150, 7, 8, 8
    action_t = 125
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
        origin_in_sequence = np.asarray(history[-1, 0, :2], dtype=np.float32).copy()
        for row in row_indices:
            row = int(row)
            bg, bg_valid = np.asarray(arrays["target_states"][row], np.float32), np.asarray(arrays["target_valid"][row], bool)
            ego, ego_valid = np.asarray(arrays["ego_future_states"][row], np.float32), np.asarray(arrays["ego_future_valid"][row], bool)
            combined = np.zeros((25, n, 6), np.float32)
            combined[:, 0], combined[:, 1:] = ego, bg
            combined_valid = np.zeros((25, n), bool)
            combined_valid[:, 0], combined_valid[:, 1:] = ego_valid, bg_valid
            combined[:, :, :2] += origin_in_sequence.reshape(1, 1, 2)
            future_states.append(combined)
            future_valid.append(combined_valid)
            future_actions.append(np.asarray(arrays["target_actions"][row], np.float32))
            origin_in_sequence = combined[-1, 0, :2].copy()
        states = np.concatenate((history, *future_states), axis=0)
        valid = np.concatenate((hist_valid, *future_valid), axis=0)
        map_override = _recording_map(
            int(arrays["recording_id"][start]), int(arrays["ego_id"][start]), int(arrays["anchor_frame"][start])
        ) if _recording_map is not None else None
        seq = adapter.adapt(
            sequence_id=str(arrays["segment_id"][start]), recording_id=str(arrays["recording_id"][start]),
            ego_id=str(arrays["ego_id"][start]), timestamps=np.arange(-24, 126, dtype=np.float32) / 25.0,
            agent_states=states, agent_valid=valid, primary_agent_index=int(arrays["primary_slot_index"][start]),
            split=_split_name(int(arrays["split_index"][start])), is_evt_tail=bool(arrays["is_evt_tail"][start]),
            map_override=map_override,
        )
        output["sequence_id"][out_i], output["agent_states"][out_i], output["agent_valid"][out_i] = seq.sequence_id, seq.agent_states, seq.agent_valid
        output["ego_index"][out_i] = seq.ego_index
        lm = min(m, seq.map_polylines.shape[0])
        output["map_polylines"][out_i, :lm], output["map_polyline_valid"][out_i, :lm] = seq.map_polylines[:lm], seq.map_polyline_valid[:lm]
        le = min(output["lane_graph_edges"].shape[1], len(seq.lane_graph_edges))
        if le:
            output["lane_graph_edges"][out_i, :le] = seq.lane_graph_edges[:le]
        output["actions_highd"][out_i] = np.concatenate(future_actions, axis=0)
        output["split_index"][out_i], output["is_evt_tail"][out_i] = int(arrays["split_index"][start]), bool(arrays["is_evt_tail"][start])
        if (out_i + 1) % 1000 == 0 or out_i + 1 == s:
            logger.info("Prepared baseline sequence %d/%d", out_i + 1, s)
    for key, value in output.items():
        np.save(root / f"{key}.npy", value, allow_pickle=False)
    manifest = {
        "cache_version": BASELINE_SEQUENCE_CACHE_SIGNATURE, "num_sequences": int(s), "frames": t,
        "history_frames": 25, "future_frames": 125, "fps": 25.0,
        "source_dataset": str(source_dir), "adapter": adapter.version,
        "uses_recording_lane_metadata": use_recording_lane_metadata,
        "top_r_lanes": int(graph_cfg.get("top_r_lanes", 3)), "arrays": list(SEQUENCE_ARRAYS),
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
    if bool(manifest.get("background_slot_stability_required", False)):
        _require_stable_background_slots(arrays["agent_valid"])
    return arrays, manifest
