"""HiQR-owned START metadata sidecar built from a read-only QR sequence cache."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from world_model.src.core.batching import SEQUENCE_FIELDS, select_sequence_indices
from world_model.src.core.data import load_world_model_dataset
from world_model.src.core.initial_behavior_anchor import (
    FrozenLegacyFlowSchema,
    behavior_anchor_from_flow_feature,
    summarize_first_second_states,
)
from world_model.src.core.sequential_dataset import (
    load_sequential_dataset,
    sequence_manifest_path,
)
from world_model.src.core.utils import ensure_dir, load_json, save_json

SIDECAR_VERSION = "hiqr_start_context_v3"
SIDECAR_ARRAYS = (
    "behavior_anchor_raw",
    "behavior_anchor_valid",
    "primary_slot_index",
)
HIQR_TRAINING_SIDECAR_ARRAYS = (
    "behavior_anchor_raw",
    "behavior_anchor_valid",
)
HIQR_SEQUENCE_FIELDS = tuple(
    name for name in SEQUENCE_FIELDS if name != "lane_graph_edges"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hiqr_sidecar_root(output_dir: str | Path, cache_owner: str | Path) -> Path:
    manifest = sequence_manifest_path(cache_owner)
    if not manifest.exists():
        raise FileNotFoundError(f"QR sequence manifest is missing: {manifest}")
    return Path(output_dir) / "hiqr_start_sidecar" / _sha256(manifest)


def _primary_slots_by_sequence(source: dict[str, np.ndarray]) -> dict[str, int]:
    source_ids = np.asarray(source["segment_id"]).astype(str)
    primary_slots = np.asarray(source["primary_slot_index"], np.int64)
    primary_by_id: dict[str, int] = {}
    for identifier, primary in zip(source_ids, primary_slots):
        prior = primary_by_id.setdefault(str(identifier), int(primary))
        if prior != int(primary):
            raise ValueError(
                f"source dataset has conflicting primary slots for {identifier}"
            )
    return primary_by_id


def _primary_slot_mapping_sha256(primary_by_id: dict[str, int]) -> str:
    digest = hashlib.sha256()
    for identifier, primary in sorted(primary_by_id.items()):
        digest.update(identifier.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(primary)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _expected_primary_slots(
    sequence_ids: np.ndarray, primary_by_id: dict[str, int]
) -> np.ndarray:
    absent = [
        identifier for identifier in sequence_ids if identifier not in primary_by_id
    ]
    if absent:
        raise KeyError(
            f"HiQR sidecar cannot find {len(absent)} QR sequence ids "
            "in the source dataset"
        )
    return np.asarray(
        [primary_by_id[identifier] for identifier in sequence_ids], np.int64
    )


def _validate_flow_tail_alignment(
    arrays: dict[str, np.ndarray],
    flow_schema: FrozenLegacyFlowSchema,
    raw: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int]:
    """Verify that B0 agrees with every overlapping frozen Flow row."""
    flow_dataset = flow_schema.source_path.parent / "dataset.npz"
    if not flow_dataset.exists():
        raise FileNotFoundError(
            f"frozen Flow dataset required for B0 validation is missing: {flow_dataset}"
        )
    with np.load(flow_dataset, allow_pickle=False) as flow:
        flow_ids = np.asarray(flow["segment_id"]).astype(str)
        cache_rows = {
            str(identifier): row
            for row, identifier in enumerate(
                np.asarray(arrays["sequence_id"]).astype(str)
            )
        }
        matched_rows = np.asarray(
            [cache_rows.get(identifier, -1) for identifier in flow_ids]
        )
        matched_flow_rows = np.flatnonzero(matched_rows >= 0)
        if not len(matched_flow_rows):
            return {"matched_flow_tail_sequences": 0, "flow_tail_max_abs_error": 0.0}
        expected = np.stack(
            [
                behavior_anchor_from_flow_feature(feature, slot_mask)[0]
                for feature, slot_mask in zip(
                    np.asarray(flow["features"])[matched_flow_rows],
                    np.asarray(flow["slot_mask"])[matched_flow_rows],
                )
            ]
        )
        expected_valid = np.asarray(flow["slot_mask"])[matched_flow_rows].astype(bool)
    observed_rows = matched_rows[matched_flow_rows]
    observed = np.asarray(raw[observed_rows], np.float32)
    observed_valid = np.asarray(valid[observed_rows], bool)
    if not np.array_equal(observed_valid, expected_valid):
        raise ValueError("HiQR B0 validity differs from the frozen Flow slot mask")
    maximum = float(
        (np.abs(observed - expected) * expected_valid[..., None]).max(initial=0.0)
    )
    if maximum > 5.0e-5:
        raise ValueError(
            "HiQR B0 differs from frozen Flow features "
            f"(max_abs_error={maximum:.3g})"
        )
    return {
        "matched_flow_tail_sequences": int(len(matched_flow_rows)),
        "flow_tail_max_abs_error": maximum,
    }


def build_hiqr_start_sidecar(
    *,
    cache_owner: str | Path,
    output_dir: str | Path,
    flow_schema: FrozenLegacyFlowSchema,
    source_dataset_dir: str | Path,
) -> dict[str, Any]:
    """Materialize HiQR's B0 inputs and Flow-only primary-slot audit data.

    The QR cache is opened read-only.  The resulting sidecar is content-bound
    to its source manifest and Flow schema so stale B0 coordinates cannot be
    silently paired with a newer cache or Flow contract.
    """
    arrays, manifest = load_sequential_dataset(cache_owner)
    source, _ = load_world_model_dataset(source_dataset_dir)
    primary_by_id = _primary_slots_by_sequence(source)
    root = hiqr_sidecar_root(output_dir, cache_owner)
    metadata_path = root / "manifest.json"
    expected = {
        "sidecar_version": SIDECAR_VERSION,
        "source_sequence_manifest_sha256": _sha256(sequence_manifest_path(cache_owner)),
        "flow_schema_sha256": flow_schema.schema_sha256,
        "num_sequences": int(len(arrays["sequence_id"])),
        "arrays": list(SIDECAR_ARRAYS),
        "source_primary_slots_sha256": _primary_slot_mapping_sha256(primary_by_id),
    }
    if metadata_path.exists() and all(
        (root / f"{name}.npy").exists() for name in SIDECAR_ARRAYS
    ):
        stored = load_json(metadata_path)
        if all(stored.get(key) == value for key, value in expected.items()):
            cached_raw = np.load(
                root / "behavior_anchor_raw.npy", mmap_mode="r", allow_pickle=False
            )
            cached_valid = np.load(
                root / "behavior_anchor_valid.npy", mmap_mode="r", allow_pickle=False
            )
            alignment = _validate_flow_tail_alignment(
                arrays, flow_schema, cached_raw, cached_valid
            )
            if any(stored.get(key) != value for key, value in alignment.items()):
                stored.update(alignment)
                save_json(stored, metadata_path)
            return stored
        raise RuntimeError(
            f"HiQR sidecar contract mismatch at {root}; "
            "use a fresh HiQR output directory"
        )
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(
            f"partial HiQR sidecar at {root}; remove only its generated "
            "directory before rebuilding"
        )
    ensure_dir(root)
    sequence_ids = np.asarray(arrays["sequence_id"]).astype(str)
    expected_primary = _expected_primary_slots(sequence_ids, primary_by_id)
    count = len(sequence_ids)
    raw_path, valid_path, primary_path = (
        root / f"{name}.npy" for name in SIDECAR_ARRAYS
    )
    raw = np.lib.format.open_memmap(
        raw_path, mode="w+", dtype=np.float32, shape=(count, 6, 6)
    )
    valid = np.lib.format.open_memmap(
        valid_path, mode="w+", dtype=bool, shape=(count, 6)
    )
    primary = np.lib.format.open_memmap(
        primary_path, mode="w+", dtype=np.int64, shape=(count,)
    )
    chunk = 1024
    for start in range(0, count, chunk):
        stop = min(count, start + chunk)
        states = torch.from_numpy(
            np.asarray(arrays["agent_states"][start:stop, 24:50, 1:], np.float32).copy()
        )
        states_valid = torch.from_numpy(
            np.asarray(arrays["agent_valid"][start:stop, 24:50, 1:], bool).copy()
        )
        row_raw, row_valid = summarize_first_second_states(states, states_valid)
        raw[start:stop], valid[start:stop] = row_raw.numpy(), row_valid.numpy()
        primary[start:stop] = expected_primary[start:stop]
    raw.flush()
    valid.flush()
    primary.flush()
    expected.update(
        {
            "source_cache_format": manifest.get("cache_format"),
            "b0_summary": "26_observed_states_S0_through_S25",
            "event_structure": "slot_mask_plus_primary_risk_slot",
            "hiqr_h0_event_structure": "slot_mask_only_causal",
            **_validate_flow_tail_alignment(arrays, flow_schema, raw, valid),
        }
    )
    save_json(expected, metadata_path)
    return expected


def load_hiqr_training_arrays(
    *,
    cache_owner: str | Path,
    output_dir: str | Path,
    flow_schema: FrozenLegacyFlowSchema,
    source_dataset_dir: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays, manifest = load_sequential_dataset(cache_owner)
    source, _ = load_world_model_dataset(source_dataset_dir)
    primary_by_id = _primary_slots_by_sequence(source)
    root = hiqr_sidecar_root(output_dir, cache_owner)
    metadata_path = root / "manifest.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"HiQR START sidecar is missing: {root}; run prepare_hiqr_sequence.py"
        )
    metadata = load_json(metadata_path)
    required = {
        "sidecar_version": SIDECAR_VERSION,
        "source_sequence_manifest_sha256": _sha256(sequence_manifest_path(cache_owner)),
        "flow_schema_sha256": flow_schema.schema_sha256,
        "num_sequences": int(len(arrays["sequence_id"])),
        "source_primary_slots_sha256": _primary_slot_mapping_sha256(primary_by_id),
        "hiqr_h0_event_structure": "slot_mask_only_causal",
    }
    if any(metadata.get(key) != value for key, value in required.items()):
        raise RuntimeError(
            "HiQR START sidecar does not match the configured QR cache "
            "or frozen Flow schema"
        )
    arrays = dict(arrays)
    arrays.update(
        {
            name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in SIDECAR_ARRAYS
        }
    )
    expected_primary = _expected_primary_slots(
        np.asarray(arrays["sequence_id"]).astype(str), primary_by_id
    )
    if not np.array_equal(arrays["primary_slot_index"], expected_primary):
        raise ValueError("HiQR sidecar primary slots differ from their source dataset")
    _validate_flow_tail_alignment(
        arrays,
        flow_schema,
        arrays["behavior_anchor_raw"],
        arrays["behavior_anchor_valid"],
    )
    return arrays, manifest


def make_hiqr_loader(
    arrays: dict[str, np.ndarray],
    split: str,
    *,
    batch_size: int,
    maximum: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
):
    """HiQR loader adds its sidecar fields without changing shared QR batching."""
    from torch.utils.data import DataLoader, Dataset

    indices = select_sequence_indices(arrays, split, maximum, seed)
    if not len(indices):
        raise RuntimeError(f"No HiQR sequences in split={split}")
    fields = (*HIQR_SEQUENCE_FIELDS, *HIQR_TRAINING_SIDECAR_ARRAYS)

    class _Dataset(Dataset):
        def __len__(self) -> int:
            return len(indices)

        def __getitem__(self, row: int):
            index = int(indices[int(row)])
            return tuple(
                torch.from_numpy(np.asarray(arrays[name][index]).copy())
                for name in fields
            )

    loader = DataLoader(
        _Dataset(),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=max(0, int(num_workers)),
        persistent_workers=int(num_workers) > 0,
        drop_last=False,
    )
    loader.field_names = fields
    return loader


def to_hiqr_batch(
    values: Sequence[torch.Tensor], names: Sequence[str], device: torch.device
) -> dict[str, torch.Tensor]:
    """Move exactly HiQR's loader fields without QR compatibility aliases."""
    return {name: value.to(device) for name, value in zip(names, values)}
