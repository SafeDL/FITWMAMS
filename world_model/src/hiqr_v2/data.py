"""Read-only canonical-cache access with HiQR-v2 stable-slot cohorts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from world_model.src.core.batching import select_sequence_indices
from world_model.src.hiqr.data import (
    HIQR_TRAINING_SIDECAR_ARRAYS,
    HIQR_SEQUENCE_FIELDS,
    load_hiqr_training_arrays,
)

V2_FIELDS = (*HIQR_SEQUENCE_FIELDS, *HIQR_TRAINING_SIDECAR_ARRAYS)


def stable_slot_mask(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Return rows whose C0-active slots remain valid through all transitions.

    ``behavior_anchor_valid`` requires a slot to survive the whole first
    second, so it cannot identify vehicles that are visible at C0 and exit
    during that second.  The cohort is therefore anchored directly to the C0
    slot mask and requires each such vehicle through S149.
    """
    valid = np.asarray(arrays["agent_valid"], bool)
    if valid.ndim != 3 or valid.shape[2] != 7:
        raise ValueError("invalid canonical HiQR-v2 validity arrays")
    anchor = 24
    c0_active = valid[:, anchor, 1:]
    full = valid[:, anchor:, 1:].all(axis=1)
    return (~c0_active | full).all(axis=1)


def cohort_manifest(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    stable = stable_slot_mask(arrays)
    valid = np.asarray(arrays["agent_valid"], bool)
    c0_active = valid[:, 24, 1:]
    survives = valid[:, 24:, 1:].all(axis=1)
    exiting_slots = c0_active & ~survives
    split = np.asarray(arrays["split_index"], np.int64)
    labels = {"train": 0, "val": 1, "test": 2}
    result: dict[str, Any] = {
        "cohort": "c0_active_backgrounds_valid_through_5p96s",
        "canonical_sequences": int(len(stable)),
        "stable_sequences": int(stable.sum()),
        "excluded_sequences": int((~stable).sum()),
        "stable_fraction": float(stable.mean()),
        "c0_active_background_slots": int(c0_active.sum()),
        "stable_cohort_active_background_slots": int(c0_active[stable].sum()),
        "excluded_sequence_active_background_slots": int(c0_active[~stable].sum()),
        "exiting_background_slots": int(exiting_slots.sum()),
        "splits": {},
    }
    for name, value in labels.items():
        rows = split == value
        result["splits"][name] = {
            "canonical_sequences": int(rows.sum()),
            "stable_sequences": int((stable & rows).sum()),
            "excluded_sequences": int((~stable & rows).sum()),
            "c0_active_background_slots": int(c0_active[rows].sum()),
            "stable_cohort_active_background_slots": int(
                c0_active[stable & rows].sum()
            ),
            "excluded_sequence_active_background_slots": int(
                c0_active[(~stable) & rows].sum()
            ),
            "exiting_background_slots": int(exiting_slots[rows].sum()),
        }
    return result


def load_hiqr_v2_arrays(
    *,
    cache_owner: str | Path,
    v1_sidecar_output_dir: str | Path,
    flow_schema,
    source_dataset_dir: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load immutable QR arrays and the already materialized V1 B0 sidecar.

    The V1 sidecar is content-addressed to the shared canonical cache, so V2
    can safely consume it without writing anything into the V1 result root.
    """
    arrays, manifest = load_hiqr_training_arrays(
        cache_owner=cache_owner,
        output_dir=v1_sidecar_output_dir,
        flow_schema=flow_schema,
        source_dataset_dir=source_dataset_dir,
    )
    return arrays, {**manifest, "hiqr_v2_cohort": cohort_manifest(arrays)}


def make_hiqr_v2_loader(
    arrays: dict[str, np.ndarray],
    split: str,
    *,
    batch_size: int,
    maximum: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
):
    """Construct a loader restricted to the immutable stable-slot cohort."""
    from torch.utils.data import DataLoader, Dataset

    selected = select_sequence_indices(arrays, split, 0, seed)
    stable = stable_slot_mask(arrays)
    selected = selected[stable[selected]]
    if maximum:
        selected = selected[: int(maximum)]
    if not len(selected):
        raise RuntimeError(f"No stable HiQR-v2 sequences in split={split}")

    class _Dataset(Dataset):
        def __len__(self) -> int:
            return len(selected)

        def __getitem__(self, row: int):
            index = int(selected[int(row)])
            return tuple(
                torch.from_numpy(np.asarray(arrays[name][index]).copy())
                for name in V2_FIELDS
            )

    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        _Dataset(),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator if shuffle else None,
        num_workers=max(0, int(num_workers)),
        persistent_workers=int(num_workers) > 0,
        drop_last=False,
    )
    loader.field_names = V2_FIELDS
    loader.cohort_manifest = cohort_manifest(arrays)
    return loader


def to_hiqr_v2_batch(
    values: Sequence[torch.Tensor], names: Sequence[str], device: torch.device
) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in zip(names, values)}
