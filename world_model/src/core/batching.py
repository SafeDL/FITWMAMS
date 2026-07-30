"""Shared batching utilities for immutable sequential world-model caches."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .data import SPLIT_TO_INDEX
from .sequential_dataset import FLOW_ANCHOR_ARRAYS


SEQUENCE_FIELDS = (
    "agent_states",
    "agent_valid",
    "ego_index",
    "map_polylines",
    "map_polyline_valid",
    "lane_graph_edges",
    "actions_highd",
    "is_evt_tail",
)
OPTIONAL_SEQUENCE_FIELDS = ("conflict_zone_features", "conflict_zone_valid")


def select_sequence_indices(
    arrays: dict[str, np.ndarray], split: str, maximum: int, seed: int
) -> np.ndarray:
    """Return a reproducibly shuffled subset for one cached dataset split."""
    indices = np.flatnonzero(np.asarray(arrays["split_index"]) == SPLIT_TO_INDEX[split])
    np.random.default_rng(int(seed)).shuffle(indices)
    return indices[: int(maximum)] if maximum > 0 else indices


def sequence_field_names(arrays: dict[str, np.ndarray]) -> tuple[str, ...]:
    """Return the ordered cache fields consumed by all sequential models."""
    return tuple(
        [
            *SEQUENCE_FIELDS,
            *[name for name in OPTIONAL_SEQUENCE_FIELDS if name in arrays],
            *[name for name in FLOW_ANCHOR_ARRAYS if name in arrays],
        ]
    )


def make_sequence_loader(
    arrays: dict[str, np.ndarray],
    split: str,
    *,
    batch_size: int,
    maximum: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
):
    """Build a PyTorch loader without copying the cache beyond each sample."""
    import torch
    from torch.utils.data import DataLoader, Dataset

    indices = select_sequence_indices(arrays, split, maximum, seed)
    if not len(indices):
        raise RuntimeError(f"No cached sequences in split={split}; prepare a larger/non-bounded cache")
    fields = sequence_field_names(arrays)

    class SequenceDataset(Dataset):
        def __len__(self) -> int:
            return len(indices)

        def __getitem__(self, item: int):
            row = int(indices[int(item)])
            return tuple(torch.from_numpy(np.asarray(arrays[name][row]).copy()) for name in fields)

    workers = max(0, int(num_workers))
    loader = DataLoader(
        SequenceDataset(),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=workers,
        persistent_workers=workers > 0,
        drop_last=False,
    )
    loader.field_names = fields
    return loader


def to_device_batch(values: Sequence[Any], names: Sequence[str], device: Any) -> dict[str, Any]:
    """Move a sequential-loader tuple to a model device and expose B0 aliases."""
    batch = {name: value.to(device) for name, value in zip(names, values)}
    if "behavior_anchor_raw" in batch:
        batch["flow_action_summary"] = batch["behavior_anchor_raw"]
        batch["flow_action_summary_normalized"] = batch["behavior_anchor_std"]
        batch["flow_action_summary_valid"] = batch["behavior_anchor_valid"].bool()
    return batch
