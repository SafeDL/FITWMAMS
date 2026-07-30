"""Tests for shared immutable-cache batching utilities."""

from __future__ import annotations

import numpy as np
import torch

from world_model.src.core.batching import (
    SEQUENCE_FIELDS,
    make_sequence_loader,
    sequence_field_names,
    to_device_batch,
)
from world_model.src.core.data import SPLIT_TO_INDEX


def _arrays() -> dict[str, np.ndarray]:
    count = 3
    arrays = {
        name: np.zeros((count, 1), dtype=np.float32)
        for name in SEQUENCE_FIELDS
    }
    arrays.update(
        {
            "agent_valid": np.ones((count, 1), dtype=bool),
            "ego_index": np.zeros((count, 1), dtype=np.int64),
            "lane_graph_edges": np.zeros((count, 1), dtype=np.int64),
            "is_evt_tail": np.zeros((count, 1), dtype=bool),
            "split_index": np.array(
                [SPLIT_TO_INDEX["train"], SPLIT_TO_INDEX["train"], SPLIT_TO_INDEX["test"]], dtype=np.int64
            ),
            "behavior_anchor_raw": np.ones((count, 2), dtype=np.float32),
            "behavior_anchor_std": np.full((count, 2), 0.5, dtype=np.float32),
            "behavior_anchor_valid": np.ones((count, 2), dtype=bool),
        }
    )
    return arrays


def test_shared_sequence_loader_preserves_field_order_and_b0_aliases() -> None:
    arrays = _arrays()
    fields = sequence_field_names(arrays)
    assert fields[: len(SEQUENCE_FIELDS)] == SEQUENCE_FIELDS
    assert fields[-3:] == ("behavior_anchor_raw", "behavior_anchor_std", "behavior_anchor_valid")

    loader = make_sequence_loader(
        arrays, "train", batch_size=2, maximum=0, shuffle=False, seed=7, num_workers=0
    )
    batch = to_device_batch(next(iter(loader)), loader.field_names, "cpu")
    assert batch["agent_states"].shape == (2, 1)
    assert batch["flow_action_summary"].equal(batch["behavior_anchor_raw"])
    assert batch["flow_action_summary_normalized"].equal(batch["behavior_anchor_std"])
    assert batch["flow_action_summary_valid"].dtype is torch.bool
