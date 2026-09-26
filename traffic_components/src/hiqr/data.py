"""Read-only cohort metadata for the canonical highD cache."""

from __future__ import annotations

from typing import Any

import numpy as np


def cohort_manifest(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Summarize the immutable recording-level split used by all models."""
    valid = np.asarray(arrays["agent_valid"], bool)
    split = np.asarray(arrays["split_index"], np.int64)
    labels = {"train": 0, "val": 1, "test": 2}
    return {
        "cohort": "canonical_cleaned_full_horizon_background_slots",
        "canonical_sequences": int(len(valid)),
        "background_slot_stability_required": True,
        "splits": {
            name: {"sequences": int(np.sum(split == index))}
            for name, index in labels.items()
        },
    }
