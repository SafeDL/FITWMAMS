"""Canonical highD response-boundary data for the hierarchical model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from diffusion.src.data import (
    DataBundle,
    load_data_bundle,
    split_rows,
)
from hierarchical_world_model.src.core.utils import load_json


def event_metadata(cache, arrays):
    """Align physical event metadata by canonical sequence identity, not row number."""
    manifest = load_json(Path(cache) / "manifest.json")
    segments = pd.read_csv(manifest["source_dataset"]).set_index("segment_id")
    return segments.loc[np.asarray(arrays["sequence_id"]).astype(str)].reset_index()


def ego_controls(source: np.ndarray, target: np.ndarray, dt_s: float) -> np.ndarray:
    """Recover realized `[a, yaw_rate]` controls from adjacent ego states."""
    source_speed = np.linalg.norm(source[..., 2:4], axis=-1)
    target_speed = np.linalg.norm(target[..., 2:4], axis=-1)
    acceleration = (target_speed - source_speed) / float(dt_s)
    source_heading = np.arctan2(source[..., 3], np.maximum(source[..., 2], 1.0e-4))
    target_heading = np.arctan2(target[..., 3], np.maximum(target[..., 2], 1.0e-4))
    difference = np.arctan2(
        np.sin(target_heading - source_heading),
        np.cos(target_heading - source_heading),
    )
    return np.stack((acceleration, difference / float(dt_s)), axis=-1).astype(
        np.float32
    )


@dataclass(frozen=True)
class ExperimentData:
    """Loaded highD arrays and recording-isolated row indices for one scope."""

    bundle: DataBundle
    train_rows: np.ndarray
    validation_rows: np.ndarray
    test_rows: np.ndarray


def prepare_experiment_data(config: dict[str, Any], config_dir: Path) -> ExperimentData:
    """Load all natural scenes in the fixed recording-level splits."""
    bundle = load_data_bundle(config, config_dir)
    split_names = {"train": "train", "validation": "val", "test": "test"}
    rows = {
        split: split_rows(bundle.arrays, source)
        for split, source in split_names.items()
    }
    return ExperimentData(
        bundle=bundle,
        train_rows=rows["train"],
        validation_rows=rows["validation"],
        test_rows=rows["test"],
    )


def artifact_identity(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve()
    stat = target.stat()
    return {
        "path": str(target),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
