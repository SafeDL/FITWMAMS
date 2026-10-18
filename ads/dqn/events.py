"""Lazy highD event source backed by the canonical sequence cache."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from hierarchical_world_model.src.randomness import WorldExogenousState

from .env import WorldEvent

ANCHOR_INDEX = 24


class HighDEventStore(Sequence[WorldEvent]):
    def __init__(
        self,
        event_csv: str | Path,
        sequence_cache: str | Path,
        *,
        world_seed: int = 0,
    ) -> None:
        frame = pd.read_csv(event_csv)
        required = {"event_id", "source_event_id", "split"}
        if missing := required - set(frame.columns):
            raise KeyError(f"event manifest is missing {sorted(missing)}")
        self.frame = frame.reset_index(drop=True)
        root = Path(sequence_cache)
        self.sequence_ids = np.load(
            root / "sequence_id.npy", mmap_mode="r", allow_pickle=False
        )
        self.states = np.load(
            root / "agent_states.npy", mmap_mode="r", allow_pickle=False
        )
        self.valid = np.load(
            root / "agent_valid.npy", mmap_mode="r", allow_pickle=False
        )
        self.maps = np.load(
            root / "map_polylines.npy", mmap_mode="r", allow_pickle=False
        )
        self.map_valid = np.load(
            root / "map_polyline_valid.npy", mmap_mode="r", allow_pickle=False
        )
        dimensions_path = root / "vehicle_dimensions.npy"
        self.vehicle_dimensions = (
            np.load(dimensions_path, mmap_mode="r", allow_pickle=False)
            if dimensions_path.exists()
            else None
        )
        if self.vehicle_dimensions is not None:
            if self.vehicle_dimensions.shape != (len(self.sequence_ids), 7, 2):
                raise ValueError(
                    "vehicle_dimensions cache must have shape [sequences,7,2]"
                )
            if not np.isfinite(self.vehicle_dimensions).all() or np.any(
                self.vehicle_dimensions <= 0.0
            ):
                raise ValueError("vehicle_dimensions cache must be finite and positive")
        self.row_by_id = {
            str(value): index for index, value in enumerate(self.sequence_ids)
        }
        self.world_seed = int(world_seed)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> WorldEvent:
        item = self.frame.iloc[int(index)]
        event_id = str(item["event_id"])
        base_event_id = str(item.get("base_event_id", event_id))
        if base_event_id not in self.row_by_id:
            raise KeyError(
                f"base event {base_event_id!r} is absent from the sequence cache"
            )
        row = self.row_by_id[base_event_id]
        material = f"ads_event:{self.world_seed}:{event_id}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
        exogenous_path = item.get("exogenous_path")
        exogenous = (
            WorldExogenousState.load(str(exogenous_path))
            if exogenous_path is not None
            and not pd.isna(exogenous_path)
            and str(exogenous_path)
            else WorldExogenousState.sample(
                1,
                seed=seed,
                response_steps=149,
            )
        )
        return WorldEvent(
            event_id=event_id,
            source_event_id=str(item["source_event_id"]),
            initial_states=np.asarray(
                self.states[row, ANCHOR_INDEX], np.float32
            ).copy(),
            valid=np.asarray(self.valid[row, ANCHOR_INDEX], bool).copy(),
            soft_reference=np.asarray(
                self.states[row, ANCHOR_INDEX + 1 :, 1:, :2], np.float32
            ).copy(),
            map_polylines=np.asarray(self.maps[row], np.float32).copy(),
            map_polyline_valid=np.asarray(self.map_valid[row], bool).copy(),
            exogenous_state=exogenous,
            vehicle_dimensions=(
                None
                if self.vehicle_dimensions is None
                else np.asarray(self.vehicle_dimensions[row], np.float32).copy()
            ),
            split=str(item["split"]),
            initial_history=np.asarray(
                self.states[row, : ANCHOR_INDEX + 1], np.float32
            ).copy(),
            initial_history_valid=np.asarray(
                self.valid[row, : ANCHOR_INDEX + 1], bool
            ).copy(),
        )


def select_event_manifest(
    bank_csv: str | Path,
    *,
    cohort: str,
    split: str,
    output_csv: str | Path,
) -> Path:
    """Write one explicit cohort/split manifest for reproducible training."""
    frame = pd.read_csv(bank_csv)
    selected_column = f"{cohort}_selected"
    if selected_column not in frame:
        raise KeyError(f"unknown event cohort {cohort!r}")
    selected = frame[(frame["split"] == split) & frame[selected_column].astype(bool)]
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(path, index=False)
    return path
