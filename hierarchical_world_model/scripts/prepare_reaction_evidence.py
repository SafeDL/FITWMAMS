#!/usr/bin/env python3
"""Build recording-isolated highD leader–follower reaction events."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from hierarchical_world_model.src.reaction_evidence import (  # noqa: E402
    SLOT_NAMES, assert_split_isolation, build_reaction_event_reference,
)


DEFAULT = ROOT / "hierarchical_world_model/config/reaction_policy.yaml"


def _source_metadata(base: dict) -> pd.DataFrame:
    owner = (ROOT / base["paths"]["sequence_cache_dir"]).resolve()
    manifest = json.loads((owner / "sequence_cache/manifest.json").read_text())
    source = pd.read_csv(manifest["source_dataset"])
    if source["segment_id"].duplicated().any():
        raise ValueError("natural segment ids must be unique")
    return source.set_index("segment_id")


def _split_arrays(experiment, rows: np.ndarray, metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    bundle = experiment.bundle
    sequence_ids = np.asarray(bundle.arrays["sequence_id"])[rows].astype(str)
    selected = metadata.loc[sequence_ids]
    agent_ids = np.column_stack((
        selected["ego_id"].to_numpy(np.int64),
        *(selected[f"{slot}_id"].to_numpy(np.int64) for slot in SLOT_NAMES),
    ))
    return {
        "agent_states": np.asarray(bundle.arrays["agent_states"])[rows],
        "agent_valid": np.asarray(bundle.arrays["agent_valid"])[rows],
        "agent_ids": agent_ids,
        "row_index": np.asarray(rows, np.int64),
        "recording_id": selected["recording_id"].to_numpy(np.int64),
        "anchor_frame": selected["anchor_frame"].to_numpy(np.int64),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--split", choices=("all", "train", "validation", "test"), default="all")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = load_protocol_config(args.config.resolve())
    base = load_protocol_config(ROOT / config["base_config"])
    experiment = prepare_experiment_data(base, ROOT)
    metadata = _source_metadata(base)
    evidence = config["evidence"]
    split_names = ("train", "validation", "test") if args.split == "all" else (args.split,)
    references = []
    train_reference = None
    if args.split in {"validation", "test"}:
        train_reference = build_reaction_event_reference(
            _split_arrays(experiment, experiment.train_rows, metadata), split="train",
            minimum_events=int(evidence["minimum_events_per_cell"]),
            minimum_recordings=int(evidence["minimum_recordings_per_cell"]),
            brake_threshold_mps2=float(evidence["brake_threshold_mps2"]),
            minimum_brake_frames=int(evidence["minimum_brake_frames"]),
            merge_gap_frames=int(evidence["merge_gap_frames"]),
        )
        references.append(train_reference)
    for split in split_names:
        rows = getattr(experiment, f"{split}_rows")
        reference = build_reaction_event_reference(
            _split_arrays(experiment, rows, metadata), split=split,
            minimum_events=int(evidence["minimum_events_per_cell"]),
            minimum_recordings=int(evidence["minimum_recordings_per_cell"]),
            brake_threshold_mps2=float(evidence["brake_threshold_mps2"]),
            minimum_brake_frames=int(evidence["minimum_brake_frames"]),
            merge_gap_frames=int(evidence["merge_gap_frames"]),
            supported_cells=(
                train_reference.supported_cells
                if split != "train" and train_reference is not None
                else None
            ),
        )
        if split == "train":
            train_reference = reference
        reference.save(args.output_dir / split)
        references.append(reference)
        print({"split": split, "events": len(reference.events.row_index), "supported_cells": reference.supported_cells})
    assert_split_isolation(*references)


if __name__ == "__main__":
    main()
