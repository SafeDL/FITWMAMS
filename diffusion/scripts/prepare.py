#!/usr/bin/env python3
"""Prepare Train normalization and the mandatory cut-in evaluation index."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import DataBundle  # noqa: E402
from diffusion.src.train import prepare_training_data  # noqa: E402
from hierarchical_world_model.src.core.utils import (  # noqa: E402
    ensure_dir,
    load_yaml,
    save_json,
    setup_logging,
)

CONFIG = ROOT / "diffusion/config.yaml"
PRE_CROSS_STEPS = (15, 20, 25, 30, 35, 45, 50)
WINDOW_STEPS = 100


def build_cutin_cohort(bundle: DataBundle, output: Path) -> dict:
    """Index upstream strict cut-ins for the mandatory Test diagnostic."""
    arrays = bundle.arrays
    source_schema = bundle.flow_schema
    natural = pd.read_csv(source_schema["source_segments_csv"])
    strict = natural.loc[natural["num_strict_cutins"] > 0].copy()
    sequence_row = {
        str(sequence_id): row
        for row, sequence_id in enumerate(np.asarray(arrays["sequence_id"]).astype(str))
    }
    slot_names = list(source_schema["slot_names"])
    events: list[tuple[int, int, int]] = []
    missing_sequence = 0
    unmodeled_target = 0
    for record in strict.itertuples(index=False):
        row = sequence_row.get(str(record.segment_id))
        if row is None:
            missing_sequence += 1
            continue
        target_id = int(record.primary_cutin_target_id)
        matching = [
            index
            for index, name in enumerate(slot_names)
            if int(getattr(record, f"{name}_id")) == target_id
        ]
        if not matching:
            unmodeled_target += 1
            continue
        crossing = int(record.primary_cutin_cross_frame - record.window_start_frame)
        if not 0 <= crossing < 150:
            raise ValueError(
                f"invalid crossing offset for {record.segment_id}: {crossing}"
            )
        events.append((int(row), int(matching[0] + 1), crossing))

    windows: list[tuple[int, int, int, int]] = []
    for row, slot, crossing in events:
        for pre in PRE_CROSS_STEPS:
            window_start = crossing - pre
            if window_start >= 0 and window_start + WINDOW_STEPS <= 149:
                windows.append((row, slot, crossing, window_start))

    index = np.asarray(windows, dtype=np.int64).reshape(-1, 4)
    output = ensure_dir(output)
    np.savez_compressed(
        output / "cut_in.npz",
        sequence_row=index[:, 0],
        target_slot_index=index[:, 1].astype(np.int8),
        crossing_step=index[:, 2].astype(np.int16),
        window_start_step=index[:, 3].astype(np.int16),
        split_index=np.asarray(arrays["split_index"])[index[:, 0]].astype(np.int8),
        is_evt_tail=np.asarray(arrays["is_evt_tail"])[index[:, 0]],
    )
    cache_manifest = Path(arrays["agent_states"].filename).parent / "manifest.json"
    event_rows = np.asarray([row for row, _, _ in events], dtype=np.int64)
    event_splits = np.asarray(arrays["split_index"])[event_rows]
    window_splits = np.asarray(arrays["split_index"])[index[:, 0]]
    manifest = {
        "name": "upstream_strict_cutin_crossing_4s",
        "source_sequence_manifest": str(cache_manifest),
        "storage": "index_only_no_trajectory_copy",
        "window_steps": WINDOW_STEPS,
        "dt_s": 0.04,
        "pre_cross_steps": list(PRE_CROSS_STEPS),
        "minimum_post_cross_steps": 50,
        "selection": [
            "consume upstream primary strict cut-in annotation",
            "use upstream full-recording laneId and all-vehicle semantics",
            "map the annotated target vehicle to one of six anchor slots",
            "inherit canonical full-horizon background-slot stability",
        ],
        "upstream_strict_segment_count": int(len(strict)),
        "missing_sequence_count": int(missing_sequence),
        "unmodeled_target_count": int(unmodeled_target),
        "event_count": len(events),
        "window_count": len(windows),
        "event_split_counts": {
            name: int(np.sum(event_splits == split))
            for split, name in enumerate(("train", "val", "test"))
        },
        "window_split_counts": {
            name: int(np.sum(window_splits == split))
            for split, name in enumerate(("train", "val", "test"))
        },
        "evt_tail_events": int(np.asarray(arrays["is_evt_tail"])[event_rows].sum()),
        "limitation": "Only the upstream primary cut-in is indexed per segment.",
    }
    save_json(manifest, output / "cut_in.json")
    return manifest


def main() -> None:
    setup_logging()
    config = load_yaml(CONFIG)
    bundle, contract = prepare_training_data(config, CONFIG.parent)
    build_cutin_cohort(bundle, Path(config["paths"]["output_dir"]) / "cohorts")
    print(
        f"Prepared lazy diffusion dataset: {len(bundle.arrays['sequence_id'])} "
        f"sequences, condition_dim={contract['condition_dim']}, "
        f"horizon={contract['horizon_steps']}"
    )


if __name__ == "__main__":
    main()
