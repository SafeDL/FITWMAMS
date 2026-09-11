#!/usr/bin/env python3
"""Build train-only empirical-coverage library and split query sets."""

from __future__ import annotations

import json
import sys

import numpy as np

from common import ROOT, load_config, result_root

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.data import prepare_experiment_data
from hierarchical_world_model.src.human_response_prior import build_human_response_support
from hierarchical_world_model.src.reaction_evidence import ReactionEventReference, assert_split_isolation


def arrays(bundle, rows: np.ndarray) -> dict[str, np.ndarray]:
    return {"agent_states": np.asarray(bundle.arrays["agent_states"])[rows], "row_index": np.asarray(rows, np.int64)}


def main() -> None:
    config, world = load_config()
    experiment = prepare_experiment_data(world, ROOT)
    event_root = ROOT / config["paths"]["event_reference"]
    references = {split: ReactionEventReference.load(event_root / split) for split in ("train", "validation", "test")}
    assert_split_isolation(*references.values())
    root = result_root(config) / "human_support"
    root.mkdir(parents=True, exist_ok=True)
    train_arrays = arrays(experiment.bundle, experiment.train_rows)
    audit = {}
    library = None
    for split in ("train", "validation", "test"):
        query_arrays = arrays(experiment.bundle, getattr(experiment, f"{split}_rows"))
        built, query, counts = build_human_response_support(
            train_reference=references["train"], split_reference=references[split],
            train_arrays=train_arrays, split_arrays=query_arrays, split=split,
        )
        library = built if library is None else library
        query.save(root / split)
        audit[split] = counts
    assert library is not None
    library.save(root / "train_library")
    (root / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
