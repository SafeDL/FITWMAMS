#!/usr/bin/env python3
"""Build train-defined conditional human-response references."""

from __future__ import annotations

import json
import sys

import numpy as np

from common import ROOT, load_config, result_root

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.data import prepare_experiment_data
from hierarchical_world_model.src.human_response_prior import build_human_response_priors
from hierarchical_world_model.src.reaction_evidence import ReactionEventReference


def split_arrays(bundle, rows: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "agent_states": np.asarray(bundle.arrays["agent_states"])[rows],
        "row_index": np.asarray(rows, np.int64),
    }


def main() -> None:
    config, base = load_config()
    experiment = prepare_experiment_data(base, ROOT)
    events_root = ROOT / config["paths"]["event_reference"]
    train_reference = ReactionEventReference.load(events_root / "train")
    train_arrays = split_arrays(experiment.bundle, experiment.train_rows)
    root = result_root(config) / "human_reference"
    root.mkdir(parents=True, exist_ok=True)
    audit = {}
    for split in ("train", "validation", "test"):
        reference = ReactionEventReference.load(events_root / split)
        rows = getattr(experiment, f"{split}_rows")
        prior, counts = build_human_response_priors(
            train_reference=train_reference,
            split_reference=reference,
            train_arrays=train_arrays,
            split_arrays=split_arrays(experiment.bundle, rows),
            split=split,
        )
        prior.save(root / split)
        audit[split] = counts
    (root / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
