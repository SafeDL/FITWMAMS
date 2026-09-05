#!/usr/bin/env python3
"""Build an auditable highD natural-braking reference for A3/A4."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from hierarchical_world_model.src.reaction_realism import build_reaction_realism_reference  # noqa: E402


DEFAULT = ROOT / "hierarchical_world_model/config/reaction_naturalistic.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine train-only highD reaction realism support cells.")
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-events", type=int, default=None)
    parser.add_argument("--allowed-cells", default=None,
                        help="comma-separated train-supported cell ids for val/test reporting")
    args = parser.parse_args()
    config = load_protocol_config(args.config.resolve())
    base = load_protocol_config(ROOT / config.get("base_config", "hierarchical_world_model/config/release.yaml"))
    experiment = prepare_experiment_data(base, ROOT)
    rows = getattr(experiment, f"{args.split}_rows")
    arrays = {
        "agent_states": experiment.bundle.arrays["agent_states"][rows],
        "agent_valid": experiment.bundle.arrays["agent_valid"][rows],
    }
    allowed = None if args.allowed_cells is None else tuple(
        int(value) for value in args.allowed_cells.split(",") if value.strip()
    )
    reference = build_reaction_realism_reference(
        arrays, rows,
        minimum_events=(int(config["training"].get("support_minimum_events", 100))
                        if args.minimum_events is None else int(args.minimum_events)),
        window_frames=int(config["training"].get("realism_window_frames", 25)),
        allowed_cells=allowed,
        rollout_steps=int(config["training"].get("rollout_steps", 149)),
        source_split=args.split,
        replay_radius_m=float(config["training"].get("influence_radius_m", 50.0)),
    )
    reference.save(args.output_dir)
    print({"output": str(args.output_dir), "supported_cells": list(reference.supported_cells),
           "event_counts": reference.event_counts})


if __name__ == "__main__":
    main()
