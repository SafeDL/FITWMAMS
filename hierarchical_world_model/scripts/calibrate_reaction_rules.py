#!/usr/bin/env python3
"""Fit independent highD IDM/MOBIL rule references from the training split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from hierarchical_world_model.src.rule_models import fit_rule_models  # noqa: E402
from world_model.src.core.utils import ensure_dir, save_json  # noqa: E402


DEFAULT = ROOT / "hierarchical_world_model/config/reaction_policy.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate highD IDM and MOBIL diagnostic references.")
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--limit", type=int, default=None, help="debug only; formal calibration must omit this")
    parser.add_argument("--epochs", type=int, default=24)
    args = parser.parse_args()
    config = load_protocol_config(args.config.resolve())
    base = load_protocol_config(ROOT / config.get("base_config", "hierarchical_world_model/config/release.yaml"))
    experiment = prepare_experiment_data(base, ROOT)
    rows = experiment.train_rows if args.limit is None else experiment.train_rows[:int(args.limit)]
    model, report = fit_rule_models(experiment.bundle.arrays, rows, epochs=args.epochs)
    output = Path(config["paths"]["rule_model"])
    ensure_dir(output.parent); model.save(output)
    save_json({
        "training_rows": int(len(rows)), "full_training_split": args.limit is None,
        "report": report, "artifact": str(output),
        "status": "formal" if args.limit is None else "debug_subset_not_for_claims",
    }, output.parent / "calibration_summary.json")
    print(output)


if __name__ == "__main__":
    main()
