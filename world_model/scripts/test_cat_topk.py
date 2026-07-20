#!/usr/bin/env python3
"""Evaluate a CAT-TopK checkpoint on its prepared START/ROLL test splits."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.cli_config import materialize_config, plot_training_losses
from world_model.src.evaluation import evaluate_world_model
from world_model.src.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config_dir = Path(__file__).parent / "configs"
    parser.add_argument("--config", default=str(config_dir / "highd_cat_topk_world_model.yaml"))
    parser.add_argument("--checkpoint", default=str(ROOT / "results/highd_world_model/cat_topk_world_model/checkpoints/best_world_model.pt"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/highd_world_model/cat_topk_world_model"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-branches", type=int, default=4)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    config, config_path = materialize_config(
        Path(args.config),
        Path(args.output_dir),
        config_name=Path(args.config).name,
        resolve_path_keys=("dataset_dir", "highd_evt_config"),
    )
    summary = evaluate_world_model(
        config,
        config_dir=config_path.parent,
        checkpoint=Path(args.checkpoint).resolve(),
        max_samples=args.max_samples,
        num_branches=args.num_branches,
    )
    plot_training_losses(
        Path(args.checkpoint).resolve().parent.parent / "training_history.csv",
        Path(config["paths"]["output_dir"]) / "training_validation_loss.png",
    )
    print(Path(config["paths"]["output_dir"]) / "evaluation_summary.json")
    return summary


if __name__ == "__main__":
    main()
