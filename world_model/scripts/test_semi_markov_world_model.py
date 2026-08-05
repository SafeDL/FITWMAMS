#!/usr/bin/env python3
"""Test the final Semi-Markov world model on the prepared highD test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.core.cli_config import materialize_config, plot_training_losses
from world_model.src.core.utils import setup_logging
from world_model.src.semi_markov.evaluation import evaluate_semi_markov_world_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs/highd_semi_markov_world_model.yaml"))
    parser.add_argument("--checkpoint", default=str(ROOT / "results/highd_world_model/semi_markov_world_model/checkpoints/best_semi_markov_relational.pt"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/highd_world_model/semi_markov_world_model"))
    parser.add_argument("--max-sequences", type=int, default=0, help="Bounded development run only; 0 evaluates the complete test split.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    config, config_path = materialize_config(
        Path(args.config), Path(args.output_dir), config_name=Path(args.config).name,
        resolve_path_keys=("source_dataset_dir", "sequence_cache_dir", "flow_checkpoint", "flow_schema"),
        drop_path_keys=("highd_evt_config",),
    )
    config.setdefault("evaluation", {})["max_sequences"] = int(args.max_sequences)
    checkpoint = Path(args.checkpoint).resolve()
    evaluate_semi_markov_world_model(config, config_dir=config_path.parent, checkpoint=checkpoint)
    plot_training_losses(checkpoint.parent.parent / "training_history.csv", Path(config["paths"]["output_dir"]) / "training_validation_loss.png")
    print(Path(config["paths"]["output_dir"]) / "semi_markov_evaluation_summary.json")


if __name__ == "__main__":
    main()
