#!/usr/bin/env python3
"""Train CAT-TopK from the prepared START/ROLL cache in a new output directory."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.cli_config import materialize_config
from world_model.src.train import train_world_model
from world_model.src.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config_dir = Path(__file__).parent / "configs"
    parser.add_argument("--config", default=str(config_dir / "highd_cat_topk_world_model.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/highd_world_model/cat_topk_world_model"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    config, config_path = materialize_config(
        Path(args.config),
        Path(args.output_dir),
        config_name=Path(args.config).name,
        resolve_path_keys=("dataset_dir", "highd_evt_config"),
    )
    summary = train_world_model(
        config,
        config_dir=config_path.parent,
        epochs=args.epochs,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        rebuild_dataset=args.rebuild_dataset,
    )
    print(summary.get("checkpoint"))


if __name__ == "__main__":
    main()
