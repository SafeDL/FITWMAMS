#!/usr/bin/env python3
"""Evaluate a trained FIRM-WM checkpoint on held-out highD sequences."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.core.cli_config import materialize_config
from world_model.src.firm.evaluation import evaluate_firm_world_model
from world_model.src.core.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs/highd_firm_world_model.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results/highd_world_model/firm_world_model"),
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    config, config_path = materialize_config(
        Path(args.config),
        Path(args.output_dir),
        config_name=Path(args.config).name,
        resolve_path_keys=("sequence_cache_dir", "flow_schema"),
    )
    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path(config["paths"]["output_dir"]) / "checkpoints/best_firm_world_model.pt"
    )
    evaluate_firm_world_model(
        config,
        config_dir=config_path.parent,
        checkpoint=checkpoint.resolve(),
        max_sequences=args.max_sequences,
    )


if __name__ == "__main__":
    main()
