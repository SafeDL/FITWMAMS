#!/usr/bin/env python3
"""Train HiQR-WM independently of QR-WM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.core.cli_config import materialize_config  # noqa: E402
from world_model.src.core.utils import setup_logging  # noqa: E402
from world_model.src.hiqr.train import train_hiqr_world_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs/highd_hiqr_world_model.yaml"),
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "results/highd_world_model/hiqr_world_model")
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="last",
        help="resume from a training state, or omit the path to use this run's last state",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    config, path = materialize_config(
        Path(args.config),
        Path(args.output_dir),
        config_name=Path(args.config).name,
        resolve_path_keys=("sequence_cache_dir", "flow_schema", "source_dataset_dir"),
    )
    resume = None
    if args.resume == "last":
        resume = (
            Path(config["paths"]["output_dir"]) / "checkpoints/last_training_state.pt"
        )
    elif args.resume:
        resume = Path(args.resume).resolve()
    train_hiqr_world_model(config, config_dir=path.parent, resume=resume)


if __name__ == "__main__":
    main()
