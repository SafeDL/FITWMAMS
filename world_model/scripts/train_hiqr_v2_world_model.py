#!/usr/bin/env python3
"""Train the independent prior-driven HiQR-v2 world model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.core.cli_config import materialize_config  # noqa: E402
from world_model.src.core.utils import setup_logging  # noqa: E402
from world_model.src.hiqr_v2.train import train_hiqr_v2_world_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs/highd_hiqr_v2_world_model.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results/highd_world_model/hiqr_v2_world_model"),
    )
    parser.add_argument("--resume", nargs="?", const="last")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    config, path = materialize_config(
        Path(args.config),
        Path(args.output_dir),
        config_name=Path(args.config).name,
        resolve_path_keys=(
            "sequence_cache_dir",
            "flow_schema",
            "source_dataset_dir",
            "hiqr_sidecar_output_dir",
        ),
    )
    resume = (
        Path(config["paths"]["output_dir"])
        / "checkpoints/last_hiqr_v2_training_state.pt"
        if args.resume == "last"
        else (None if args.resume is None else Path(args.resume).resolve())
    )
    train_hiqr_v2_world_model(config, config_dir=path.parent, resume=resume)


if __name__ == "__main__":
    main()
