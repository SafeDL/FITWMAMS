#!/usr/bin/env python3
"""Train the maintained hierarchical traffic world model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from hierarchical_world_model.src.train import train_world_model  # noqa: E402
from world_model.src.core.utils import setup_logging  # noqa: E402

CONFIG = (
    ROOT
    / "hierarchical_world_model/config/release.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the maintained hierarchical traffic world model."
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--stage", choices=("base", "stochastic_heads"), default="base")
    args = parser.parse_args()
    config_path = args.config.resolve()
    setup_logging()
    train_world_model(load_protocol_config(config_path), config_dir=ROOT, stage=args.stage)


if __name__ == "__main__":
    main()
