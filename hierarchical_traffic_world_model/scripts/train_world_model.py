#!/usr/bin/env python3
"""Train the maintained hierarchical traffic world model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_traffic_world_model.src.train import train_world_model  # noqa: E402
from world_model.src.core.utils import load_yaml, setup_logging  # noqa: E402

CONFIG = (
    ROOT
    / "hierarchical_traffic_world_model/configs/highd_hierarchical_world_model.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    setup_logging()
    train_world_model(load_yaml(config_path), config_dir=config_path.parent)


if __name__ == "__main__":
    main()
