#!/usr/bin/env python3
"""Evaluate the maintained hierarchical traffic world model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.evaluation import (  # noqa: E402
    evaluate_world_model,
)
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from world_model.src.core.utils import setup_logging  # noqa: E402

CONFIG = (
    ROOT
    / "hierarchical_world_model/config/release.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate factual fidelity, stochasticity and interventions on highD."
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    setup_logging()
    evaluate_world_model(load_protocol_config(config_path), config_dir=ROOT)


if __name__ == "__main__":
    main()
