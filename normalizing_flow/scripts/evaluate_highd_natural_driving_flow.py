#!/usr/bin/env python3
"""Evaluate, sample, and visualize the highD full-natural-driving flow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from normalizing_flow.src.evaluation import evaluate_natural_flow  # noqa: E402
from normalizing_flow.src.utils import load_yaml, setup_logging  # noqa: E402

CONFIG_PATH = (
    ROOT / "normalizing_flow/configs/highd_natural_driving_flow.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate and sample the maintained highD natural-driving Flow."
    )
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    evaluate_natural_flow(
        config,
        config_dir=config_path.parent,
        repo_root=ROOT,
    )


if __name__ == "__main__":
    main()
