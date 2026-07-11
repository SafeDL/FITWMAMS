#!/usr/bin/env python3
"""Build the highD car-following natural-prior dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

from diffusion.src.data import build_action_dataset
from diffusion.src.utils import load_yaml, setup_logging


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "natural_following.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    args = parser.parse_args()
    setup_logging(args.log_level)
    cfg_path = Path(args.config).resolve()
    build_action_dataset(load_yaml(cfg_path), config_dir=cfg_path.parent)


if __name__ == "__main__":
    main()
