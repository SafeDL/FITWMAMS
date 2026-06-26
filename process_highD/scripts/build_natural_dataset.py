#!/usr/bin/env python3
"""Build a highD natural-prior dataset."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import build_action_dataset
from diffusion.src.utils import load_yaml


DEFAULT_CONFIG_PATH = (ROOT / "diffusion" / "scripts" / "configs" / "natural_following.yaml")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to natural diffusion dataset config.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    cfg_path = Path(args.config).resolve()
    build_action_dataset(load_yaml(cfg_path), config_dir=cfg_path.parent)


if __name__ == "__main__":
    main()
