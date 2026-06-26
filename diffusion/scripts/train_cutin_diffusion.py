#!/usr/bin/env python3
"""Train the highD cut-in action diffusion prior."""
from __future__ import annotations

from pathlib import Path

from diffusion.src.train import train_action_diffusion
from diffusion.src.utils import load_yaml, setup_logging


CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "natural_cutin.yaml"
LOG_LEVEL = "INFO"


def main() -> None:
    setup_logging(LOG_LEVEL)
    config = load_yaml(CONFIG_PATH)
    train_action_diffusion(config, config_dir=CONFIG_PATH.parent)


if __name__ == "__main__":
    main()
