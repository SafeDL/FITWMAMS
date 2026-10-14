#!/usr/bin/env python3
"""Train the conditional background position-residual diffusion model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.core.utils import (
    load_yaml,
    setup_logging,
)  # noqa: E402
from diffusion.src.train import train_background_diffusion  # noqa: E402

CONFIG = ROOT / "diffusion/config.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    setup_logging()
    config = load_yaml(CONFIG)
    last = Path(config["paths"]["output_dir"]) / "checkpoints/last_training_state.pt"
    train_background_diffusion(
        config,
        config_dir=CONFIG.parent,
        resume=last if args.resume else None,
    )


if __name__ == "__main__":
    main()
