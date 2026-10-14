#!/usr/bin/env python3
"""Evaluate a highD joint-background diffusion model."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.core.utils import (
    load_yaml,
    setup_logging,
)  # noqa: E402
from diffusion.src.evaluation import evaluate_background_diffusion  # noqa: E402

CONFIG = ROOT / "diffusion/config.yaml"


def main() -> None:
    setup_logging()
    evaluate_background_diffusion(load_yaml(CONFIG), config_dir=CONFIG.parent)


if __name__ == "__main__":
    main()
