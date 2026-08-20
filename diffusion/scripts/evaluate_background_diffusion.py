#!/usr/bin/env python3
"""Evaluate a highD joint-background diffusion model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.core.utils import load_yaml, setup_logging  # noqa: E402
from diffusion.src.evaluation import evaluate_background_diffusion  # noqa: E402

DEFAULT_CONFIG = ROOT / "diffusion/configs/highd_background_diffusion.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint")
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--samples-per-condition", type=int)
    parser.add_argument("--ddim-steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(config["paths"]["output_dir"]).resolve()
    )
    config["paths"]["output_dir"] = str(output)
    if args.max_sequences is not None:
        config["evaluation"]["max_sequences"] = int(args.max_sequences)
    if args.samples_per_condition is not None:
        config["evaluation"]["samples_per_condition"] = int(args.samples_per_condition)
    if args.ddim_steps is not None:
        config["evaluation"]["ddim_steps"] = int(args.ddim_steps)
    if args.guidance_scale is not None:
        config["evaluation"]["guidance_scale"] = float(args.guidance_scale)
    checkpoint = None if args.checkpoint is None else Path(args.checkpoint).resolve()
    evaluate_background_diffusion(
        config,
        config_dir=config_path.parent,
        checkpoint=checkpoint,
    )


if __name__ == "__main__":
    main()
