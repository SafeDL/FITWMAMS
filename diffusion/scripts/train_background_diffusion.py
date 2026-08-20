#!/usr/bin/env python3
"""Train the state-knot-conditioned background diffusion model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.core.utils import load_yaml, setup_logging  # noqa: E402
from diffusion.src.train import train_background_diffusion  # noqa: E402

DEFAULT_CONFIG = ROOT / "diffusion/configs/highd_background_diffusion.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", nargs="?", const="last")
    parser.add_argument("--warm-start")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-sequences", type=int)
    parser.add_argument("--max-val-sequences", type=int)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(config["paths"]["output_dir"]).resolve()
    )
    config["paths"]["output_dir"] = str(output_dir)
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)
    if args.max_train_sequences is not None:
        config["dataset"]["max_train_sequences"] = int(args.max_train_sequences)
    if args.max_val_sequences is not None:
        config["dataset"]["max_val_sequences"] = int(args.max_val_sequences)
    last = Path(config["paths"]["output_dir"]) / "checkpoints/last_training_state.pt"
    resume = (
        last
        if args.resume == "last"
        else (None if args.resume is None else Path(args.resume).resolve())
    )
    warm_start = None if args.warm_start is None else Path(args.warm_start).resolve()
    if resume is not None and warm_start is not None:
        parser.error("--resume and --warm-start are mutually exclusive")
    train_background_diffusion(
        config,
        config_dir=config_path.parent,
        resume=resume,
        warm_start=warm_start,
    )


if __name__ == "__main__":
    main()
