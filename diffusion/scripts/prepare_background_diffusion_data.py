#!/usr/bin/env python3
"""Validate the sequence cache and fit the diffusion data contract only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.train import prepare_training_data  # noqa: E402
from world_model.src.core.utils import load_yaml, setup_logging  # noqa: E402

DEFAULT_CONFIG = ROOT / "diffusion/configs/highd_background_diffusion.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir")
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
    bundle, contract = prepare_training_data(config, config_path.parent)
    print(
        f"Prepared lazy diffusion dataset: {len(bundle.arrays['sequence_id'])} "
        f"sequences, condition_dim={contract['condition_dim']}, "
        f"horizon={contract['horizon_steps']}"
    )


if __name__ == "__main__":
    main()
