#!/usr/bin/env python3
"""Prepare QR's canonical 149-transition highD cache without training.

The cache retains all 150 observed state points in each nominal six-second
highD segment: 25 START transitions followed by 124 ROLL transitions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.core.sequential_dataset import (
    QR_START_ROLL_PROTOCOL,
    prepare_sequential_dataset,
    sequence_cache_owner_dir,
)
from world_model.src.core.utils import load_yaml, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs/highd_qr_world_model.yaml"))
    parser.add_argument("--rebuild", action="store_true", help="Replace the configured QR cache.")
    parser.add_argument("--max-sequences", type=int, default=None, help="Optional bounded development cache; formal training requires 0.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    if str(config.get("dataset", {}).get("sequence_protocol", "")) != QR_START_ROLL_PROTOCOL:
        raise ValueError(f"{config_path} must set dataset.sequence_protocol={QR_START_ROLL_PROTOCOL!r}")
    manifest = prepare_sequential_dataset(
        config, config_dir=config_path.parent, rebuild=args.rebuild, max_sequences=args.max_sequences,
    )
    owner = sequence_cache_owner_dir(config, config_dir=config_path.parent)
    print(f"Prepared QR cache: {owner / 'sequence_cache'}")
    print(
        f"{manifest['future_transition_frames']} transitions = "
        f"{manifest['start_reconstruction_seconds']:.2f}s START + {manifest['roll_seconds']:.2f}s ROLL"
    )


if __name__ == "__main__":
    main()
