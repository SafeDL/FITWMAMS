#!/usr/bin/env python3
"""Build the canonical 5.96-second highD sequence cache without training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.core.sequential_dataset import (  # noqa: E402
    CANONICAL_SEQUENCE_PROTOCOL,
    prepare_sequential_dataset,
    sequence_cache_owner_dir,
)
from world_model.src.core.utils import load_yaml, setup_logging  # noqa: E402

CONFIG = ROOT / "process_highD/scripts/configs/highd_sequences.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    protocol = str(config.get("dataset", {}).get("sequence_protocol", ""))
    if protocol != CANONICAL_SEQUENCE_PROTOCOL:
        raise ValueError(
            f"dataset.sequence_protocol must be {CANONICAL_SEQUENCE_PROTOCOL!r}"
        )
    manifest = prepare_sequential_dataset(
        config,
        config_dir=config_path.parent,
        rebuild=args.rebuild,
        max_sequences=args.max_sequences,
    )
    owner = sequence_cache_owner_dir(config, config_dir=config_path.parent)
    print(f"Prepared canonical sequence cache: {owner / 'sequence_cache'}")
    print(
        f"{manifest['num_sequences']} sequences, "
        f"{manifest['future_transition_frames']} transitions each"
    )


if __name__ == "__main__":
    main()
