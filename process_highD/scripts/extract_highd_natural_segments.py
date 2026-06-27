#!/usr/bin/env python3
"""Extract fixed-length natural highD local traffic segments."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from process_highD.src.natural_evt_pipeline import build_natural_segments_dataset  # noqa: E402


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "highd_natural_evt.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract fixed-length natural highD driving segments."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--recordings",
        default=None,
        help='Optional override: "all" or comma-separated recording IDs.',
    )
    parser.add_argument(
        "--with-evt",
        action="store_true",
        help="Also fit EVT after segment extraction.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    build_natural_segments_dataset(
        config_path=Path(args.config).resolve(),
        recording_override=args.recordings,
        fit_evt=bool(args.with_evt),
    )


if __name__ == "__main__":
    main()
