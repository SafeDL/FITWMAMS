#!/usr/bin/env python3
"""Select high-risk natural highD segment contexts for review/playback."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from process_highD.src.natural_evt_pipeline import select_natural_tail_contexts  # noqa: E402


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "highd_natural_evt.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--min-event-risk",
        type=float,
        default=None,
        help="Risk cutoff. Defaults to strict EVT POT exceedance threshold u when available.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Maximum number of sorted contexts to write; default 0 writes all.",
    )
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    select_natural_tail_contexts(
        config_path=Path(args.config).resolve(),
        min_event_risk=args.min_event_risk,
        top_k=int(args.top_k),
        output_csv=Path(args.output_csv).resolve() if args.output_csv else None,
    )


if __name__ == "__main__":
    main()
