#!/usr/bin/env python3
"""Fit POT/GPD EVT for highD natural equal-length segment risks."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from process_highD.src.natural_evt_pipeline import refit_natural_evt  # noqa: E402


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "highd_natural_evt.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    refit_natural_evt(config_path=Path(args.config).resolve())


if __name__ == "__main__":
    main()
