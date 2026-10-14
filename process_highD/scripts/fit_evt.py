#!/usr/bin/env python3
"""Fit POT/GPD to risks from the canonical preprocessed segment table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from process_highD.src.evt_pipeline import fit_evt  # noqa: E402

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    summary = fit_evt(args.config.resolve())
    print(
        json.dumps(
            {
                "calibration_scope": summary["calibration_scope"],
                "calibration_segments": summary["source"]["calibration_segments"],
                "u": summary["u"],
                "xi": summary["xi"],
                "beta": summary["beta"],
                "exceedance_rate": summary["exceedance_rate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
