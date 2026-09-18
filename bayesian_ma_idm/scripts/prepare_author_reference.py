#!/usr/bin/env python3
"""Export the public author's fixed 20-pair, 5 Hz highD reference cohort."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from bayesian_ma_idm.src.author_reference import prepare_author_reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="/tmp/IDM_Bayesian_Calibration/data/cache")
    parser.add_argument("--output-dir", default=str(ROOT / "bayesian_ma_idm/evidence/paper_reference"))
    args = parser.parse_args()
    print(prepare_author_reference(args.cache_dir, args.output_dir))


if __name__ == "__main__":
    main()
