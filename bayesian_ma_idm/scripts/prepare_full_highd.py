#!/usr/bin/env python3
"""Materialize every source-compatible ≥50 s highD following pair at native 25 Hz."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from bayesian_ma_idm.src.data import build_author_full_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(ROOT / "highD_dataset/Matlab/data"))
    parser.add_argument("--output-dir", default=str(ROOT / "bayesian_ma_idm/evidence/highd_cohort"))
    parser.add_argument("--minimum-seconds", type=float, default=50.0)
    parser.add_argument("--velocity-policy", choices=("strict", "project_nonnegative"), default="strict",
                        help="strict physical filter, or causal v=max(v,0) projection for highD standstill jitter")
    parser.add_argument("--lane-change-policy", choices=("correct_meta", "source_loader_index"), default="correct_meta",
                        help="correct vehicle-ID metadata, or exact public loader's row-index behavior")
    args = parser.parse_args()
    print(build_author_full_dataset(args.raw_dir, args.output_dir, minimum_seconds=args.minimum_seconds,
                                    velocity_policy=args.velocity_policy, lane_change_policy=args.lane_change_policy))


if __name__ == "__main__":
    main()
