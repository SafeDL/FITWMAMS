#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from reproduction.evaluation.driver_reproduction.matched_evaluation import evaluate_matched


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--futures", type=int, default=64)
    parser.add_argument("--output-dir", default=str(ROOT / "results/driver_reproduction/matched_highd"))
    args = parser.parse_args()
    print(evaluate_matched(
        ROOT / "reproduction/models/multi_regime_bidm/evidence/dataset/highd_multi_regime_cohort_25hz.npz",
        ROOT / "reproduction/evaluation/driver_reproduction/artifacts/matched/b_idm_train25.npz",
        ROOT / "reproduction/evaluation/driver_reproduction/artifacts/matched/ma_idm_train25.npz",
        ROOT / "reproduction/evaluation/driver_reproduction/artifacts/matched/dynamic_ar5_train25.npz",
        ROOT / "reproduction/models/multi_regime_bidm/evidence/segmentation",
        ROOT / "reproduction/models/multi_regime_bidm/evidence/posterior",
        args.output_dir,
        futures=args.futures,
    ))


if __name__ == "__main__":
    main()
