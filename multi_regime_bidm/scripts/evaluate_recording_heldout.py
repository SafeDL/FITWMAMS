#!/usr/bin/env python3
"""Evaluate causal filtered multi-regime B-IDM on held-out highD recordings."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from multi_regime_bidm.src.evaluation import evaluate

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=str(ROOT / "multi_regime_bidm/evidence/dataset/highd_multi_regime_cohort_25hz.npz"))
    p.add_argument("--stage-a", default=str(ROOT / "multi_regime_bidm/evidence/segmentation"))
    p.add_argument("--posterior-dir", default=str(ROOT / "multi_regime_bidm/evidence/posterior"))
    p.add_argument("--output-dir", default=str(ROOT / "multi_regime_bidm/evidence/heldout"))
    p.add_argument("--futures", type=int, default=64)
    a = p.parse_args(); print(evaluate(a.dataset, a.stage_a, a.posterior_dir, a.output_dir, futures=a.futures))
if __name__ == "__main__": main()
