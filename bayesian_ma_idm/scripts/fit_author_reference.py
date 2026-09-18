#!/usr/bin/env python3
"""Fit the paper's public 20-pair MA-IDM protocol at original 5 Hz."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from bayesian_ma_idm.src.author_reference_pymc import fit_author_reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "bayesian_ma_idm/evidence/paper_reference/zhang_sun_author_20pairs_5hz.npz"))
    parser.add_argument("--output-dir", default=str(ROOT / "bayesian_ma_idm/evidence/paper_reference/nuts"))
    parser.add_argument("--tune", type=int, default=5000); parser.add_argument("--draws", type=int, default=2500)
    parser.add_argument("--chains", type=int, default=2); parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--target-accept", type=float, default=.90); parser.add_argument("--seed", type=int, default=16)
    args = parser.parse_args()
    print(fit_author_reference(args.dataset, args.output_dir, tune=args.tune, draws=args.draws, chains=args.chains,
                              cores=args.cores, target_accept=args.target_accept, seed=args.seed))


if __name__ == "__main__":
    main()
