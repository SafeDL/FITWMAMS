from __future__ import annotations
import argparse
from dynamic_ar_idm.evaluate import evaluate_paper_cohort

parser = argparse.ArgumentParser(description="25 Hz closed-loop full-cohort Dynamic IDM evaluation")
parser.add_argument("--dataset", default="dynamic_ar_idm/artifacts/full_data/full_highd_25hz.npz")
parser.add_argument("--posterior", default="dynamic_ar_idm/artifacts/full_posterior/dynamic_ar5_full_posterior.npz")
parser.add_argument("--output", default="dynamic_ar_idm/artifacts/full_evaluation/natural_metrics_full_stable_map_rho.json")
parser.add_argument("--futures", type=int, default=64)
parser.add_argument("--max-horizon", type=int, default=5)
parser.add_argument("--rho-mode", choices=("posterior", "map"), default="map")
args = parser.parse_args()
print(evaluate_paper_cohort(args.dataset, args.posterior, args.output, futures=args.futures, max_horizon_s=args.max_horizon, rho_mode=args.rho_mode))
