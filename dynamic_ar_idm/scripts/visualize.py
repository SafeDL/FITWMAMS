from __future__ import annotations
import argparse
from dynamic_ar_idm.visualize import make_figures

parser = argparse.ArgumentParser(description="Draw Dynamic-AR IDM paper-style evaluation figures")
parser.add_argument("--dataset", default="dynamic_ar_idm/artifacts/full_data/full_highd_25hz.npz")
parser.add_argument("--posterior", default="dynamic_ar_idm/artifacts/full_posterior/dynamic_ar5_full_posterior.npz")
parser.add_argument("--metrics", default="dynamic_ar_idm/artifacts/full_evaluation/natural_metrics_full_stable_map_rho.json")
parser.add_argument("--output-dir", default="dynamic_ar_idm/artifacts/full_evaluation/figures")
args = parser.parse_args()
for path in make_figures(args.dataset, args.posterior, args.metrics, args.output_dir): print(path)
