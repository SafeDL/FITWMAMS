from __future__ import annotations
import argparse
from dynamic_ar_idm.ring import plot_ring, simulate_ring

parser = argparse.ArgumentParser(description="Paper Fig. 10-style Dynamic IDM ring simulation")
parser.add_argument("--posterior", default="dynamic_ar_idm/artifacts/full_posterior/dynamic_ar5_full_posterior.npz")
parser.add_argument("--seconds", type=float, default=3000.)
parser.add_argument("--vehicles", type=int, default=32)
parser.add_argument("--output", default="dynamic_ar_idm/artifacts/full_evaluation/ring_full_32.npz")
parser.add_argument("--figure", default="dynamic_ar_idm/artifacts/full_evaluation/figures/figure10_ring_full.png")
args = parser.parse_args()
path = simulate_ring(args.posterior, args.output, seconds=args.seconds, vehicles=args.vehicles)
print(plot_ring(path, args.figure))
