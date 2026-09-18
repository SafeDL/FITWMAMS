from __future__ import annotations

import argparse

from dynamic_ar_idm.nuts_audit import audit_chainwise_predictions


parser = argparse.ArgumentParser(description="Chain-separated predictive audit for AR(0)/AR(5) NUTS fits")
parser.add_argument("--dataset", default="dynamic_ar_idm/artifacts/data/paper_highd_20pairs_25hz.npz")
parser.add_argument("--ar0", default="dynamic_ar_idm/artifacts/nuts/dynamic_ar0_paper_nuts_long.nc")
parser.add_argument("--ar5", default="dynamic_ar_idm/artifacts/nuts/dynamic_ar5_paper_nuts_long.nc")
parser.add_argument("--output", default="dynamic_ar_idm/artifacts/nuts/nuts_chainwise_predictive_audit.json")
parser.add_argument("--futures", type=int, default=64)
parser.add_argument("--max-horizon", type=int, default=5)
parser.add_argument("--chain-draws", type=int, default=512)
args = parser.parse_args()

print(audit_chainwise_predictions(args.dataset, args.ar0, args.ar5, args.output,
                                  futures=args.futures, max_horizon_s=args.max_horizon,
                                  chain_draws=args.chain_draws))
