#!/usr/bin/env python3
"""Fit hierarchical and independently pooled Stage-B posteriors by train style."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesian_ma_idm.src.data import load_ragged_pairs
from multi_regime_bidm.src.pymc_stage_b import fit_style_nuts, fit_style_pooled_nuts
from multi_regime_bidm.src.style import decision_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "multi_regime_bidm/evidence/dataset/highd_multi_regime_cohort_25hz.npz"))
    parser.add_argument("--stage-dir", default=str(ROOT / "multi_regime_bidm/evidence/segmentation"))
    parser.add_argument("--output-dir", default=str(ROOT / "multi_regime_bidm/evidence/posterior"))
    parser.add_argument("--max-per-regime", type=int, default=400)
    parser.add_argument("--tune", type=int, default=600)
    parser.add_argument("--draws", type=int, default=600)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=.99)
    parser.add_argument("--pooled-only", action="store_true",
                        help="fit only the separately pooled controls, retaining existing hierarchical artifacts")
    args = parser.parse_args()
    labels = np.load(Path(args.stage_dir) / "stage_a_offline_labels.npz", allow_pickle=False)
    offsets, packed = labels["regime_offsets"], labels["offline_regime"]
    pair_index, style = labels["pair_index"], labels["style_id"]
    _, pairs = load_ragged_pairs(args.dataset)
    grids = [decision_grid(pairs[int(index)]) for index in pair_index]
    reports = []
    for style_id in range(3):
        members = np.flatnonzero(style == style_id)
        observation = np.concatenate([grids[index][0] for index in members])
        action = np.concatenate([grids[index][1] for index in members])
        regime = np.concatenate([packed[offsets[index]:offsets[index + 1]] for index in members])
        if not args.pooled_only:
            report = fit_style_nuts(observation, action, regime, args.output_dir, style_id=style_id,
                                    maximum_per_regime=args.max_per_regime, tune=args.tune, draws=args.draws,
                                    chains=args.chains, target_accept=args.target_accept)
            reports.append(json.loads(report.read_text(encoding="utf-8")))
            print(report)
        pooled_report = fit_style_pooled_nuts(
            observation,
            action,
            regime,
            args.output_dir,
            style_id=style_id,
            maximum_per_regime=args.max_per_regime,
            tune=args.tune,
            draws=args.draws,
            chains=args.chains,
            target_accept=args.target_accept,
        )
        reports.append(json.loads(pooled_report.read_text(encoding="utf-8")))
        print(pooled_report)
    output = Path(args.output_dir)
    reports = [json.loads(path.read_text(encoding="utf-8"))
               for path in sorted(output.glob("style_*_nuts_report.json"))]
    passed = all(item["divergences"] == 0 and item["max_rhat"] <= 1.01 and item["min_ess_bulk"] >= 100
                 for item in reports)
    summary = {"kind": "retained Stage-B NUTS diagnostic", "sampler_diagnostics_passed": bool(passed),
               "acceptance_rule": "all hierarchical and pooled style fits: divergences=0, max R-hat <= 1.01, min bulk ESS >= 100",
               "decision": "do_not_evaluate" if not passed else "evaluation_only_not_deployment_approved",
               "fits": [{"style_id": item["style_id"],
                          "model": item.get("model", "hierarchical_regime_b_idm"),
                          "max_rhat": item["max_rhat"], "min_ess_bulk": item["min_ess_bulk"],
                          "divergences": item["divergences"]} for item in reports]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "posterior_diagnostics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
