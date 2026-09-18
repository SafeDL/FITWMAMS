"""Validate the retained all-highD artifact before fitting/evaluation claims."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from dynamic_ar_idm.data import load_pairs

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="dynamic_ar_idm/artifacts/full_data/full_highd_25hz.npz")
parser.add_argument("--output", default="dynamic_ar_idm/artifacts/full_data/full_cohort_validation.json")
args = parser.parse_args()
pairs = load_pairs(args.dataset)
gaps = np.concatenate([pair["leader_x"] - pair["follower_x"] - float(pair["length_sum"]) for pair in pairs])
report = {"dataset": args.dataset, "pair_count": len(pairs), "recording_ids": sorted({int(pair["recording_id"]) for pair in pairs}),
          "decision_count": int(sum(len(pair["decision"]) for pair in pairs)), "native_fps": 25, "decision_fps": 5,
          "all_native_arrays_finite": bool(all(np.all(np.isfinite(pair[key])) for pair in pairs for key in ("follower_x", "follower_v", "leader_x", "leader_v"))),
          "decision_grid_is_five_native_ticks": bool(all(np.all(np.diff(pair["decision"]) == 5) for pair in pairs)),
          "minimum_native_gap_m": float(np.min(gaps)), "minimum_decisions_per_pair": int(min(len(pair["decision"]) for pair in pairs)),
          "selection_count_matches_project_manifest": len(pairs) == 251}
Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
print(args.output)
