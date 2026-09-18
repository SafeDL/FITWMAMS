#!/usr/bin/env python3
"""Aggregate independently held-out class-conditioned MA-IDM folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="bayesian_ma_idm/evidence/class_conditioned")
    parser.add_argument("--recordings", default="25,26,36")
    parser.add_argument("--output", default="bayesian_ma_idm/evidence/class_conditioned/summary.json")
    args = parser.parse_args(); base = Path(args.base_dir)
    reports = []
    for recording in (int(value) for value in args.recordings.split(",")):
        path = base / f"holdout_recording_{recording}" / f"class_conditional_holdout_{recording}_summary.json"
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    total = sum(report["pairs"] for report in reports)
    output = {"extension": "vehicle-class-conditioned independent MA-IDM populations; not original paper hierarchy",
              "evaluation": "recording-held-out, class-specific training; one OOF posterior per pair", "pairs": total,
              "holdout_recordings": [report["holdout"] for report in reports], "futures": reports[0]["futures"],
              "aggregation": "pair-weighted mean of held-out fold summaries", "folds": reports, "pair_weighted_horizons": {}}
    for horizon in ("3.0", "5.0"):
        keys = reports[0]["horizons"][horizon]
        output["pair_weighted_horizons"][horizon] = {
            key: sum(report["pairs"] * report["horizons"][horizon][key] for report in reports) / total for key in keys}
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
