#!/usr/bin/env python3
"""Aggregate final OOF reports from the gap-regime conformal wrapper."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="bayesian_ma_idm/evidence/calibrated_intervals")
    parser.add_argument("--recordings", default="25,26,36")
    parser.add_argument("--output", default="bayesian_ma_idm/evidence/calibrated_intervals/summary.json")
    args = parser.parse_args(); directory = Path(args.input_dir)
    selections = [json.loads((directory / f"gap_regime_temporal_conformal_holdout_{int(recording)}.json").read_text(encoding="utf-8"))
                  for recording in args.recordings.split(",")]
    reports = [json.loads((directory / selection["oof_report"]).read_text(encoding="utf-8")) for selection in selections]
    total = sum(report["pairs"] for report in reports)
    output = {"extension": "gap-regime temporal split-conformal position band; not paper MA-IDM",
              "protocol": "outer recording held out; early-train MA-IDM; training-only gap-bin offsets; final OOF reports only",
              "pairs": total, "holdout_recordings": [selection["holdout_recording"] for selection in selections],
              "shortfall_quantile": selections[0]["shortfall_quantile"], "gap_bins": selections[0]["gap_bins"],
              "selections": selections, "pair_weighted_horizons": {}}
    for horizon in ("3.0", "5.0"):
        output["pair_weighted_horizons"][horizon] = {
            metric: sum(report["pairs"] * report["horizons"][horizon][metric]["mean"] for report in reports) / total
            for metric in ("position_rmse_m", "speed_rmse_mps", "acceleration_crps_mps2", "position_90_coverage", "mean_interval_width_m")}
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
