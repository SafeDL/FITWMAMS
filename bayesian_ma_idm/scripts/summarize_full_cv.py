#!/usr/bin/env python3
"""Create one auditable pair-weighted summary from recording-held-out CV files."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


METRICS = ("position_rmse_m", "speed_rmse_mps", "acceleration_crps_mps2",
           "position_90_coverage", "mean_interval_width_m")


def summarize(directory: str | Path, output: str | Path, *, models: tuple[str, ...] = ("b_idm", "ma_idm"),
              report_regex: str | None = None) -> Path:
    directory, output = Path(directory), Path(output)
    model_names, summaries = models, {}
    for model in model_names:
        paths = sorted(directory.glob(f"{model}_holdout_*.json"))
        if report_regex is not None:
            pattern = re.compile(report_regex.format(model=re.escape(model)))
            paths = [path for path in paths if pattern.fullmatch(path.name)]
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        if not rows or any("horizons" not in row for row in rows):
            raise RuntimeError(f"Missing completed {model} held-out reports in {directory}")
        pair_count = sum(int(row["pairs"]) for row in rows)
        if pair_count <= 0:
            raise RuntimeError("cannot weight an empty CV result")
        summaries[model] = {
            "fit_backends": sorted({str(row["fit_backend"]) for row in rows}),
            "held_out_recordings": [row["evaluation_recordings"] for row in rows],
            "pairs": pair_count, "anchors": sum(int(row["anchors"]) for row in rows),
            "futures": sorted({int(row["futures"]) for row in rows}),
            "pair_weighted_horizons": {
                horizon: {
                    **{metric: sum(int(row["pairs"]) * float(row["horizons"][horizon][metric]["mean"])
                                   for row in rows) / pair_count for metric in METRICS},
                    **({"position_interval_calibration": {
                        level: sum(int(row["pairs"]) * float(row["horizons"][horizon]["position_interval_calibration"][level]["mean"])
                                   for row in rows) / pair_count
                        for level in rows[0]["horizons"][horizon]["position_interval_calibration"]}}
                       if all("position_interval_calibration" in row["horizons"][horizon] for row in rows) else {}),
                }
                for horizon in ("3.0", "5.0")
            },
        }
    result = {
        "protocol": "recording-held-out rolling-origin; pair-weighted means; native 25 Hz plant / 5 Hz driver action",
        "source_directory": str(directory), "report_regex": report_regex, "models": summaries,
        "interpretation": "MA-IDM is a point/CRPS comparison against B-IDM. Position coverage is an acceptance diagnostic; values below .90 are not calibrated.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(ROOT / "bayesian_ma_idm/evidence/population"))
    parser.add_argument("--output", default=str(ROOT / "bayesian_ma_idm/evidence/population/summary.json"))
    parser.add_argument("--models", default="b_idm,ma_idm", help="comma-separated models present in input-dir")
    parser.add_argument("--report-regex", help="Optional full-match regex for report basenames; use {model} as a placeholder.")
    args = parser.parse_args()
    print(summarize(args.input_dir, args.output, models=tuple(args.models.split(",")), report_regex=args.report_regex))


if __name__ == "__main__":
    main()
