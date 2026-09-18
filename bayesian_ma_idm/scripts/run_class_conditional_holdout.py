#!/usr/bin/env python3
"""Evaluate a vehicle-class-conditioned MA-IDM population extension.

This is deliberately separate from Zhang--Sun's single hierarchical population:
the paper identifies different car/truck behaviour and names vehicle dynamics as
future work.  It is an OOF extension, not a replacement for the paper result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesian_ma_idm.src.data import load_ragged_pairs
from bayesian_ma_idm.src.full_evaluation import evaluate_full_population
from bayesian_ma_idm.src.full_model import fit_full_population


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "bayesian_ma_idm/evidence/highd_cohort/highd_author_full_25hz.npz"))
    parser.add_argument("--holdouts", default="25,26,36")
    parser.add_argument("--output-dir", default=str(ROOT / "bayesian_ma_idm/evidence/class_conditioned"))
    parser.add_argument("--futures", type=int, default=64)
    args = parser.parse_args(); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    _, pairs = load_ragged_pairs(args.dataset)
    recordings = sorted({int(pair["recording_id"]) for pair in pairs})
    for holdout in (int(value) for value in args.holdouts.split(",")):
        evaluate_holdout(args.dataset, pairs, recordings, holdout, output / f"holdout_recording_{holdout}", args.futures)


def evaluate_holdout(dataset: str, pairs: list[dict], recordings: list[int], holdout: int,
                     output: Path, futures: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    train = [recording for recording in recordings if recording != holdout]
    classes = sorted({str(pair["vehicle_class"]) for pair in pairs if int(pair["recording_id"]) == holdout})
    reports = {}
    for vehicle_class in classes:
        posterior = output / f"ma_idm_{vehicle_class.lower()}_holdout_{holdout}.npz"
        report = output / f"ma_idm_{vehicle_class.lower()}_holdout_{holdout}.json"
        fit_full_population(dataset, posterior, model="ma_idm", training_recordings=train,
                            training_vehicle_classes=[vehicle_class])
        evaluate_full_population(dataset, posterior, report, evaluation_recordings=[holdout],
                                 evaluation_vehicle_classes=[vehicle_class], futures=futures)
        reports[vehicle_class] = json.loads(report.read_text(encoding="utf-8"))
    total_pairs = sum(report["pairs"] for report in reports.values())
    combined = {"extension": "vehicle-class-conditioned independent MA-IDM populations; not original paper hierarchy",
                "holdout": holdout, "classes": classes, "pairs": total_pairs, "futures": futures,
                "aggregation": "pair-weighted mean of independently evaluated class reports; bootstrap intervals remain class reports",
                "horizons": {}}
    for horizon in ("3.0", "5.0"):
        combined["horizons"][horizon] = {}
        keys = ("position_rmse_m", "speed_rmse_mps", "acceleration_crps_mps2", "position_90_coverage", "mean_interval_width_m")
        for key in keys:
            combined["horizons"][horizon][key] = sum(report["pairs"] * report["horizons"][horizon][key]["mean"]
                                                        for report in reports.values()) / total_pairs
        for level in ("0.5", "0.8", "0.9", "0.95"):
            combined["horizons"][horizon][f"coverage_{level}"] = sum(
                report["pairs"] * report["horizons"][horizon]["position_interval_calibration"][level]["mean"]
                for report in reports.values()) / total_pairs
    path = output / f"class_conditional_holdout_{holdout}_summary.json"
    path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
