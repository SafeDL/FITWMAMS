"""Audit a completed 25 Hz highD evaluation without re-running trajectories.

The paper's front-to-rear human comparison is literature based.  This script
therefore deliberately labels highD as an external-domain validation of the
longitudinal adapter, rather than converting its RMSE numbers into a claim of
paper-level human fit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from active_inference_driver.data import leader_brake_events, load_highd_pairs


METRICS = ("speed_rmse", "gap_rmse", "min_gap", "model_response_s", "human_response_s")


def _mean_or_none(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=Path("dynamic_ar_idm/artifacts/full_data/full_highd_25hz.npz"))
    parser.add_argument("--evaluation", type=Path, required=True,
                        help="path to highd_evaluation.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    available_events = sum(len(leader_brake_events(pair)) for pair in load_highd_pairs(args.dataset))
    rows = evaluation["event_rows"]
    variants = sorted({row["variant"] for row in rows})
    summaries: dict[str, dict[str, object]] = {}
    for variant in variants:
        subset = [row for row in rows if row["variant"] == variant]
        summaries[variant] = {
            "rows": len(subset),
            "collisions": int(sum(bool(row["collision"]) for row in subset)),
            "collision_rate": float(np.mean([bool(row["collision"]) for row in subset])),
            **{metric: _mean_or_none([row[metric] for row in subset]) for metric in METRICS},
        }

    report = {
        "dataset": str(args.dataset),
        "event_coverage": {
            "detected_leader_brake_events": available_events,
            "evaluated_events": evaluation["events"],
            "complete": evaluation["events"] == available_events,
            "window": {"pre_event_s": 2.0, "post_event_s": 5.0, "plant_hz": 25},
        },
        "adapter_contract": {
            "native_decision_dt_s": 0.2,
            "plant_ticks_per_decision": 5,
            "internal_horizon_s": 6.0,
            "no_steering_adaptation": True,
        },
        "summaries": summaries,
        "interpretation_boundary": {
            "paper_human_metric": "The paper validates front-to-rear response against published human-data summaries using a piecewise-linear velocity fit.",
            "highd_metric": "This report is an external 25 Hz closed-loop evaluation of the independent longitudinal adapter; it is not a replacement for that human-fit test.",
            "future_action_firewall": "At each 0.2 s decision the driver sees current observations only; recorded leader future positions are applied solely by the evaluation plant.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"coverage": report["event_coverage"], "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
