#!/usr/bin/env python3
"""Aggregate fixed-event policy evaluations without pooling away seed variance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", nargs=3, action="append", metavar=("ALGORITHM", "SEED", "CSV"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    event_sets: dict[str, set[str]] = {}
    for algorithm, seed, path in args.evaluation:
        frame = pd.read_csv(path)
        required = {"event_id", "failure", "progress_m", "reward"}
        if missing := required - set(frame):
            raise KeyError(f"{path} lacks {sorted(missing)}")
        current = set(frame["event_id"].astype(str))
        event_sets.setdefault(algorithm, current)
        if event_sets[algorithm] != current:
            raise ValueError(f"{algorithm} seed evaluations use different event sets")
        rows.append(
            {
                "algorithm": algorithm,
                "seed": int(seed),
                "events": int(len(frame)),
                "failures": int(frame["failure"].sum()),
                "failure_rate": float(frame["failure"].mean()),
                "mean_progress_m": float(frame["progress_m"].mean()),
                "mean_reward": float(frame["reward"].mean()),
                "mean_abs_acceleration_mps2": float(
                    frame["mean_abs_acceleration_mps2"].mean()
                ),
                "mean_action_switches": float(frame["action_switches"].mean()),
            }
        )
    runs = pd.DataFrame(rows).sort_values(["algorithm", "seed"])
    reference_event_set = next(iter(event_sets.values()))
    if any(values != reference_event_set for values in event_sets.values()):
        raise ValueError("algorithms use different fixed evaluation events")
    aggregate = {}
    for algorithm, group in runs.groupby("algorithm", sort=True):
        metrics = {}
        for column in (
            "failure_rate", "mean_progress_m", "mean_reward",
            "mean_abs_acceleration_mps2", "mean_action_switches",
        ):
            metrics[column] = {
                "mean": float(group[column].mean()),
                "std_across_seeds": (
                    float(group[column].std(ddof=1)) if len(group) > 1 else None
                ),
                "raw": [float(value) for value in group[column]],
            }
        aggregate[algorithm] = {
            "seed_count": int(len(group)),
            "events_per_seed": int(group["events"].iloc[0]),
            "metrics": metrics,
        }
    result = {
        "algorithms": aggregate,
        "event_set_overlap": {
            f"{left}:{right}": len(event_sets[left] & event_sets[right])
            for left in event_sets for right in event_sets if left < right
        },
        "inference_unit": "training_seed; the same fixed events are paired within each seed",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output_dir / "seed_runs.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
