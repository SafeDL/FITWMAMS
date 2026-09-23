"""Audit D3 lateral stimulus completion before freezing evaluator doses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from interactive_behavior_world_model.data.causal_cache import load_causal_cache
from interactive_behavior_world_model.data.manifest import load_benchmark_config
from interactive_behavior_world_model.evaluation.stimuli import lateral_stimulus_trajectories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/benchmark_v1.yaml",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--doses", type=float, nargs="+", default=(0.6, 0.9, 1.2))
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    config, _ = load_benchmark_config(args.config)
    arrays, _, _ = load_causal_cache(args.config)
    root = Path(config["paths"]["output_dir"])
    frame = pd.read_csv(root / "lateral_probe_manifest.csv")
    frame = frame[
        frame.split_index == {"validation": 1, "test": 2}[args.split]
    ].reset_index(drop=True)
    report = {"split": args.split, "anchors": len(frame), "doses": {}}
    for dose in args.doses:
        errors = []
        shifts = []
        for begin in range(0, len(frame), args.batch_size):
            batch = frame.iloc[begin : begin + args.batch_size]
            rows = batch.scenario_row.to_numpy(np.int64)
            initial = np.asarray(arrays["agent_states"][rows, 24])
            valid = np.asarray(arrays["agent_valid"][rows, 24])
            index = batch.stimulus_agent_index.to_numpy(np.int64)
            _, changed = lateral_stimulus_trajectories(
                initial,
                valid,
                index,
                batch.target_lane_center_y_m.to_numpy(np.float32),
                np.full(len(batch), dose, np.float32),
            )
            local = np.arange(len(batch))
            final_y = changed[local, -1, index, 1]
            errors.extend(np.abs(final_y - batch.target_lane_center_y_m.to_numpy()))
            shifts.extend(np.abs(final_y - initial[local, index, 1]))
        error = np.asarray(errors)
        shift = np.asarray(shifts)
        report["doses"][f"{dose:g}"] = {
            "target_absolute_error_quantiles_m": np.quantile(
                error, (0.1, 0.5, 0.9)
            ).tolist(),
            "lateral_shift_quantiles_m": np.quantile(shift, (0.1, 0.5, 0.9)).tolist(),
            "completion_within_0p75m": float((error < 0.75).mean()),
        }
    output = root / f"lateral_stimulus_audit_{args.split}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
