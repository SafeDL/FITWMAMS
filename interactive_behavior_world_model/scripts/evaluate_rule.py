"""Run the deterministic IDM lane-keeping baseline on T1."""

from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch

from interactive_behavior_world_model.data.causal_cache import load_causal_cache
from interactive_behavior_world_model.data.manifest import load_benchmark_config
from interactive_behavior_world_model.evaluation.geometry import collision_matrix
from interactive_behavior_world_model.evaluation.metrics import (
    ensemble_channel_metrics,
    fair_energy_score,
    trajectory_errors,
)
from interactive_behavior_world_model.evaluation.rule_rollout import rollout_idm_lane_keep
from interactive_behavior_world_model.scripts.evaluate import (
    EVAL_INDICES,
    _metric_features,
    _metric_scales,
)


def evaluate(args):
    config, _ = load_benchmark_config(args.config)
    arrays, metadata, _ = load_causal_cache(args.config)
    rows = np.flatnonzero(
        metadata["split_index"] == {"validation": 1, "test": 2}[args.split]
    )
    if args.limit:
        rows = rows[: args.limit]
    scales = _metric_scales(arrays, metadata, Path(config["paths"]["output_dir"]))
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    records = []
    started = time.time()
    output = (
        Path(config["paths"]["output_dir"])
        / "methods/idm_lane_keep/0"
        / args.split
        / "t1_natural"
    )
    output.mkdir(parents=True, exist_ok=True)
    for begin in range(0, len(rows), args.batch_size):
        take = rows[begin : begin + args.batch_size]
        result = rollout_idm_lane_keep(
            np.asarray(arrays["agent_states"][take, :25]),
            np.asarray(arrays["agent_valid"][take, :25]),
            np.asarray(arrays["agent_states"][take, 25:, 0]),
            metadata["lengths_m"][take],
            metadata["widths_m"][take],
            np.asarray(arrays["map_polylines"][take]),
            np.asarray(arrays["map_polyline_valid"][take]),
            futures=args.futures,
            device=device,
        )
        target = np.asarray(arrays["agent_states"][take, 25:])
        target_valid = np.asarray(arrays["agent_valid"][take, 25:])
        initial = np.asarray(arrays["agent_states"][take, 24])
        generated = _metric_features(result.states, initial[:, None])
        observed = _metric_features(target[:, None], initial[:, None])[:, 0]
        collisions = collision_matrix(
            result.states,
            np.broadcast_to(target_valid[:, None], result.states.shape[:-1]),
            metadata["lengths_m"][take, None, None],
            metadata["widths_m"][take, None, None],
        )
        initial_collisions = collision_matrix(
            initial,
            target_valid[:, 0],
            metadata["lengths_m"][take],
            metadata["widths_m"][take],
        )
        for local, row in enumerate(take):
            active = target_valid[local, EVAL_INDICES, 1:]
            error = trajectory_errors(
                result.states[local, :, :, 1:],
                target[local, :, 1:],
                target_valid[local, :, 1:],
            )
            calibration = ensemble_channel_metrics(
                generated[local],
                observed[local],
                active,
                ("dx_m", "dy_m", "vx_mps", "vy_mps"),
            )
            new = collisions[local] & ~initial_collisions[local][None, None]
            records.append(
                {
                    "benchmark_id": config["benchmark"]["id"],
                    "method_id": "idm_lane_keep",
                    "fit_seed": 0,
                    "scenario_id": str(metadata["sequence_id"][row]),
                    "recording_id": int(metadata["recording_id"][row]),
                    "suite": "t1_natural",
                    "requested_futures": args.futures,
                    "completed_futures": args.futures,
                    "fair_energy_score": fair_energy_score(
                        generated[local], observed[local], active, scales
                    ),
                    **error,
                    **calibration,
                    "new_collision_probability": float(new.any((-1, -2, -3)).mean()),
                    "action_rewrite_rate": 0.0,
                    "failure_type": "",
                    "fallback_used": False,
                }
            )
    frame = pd.DataFrame(records)
    frame.to_csv(output / "per_scene_metrics.csv", index=False)
    names = [
        name
        for name in frame.select_dtypes("number").columns
        if name
        not in {"fit_seed", "recording_id", "requested_futures", "completed_futures"}
    ]
    summary = {
        "benchmark_id": config["benchmark"]["id"],
        "method_id": "idm_lane_keep",
        "fit_seed": 0,
        "split": args.split,
        "scenarios": len(frame),
        "completion_rate": len(frame) / max(len(rows), 1),
        "futures_per_scenario": args.futures,
        "wall_time_seconds": time.time() - started,
        "metrics": {name: float(frame[name].mean()) for name in names},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/benchmark_v1.yaml",
    )
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--futures", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    print(json.dumps(evaluate(a), indent=2))


if __name__ == "__main__":
    main()
