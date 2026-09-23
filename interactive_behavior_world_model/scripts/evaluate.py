"""Run the common T1 suite for a trained shared-BC checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from interactive_behavior_world_model.data.causal_cache import load_causal_cache
from interactive_behavior_world_model.data.manifest import load_benchmark_config
from interactive_behavior_world_model.evaluation.geometry import collision_matrix
from interactive_behavior_world_model.evaluation.metrics import (
    ensemble_channel_metrics,
    fair_energy_score,
    trajectory_errors,
)
from interactive_behavior_world_model.evaluation.rollout import rollout_shared_bc
from interactive_behavior_world_model.policies.shared_bc import SharedBCModel

EVAL_INDICES = np.asarray([*range(4, 145, 5), 148], np.int64)


def _metric_features(states: np.ndarray, initial: np.ndarray) -> np.ndarray:
    selected = states[..., EVAL_INDICES, 1:, :]
    displacement = selected[..., :2] - initial[..., None, 1:, :2]
    return np.concatenate((displacement, selected[..., 2:4]), axis=-1)


def _metric_scales(arrays, metadata, output: Path) -> np.ndarray:
    schema_path = output / "metric_schema.json"
    if schema_path.exists():
        return np.asarray(
            json.loads(schema_path.read_text())["t1_feature_scale"], np.float64
        )
    rows = np.flatnonzero(metadata["split_index"] == 0)
    total = np.zeros(4, np.float64)
    square = np.zeros(4, np.float64)
    count = 0
    for start in range(0, len(rows), 512):
        take = rows[start : start + 512]
        states = np.asarray(arrays["agent_states"][take])
        values = _metric_features(states[:, None, 25:], states[:, None, 24])[:, 0]
        valid = np.asarray(arrays["agent_valid"][take, 25:])[:, EVAL_INDICES, 1:]
        selected = values[valid]
        total += selected.sum(0)
        square += np.square(selected).sum(0)
        count += len(selected)
    mean = total / count
    scale = np.sqrt(np.maximum(square / count - np.square(mean), 1.0e-6))
    schema = {
        "normalization_source": "strict_causal_train_split_only",
        "t1_features": ["dx_m", "dy_left_m", "vx_mps", "vy_left_mps"],
        "t1_evaluation_future_indices_zero_based": EVAL_INDICES.tolist(),
        "t1_feature_mean": mean.tolist(),
        "t1_feature_scale": scale.tolist(),
        "normalization_count": int(count),
        "fair_energy_dimension_normalization": "sqrt(valid_dimensions)",
    }
    schema_path.write_text(json.dumps(schema, indent=2) + "\n")
    return scale


def evaluate(args: argparse.Namespace) -> dict:
    config, _ = load_benchmark_config(args.config)
    arrays, metadata, _ = load_causal_cache(args.config)
    split_value = {"validation": 1, "val": 1, "test": 2}[args.split]
    rows = np.flatnonzero(metadata["split_index"] == split_value)
    if args.limit:
        rows = rows[: args.limit]
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    method_id = str(checkpoint.get("method_id", "shared_bc"))
    model = SharedBCModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)
    benchmark_output = Path(config["paths"]["output_dir"])
    scales = _metric_scales(arrays, metadata, benchmark_output)
    budget_suffix = "" if args.futures == 4 else f"_k{args.futures}"
    method_output = args.checkpoint.parent / args.split / f"t1_natural{budget_suffix}"
    method_output.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    started = time.time()

    for begin in range(0, len(rows), args.batch_size):
        take = rows[begin : begin + args.batch_size]
        result = rollout_shared_bc(
            model,
            checkpoint,
            np.asarray(arrays["agent_states"][take, :25]),
            np.asarray(arrays["agent_valid"][take, :25]),
            np.asarray(arrays["agent_states"][take, 25:, 0]),
            metadata["agent_ids"][take],
            metadata["lengths_m"][take],
            metadata["widths_m"][take],
            np.asarray(arrays["map_polylines"][take]),
            np.asarray(arrays["map_polyline_valid"][take]),
            metadata["sequence_id"][take],
            benchmark_id=config["benchmark"]["id"],
            fit_seed=int(checkpoint["fit_seed"]),
            futures=args.futures,
            device=device,
        )
        target = np.asarray(arrays["agent_states"][take, 25:])
        target_valid = np.asarray(arrays["agent_valid"][take, 25:])
        initial = np.asarray(arrays["agent_states"][take, 24])
        generated_features = _metric_features(result.states, initial[:, None])
        target_features = _metric_features(target[:, None], initial[:, None])[:, 0]
        generated_collision = collision_matrix(
            result.states,
            np.broadcast_to(target_valid[:, None], result.states.shape[:-1]),
            metadata["lengths_m"][take, None, None],
            metadata["widths_m"][take, None, None],
        )
        initial_collision = collision_matrix(
            initial,
            np.asarray(arrays["agent_valid"][take, 24]),
            metadata["lengths_m"][take],
            metadata["widths_m"][take],
        )
        for local, row in enumerate(take):
            active = target_valid[local, EVAL_INDICES, 1:]
            fes = fair_energy_score(
                generated_features[local], target_features[local], active, scales
            )
            error = trajectory_errors(
                result.states[local, :, :, 1:],
                target[local, :, 1:],
                target_valid[local, :, 1:],
            )
            calibration = ensemble_channel_metrics(
                generated_features[local],
                target_features[local],
                active,
                ("dx_m", "dy_m", "vx_mps", "vy_mps"),
            )
            initial_pairs = initial_collision[local][None, None]
            new_collision = generated_collision[local] & ~initial_pairs
            valid_actions = np.broadcast_to(
                target_valid[local, 0, 1:], result.rewrite[local].shape
            )
            records.append(
                {
                    "benchmark_id": config["benchmark"]["id"],
                    "method_id": method_id,
                    "fit_seed": int(checkpoint["fit_seed"]),
                    "scenario_id": str(metadata["sequence_id"][row]),
                    "recording_id": int(metadata["recording_id"][row]),
                    "suite": "t1_natural",
                    "requested_futures": args.futures,
                    "completed_futures": args.futures,
                    "fair_energy_score": fes,
                    **error,
                    **calibration,
                    "new_collision_probability": float(
                        new_collision.any((-1, -2, -3)).mean()
                    ),
                    "action_rewrite_rate": float(
                        result.rewrite[local][valid_actions].mean()
                    ),
                    "failure_type": "",
                    "fallback_used": False,
                }
            )
        if begin and begin % (args.batch_size * 20) == 0:
            print(f"completed {begin}/{len(rows)}", flush=True)
    frame = pd.DataFrame.from_records(records)
    frame.to_csv(method_output / "per_scene_metrics.csv", index=False)
    metric_names = [
        "fair_energy_score",
        "sample_mean_ADE_m",
        "joint_min_ADE_m",
        "sample_mean_FDE_m",
        "joint_min_corresponding_FDE_m",
        "mean_pairwise_trajectory_distance_m",
        "new_collision_probability",
        "action_rewrite_rate",
    ]
    metric_names += [
        name
        for name in frame.columns
        if name.startswith(("fair_CRPS_", "coverage_90_", "interval_width_90_"))
    ]
    summary = {
        "benchmark_id": config["benchmark"]["id"],
        "method_id": method_id,
        "fit_seed": int(checkpoint["fit_seed"]),
        "split": args.split,
        "scenarios": len(frame),
        "requested_scenarios": len(rows),
        "completion_rate": float(len(frame) / max(len(rows), 1)),
        "futures_per_scenario": args.futures,
        "wall_time_seconds": time.time() - started,
        "metrics": {name: float(frame[name].mean()) for name in metric_names},
    }
    (method_output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/benchmark_v1.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--futures", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
