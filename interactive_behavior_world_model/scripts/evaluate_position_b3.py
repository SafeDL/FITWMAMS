"""Run common T1 for B3 position diffusion plus waypoint tracking."""

from __future__ import annotations

import argparse, json, time
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
from interactive_behavior_world_model.evaluation.position_rollout import rollout_position_diffusion
from interactive_behavior_world_model.policies.rolling_action_policy import (
    ActionDiffusionConfig,
    RollingActionDiffusion,
)
from interactive_behavior_world_model.scripts.evaluate import (
    EVAL_INDICES,
    _metric_features,
    _metric_scales,
)


def evaluate(args):
    config, _ = load_benchmark_config(args.config)
    arrays, metadata, _ = load_causal_cache(args.config)
    rows = np.flatnonzero(
        metadata["split_index"] == (1 if args.split == "validation" else 2)
    )
    rows = rows[: args.limit] if args.limit else rows
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = RollingActionDiffusion(
        ActionDiffusionConfig(**checkpoint["model_config"]),
        checkpoint["action_mean"],
        checkpoint["action_std"],
    )
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)
    scales = _metric_scales(arrays, metadata, Path(config["paths"]["output_dir"]))
    method_id = "position_track_b3"
    output = (
        args.checkpoint.parent
        / args.split
        / f"t1_natural_{method_id}_ddim{args.denoising_steps}"
    )
    output.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.time()
    for begin in range(0, len(rows), args.batch_size):
        take = rows[begin : begin + args.batch_size]
        result = rollout_position_diffusion(
            model,
            checkpoint,
            np.asarray(arrays["agent_states"][take, :25]),
            np.asarray(arrays["agent_valid"][take, :25]),
            np.asarray(arrays["agent_states"][take, 25:, 0]),
            metadata["lengths_m"][take],
            metadata["widths_m"][take],
            np.asarray(arrays["map_polylines"][take]),
            np.asarray(arrays["map_polyline_valid"][take]),
            metadata["sequence_id"][take],
            benchmark_id=config["benchmark"]["id"],
            fit_seed=int(checkpoint["fit_seed"]),
            futures=args.futures,
            inference_steps=args.denoising_steps,
            warm_start=args.warm_start,
            device=device,
            tracker_lookahead_steps=args.tracker_lookahead_steps,
        )
        target = np.asarray(arrays["agent_states"][take, 25:])
        target_valid = np.asarray(arrays["agent_valid"][take, 25:])
        initial = np.asarray(arrays["agent_states"][take, 24])
        generated_features = _metric_features(result.states, initial[:, None])
        target_features = _metric_features(target[:, None], initial[:, None])[:, 0]
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
                generated_features[local],
                target_features[local],
                active,
                ("dx_m", "dy_m", "vx_mps", "vy_mps"),
            )
            new_collision = collisions[local] & ~initial_collisions[local][None, None]
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
                    "fair_energy_score": fair_energy_score(
                        generated_features[local],
                        target_features[local],
                        active,
                        scales,
                    ),
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
                    "nn_evaluations": 30 * args.denoising_steps,
                }
            )
        if begin and begin % (args.batch_size * 10) == 0:
            print(f"completed {begin}/{len(rows)}", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "per_scene_metrics.csv", index=False)
    names = [
        "fair_energy_score",
        "sample_mean_ADE_m",
        "joint_min_ADE_m",
        "sample_mean_FDE_m",
        "joint_min_corresponding_FDE_m",
        "mean_pairwise_trajectory_distance_m",
        "new_collision_probability",
        "action_rewrite_rate",
    ] + [
        name
        for name in frame
        if name.startswith(("fair_CRPS_", "coverage_90_", "interval_width_90_"))
    ]
    summary = {
        "benchmark_id": config["benchmark"]["id"],
        "method_id": method_id,
        "fit_seed": int(checkpoint["fit_seed"]),
        "split": args.split,
        "scenarios": len(frame),
        "requested_scenarios": len(rows),
        "completion_rate": len(frame) / max(len(rows), 1),
        "futures_per_scenario": args.futures,
        "denoising_steps": args.denoising_steps,
        "warm_start": args.warm_start,
        "tracker": "fixed_lookahead_waypoint_velocity_heading",
        "tracker_lookahead_steps": args.tracker_lookahead_steps,
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
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--futures", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--denoising-steps", type=int, choices=(4, 8, 16), default=8)
    p.add_argument("--warm-start", action="store_true")
    p.add_argument("--tracker-lookahead-steps", type=int, choices=(3, 4, 5), default=5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
