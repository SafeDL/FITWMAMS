"""Run T1 for C1/C2/C3 semantic candidate selection."""

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
from interactive_behavior_world_model.evaluation.semantic_rollout import rollout_semantic_policy
from interactive_behavior_world_model.policies.rolling_action_policy import (
    ActionDiffusionConfig,
    RollingActionDiffusion,
)
from interactive_behavior_world_model.scripts.evaluate import (
    EVAL_INDICES,
    _metric_features,
    _metric_scales,
)


def validate_semantic_scene_metrics(
    frame: pd.DataFrame, *, expected_scenarios: int | None = None
) -> None:
    """Validate a full-denominator C-policy T1 artifact."""
    required = {"scenario_id", "requested_futures", "completed_futures"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"semantic T1 metrics miss columns: {sorted(missing)}")
    if frame.scenario_id.astype(str).duplicated().any():
        raise ValueError("semantic T1 metrics contain duplicate scenario IDs")
    if not (frame.requested_futures == frame.completed_futures).all():
        raise ValueError("semantic T1 metrics contain incomplete futures")
    if (
        "failure_type" in frame
        and frame.failure_type.fillna("").astype(str).str.strip().ne("").any()
    ):
        raise ValueError("semantic T1 metrics contain a recorded failure")
    numeric = (
        frame.drop(columns=["failure_type"], errors="ignore")
        .select_dtypes("number")
        .to_numpy(dtype=np.float64, copy=False)
    )
    if not np.isfinite(numeric).all():
        raise ValueError("semantic T1 metrics contain NaN or Inf")
    if expected_scenarios is not None and len(frame) != int(expected_scenarios):
        raise ValueError(
            f"semantic T1 metrics rows={len(frame)}, expected={expected_scenarios}"
        )


def evaluate(args):
    config, _ = load_benchmark_config(args.config)
    arrays, metadata, _ = load_causal_cache(args.config)
    split_value = {"validation": 1, "test": 2}[args.split]
    rows = np.flatnonzero(metadata["split_index"] == split_value)
    if args.limit:
        rows = rows[: args.limit]
    selected_rows = rows.copy()
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
    method_id = "semantic_c2" if args.response_aware else "semantic_c1"
    if args.response_aware and args.deterministic_selection:
        method_id = "semantic_c3_deterministic"
    partial_suffix = f"_partial{args.limit}" if args.limit else ""
    output = (
        args.checkpoint.parent
        / args.split
        / f"t1_{method_id}_c{args.candidates}_r{args.response_samples}_ddim{args.denoising_steps}{partial_suffix}"
    )
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "per_scene_metrics.csv"
    records = []
    if args.resume and metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        validate_semantic_scene_metrics(existing)
        records = existing.to_dict("records")
        completed = set(existing.scenario_id.astype(str))
        rows = np.asarray(
            [row for row in rows if str(metadata["sequence_id"][row]) not in completed],
            dtype=np.int64,
        )
        print(
            f"resuming with {len(records)} completed; {len(rows)} remaining", flush=True
        )
    started = time.time()
    for begin in range(0, len(rows), args.batch_size):
        take = rows[begin : begin + args.batch_size]
        result = rollout_semantic_policy(
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
            candidates=args.candidates,
            response_samples=args.response_samples,
            response_aware=args.response_aware,
            stochastic_selection=not args.deterministic_selection,
            temperature=args.temperature,
            device=device,
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
            new_collision = collisions[local] & ~initial_collisions[local][None, None]
            calibration = ensemble_channel_metrics(
                generated_features[local],
                target_features[local],
                active,
                ("dx_m", "dy_m", "vx_mps", "vy_mps"),
            )
            records.append(
                {
                    "method_id": method_id,
                    "fit_seed": int(checkpoint["fit_seed"]),
                    "scenario_id": str(metadata["sequence_id"][row]),
                    "recording_id": int(metadata["recording_id"][row]),
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
                    "action_rewrite_rate": float(result.rewrite[local].mean()),
                    "nn_evaluations": int(
                        args.futures
                        * 30
                        * args.denoising_steps
                        * args.candidates
                        * (1 + (args.response_samples if args.response_aware else 0))
                    ),
                    "proxy_steps": int(
                        args.futures
                        * 30
                        * args.candidates
                        * (
                            15
                            + (
                                1 + 15 * args.response_samples
                                if args.response_aware
                                else 0
                            )
                        )
                        * 5
                    ),
                    "requested_futures": args.futures,
                    "completed_futures": args.futures,
                    "failure_type": "",
                }
            )
        if begin and begin % (args.batch_size * 5) == 0:
            print(f"completed {begin}/{len(rows)}", flush=True)
        if args.resume and (
            (begin // args.batch_size + 1) % 5 == 0 or begin + len(take) >= len(rows)
        ):
            frame = pd.DataFrame(records)
            temporary = metrics_path.with_suffix(".csv.tmp")
            frame.to_csv(temporary, index=False)
            temporary.replace(metrics_path)
    frame = pd.DataFrame(records)
    validate_semantic_scene_metrics(frame, expected_scenarios=len(selected_rows))
    frame.to_csv(metrics_path, index=False)
    names = [
        "fair_energy_score",
        "sample_mean_ADE_m",
        "joint_min_ADE_m",
        "sample_mean_FDE_m",
        "joint_min_corresponding_FDE_m",
        "mean_pairwise_trajectory_distance_m",
        "new_collision_probability",
        "action_rewrite_rate",
    ]
    names += [
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
        "eligible_scenarios": int((metadata["split_index"] == split_value).sum()),
        "is_partial": bool(args.limit),
        "completion_rate": len(frame) / max(len(selected_rows), 1),
        "futures_per_scenario": args.futures,
        "candidates": args.candidates,
        "response_samples": args.response_samples,
        "proxy_steps": int(
            len(frame)
            * args.futures
            * 30
            * args.candidates
            * (15 + (1 + 15 * args.response_samples if args.response_aware else 0))
            * 5
        ),
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
    p.add_argument("--futures", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--denoising-steps", type=int, choices=(4, 8, 16), default=4)
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--response-samples", type=int, default=2)
    p.add_argument("--response-aware", action="store_true")
    p.add_argument("--deterministic-selection", action="store_true")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
