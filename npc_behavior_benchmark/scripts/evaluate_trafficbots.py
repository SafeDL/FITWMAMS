"""Evaluate TrafficBots V1.5 on the strict-causal all-background T1 suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from npc_behavior_benchmark.data.causal_cache import load_causal_cache
from npc_behavior_benchmark.data.manifest import load_benchmark_config
from npc_behavior_benchmark.evaluation.geometry import collision_matrix
from npc_behavior_benchmark.evaluation.metrics import (
    ensemble_channel_metrics,
    fair_energy_score,
    trajectory_errors,
)
from npc_behavior_benchmark.scripts.evaluate import (
    EVAL_INDICES,
    _metric_features,
    _metric_scales,
)
from external_model_baselines.models.trafficbots.config import (
    load_config as load_trafficbots_config,
)
from external_model_baselines.models.trafficbots.data import adapt_highd_batch
from external_model_baselines.models.trafficbots.evaluation import load_checkpoint
from external_model_baselines.models.trafficbots.rollout import TrafficBotsHighDRollout


def _batch(
    states: np.ndarray, valid: np.ndarray, maps: np.ndarray, map_valid: np.ndarray
) -> dict:
    adapted = adapt_highd_batch(states, valid, maps, map_valid)
    batch = {key: torch.from_numpy(value.copy()) for key, value in adapted.items()}
    batch["canonical/states"] = torch.from_numpy(states.copy())
    batch["canonical/valid"] = torch.from_numpy(valid.copy())
    return batch


def _seed(benchmark_id: str, fit_seed: int, begin: int, rollout_id: int) -> int:
    payload = f"{benchmark_id}|trafficbots|{fit_seed}|{begin}|{rollout_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31 - 1)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    config, _ = load_benchmark_config(args.config)
    tb_config = load_trafficbots_config(args.trafficbots_config)
    arrays, metadata, _ = load_causal_cache(args.config)
    split_value = 1 if args.split == "validation" else 2
    rows = np.flatnonzero(metadata["split_index"] == split_value)
    if args.limit:
        rows = rows[: args.limit]
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    module = load_checkpoint(tb_config, args.checkpoint).to(device).eval()
    runner = TrafficBotsHighDRollout(module, common_backend=True)
    training_seed = int(
        args.training_seed
        if args.training_seed is not None
        else tb_config["training"].get("seed", tb_config["experiment"]["seed"])
    )
    fit_seed = int(args.fit_seed)
    method_id = str(args.method_id)
    output = (
        Path(config["paths"]["output_dir"])
        / f"methods/{method_id}/{fit_seed}"
        / args.split
        / "t1_natural"
    )
    output.mkdir(parents=True, exist_ok=True)
    scales = _metric_scales(arrays, metadata, Path(config["paths"]["output_dir"]))
    metric_path = output / "per_scene_metrics.csv"
    records = []
    if args.resume and metric_path.is_file():
        existing = pd.read_csv(metric_path)
        required = {"scenario_id", "requested_futures", "training_seed"}
        if not required.issubset(existing):
            raise ValueError(
                f"cannot resume incomplete TrafficBots T1 schema at {metric_path}"
            )
        if (existing["requested_futures"] != args.futures).any() or (
            existing["training_seed"] != training_seed
        ).any():
            raise ValueError(
                "TrafficBots T1 resume contract differs from existing rows"
            )
        if existing["scenario_id"].duplicated().any():
            raise ValueError(
                "TrafficBots T1 resume file contains duplicate scenario IDs"
            )
        records = existing.to_dict("records")
    completed = {str(record["scenario_id"]) for record in records}

    def checkpoint_rows() -> None:
        frame = pd.DataFrame(records)
        if frame["scenario_id"].duplicated().any():
            raise RuntimeError(
                "duplicate scenario written during TrafficBots T1 evaluation"
            )
        temporary = metric_path.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(metric_path)

    started = time.time()
    for begin in range(0, len(rows), args.batch_size):
        take = rows[begin : begin + args.batch_size]
        take = np.asarray(
            [row for row in take if str(metadata["sequence_id"][row]) not in completed],
            np.int64,
        )
        if not len(take):
            continue
        states = np.asarray(arrays["agent_states"][take, 24:174], np.float32)
        valid = np.asarray(arrays["agent_valid"][take, 24:174], bool)
        batch = _batch(
            states,
            valid,
            np.asarray(arrays["map_polylines"][take]),
            np.asarray(arrays["map_polyline_valid"][take]),
        )
        futures, rewrites = [], []
        for rollout_id in range(args.futures):
            seed = _seed(config["benchmark"]["id"], training_seed, begin, rollout_id)
            np.random.seed(seed)
            torch.manual_seed(seed)
            result = runner.run(batch, deterministic=False)
            futures.append(result.states.detach().cpu().numpy())
            requested = result.reference_actions.detach().cpu().numpy()
            applied = result.background_actions.detach().cpu().numpy()
            rewrites.append(np.any(np.abs(requested - applied) > 1.0e-6, axis=-1))
        generated = np.stack(futures, 1)
        rewrite = np.stack(rewrites, 1)
        target, target_valid, initial = states[:, 1:], valid[:, 1:], states[:, 0]
        generated_features = _metric_features(generated, initial[:, None])
        target_features = _metric_features(target[:, None], initial[:, None])[:, 0]
        generated_collision = collision_matrix(
            generated,
            np.broadcast_to(target_valid[:, None], generated.shape[:-1]),
            metadata["lengths_m"][take, None, None],
            metadata["widths_m"][take, None, None],
        )
        initial_collision = collision_matrix(
            initial,
            valid[:, 0],
            metadata["lengths_m"][take],
            metadata["widths_m"][take],
        )
        for local, row in enumerate(take):
            active = target_valid[local, EVAL_INDICES, 1:]
            error = trajectory_errors(
                generated[local, :, :, 1:],
                target[local, :, 1:],
                target_valid[local, :, 1:],
            )
            calibration = ensemble_channel_metrics(
                generated_features[local],
                target_features[local],
                active,
                ("dx_m", "dy_m", "vx_mps", "vy_mps"),
            )
            new_collision = (
                generated_collision[local] & ~initial_collision[local][None, None]
            )
            records.append(
                {
                    "benchmark_id": config["benchmark"]["id"],
                    "method_id": method_id,
                    "fit_seed": fit_seed,
                    "training_seed": training_seed,
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
                    "action_rewrite_rate": float(rewrite[local].mean()),
                    "failure_type": "",
                    "fallback_used": False,
                }
            )
            completed.add(str(metadata["sequence_id"][row]))
        checkpoint_rows()
        if begin and begin % (args.batch_size * 10) == 0:
            print(f"completed {begin}/{len(rows)}", flush=True)
    frame = pd.DataFrame(records)
    if len(frame) != len(rows):
        raise RuntimeError(
            f"TrafficBots T1 completion mismatch: {len(frame)}/{len(rows)}"
        )
    expected_ids = {str(metadata["sequence_id"][row]) for row in rows}
    observed_ids = set(frame["scenario_id"].astype(str))
    if observed_ids != expected_ids:
        raise RuntimeError(
            f"TrafficBots T1 scenario-ID mismatch: missing={len(expected_ids - observed_ids)}, "
            f"unexpected={len(observed_ids - expected_ids)}"
        )
    numeric = frame.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        raise RuntimeError("TrafficBots T1 produced non-finite formal metrics")
    metric_names = [
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
        "fit_seed": fit_seed,
        "training_seed": training_seed,
        "split": args.split,
        "scenarios": len(frame),
        "requested_scenarios": len(rows),
        "completion_rate": len(frame) / max(len(rows), 1),
        "futures_per_scenario": args.futures,
        "backend": "shared_kinematic_traffic_dynamics",
        "background_scope": "all_six",
        "wall_time_seconds": time.time() - started,
        "metrics": {name: float(frame[name].mean()) for name in metric_names},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/benchmark_v1.yaml",
    )
    parser.add_argument(
        "--trafficbots-config",
        type=Path,
        default=Path("external_model_baselines/models/trafficbots/config/highd.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/baselines/trafficbots_highd/checkpoints/best.ckpt"),
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--futures", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fit-seed", type=int, default=0)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--method-id", default="trafficbots_v15_all_background")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="atomically resume a matching partial per-scene file",
    )
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
