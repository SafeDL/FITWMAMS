"""Measure common-backend TrafficBots batch-one causal decision latency."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from interactive_behavior_world_model.data.causal_cache import load_causal_cache
from interactive_behavior_world_model.scripts.evaluate_trafficbots import _batch
from reproduction.models.trafficbots.config import (
    load_config as load_trafficbots_config,
)
from reproduction.models.trafficbots.evaluation import load_checkpoint
from reproduction.models.trafficbots.rollout import TrafficBotsHighDRollout


@torch.no_grad()
def benchmark(args: argparse.Namespace) -> dict:
    arrays, metadata, _ = load_causal_cache(args.config)
    row = int(np.flatnonzero(metadata["split_index"] == 1)[0])
    states = np.asarray(arrays["agent_states"][row : row + 1, 24:174], np.float32)
    valid = np.asarray(arrays["agent_valid"][row : row + 1, 24:174], bool)
    batch = _batch(
        states,
        valid,
        np.asarray(arrays["map_polylines"][row : row + 1]),
        np.asarray(arrays["map_polyline_valid"][row : row + 1]),
    )
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    config = load_trafficbots_config(args.trafficbots_config)
    started = time.perf_counter()
    module = load_checkpoint(config, args.checkpoint).to(device).eval()
    runner = TrafficBotsHighDRollout(
        module, require_follower_excluded=False, common_backend=True
    )
    initialization_ms = (time.perf_counter() - started) * 1000.0
    for _ in range(3):
        runner.run(batch, deterministic=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    values = []
    for index in range(args.iterations):
        torch.manual_seed(20260919 + index)
        tick = time.perf_counter()
        runner.run(batch, deterministic=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        values.append((time.perf_counter() - tick) * 1000.0 / 149.0)
    report = {
        "method_id": str(args.method_id),
        "fit_seed": int(args.fit_seed),
        "training_seed": int(
            args.training_seed
            if args.training_seed is not None
            else config["training"]["seed"]
        ),
        "kind": "trafficbots",
        "batch_size": 1,
        "decisions_per_timing": 149,
        "iterations": args.iterations,
        "p50_latency_ms_per_decision": float(np.quantile(values, 0.5)),
        "p95_latency_ms_per_decision": float(np.quantile(values, 0.95)),
        "mean_latency_ms_per_decision": float(np.mean(values)),
        "initialization_ms": initialization_ms,
        "model_parameter_mib": float(
            sum(value.numel() * value.element_size() for value in module.parameters())
            / 2**20
        ),
        "peak_incremental_cuda_memory_mib": (
            float(torch.cuda.max_memory_allocated() / 2**20)
            if device.type == "cuda"
            else 0.0
        ),
        "backend": "shared_kinematic_traffic_dynamics",
        "background_scope": "all_six",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


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
        default=Path("reproduction/models/trafficbots/config/highd.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/baselines/trafficbots_highd/checkpoints/best.ckpt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/interactive_behavior_world_model/benchmark_v1/methods/trafficbots_v15_all_background/0/validation/latency_trafficbots.json"
        ),
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--method-id", default="trafficbots_v15_all_background")
    parser.add_argument("--fit-seed", type=int, default=0)
    parser.add_argument("--training-seed", type=int)
    args = parser.parse_args()
    print(json.dumps(benchmark(args), indent=2))


if __name__ == "__main__":
    main()
