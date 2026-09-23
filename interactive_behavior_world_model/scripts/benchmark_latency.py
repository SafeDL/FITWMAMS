"""Measure batch-one causal decision latency and peak CUDA memory."""

from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np, torch

from interactive_behavior_world_model.data.causal_cache import load_causal_cache
from interactive_behavior_world_model.data.manifest import load_benchmark_config
from interactive_behavior_world_model.evaluation.action_rollout import rollout_action_diffusion
from interactive_behavior_world_model.evaluation.rollout import rollout_shared_bc
from interactive_behavior_world_model.evaluation.semantic_rollout import rollout_semantic_policy
from interactive_behavior_world_model.policies.rolling_action_policy import (
    ActionDiffusionConfig,
    RollingActionDiffusion,
)
from interactive_behavior_world_model.policies.shared_bc import SharedBCModel


def benchmark(args):
    config, _ = load_benchmark_config(args.config)
    arrays, metadata, _ = load_causal_cache(args.config)
    row = int(np.flatnonzero(metadata["split_index"] == 1)[0])
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    started = time.perf_counter()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if args.kind == "shared":
        model = SharedBCModel(**checkpoint["model_config"])
    else:
        model = RollingActionDiffusion(
            ActionDiffusionConfig(**checkpoint["model_config"]),
            checkpoint["action_mean"],
            checkpoint["action_std"],
        )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    initialization_ms = (time.perf_counter() - started) * 1000
    horizon = 10
    history = np.asarray(arrays["agent_states"][row : row + 1, :25])
    valid = np.asarray(arrays["agent_valid"][row : row + 1, :25])
    ego = np.asarray(arrays["agent_states"][row : row + 1, 25 : 25 + horizon, 0])
    lengths = metadata["lengths_m"][row : row + 1]
    widths = metadata["widths_m"][row : row + 1]
    maps = np.asarray(arrays["map_polylines"][row : row + 1])
    map_valid = np.asarray(arrays["map_polyline_valid"][row : row + 1])
    ids = metadata["sequence_id"][row : row + 1]

    def run():
        if args.kind == "shared":
            return rollout_shared_bc(
                model,
                checkpoint,
                history,
                valid,
                ego,
                metadata["agent_ids"][row : row + 1],
                lengths,
                widths,
                maps,
                map_valid,
                ids,
                benchmark_id=config["benchmark"]["id"],
                fit_seed=int(checkpoint["fit_seed"]),
                futures=1,
                device=device,
            )
        if args.kind == "semantic":
            return rollout_semantic_policy(
                model,
                checkpoint,
                history,
                valid,
                ego,
                lengths,
                widths,
                maps,
                map_valid,
                ids,
                benchmark_id=config["benchmark"]["id"],
                fit_seed=int(checkpoint["fit_seed"]),
                futures=1,
                inference_steps=args.denoising_steps,
                candidates=args.candidates,
                response_samples=args.response_samples,
                response_aware=args.response_aware,
                stochastic_selection=True,
                temperature=0.5,
                device=device,
            )
        return rollout_action_diffusion(
            model,
            checkpoint,
            history,
            valid,
            ego,
            lengths,
            widths,
            maps,
            map_valid,
            ids,
            benchmark_id=config["benchmark"]["id"],
            fit_seed=int(checkpoint["fit_seed"]),
            futures=1,
            inference_steps=args.denoising_steps,
            warm_start=args.warm_start,
            device=device,
        )

    for _ in range(3):
        run()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    values = []
    for _ in range(args.iterations):
        tick = time.perf_counter()
        run()
        if device.type == "cuda":
            torch.cuda.synchronize()
        values.append((time.perf_counter() - tick) * 1000 / 2.0)
    if args.kind == "semantic":
        method = "semantic_c2" if args.response_aware else "semantic_c1"
    else:
        # Preserve the trained method identity (notably B2 checkpoints) in the
        # cost report.  Warm/cold is a runtime configuration, not a method ID.
        method = str(
            checkpoint.get(
                "method_id", "rolling_action_b1" if args.warm_start else args.kind
            )
        )
    sampler = (
        "euler"
        if args.kind != "shared" and model.config.training_objective == "flow_matching"
        else "ddim"
    )
    label = (
        f"{args.kind}_{'warm' if args.warm_start else 'cold'}_{sampler}{args.denoising_steps}"
        if args.kind != "shared"
        else args.kind
    )
    report = {
        "method_id": method,
        "fit_seed": int(checkpoint["fit_seed"]),
        "kind": args.kind,
        "batch_size": 1,
        "decisions_per_timing": 2,
        "iterations": args.iterations,
        "p50_latency_ms_per_decision": float(np.quantile(values, 0.5)),
        "p95_latency_ms_per_decision": float(np.quantile(values, 0.95)),
        "mean_latency_ms_per_decision": float(np.mean(values)),
        "initialization_ms": initialization_ms,
        "model_parameter_mib": float(
            sum(value.numel() * value.element_size() for value in model.parameters())
            / 2**20
        ),
        "peak_incremental_cuda_memory_mib": (
            float(torch.cuda.max_memory_allocated() / 2**20)
            if device.type == "cuda"
            else 0.0
        ),
        "denoising_steps": args.denoising_steps,
        "network_evaluations_per_decision": args.denoising_steps,
        "sampler": sampler,
        "candidates": args.candidates if args.kind == "semantic" else 0,
        "response_samples": args.response_samples if args.kind == "semantic" else 0,
    }
    output = args.checkpoint.parent / "validation" / f"latency_{label}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/benchmark_v1.yaml",
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--kind", choices=("shared", "action", "semantic"), required=True)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--denoising-steps", type=int, default=4)
    p.add_argument("--warm-start", action="store_true")
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--response-samples", type=int, default=2)
    p.add_argument("--response-aware", action="store_true")
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    print(json.dumps(benchmark(a), indent=2))


if __name__ == "__main__":
    main()
