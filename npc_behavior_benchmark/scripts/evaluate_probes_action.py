"""Evaluate shared, B-family, or C-policy rollouts on paired longitudinal D3 stimuli."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from npc_behavior_benchmark.data.causal_cache import load_causal_cache
from npc_behavior_benchmark.data.manifest import load_benchmark_config
from npc_behavior_benchmark.evaluation.action_rollout import rollout_action_diffusion
from npc_behavior_benchmark.evaluation.geometry import collision_matrix
from npc_behavior_benchmark.evaluation.rollout import rollout_shared_bc
from npc_behavior_benchmark.evaluation.semantic_rollout import rollout_semantic_policy
from npc_behavior_benchmark.evaluation.stimuli import (
    longitudinal_stimulus_trajectories,
    sustained_response_latency,
)
from npc_behavior_benchmark.policies.rolling_action_policy import (
    ActionDiffusionConfig,
    RollingActionDiffusion,
    runtime_action_method_id,
)
from npc_behavior_benchmark.policies.shared_bc import SharedBCModel

CONDITIONS = (
    ("brake", 1.5, -1.5),
    ("brake", 3.0, -3.0),
    ("brake", 5.0, -5.0),
    ("speed_recovery", 0.5, 0.5),
    ("speed_recovery", 1.0, 1.0),
    ("speed_recovery", 2.0, 2.0),
)


def _expanded_probes(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for family, dose, signed in CONDITIONS:
        part = frame.copy()
        part["stimulus_family"] = family
        part["dose_mps2"] = dose
        part["signed_acceleration_mps2"] = signed
        parts.append(part)
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["recording_id", "probe_id", "stimulus_family", "dose_mps2"])
        .reset_index(drop=True)
    )


def evaluate(args):
    config, _ = load_benchmark_config(args.config)
    output_root = Path(config["paths"]["output_dir"])
    arrays, metadata, _ = load_causal_cache(args.config)
    anchors = pd.read_csv(output_root / "probe_manifest.csv")
    anchors = anchors[
        anchors.split_index == {"validation": 1, "test": 2}[args.split]
    ].copy()
    eligible_anchors = len(anchors)
    if args.limit:
        anchors = anchors.iloc[: args.limit].copy()
    probes = _expanded_probes(anchors)
    selected_probes = probes.copy()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    is_action_diffusion = "action_horizon" in checkpoint["model_config"]
    if is_action_diffusion:
        model = RollingActionDiffusion(
            ActionDiffusionConfig(**checkpoint["model_config"]),
            checkpoint["action_mean"],
            checkpoint["action_std"],
        )
    else:
        model = SharedBCModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)
    trained_method = str(
        checkpoint.get(
            "method_id", "rolling_action_b0" if is_action_diffusion else "shared_bc"
        )
    )
    semantic = args.policy_kind == "semantic"
    if semantic and not is_action_diffusion:
        raise ValueError("semantic policy requires a rolling-action checkpoint")
    if semantic:
        method_id = "semantic_c2" if args.response_aware else "semantic_c1"
        if args.response_aware and args.deterministic_selection:
            method_id = "semantic_c3_deterministic"
    else:
        method_id = (
            runtime_action_method_id(trained_method, args.warm_start)
            if is_action_diffusion
            else trained_method
        )
    sampler = (
        "euler"
        if is_action_diffusion and model.config.training_objective == "flow_matching"
        else "ddim"
    )
    suffix = f"_{sampler}{args.denoising_steps}" if is_action_diffusion else ""
    if semantic:
        suffix = f"_c{args.candidates}_r{args.response_samples}{suffix}"
    suffix += f"_partial{args.limit}" if args.limit else ""
    output = args.checkpoint.parent / args.split / f"t2b_probes_{method_id}{suffix}"
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "per_probe_metrics.csv"
    records = []
    if args.resume and metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        records = existing.to_dict("records")
        completed = {
            (str(row.probe_id), str(row.stimulus_family), float(row.dose_mps2))
            for row in existing.itertuples(index=False)
        }
        keep = [
            (str(row.probe_id), str(row.stimulus_family), float(row.dose_mps2))
            not in completed
            for row in probes.itertuples(index=False)
        ]
        probes = probes[np.asarray(keep, dtype=bool)].copy()
        print(
            f"resuming with {len(records)} completed; {len(probes)} remaining",
            flush=True,
        )
    started = time.time()
    for begin in range(0, len(probes), args.batch_size):
        batch = probes.iloc[begin : begin + args.batch_size]
        rows = batch.scenario_row.to_numpy(np.int64)
        initial = np.asarray(arrays["agent_states"][rows, 24])
        valid = np.asarray(arrays["agent_valid"][rows, 24])
        stimulus = batch.stimulus_agent_index.to_numpy(np.int64)
        baseline_external, changed_external = longitudinal_stimulus_trajectories(
            initial,
            valid,
            stimulus,
            batch.signed_acceleration_mps2.to_numpy(np.float32),
        )
        external_mask = np.zeros((len(batch), 7), bool)
        external_mask[np.arange(len(batch)), stimulus] = True
        # All doses within a family reuse the same internal random stream.  The
        # natural/intervention pair is therefore matched, and dose trends are
        # not obscured by a different sampled future at each intensity.
        scenario_keys = np.asarray(
            [f"{p}:{f}" for p, f in zip(batch.probe_id, batch.stimulus_family)]
        )
        rollout_args = (
            model,
            checkpoint,
            np.asarray(arrays["agent_states"][rows, :25]),
            np.asarray(arrays["agent_valid"][rows, :25]),
            np.asarray(arrays["agent_states"][rows, 25:100, 0]),
        )
        common = dict(
            lengths_m=metadata["lengths_m"][rows],
            widths_m=metadata["widths_m"][rows],
            map_polylines=np.asarray(arrays["map_polylines"][rows]),
            map_valid=np.asarray(arrays["map_polyline_valid"][rows]),
            scenario_ids=scenario_keys,
            benchmark_id=config["benchmark"]["id"],
            fit_seed=int(checkpoint["fit_seed"]),
            futures=args.futures,
            device=device,
            exogenous_mask=external_mask,
        )
        if semantic:
            common.update(
                inference_steps=args.denoising_steps,
                candidates=args.candidates,
                response_samples=args.response_samples,
                response_aware=args.response_aware,
                stochastic_selection=not args.deterministic_selection,
                temperature=args.temperature,
            )
            natural = rollout_semantic_policy(
                *rollout_args, **common, exogenous_future=baseline_external
            )
            changed = rollout_semantic_policy(
                *rollout_args, **common, exogenous_future=changed_external
            )
        elif is_action_diffusion:
            common.update(
                inference_steps=args.denoising_steps, warm_start=args.warm_start
            )
            natural = rollout_action_diffusion(
                *rollout_args, **common, exogenous_future=baseline_external
            )
            changed = rollout_action_diffusion(
                *rollout_args, **common, exogenous_future=changed_external
            )
        else:
            shared_args = (*rollout_args, metadata["agent_ids"][rows])
            natural = rollout_shared_bc(
                *shared_args, **common, exogenous_future=baseline_external
            )
            changed = rollout_shared_bc(
                *shared_args, **common, exogenous_future=changed_external
            )
        for local, probe in enumerate(batch.itertuples(index=False)):
            response_index = int(probe.response_agent_index)
            stimulus_index = int(probe.stimulus_agent_index)
            action_slot = response_index - 1
            acceleration_delta = (
                changed.applied_actions[local, :, :, action_slot, 0]
                - natural.applied_actions[local, :, :, action_slot, 0]
            )
            expected_sign = -1.0 if probe.stimulus_family == "brake" else 1.0
            threshold_results = {
                threshold: sustained_response_latency(
                    acceleration_delta,
                    np.full(args.futures, expected_sign),
                    threshold_mps2=threshold,
                )
                for threshold in (0.1, 0.2, 0.3)
            }
            responded, latency = threshold_results[0.2]
            aligned = acceleration_delta * expected_sign
            natural_gap = (
                natural.states[local, ..., stimulus_index, 0]
                - natural.states[local, ..., response_index, 0]
            )
            changed_gap = (
                changed.states[local, ..., stimulus_index, 0]
                - changed.states[local, ..., response_index, 0]
            )
            half_length = 0.5 * (
                metadata["lengths_m"][int(probe.scenario_row), stimulus_index]
                + metadata["lengths_m"][int(probe.scenario_row), response_index]
            )
            natural_gap -= half_length
            changed_gap -= half_length
            response_shift = np.linalg.norm(
                changed.states[local, ..., response_index, :2]
                - natural.states[local, ..., response_index, :2],
                axis=-1,
            )
            unrelated = valid[local].copy()
            unrelated[[0, stimulus_index, response_index]] = False
            if unrelated.any():
                unrelated_shift = np.linalg.norm(
                    changed.states[local, ..., unrelated, :2]
                    - natural.states[local, ..., unrelated, :2],
                    axis=-1,
                ).mean()
            else:
                unrelated_shift = 0.0
            future_valid = np.broadcast_to(
                valid[local], natural.states[local].shape[:-1]
            )
            initial_collision = collision_matrix(
                initial[local],
                valid[local],
                metadata["lengths_m"][int(probe.scenario_row)],
                metadata["widths_m"][int(probe.scenario_row)],
            )
            collision_natural = (
                collision_matrix(
                    natural.states[local],
                    future_valid,
                    metadata["lengths_m"][int(probe.scenario_row)],
                    metadata["widths_m"][int(probe.scenario_row)],
                )
                & ~initial_collision[None, None]
            )
            collision_changed = (
                collision_matrix(
                    changed.states[local],
                    future_valid,
                    metadata["lengths_m"][int(probe.scenario_row)],
                    metadata["widths_m"][int(probe.scenario_row)],
                )
                & ~initial_collision[None, None]
            )
            record = {
                "probe_id": probe.probe_id,
                "scenario_id": probe.scenario_id,
                "recording_id": int(probe.recording_id),
                "method_id": method_id,
                "fit_seed": int(checkpoint["fit_seed"]),
                "stimulus_family": probe.stimulus_family,
                "dose_mps2": float(probe.dose_mps2),
                "requested_pairs": args.futures,
                "completed_pairs": args.futures,
                "response_probability": float(responded.mean()),
                "conditional_response_latency_s": (
                    float(np.nanmean(latency)) if responded.any() else np.nan
                ),
                "right_censored_latency_s": float(
                    np.where(responded, latency, 3.0).mean()
                ),
                "response_probability_threshold_0p1": float(
                    threshold_results[0.1][0].mean()
                ),
                "response_probability_threshold_0p3": float(
                    threshold_results[0.3][0].mean()
                ),
                "right_censored_latency_threshold_0p1_s": float(
                    np.where(
                        threshold_results[0.1][0], threshold_results[0.1][1], 3.0
                    ).mean()
                ),
                "right_censored_latency_threshold_0p3_s": float(
                    np.where(
                        threshold_results[0.3][0], threshold_results[0.3][1], 3.0
                    ).mean()
                ),
                "peak_aligned_acceleration_response_mps2": float(
                    aligned.max(-1).mean()
                ),
                "cumulative_aligned_acceleration_response_mps": float(
                    (aligned * 0.2).sum(-1).mean()
                ),
                "mean_response_position_change_m": float(response_shift.mean()),
                "final_response_position_change_m": float(
                    response_shift[..., -1].mean()
                ),
                "natural_minimum_gap_m": float(natural_gap.min(-1).mean()),
                "stimulated_minimum_gap_m": float(changed_gap.min(-1).mean()),
                "minimum_gap_change_m": float(
                    (changed_gap.min(-1) - natural_gap.min(-1)).mean()
                ),
                "unrelated_agent_position_change_m": float(unrelated_shift),
                "natural_new_collision_probability": float(
                    collision_natural.any((-1, -2, -3)).mean()
                ),
                "stimulated_new_collision_probability": float(
                    collision_changed.any((-1, -2, -3)).mean()
                ),
                "failure_type": "",
                "fallback_used": False,
            }
            if semantic:
                record.update(
                    {
                        "nn_evaluations": int(
                            2
                            * args.futures
                            * 15
                            * args.denoising_steps
                            * args.candidates
                            * (
                                1
                                + (args.response_samples if args.response_aware else 0)
                            )
                        ),
                        "proxy_steps": int(
                            2
                            * args.futures
                            * 15
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
                    }
                )
            records.append(record)
        if begin and begin % (args.batch_size * 10) == 0:
            print(f"completed {begin}/{len(probes)}", flush=True)
        if args.resume and (
            (begin // args.batch_size + 1) % 5 == 0 or begin + len(batch) >= len(probes)
        ):
            frame = pd.DataFrame(records)
            temporary = metrics_path.with_suffix(".csv.tmp")
            frame.to_csv(temporary, index=False)
            temporary.replace(metrics_path)
    frame = pd.DataFrame(records)
    frame.to_csv(metrics_path, index=False)
    metric_names = [
        "response_probability",
        "conditional_response_latency_s",
        "right_censored_latency_s",
        "response_probability_threshold_0p1",
        "response_probability_threshold_0p3",
        "right_censored_latency_threshold_0p1_s",
        "right_censored_latency_threshold_0p3_s",
        "peak_aligned_acceleration_response_mps2",
        "cumulative_aligned_acceleration_response_mps",
        "mean_response_position_change_m",
        "final_response_position_change_m",
        "natural_minimum_gap_m",
        "stimulated_minimum_gap_m",
        "minimum_gap_change_m",
        "unrelated_agent_position_change_m",
        "natural_new_collision_probability",
        "stimulated_new_collision_probability",
    ]
    grouped = frame.groupby(["stimulus_family", "dose_mps2"], dropna=False)
    by_condition = {}
    for key, group in grouped:
        by_condition[f"{key[0]}_{key[1]:g}"] = {
            "probes": int(len(group)),
            **{name: float(group[name].mean()) for name in metric_names},
        }
    summary = {
        "benchmark_id": config["benchmark"]["id"],
        "probe_manifest_version": "npc_interaction_d3_longitudinal_v1",
        "method_id": method_id,
        "fit_seed": int(checkpoint["fit_seed"]),
        "split": args.split,
        "anchors": int(len(anchors)),
        "eligible_anchors": int(eligible_anchors),
        "probe_conditions": int(len(frame)),
        "is_partial": bool(args.limit),
        "completion_rate": len(frame) / max(len(selected_probes), 1),
        "paired_futures": args.futures,
        "matched_pair_noise": True,
        "matched_noise_across_doses": True,
        "horizon_s": 3.0,
        "denoising_steps": args.denoising_steps if is_action_diffusion else None,
        "sampler": sampler if is_action_diffusion else None,
        "network_evaluations_per_decision": (
            args.denoising_steps if is_action_diffusion else None
        ),
        "warm_start": args.warm_start if is_action_diffusion and not semantic else None,
        "candidates": args.candidates if semantic else None,
        "response_samples": args.response_samples if semantic else None,
        "response_aware": args.response_aware if semantic else None,
        "stochastic_selection": (
            (not args.deterministic_selection) if semantic else None
        ),
        "nn_evaluations": int(frame["nn_evaluations"].sum()) if semantic else None,
        "proxy_steps": int(frame["proxy_steps"].sum()) if semantic else None,
        "response_threshold_mps2": 0.2,
        "response_threshold_sensitivity_mps2": [0.1, 0.3],
        "response_consecutive_decisions": 2,
        "condition_metrics": by_condition,
        "overall_metrics": {name: float(frame[name].mean()) for name in metric_names},
        "wall_time_seconds": time.time() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/benchmark_v1.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--futures", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--denoising-steps", type=int, choices=(4, 8, 16), default=8)
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument(
        "--policy-kind", choices=("action", "semantic"), default="action"
    )
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--response-samples", type=int, default=2)
    parser.add_argument("--response-aware", action="store_true")
    parser.add_argument("--deterministic-selection", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
