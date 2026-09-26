"""Evaluate shared, B-family, or C-policy rollouts on paired lateral D3 stimuli."""

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
from npc_behavior_benchmark.evaluation.geometry import collision_matrix, straight_road_offroad
from npc_behavior_benchmark.evaluation.rollout import rollout_shared_bc
from npc_behavior_benchmark.evaluation.semantic_rollout import rollout_semantic_policy
from npc_behavior_benchmark.evaluation.stimuli import (
    lateral_stimulus_trajectories,
    sustained_response_latency,
)
from npc_behavior_benchmark.policies.rolling_action_policy import (
    ActionDiffusionConfig,
    RollingActionDiffusion,
    runtime_action_method_id,
)
from npc_behavior_benchmark.policies.shared_bc import SharedBCModel

DOSES = (0.6, 0.9, 1.2)


def validate_lateral_probe_metrics(
    frame: pd.DataFrame, *, expected_conditions: int | None = None
) -> None:
    """Enforce the full-denominator D3 artifact contract before reporting it.

    A resumed CSV is authoritative input to the final result, so a duplicate
    probe-dose key or a non-finite metric must fail loudly instead of being
    hidden by an average.  The caller supplies the frozen denominator only
    for a completed (rather than an in-progress) run.
    """
    required = {"probe_id", "controller_rate_rps", "requested_pairs", "completed_pairs"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"lateral probe metrics miss columns: {sorted(missing)}")
    if frame.duplicated(["probe_id", "controller_rate_rps"]).any():
        raise ValueError("lateral probe metrics contain duplicate probe-dose keys")
    if not (frame.requested_pairs == frame.completed_pairs).all():
        raise ValueError("lateral probe metrics contain incomplete paired futures")
    # Empty string failure markers round-trip through CSV as an all-NaN float
    # column.  They are operational status, not a numerical metric; validate
    # their semantic content separately before checking real numeric fields.
    if (
        "failure_type" in frame
        and frame.failure_type.fillna("").astype(str).str.strip().ne("").any()
    ):
        raise ValueError("lateral probe metrics contain a recorded failure")
    numeric = (
        frame.drop(columns=["failure_type"], errors="ignore")
        .select_dtypes("number")
        .to_numpy(dtype=np.float64, copy=False)
    )
    if not np.isfinite(numeric).all():
        raise ValueError("lateral probe metrics contain NaN or Inf")
    if expected_conditions is not None and len(frame) != int(expected_conditions):
        raise ValueError(
            f"lateral probe metrics rows={len(frame)}, expected={expected_conditions}"
        )


def _expanded(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for dose in DOSES:
        part = frame.copy()
        part["controller_rate_rps"] = dose
        parts.append(part)
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["recording_id", "probe_id", "controller_rate_rps"])
        .reset_index(drop=True)
    )


def _combined_response(
    acceleration_delta: np.ndarray,
    yaw_delta: np.ndarray,
    family: str,
    avoidance_sign: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    expected_acceleration_sign = -1.0 if family == "merge" else 1.0
    longitudinal = sustained_response_latency(
        acceleration_delta,
        np.full(len(acceleration_delta), expected_acceleration_sign),
        threshold_mps2=0.2,
    )
    lateral = sustained_response_latency(
        yaw_delta,
        np.full(len(yaw_delta), avoidance_sign),
        threshold_mps2=0.015,
    )
    if family == "merge":
        responded = longitudinal[0] | lateral[0]
        latency = np.fmin(
            np.where(longitudinal[0], longitudinal[1], np.nan),
            np.where(lateral[0], lateral[1], np.nan),
        )
    else:
        responded, latency = longitudinal
    return responded, latency, longitudinal[0], lateral[0]


def evaluate(args: argparse.Namespace) -> dict:
    config, _ = load_benchmark_config(args.config)
    root = Path(config["paths"]["output_dir"])
    arrays, metadata, _ = load_causal_cache(args.config)
    anchors = pd.read_csv(root / "lateral_probe_manifest.csv")
    anchors = anchors[
        anchors.split_index == {"validation": 1, "test": 2}[args.split]
    ].copy()
    eligible_anchors = len(anchors)
    if args.limit:
        anchors = anchors.iloc[: args.limit].copy()
    probes = _expanded(anchors)
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
    trained_method = str(checkpoint.get("method_id", "rolling_action_b0"))
    semantic = args.policy_kind == "semantic"
    if semantic and not is_action_diffusion:
        raise ValueError("semantic policy requires a rolling-action checkpoint")
    if semantic:
        method_id = "semantic_c2" if args.response_aware else "semantic_c1"
        if args.response_aware and args.deterministic_selection:
            method_id = "semantic_c3_deterministic"
    elif is_action_diffusion:
        method_id = runtime_action_method_id(trained_method, args.warm_start)
    else:
        method_id = trained_method
    sampler = (
        "euler"
        if is_action_diffusion and model.config.training_objective == "flow_matching"
        else "ddim"
    )
    partial_suffix = f"_partial{args.limit}" if args.limit else ""
    semantic_suffix = (
        f"_c{args.candidates}_r{args.response_samples}" if semantic else ""
    )
    denoising_suffix = (
        f"_{sampler}{args.denoising_steps}" if is_action_diffusion else ""
    )
    output = (
        args.checkpoint.parent
        / args.split
        / f"t2b_lateral_probes_{method_id}{semantic_suffix}{denoising_suffix}{partial_suffix}"
    )
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "per_probe_metrics.csv"
    records = []
    if args.resume and metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        validate_lateral_probe_metrics(existing)
        records = existing.to_dict("records")
        completed = {
            (str(row.probe_id), float(row.controller_rate_rps))
            for row in existing.itertuples(index=False)
        }
        keep = [
            (str(row.probe_id), float(row.controller_rate_rps)) not in completed
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
        baseline_external, changed_external = lateral_stimulus_trajectories(
            initial,
            valid,
            stimulus,
            batch.target_lane_center_y_m.to_numpy(np.float32),
            batch.controller_rate_rps.to_numpy(np.float32),
        )
        external_mask = np.zeros((len(batch), 7), bool)
        external_mask[np.arange(len(batch)), stimulus] = True
        scenario_keys = np.asarray(
            [
                f"{probe}:{family}"
                for probe, family in zip(batch.probe_id, batch.stimulus_family)
            ]
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
            inference_steps=args.denoising_steps,
            device=device,
            exogenous_mask=external_mask,
        )
        if semantic:
            common.update(
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
            common["warm_start"] = args.warm_start
            natural = rollout_action_diffusion(
                *rollout_args, **common, exogenous_future=baseline_external
            )
            changed = rollout_action_diffusion(
                *rollout_args, **common, exogenous_future=changed_external
            )
        else:
            common.pop("inference_steps")
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
            yaw_delta = (
                changed.applied_actions[local, :, :, action_slot, 1]
                - natural.applied_actions[local, :, :, action_slot, 1]
            )
            side = np.sign(
                initial[local, stimulus_index, 1] - initial[local, response_index, 1]
            )
            avoidance_sign = float(-side if side else 1.0)
            responded, latency, longitudinal_response, lateral_response = (
                _combined_response(
                    acceleration_delta,
                    yaw_delta,
                    str(probe.stimulus_family),
                    avoidance_sign,
                )
            )
            response_shift = np.linalg.norm(
                changed.states[local, ..., response_index, :2]
                - natural.states[local, ..., response_index, :2],
                axis=-1,
            )
            unrelated = valid[local].copy()
            unrelated[[0, stimulus_index, response_index]] = False
            unrelated_shift = (
                float(
                    np.linalg.norm(
                        changed.states[local, ..., unrelated, :2]
                        - natural.states[local, ..., unrelated, :2],
                        axis=-1,
                    ).mean()
                )
                if unrelated.any()
                else 0.0
            )
            future_valid = np.broadcast_to(
                valid[local], natural.states[local].shape[:-1]
            )
            lengths = metadata["lengths_m"][int(probe.scenario_row)]
            widths = metadata["widths_m"][int(probe.scenario_row)]
            initial_collision = collision_matrix(
                initial[local], valid[local], lengths, widths
            )
            natural_collision = (
                collision_matrix(natural.states[local], future_valid, lengths, widths)
                & ~initial_collision[None, None]
            )
            changed_collision = (
                collision_matrix(changed.states[local], future_valid, lengths, widths)
                & ~initial_collision[None, None]
            )
            natural_offroad = straight_road_offroad(
                natural.states[local],
                future_valid,
                lengths,
                widths,
                np.asarray(arrays["map_polylines"][int(probe.scenario_row)]),
                np.asarray(arrays["map_polyline_valid"][int(probe.scenario_row)]),
            )
            changed_offroad = straight_road_offroad(
                changed.states[local],
                future_valid,
                lengths,
                widths,
                np.asarray(arrays["map_polylines"][int(probe.scenario_row)]),
                np.asarray(arrays["map_polyline_valid"][int(probe.scenario_row)]),
            )
            final_stimulus_y = changed_external[local, -1, stimulus_index, 1]
            record = {
                "probe_id": probe.probe_id,
                "scenario_id": probe.scenario_id,
                "recording_id": int(probe.recording_id),
                "probe_manifest_version": "npc_interaction_d3_lateral_v1",
                "method_id": method_id,
                "fit_seed": int(checkpoint["fit_seed"]),
                "stimulus_family": probe.stimulus_family,
                "controller_rate_rps": float(probe.controller_rate_rps),
                "requested_pairs": args.futures,
                "completed_pairs": args.futures,
                "response_probability": float(responded.mean()),
                "right_censored_latency_s": float(
                    np.where(responded, latency, 3.0).mean()
                ),
                "longitudinal_response_probability": float(
                    longitudinal_response.mean()
                ),
                "lateral_avoidance_probability": float(lateral_response.mean()),
                "peak_braking_response_mps2": float(
                    (-acceleration_delta).max(-1).mean()
                ),
                "peak_aligned_lateral_response_rps": float(
                    (yaw_delta * avoidance_sign).max(-1).mean()
                ),
                "mean_response_position_change_m": float(response_shift.mean()),
                "final_response_position_change_m": float(
                    response_shift[..., -1].mean()
                ),
                "unrelated_agent_position_change_m": unrelated_shift,
                "stimulus_final_target_error_m": float(
                    abs(final_stimulus_y - probe.target_lane_center_y_m)
                ),
                "natural_new_collision_probability": float(
                    natural_collision.any((-1, -2, -3)).mean()
                ),
                "stimulated_new_collision_probability": float(
                    changed_collision.any((-1, -2, -3)).mean()
                ),
                "natural_offroad_probability": float(
                    natural_offroad.any((-1, -2)).mean()
                ),
                "stimulated_offroad_probability": float(
                    changed_offroad.any((-1, -2)).mean()
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
    validate_lateral_probe_metrics(frame, expected_conditions=len(selected_probes))
    frame.to_csv(metrics_path, index=False)
    numeric = [
        name
        for name in frame.select_dtypes("number")
        if name
        not in {
            "fit_seed",
            "recording_id",
            "requested_pairs",
            "completed_pairs",
            "controller_rate_rps",
        }
    ]
    conditions = {}
    for key, group in frame.groupby(["stimulus_family", "controller_rate_rps"]):
        conditions[f"{key[0]}_{key[1]:g}"] = {
            "probes": len(group),
            **{name: float(group[name].mean()) for name in numeric},
        }
    summary = {
        "benchmark_id": config["benchmark"]["id"],
        "probe_manifest_version": "npc_interaction_d3_lateral_v1",
        "method_id": method_id,
        "fit_seed": int(checkpoint["fit_seed"]),
        "split": args.split,
        "anchors": len(anchors),
        "eligible_anchors": eligible_anchors,
        "probe_conditions": len(frame),
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
        "response_definition": "merge: sustained braking OR away-yaw; cut_out: sustained acceleration",
        "condition_metrics": conditions,
        "overall_metrics": {name: float(frame[name].mean()) for name in numeric},
        "wall_time_seconds": time.time() - started,
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--futures", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--denoising-steps", type=int, choices=(4, 8, 16), default=8)
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--policy-kind", choices=("action", "semantic"), default="action"
    )
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--response-samples", type=int, default=2)
    parser.add_argument("--response-aware", action="store_true")
    parser.add_argument("--deterministic-selection", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    print(json.dumps(evaluate(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
