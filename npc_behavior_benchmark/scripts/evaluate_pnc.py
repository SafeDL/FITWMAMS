"""Run T3 fixed-PNC closed-loop evaluation for shared, action, or semantic policies."""

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
from npc_behavior_benchmark.evaluation.semantic_rollout import rollout_semantic_policy
from npc_behavior_benchmark.evaluation.geometry import collision_matrix, straight_road_offroad
from npc_behavior_benchmark.evaluation.pnc import PNC_IDS
from npc_behavior_benchmark.evaluation.rollout import rollout_shared_bc
from npc_behavior_benchmark.policies.rolling_action_policy import (
    ActionDiffusionConfig,
    RollingActionDiffusion,
    runtime_action_method_id,
)
from npc_behavior_benchmark.policies.shared_bc import SharedBCModel
from npc_behavior_benchmark.policies.learned_pnc import StructuredBCPNC


def validate_pnc_metrics(
    frame: pd.DataFrame,
    *,
    event_cohort: bool,
    expected_scenarios: int | None = None,
) -> None:
    """Validate a resumable T3 artifact before it is summarized or reused."""
    identity = "event_id" if event_cohort else "scenario_id"
    required = {identity, "requested_futures", "completed_futures"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"T3 metrics miss columns: {sorted(missing)}")
    if frame[identity].astype(str).duplicated().any():
        raise ValueError(f"T3 metrics contain duplicate {identity} values")
    if not (frame.requested_futures == frame.completed_futures).all():
        raise ValueError("T3 metrics contain incomplete futures")
    if (
        "failure_type" in frame
        and frame.failure_type.fillna("").astype(str).str.strip().ne("").any()
    ):
        raise ValueError("T3 metrics contain a recorded failure")
    numeric = (
        frame.drop(columns=["failure_type"], errors="ignore")
        .select_dtypes("number")
        .to_numpy(dtype=np.float64, copy=False)
    )
    if not np.isfinite(numeric).all():
        raise ValueError("T3 metrics contain NaN or Inf")
    if expected_scenarios is not None and len(frame) != int(expected_scenarios):
        raise ValueError(f"T3 metrics rows={len(frame)}, expected={expected_scenarios}")


def evaluate(args: argparse.Namespace) -> dict:
    config, _ = load_benchmark_config(args.config)
    arrays, metadata, _ = load_causal_cache(args.config)
    output_root = Path(config["paths"]["output_dir"])
    split_value = {"validation": 1, "test": 2}[args.split]
    events = None
    if args.cohort == "d1_events":
        all_events = pd.read_csv(output_root / "event_manifest.csv")
        events = all_events[
            (all_events.split_index == split_value)
            & (all_events.response_agent_index > 0)
        ].reset_index(drop=True)
        rows = np.arange(len(events), dtype=np.int64)
        horizon = 75
    else:
        rows = np.flatnonzero(metadata["split_index"] == split_value)
        horizon = 149
    eligible_scenarios = len(rows)
    if args.limit:
        rows = rows[: args.limit]
    selected_scenarios = len(rows)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if args.policy_kind in {"action", "semantic"}:
        model = RollingActionDiffusion(
            ActionDiffusionConfig(**checkpoint["model_config"]),
            checkpoint["action_mean"],
            checkpoint["action_std"],
        )
        if args.policy_kind == "semantic":
            method_id = "semantic_c2" if args.response_aware else "semantic_c1"
            if args.response_aware and args.deterministic_selection:
                method_id = "semantic_c3_deterministic"
        else:
            trained_method = str(checkpoint.get("method_id", "rolling_action_b0"))
            method_id = runtime_action_method_id(trained_method, args.warm_start)
    else:
        model = SharedBCModel(**checkpoint["model_config"])
        method_id = str(checkpoint.get("method_id", "shared_bc"))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    is_action_model = args.policy_kind in {"action", "semantic"}
    sampler = (
        "euler"
        if is_action_model and model.config.training_objective == "flow_matching"
        else "ddim"
    )
    ego_model = None
    ego_checkpoint = None
    if args.pnc in {"structured_bc", "structured_bc_lane_stable"}:
        if args.pnc_checkpoint is None:
            raise ValueError("--pnc-checkpoint is required for structured_bc")
        ego_checkpoint = torch.load(args.pnc_checkpoint, map_location="cpu")
        ego_model = StructuredBCPNC(**ego_checkpoint["model_config"])
        ego_model.load_state_dict(ego_checkpoint["model_state"])
        ego_model.to(device).eval()
    denoising_suffix = f"_{sampler}{args.denoising_steps}" if is_action_model else ""
    semantic_suffix = (
        f"_c{args.candidates}_r{args.response_samples}"
        if args.policy_kind == "semantic"
        else ""
    )
    partial_suffix = f"_partial{args.limit}" if args.limit else ""
    tag_suffix = f"_{args.output_tag}" if args.output_tag else ""
    cohort_suffix = "_d1_events" if args.cohort == "d1_events" else ""
    output = (
        args.checkpoint.parent
        / args.split
        / f"t3_fixed_pnc{cohort_suffix}_{method_id}_{args.pnc}{semantic_suffix}{denoising_suffix}{tag_suffix}{partial_suffix}"
    )
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "per_scene_metrics.csv"
    records = []
    if args.resume and metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        validate_pnc_metrics(existing, event_cohort=events is not None)
        records = existing.to_dict("records")
        if events is None:
            completed = set(existing.scenario_id.astype(str))
            rows = np.asarray(
                [
                    row
                    for row in rows
                    if str(metadata["sequence_id"][row]) not in completed
                ],
                dtype=np.int64,
            )
        else:
            completed = set(existing.event_id.astype(str))
            rows = np.asarray(
                [
                    row
                    for row in rows
                    if str(events.iloc[row].event_id) not in completed
                ],
                dtype=np.int64,
            )
        print(
            f"resuming with {len(records)} completed; {len(rows)} remaining", flush=True
        )
    started = time.time()
    for begin in range(0, len(rows), args.batch_size):
        take = rows[begin : begin + args.batch_size]
        if events is None:
            source_rows = take
            onsets = np.full(len(take), 24, np.int64)
            identifiers = metadata["sequence_id"][source_rows]
        else:
            selected_events = events.iloc[take]
            source_rows = selected_events.scenario_row.to_numpy(np.int64)
            onsets = selected_events.local_onset_frame.to_numpy(np.int64)
            identifiers = selected_events.event_id.astype(str).to_numpy()
        histories = np.stack(
            [
                np.asarray(arrays["agent_states"][row, onset - 24 : onset + 1])
                for row, onset in zip(source_rows, onsets)
            ]
        )
        history_valid = np.stack(
            [
                np.asarray(arrays["agent_valid"][row, onset - 24 : onset + 1])
                for row, onset in zip(source_rows, onsets)
            ]
        )
        ego_futures = np.stack(
            [
                np.asarray(
                    arrays["agent_states"][row, onset + 1 : onset + 1 + horizon, 0]
                )
                for row, onset in zip(source_rows, onsets)
            ]
        )
        common = (
            histories,
            history_valid,
            ego_futures,
            metadata["lengths_m"][source_rows],
            metadata["widths_m"][source_rows],
            np.asarray(arrays["map_polylines"][source_rows]),
            np.asarray(arrays["map_polyline_valid"][source_rows]),
            identifiers,
        )
        if args.policy_kind == "action":
            result = rollout_action_diffusion(
                model,
                checkpoint,
                *common,
                benchmark_id=config["benchmark"]["id"],
                fit_seed=int(checkpoint["fit_seed"]),
                futures=args.futures,
                inference_steps=args.denoising_steps,
                warm_start=args.warm_start,
                ego_policy_id=args.pnc,
                ego_policy_model=ego_model,
                ego_policy_checkpoint=ego_checkpoint,
                device=device,
            )
        elif args.policy_kind == "semantic":
            result = rollout_semantic_policy(
                model,
                checkpoint,
                *common,
                benchmark_id=config["benchmark"]["id"],
                fit_seed=int(checkpoint["fit_seed"]),
                futures=args.futures,
                inference_steps=args.denoising_steps,
                candidates=args.candidates,
                response_samples=args.response_samples,
                response_aware=args.response_aware,
                stochastic_selection=not args.deterministic_selection,
                temperature=args.temperature,
                ego_policy_id=args.pnc,
                ego_policy_model=ego_model,
                ego_policy_checkpoint=ego_checkpoint,
                device=device,
            )
        else:
            history, valid, ego, lengths, widths, maps, map_valid, ids = common
            result = rollout_shared_bc(
                model,
                checkpoint,
                history,
                valid,
                ego,
                metadata["agent_ids"][source_rows],
                lengths,
                widths,
                maps,
                map_valid,
                ids,
                benchmark_id=config["benchmark"]["id"],
                fit_seed=int(checkpoint["fit_seed"]),
                futures=args.futures,
                ego_policy_id=args.pnc,
                ego_policy_model=ego_model,
                ego_policy_checkpoint=ego_checkpoint,
                device=device,
            )
        target_valid = np.stack(
            [
                np.asarray(arrays["agent_valid"][row, onset + 1 : onset + 1 + horizon])
                for row, onset in zip(source_rows, onsets)
            ]
        )
        initial = np.stack(
            [
                np.asarray(arrays["agent_states"][row, onset])
                for row, onset in zip(source_rows, onsets)
            ]
        )
        initial_valid = np.stack(
            [
                np.asarray(arrays["agent_valid"][row, onset])
                for row, onset in zip(source_rows, onsets)
            ]
        )
        collisions = collision_matrix(
            result.states,
            np.broadcast_to(target_valid[:, None], result.states.shape[:-1]),
            metadata["lengths_m"][source_rows, None, None],
            metadata["widths_m"][source_rows, None, None],
        )
        initial_collisions = collision_matrix(
            initial,
            initial_valid,
            metadata["lengths_m"][source_rows],
            metadata["widths_m"][source_rows],
        )
        for local, row in enumerate(source_rows):
            new_collision = collisions[local] & ~initial_collisions[local][None, None]
            valid = np.broadcast_to(
                target_valid[local, None], result.states[local].shape[:-1]
            )
            offroad = straight_road_offroad(
                result.states[local],
                valid,
                metadata["lengths_m"][row],
                metadata["widths_m"][row],
                np.asarray(arrays["map_polylines"][row]),
                np.asarray(arrays["map_polyline_valid"][row]),
            )
            collision_time = new_collision.any((-1, -2))
            has_collision = collision_time.any(-1)
            stop = np.where(
                has_collision, collision_time.argmax(-1), result.states.shape[2] - 1
            )
            time_mask = np.arange(result.states.shape[2])[None] <= stop[:, None]
            member = np.arange(args.futures)
            progress = result.states[local, member, stop, 0, 0] - initial[local, 0, 0]
            initial_speed = np.linalg.norm(initial[local, 0, 2:4])
            final_speed = np.linalg.norm(
                result.states[local, member, stop, 0, 2:4], axis=-1
            )
            jerk = np.diff(result.applied_actions[local, ..., 0], axis=1) / 0.2
            active_actions = target_valid[local, 0, 1:]
            jerk_time = (np.arange(jerk.shape[1]) + 1)[None] * 5 <= stop[:, None]
            jerk_valid = jerk_time[..., None] & active_actions[None, None]
            record = {
                "benchmark_id": config["benchmark"]["id"],
                "method_id": method_id,
                "fit_seed": int(checkpoint["fit_seed"]),
                "scenario_id": str(metadata["sequence_id"][row]),
                "recording_id": int(metadata["recording_id"][row]),
                "suite": "t3_fixed_pnc",
                "ego_policy_id": args.pnc,
                "requested_futures": args.futures,
                "completed_futures": args.futures,
                "ego_collision_probability": float(
                    (new_collision[..., 0, 1:].any(-1) & time_mask).any(-1).mean()
                ),
                "npc_collision_probability": float(
                    (new_collision[..., 1:, 1:].any((-1, -2)) & time_mask)
                    .any(-1)
                    .mean()
                ),
                "ego_offroad_probability": float(
                    (offroad[..., 0] & time_mask).any(-1).mean()
                ),
                "npc_offroad_probability": float(
                    (offroad[..., 1:].any(-1) & time_mask).any(-1).mean()
                ),
                "ego_progress_m": float(progress.mean()),
                "ego_final_speed_loss_mps": float((initial_speed - final_speed).mean()),
                "npc_jerk_5hz_abs_mean_mps3": (
                    float(np.abs(jerk)[jerk_valid].mean()) if jerk_valid.any() else 0.0
                ),
                "action_rewrite_rate": (
                    float(result.rewrite[..., active_actions].mean())
                    if active_actions.any()
                    else 0.0
                ),
                "failure_type": "",
                "fallback_used": False,
            }
            if args.policy_kind == "semantic":
                record.update(
                    {
                        "nn_evaluations": int(
                            args.futures
                            * ((horizon + 4) // 5)
                            * args.denoising_steps
                            * args.candidates
                            * (
                                1
                                + (args.response_samples if args.response_aware else 0)
                            )
                        ),
                        "proxy_steps": int(
                            args.futures
                            * ((horizon + 4) // 5)
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
            if events is not None:
                event = events.iloc[int(take[local])]
                record.update(
                    {
                        "event_id": str(event.event_id),
                        "event_type": str(event.event_type),
                    }
                )
            records.append(record)
        if begin and begin % (args.batch_size * 10) == 0:
            print(f"completed {begin}/{len(rows)}", flush=True)
        if args.resume and (
            (begin // args.batch_size + 1) % 5 == 0 or begin + len(take) >= len(rows)
        ):
            frame = pd.DataFrame(records)
            temporary = metrics_path.with_suffix(".csv.tmp")
            frame.to_csv(temporary, index=False)
            temporary.replace(metrics_path)
    frame = pd.DataFrame(records)
    validate_pnc_metrics(
        frame, event_cohort=events is not None, expected_scenarios=selected_scenarios
    )
    frame.to_csv(metrics_path, index=False)
    names = [
        name
        for name in frame.select_dtypes("number").columns
        if name
        not in {"fit_seed", "recording_id", "requested_futures", "completed_futures"}
    ]
    summary = {
        "benchmark_id": config["benchmark"]["id"],
        "method_id": method_id,
        "fit_seed": int(checkpoint["fit_seed"]),
        "split": args.split,
        "suite": "t3_fixed_pnc",
        "cohort": args.cohort,
        "ego_policy_id": args.pnc,
        "scenarios": len(frame),
        "eligible_scenarios": eligible_scenarios,
        "is_partial": bool(args.limit),
        "denoising_steps": args.denoising_steps if is_action_model else None,
        "sampler": sampler if is_action_model else None,
        "network_evaluations_per_decision": (
            args.denoising_steps if is_action_model else None
        ),
        "candidates": args.candidates if args.policy_kind == "semantic" else None,
        "response_samples": (
            args.response_samples if args.policy_kind == "semantic" else None
        ),
        "pnc_checkpoint": (
            str(args.pnc_checkpoint) if args.pnc_checkpoint is not None else None
        ),
        "output_tag": args.output_tag,
        "termination": "metrics_truncated_at_first_new_collision",
        "horizon_frames": horizon,
        "completion_rate": len(frame) / max(eligible_scenarios, 1),
        "futures_per_scenario": args.futures,
        "wall_time_seconds": time.time() - started,
        "metrics": {name: float(frame[name].mean()) for name in names},
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
    parser.add_argument(
        "--policy-kind", choices=("shared", "action", "semantic"), required=True
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--pnc", choices=PNC_IDS, default="idm_cautious")
    parser.add_argument("--cohort", choices=("d0", "d1_events"), default="d0")
    parser.add_argument("--pnc-checkpoint", type=Path)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--response-samples", type=int, default=2)
    parser.add_argument("--response-aware", action="store_true")
    parser.add_argument("--deterministic-selection", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--futures", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
