"""Evaluate explicit-goal T4a behavior control for a conditioned action model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from npc_behavior_benchmark.data.causal_cache import load_causal_cache
from npc_behavior_benchmark.data.dataset import event_behavior_condition
from npc_behavior_benchmark.data.manifest import load_benchmark_config
from npc_behavior_benchmark.evaluation.action_rollout import rollout_action_diffusion
from npc_behavior_benchmark.policies.rolling_action_policy import (
    ActionDiffusionConfig,
    RollingActionDiffusion,
)
from npc_behavior_benchmark.scripts.evaluate_action import (
    checkpoint_sha256,
    prepare_run_contract,
)

AMPLITUDE_TARGET_MPS = 2.0
LATERAL_COMPLETION_M = 1.75
HORIZON_FRAMES = 75


def validate_controlled_metrics(
    frame: pd.DataFrame, *, expected_events: int | None = None
) -> None:
    """Reject incomplete or malformed T4a records before reporting control."""
    required = {"event_id", "requested_futures", "completed_futures"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"T4a metrics miss columns: {sorted(missing)}")
    if frame.event_id.astype(str).duplicated().any():
        raise ValueError("T4a metrics contain duplicate event IDs")
    if not (frame.requested_futures == frame.completed_futures).all():
        raise ValueError("T4a metrics contain incomplete futures")
    if (
        "failure_type" in frame
        and frame.failure_type.fillna("").astype(str).str.strip().ne("").any()
    ):
        raise ValueError("T4a metrics contain a recorded failure")
    numeric = (
        frame.drop(columns=["failure_type"], errors="ignore")
        .select_dtypes("number")
        .to_numpy(dtype=np.float64, copy=False)
    )
    if not np.isfinite(numeric).all():
        raise ValueError("T4a metrics contain NaN or Inf")
    if expected_events is not None and len(frame) != int(expected_events):
        raise ValueError(f"T4a metrics rows={len(frame)}, expected={expected_events}")


def _mode_metrics(
    states: np.ndarray, initial: np.ndarray, target: int, mode: str
) -> dict[str, float]:
    """Score a fixed explicit goal; no logged future participates here."""
    value = np.asarray(states)[:, :, target]
    speed = np.linalg.norm(value[..., 2:4], axis=-1)
    initial_speed = float(np.linalg.norm(initial[target, 2:4]))
    lateral = value[..., 1] - float(initial[target, 1])
    if mode == "brake":
        achieved = initial_speed - speed.min(1)
        threshold = AMPLITUDE_TARGET_MPS
    elif mode == "recover":
        achieved = speed.max(1) - initial_speed
        threshold = AMPLITUDE_TARGET_MPS
    elif mode == "lane_left":
        achieved = lateral.max(1)
        threshold = LATERAL_COMPLETION_M
    elif mode == "lane_right":
        achieved = -lateral.min(1)
        threshold = LATERAL_COMPLETION_M
    else:
        raise ValueError(f"unsupported mode {mode!r}")
    completed = achieved >= threshold
    # Completion onset is measured from execution start; non-completion is
    # right-censored at the common 3 s horizon.
    if mode == "brake":
        progress = initial_speed - speed
    elif mode == "recover":
        progress = speed - initial_speed
    elif mode == "lane_left":
        progress = lateral
    else:
        progress = -lateral
    crossing = np.argmax(progress >= threshold, axis=1)
    completion_time = np.where(completed, (crossing + 1) * 0.04, HORIZON_FRAMES * 0.04)
    return {
        "completion_probability": float(completed.mean()),
        "right_censored_completion_time_s": float(completion_time.mean()),
        "mean_achieved_amplitude": float(achieved.mean()),
        "mean_amplitude_error": float(np.abs(achieved - threshold).mean()),
    }


def evaluate(args: argparse.Namespace) -> dict:
    config, _ = load_benchmark_config(args.config)
    root = Path(config["paths"]["output_dir"])
    arrays, metadata, _ = load_causal_cache(args.config)
    events = pd.read_csv(root / "event_manifest.csv")
    split_value = {"validation": 1, "test": 2}[args.split]
    events = events[events.split_index == split_value].copy()
    events["condition"] = [
        event_behavior_condition(row) for _, row in events.iterrows()
    ]
    events = events[events.condition.notna()].reset_index(drop=True)
    eligible_events = len(events)
    if args.limit:
        events = events.iloc[: args.limit].copy()
    selected_events = len(events)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if int(checkpoint["model_config"].get("feature_dim", 12)) != 16:
        raise ValueError("T4a requires a 16-feature explicitly conditioned checkpoint")
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
    suffix = f"_partial{args.limit}" if args.limit else ""
    output = (
        args.checkpoint.parent
        / args.split
        / f"t4a_controlled_{checkpoint.get('method_id', 'controlled')}_ddim{args.denoising_steps}{suffix}"
    )
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "per_event_metrics.csv"
    contract = {
        "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        "split": args.split,
        "futures": args.futures,
        "denoising_steps": args.denoising_steps,
        "limit": args.limit,
        "suite": "t4a_controlled",
    }
    prepare_run_contract(
        output / "run_contract.json",
        contract,
        resume_with_metrics=args.resume and metrics_path.exists(),
    )
    records = []
    completed: set[str] = set()
    if args.resume and metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        validate_controlled_metrics(existing)
        records = existing.to_dict("records")
        completed = set(existing.event_id.astype(str))
    events = events[~events.event_id.astype(str).isin(completed)].reset_index(drop=True)
    started = time.time()
    for begin in range(0, len(events), args.batch_size):
        batch = events.iloc[begin : begin + args.batch_size]
        rows = batch.scenario_row.to_numpy(np.int64)
        onset = batch.local_onset_frame.to_numpy(np.int64)
        history = np.stack(
            [
                np.asarray(arrays["agent_states"][row, point - 24 : point + 1])
                for row, point in zip(rows, onset)
            ]
        )
        valid = np.stack(
            [
                np.asarray(arrays["agent_valid"][row, point - 24 : point + 1])
                for row, point in zip(rows, onset)
            ]
        )
        ego_future = np.stack(
            [
                np.asarray(
                    arrays["agent_states"][
                        row, point + 1 : point + 1 + HORIZON_FRAMES, 0
                    ]
                )
                for row, point in zip(rows, onset)
            ]
        )
        rollout = rollout_action_diffusion(
            model,
            checkpoint,
            history,
            valid,
            ego_future,
            metadata["lengths_m"][rows],
            metadata["widths_m"][rows],
            np.asarray(arrays["map_polylines"][rows]),
            np.asarray(arrays["map_polyline_valid"][rows]),
            batch.event_id.to_numpy(),
            benchmark_id=config["benchmark"]["id"],
            fit_seed=int(checkpoint["fit_seed"]),
            futures=args.futures,
            inference_steps=args.denoising_steps,
            warm_start=False,
            device=device,
            behavior_conditions=batch.condition.tolist(),
        )
        for local, event in enumerate(batch.itertuples(index=False)):
            condition = event.condition
            initial = history[local, -1]
            target = int(condition["target_agent_index"])
            metrics = _mode_metrics(
                rollout.states[local], initial, target, str(condition["mode"])
            )
            records.append(
                {
                    "event_id": str(event.event_id),
                    "event_type": str(event.event_type),
                    "recording_id": int(event.recording_id),
                    "mode": str(condition["mode"]),
                    "target_agent_index": target,
                    "method_id": str(checkpoint["method_id"]),
                    "fit_seed": int(checkpoint["fit_seed"]),
                    "requested_futures": args.futures,
                    "completed_futures": args.futures,
                    "failure_type": "",
                    **metrics,
                }
            )
        if args.resume:
            pd.DataFrame(records).to_csv(
                metrics_path.with_suffix(".csv.tmp"), index=False
            )
            metrics_path.with_suffix(".csv.tmp").replace(metrics_path)
    frame = pd.DataFrame(records)
    validate_controlled_metrics(frame, expected_events=selected_events)
    frame.to_csv(metrics_path, index=False)
    summary = {
        "suite": "t4a_controlled",
        "split": args.split,
        "eligible_events": eligible_events,
        "events": len(frame),
        "completion_rate": len(frame) / max(selected_events, 1),
        "is_partial": bool(args.limit),
        "futures": args.futures,
        "metrics_by_mode": (
            frame.groupby("mode")[
                [
                    "completion_probability",
                    "right_censored_completion_time_s",
                    "mean_achieved_amplitude",
                    "mean_amplitude_error",
                ]
            ]
            .mean()
            .to_dict("index")
            if len(frame)
            else {}
        ),
        "wall_time_seconds": time.time() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/benchmark_v1.yaml",
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--futures", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--denoising-steps", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="auto")
    print(json.dumps(evaluate(p.parse_args()), indent=2))


if __name__ == "__main__":
    main()
