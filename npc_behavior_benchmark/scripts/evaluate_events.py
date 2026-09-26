"""Run T2a logged-event response scoring for B0/B1."""

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
from npc_behavior_benchmark.evaluation.metrics import (
    ensemble_channel_metrics,
    fair_energy_score,
    trajectory_errors,
)
from npc_behavior_benchmark.policies.rolling_action_policy import (
    ActionDiffusionConfig,
    RollingActionDiffusion,
    runtime_action_method_id,
)
from npc_behavior_benchmark.scripts.evaluate_action import (
    checkpoint_sha256,
    prepare_run_contract,
)

POINTS = np.arange(4, 75, 5, dtype=np.int64)


def validate_event_metrics(
    frame: pd.DataFrame, *, expected_events: int | None = None
) -> None:
    """Reject incomplete T2a output before it can enter a formal aggregate."""
    required = {"event_id", "requested_futures", "completed_futures"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"T2a metrics miss columns: {sorted(missing)}")
    if frame.event_id.astype(str).duplicated().any():
        raise ValueError("T2a metrics contain duplicate event IDs")
    if not (frame.requested_futures == frame.completed_futures).all():
        raise ValueError("T2a metrics contain incomplete futures")
    if (
        "failure_type" in frame
        and frame.failure_type.fillna("").astype(str).str.strip().ne("").any()
    ):
        raise ValueError("T2a metrics contain a recorded failure")
    numeric = (
        frame.drop(columns=["failure_type"], errors="ignore")
        .select_dtypes("number")
        .to_numpy(dtype=np.float64, copy=False)
    )
    if not np.isfinite(numeric).all():
        raise ValueError("T2a metrics contain NaN or Inf")
    if expected_events is not None and len(frame) != int(expected_events):
        raise ValueError(f"T2a metrics rows={len(frame)}, expected={expected_events}")


def _relevant_neighbor(initial: np.ndarray, valid: np.ndarray, response: int) -> int:
    distance = np.linalg.norm(initial[:, :2] - initial[response, :2], axis=-1)
    distance[~valid] = np.inf
    distance[response] = np.inf
    return int(np.argmin(distance))


def _response_features(
    states,
    initial,
    event_type: str,
    stimulus: int,
    response: int,
    relevant: int,
    lengths,
):
    selected = states[..., POINTS, :, :]
    response_state = selected[..., response, :]
    displacement = response_state[..., :2] - initial[response, :2]
    if event_type in {"following_brake", "cut_in"} and stimulus != response:
        gap = (
            selected[..., stimulus, 0]
            - response_state[..., 0]
            - 0.5 * (lengths[stimulus] + lengths[response])
        )
    else:
        gap = np.linalg.norm(
            selected[..., relevant, :2] - response_state[..., :2], axis=-1
        )
    return np.concatenate(
        (displacement, response_state[..., 2:4], gap[..., None]), axis=-1
    )


def _behavior_brier(
    states, target, response: int, initial_response: np.ndarray
) -> dict[str, float]:
    """Sample-frequency Brier scores for lateral intent and braking."""
    generated = np.asarray(states)[..., response, :]
    observed = np.asarray(target)[..., response, :]
    dy = generated[:, -1, 1] - initial_response[1]
    target_dy = observed[-1, 1] - initial_response[1]
    lane_class = np.where(dy < -1.75, 0, np.where(dy > 1.75, 2, 1))
    target_lane_class = int(0 if target_dy < -1.75 else 2 if target_dy > 1.75 else 1)
    probabilities = np.bincount(lane_class, minlength=3) / len(lane_class)
    lane_observed = np.zeros(3)
    lane_observed[target_lane_class] = 1.0
    initial_speed = np.linalg.norm(initial_response[2:4])
    braking = np.linalg.norm(generated[..., 2:4], axis=-1).min(-1) < initial_speed - 2.0
    target_braking = float(
        np.linalg.norm(observed[..., 2:4], axis=-1).min() < initial_speed - 2.0
    )
    return {
        "lane_behavior_brier": float(np.square(probabilities - lane_observed).sum()),
        "braking_behavior_brier": float((braking.mean() - target_braking) ** 2),
        "sample_lane_change_probability": float(1.0 - probabilities[1]),
        "sample_braking_probability": float(braking.mean()),
    }


def _event_scales(arrays, metadata, events: pd.DataFrame, schema_path: Path):
    schema = json.loads(schema_path.read_text()) if schema_path.exists() else {}
    if "t2a_feature_scale" in schema:
        return np.asarray(schema["t2a_feature_scale"], np.float64)
    total = np.zeros(5, np.float64)
    square = np.zeros(5, np.float64)
    count = 0
    for event in events[events.split_index == 0].itertuples(index=False):
        row, onset = int(event.scenario_row), int(event.local_onset_frame)
        initial = np.asarray(arrays["agent_states"][row, onset])
        valid = np.asarray(arrays["agent_valid"][row, onset])
        relevant = _relevant_neighbor(initial, valid, int(event.response_agent_index))
        future = np.asarray(arrays["agent_states"][row, onset + 1 : onset + 76])[None]
        value = _response_features(
            future,
            initial,
            event.event_type,
            int(event.stimulus_agent_index),
            int(event.response_agent_index),
            relevant,
            metadata["lengths_m"][row],
        )[0]
        total += value.sum(0)
        square += np.square(value).sum(0)
        count += len(value)
    mean = total / count
    scale = np.sqrt(np.maximum(square / count - np.square(mean), 1.0e-6))
    schema.update(
        {
            "t2a_features": [
                "response_dx_m",
                "response_dy_left_m",
                "response_vx_mps",
                "response_vy_left_mps",
                "relation_gap_m",
            ],
            "t2a_feature_mean": mean.tolist(),
            "t2a_feature_scale": scale.tolist(),
            "t2a_normalization_count": int(count),
            "t2a_points_zero_based": POINTS.tolist(),
        }
    )
    schema_path.write_text(json.dumps(schema, indent=2) + "\n")
    return scale


def evaluate(args):
    config, _ = load_benchmark_config(args.config)
    output_root = Path(config["paths"]["output_dir"])
    arrays, metadata, _ = load_causal_cache(args.config)
    events = pd.read_csv(output_root / "event_manifest.csv")
    split_value = {"validation": 1, "test": 2}[args.split]
    events = events[
        (events.split_index == split_value) & (events.response_agent_index > 0)
    ].copy()
    eligible_events = len(events)
    if args.limit:
        events = events.iloc[: args.limit].copy()
    selected_events = events.copy()
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
    scales = _event_scales(
        arrays,
        metadata,
        pd.read_csv(output_root / "event_manifest.csv"),
        output_root / "metric_schema.json",
    )
    trained_method = str(checkpoint.get("method_id", "rolling_action_b0"))
    method_id = runtime_action_method_id(trained_method, args.warm_start)
    sampler = "euler" if model.config.training_objective == "flow_matching" else "ddim"
    partial_suffix = f"_partial{args.limit}" if args.limit else ""
    output = (
        args.checkpoint.parent
        / args.split
        / f"t2a_events_{method_id}_{sampler}{args.denoising_steps}{partial_suffix}"
    )
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "per_event_metrics.csv"
    checkpoint_hash = checkpoint_sha256(args.checkpoint)
    run_contract = {
        "checkpoint_sha256": checkpoint_hash,
        "method_id": method_id,
        "fit_seed": int(checkpoint["fit_seed"]),
        "split": args.split,
        "futures": int(args.futures),
        "denoising_steps": int(args.denoising_steps),
        "warm_start": bool(args.warm_start),
        "sampler": sampler,
        "limit": int(args.limit),
    }
    prepare_run_contract(
        output / "run_contract.json",
        run_contract,
        resume_with_metrics=args.resume and metrics_path.exists(),
    )
    records = []
    if args.resume and metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        validate_event_metrics(existing)
        expected = {
            "method_id": str(method_id),
            "fit_seed": int(checkpoint["fit_seed"]),
            "requested_futures": int(args.futures),
            "completed_futures": int(args.futures),
        }
        for column, value in expected.items():
            if column not in existing or not (existing[column] == value).all():
                raise ValueError(
                    f"resume file {column} does not match current run: {metrics_path}"
                )
        records = existing.to_dict("records")
        completed = set(existing.event_id.astype(str))
        events = events[~events.event_id.astype(str).isin(completed)].copy()
        print(
            f"resuming with {len(records)} completed; {len(events)} remaining",
            flush=True,
        )
    started = time.time()
    for begin in range(0, len(events), args.batch_size):
        batch_events = events.iloc[begin : begin + args.batch_size]
        histories = []
        history_valid = []
        future = []
        masks = []
        ids = []
        lengths = []
        widths = []
        maps = []
        map_valid = []
        for event in batch_events.itertuples(index=False):
            row, onset = int(event.scenario_row), int(event.local_onset_frame)
            histories.append(
                np.asarray(arrays["agent_states"][row, onset - 24 : onset + 1])
            )
            history_valid.append(
                np.asarray(arrays["agent_valid"][row, onset - 24 : onset + 1])
            )
            future.append(
                np.asarray(arrays["agent_states"][row, onset + 1 : onset + 76])
            )
            mask = np.zeros(7, bool)
            mask[0] = True
            if int(event.stimulus_agent_index) != int(event.response_agent_index):
                mask[int(event.stimulus_agent_index)] = True
            masks.append(mask)
            ids.append(str(event.event_id))
            lengths.append(metadata["lengths_m"][row])
            widths.append(metadata["widths_m"][row])
            maps.append(np.asarray(arrays["map_polylines"][row]))
            map_valid.append(np.asarray(arrays["map_polyline_valid"][row]))
        future_np = np.stack(future)
        masks_np = np.stack(masks)
        result = rollout_action_diffusion(
            model,
            checkpoint,
            np.stack(histories),
            np.stack(history_valid),
            future_np[:, :, 0],
            np.stack(lengths),
            np.stack(widths),
            np.stack(maps),
            np.stack(map_valid),
            np.asarray(ids),
            benchmark_id=config["benchmark"]["id"],
            fit_seed=int(checkpoint["fit_seed"]),
            futures=args.futures,
            inference_steps=args.denoising_steps,
            warm_start=args.warm_start,
            device=device,
            exogenous_future=future_np,
            exogenous_mask=masks_np,
        )
        for local, event in enumerate(batch_events.itertuples(index=False)):
            row, onset = int(event.scenario_row), int(event.local_onset_frame)
            response, stimulus = int(event.response_agent_index), int(
                event.stimulus_agent_index
            )
            initial = np.asarray(arrays["agent_states"][row, onset])
            valid = np.asarray(arrays["agent_valid"][row, onset])
            relevant = _relevant_neighbor(initial, valid, response)
            generated = _response_features(
                result.states[local],
                initial,
                event.event_type,
                stimulus,
                response,
                relevant,
                metadata["lengths_m"][row],
            )
            target = _response_features(
                future_np[local : local + 1],
                initial,
                event.event_type,
                stimulus,
                response,
                relevant,
                metadata["lengths_m"][row],
            )[0]
            target_response = future_np[local, :, response : response + 1]
            error = trajectory_errors(
                result.states[local, :, :, response : response + 1],
                target_response,
                np.ones((75, 1), bool),
            )
            calibration = ensemble_channel_metrics(
                generated,
                target,
                np.ones(len(POINTS), bool),
                ("dx_m", "dy_m", "vx_mps", "vy_mps", "gap_m"),
            )
            behavior = _behavior_brier(
                result.states[local], future_np[local], response, initial[response]
            )
            records.append(
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "recording_id": int(event.recording_id),
                    "method_id": method_id,
                    "fit_seed": int(checkpoint["fit_seed"]),
                    "requested_futures": args.futures,
                    "completed_futures": args.futures,
                    "fair_energy_score": fair_energy_score(
                        generated[:, :, None],
                        target[:, None],
                        np.ones((len(POINTS), 1), bool),
                        scales,
                    ),
                    **error,
                    **calibration,
                    **behavior,
                    "failure_type": "",
                    "fallback_used": False,
                }
            )
        if args.resume and (
            (begin // args.batch_size + 1) % 5 == 0
            or begin + len(batch_events) >= len(events)
        ):
            frame = pd.DataFrame(records)
            temporary = metrics_path.with_suffix(".csv.tmp")
            frame.to_csv(temporary, index=False)
            temporary.replace(metrics_path)
    frame = pd.DataFrame(records)
    validate_event_metrics(frame, expected_events=len(selected_events))
    frame.to_csv(metrics_path, index=False)
    family = (
        frame.groupby("event_type")["fair_energy_score"]
        .agg(["count", "mean"])
        .reset_index()
    )
    summary = {
        "benchmark_id": config["benchmark"]["id"],
        "method_id": method_id,
        "fit_seed": int(checkpoint["fit_seed"]),
        "split": args.split,
        "events": len(frame),
        "eligible_events": int(eligible_events),
        "is_partial": bool(args.limit),
        "completion_rate": len(frame) / max(len(selected_events), 1),
        "futures_per_event": args.futures,
        "denoising_steps": args.denoising_steps,
        "network_evaluations_per_decision": args.denoising_steps,
        "sampler": sampler,
        "checkpoint_sha256": checkpoint_hash,
        "event_macro_fair_energy_score": float(family["mean"].mean()),
        "event_families": {
            row.event_type: {
                "events": int(row.count),
                "fair_energy_score": float(row.mean),
            }
            for row in family.itertuples(index=False)
        },
        "wall_time_seconds": time.time() - started,
    }
    summary["calibration_metrics"] = {
        name: float(frame[name].mean())
        for name in frame.columns
        if name.startswith(("fair_CRPS_", "coverage_90_", "interval_width_90_"))
    }
    summary["behavior_metrics"] = {
        name: float(frame[name].mean())
        for name in (
            "lane_behavior_brier",
            "braking_behavior_brier",
            "sample_lane_change_probability",
            "sample_braking_probability",
        )
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
    parser.add_argument("--futures", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--denoising-steps", type=int, choices=(4, 8, 16), default=4)
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
