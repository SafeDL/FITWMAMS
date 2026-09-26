"""Run T2a logged-event response scoring for C1/C2/C3."""

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
from npc_behavior_benchmark.evaluation.metrics import (
    ensemble_channel_metrics,
    fair_energy_score,
    trajectory_errors,
)
from npc_behavior_benchmark.evaluation.semantic_rollout import rollout_semantic_policy
from npc_behavior_benchmark.policies.rolling_action_policy import (
    ActionDiffusionConfig,
    RollingActionDiffusion,
)
from npc_behavior_benchmark.scripts.evaluate_events import (
    POINTS,
    _behavior_brier,
    _event_scales,
    _relevant_neighbor,
    _response_features,
)


def validate_semantic_event_metrics(
    frame: pd.DataFrame, *, expected_events: int | None = None
) -> None:
    """Validate a full-denominator C-policy T2a artifact."""
    required = {"event_id", "requested_futures", "completed_futures"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"semantic T2a metrics miss columns: {sorted(missing)}")
    if frame.event_id.astype(str).duplicated().any():
        raise ValueError("semantic T2a metrics contain duplicate event IDs")
    if not (frame.requested_futures == frame.completed_futures).all():
        raise ValueError("semantic T2a metrics contain incomplete futures")
    if (
        "failure_type" in frame
        and frame.failure_type.fillna("").astype(str).str.strip().ne("").any()
    ):
        raise ValueError("semantic T2a metrics contain a recorded failure")
    numeric = (
        frame.drop(columns=["failure_type"], errors="ignore")
        .select_dtypes("number")
        .to_numpy(dtype=np.float64, copy=False)
    )
    if not np.isfinite(numeric).all():
        raise ValueError("semantic T2a metrics contain NaN or Inf")
    if expected_events is not None and len(frame) != int(expected_events):
        raise ValueError(
            f"semantic T2a metrics rows={len(frame)}, expected={expected_events}"
        )


def evaluate(args):
    config, _ = load_benchmark_config(args.config)
    root = Path(config["paths"]["output_dir"])
    arrays, metadata, _ = load_causal_cache(args.config)
    all_events = pd.read_csv(root / "event_manifest.csv")
    split_value = {"validation": 1, "test": 2}[args.split]
    events = all_events[
        (all_events.split_index == split_value) & (all_events.response_agent_index > 0)
    ].copy()
    if args.limit:
        events = events.iloc[: args.limit]
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
    scales = _event_scales(arrays, metadata, all_events, root / "metric_schema.json")
    method_id = "semantic_c2" if args.response_aware else "semantic_c1"
    if args.response_aware and args.deterministic_selection:
        method_id = "semantic_c3_deterministic"
    partial_suffix = f"_partial{args.limit}" if args.limit else ""
    output = (
        args.checkpoint.parent
        / args.split
        / f"t2a_events_{method_id}_c{args.candidates}_r{args.response_samples}_ddim{args.denoising_steps}{partial_suffix}"
    )
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "per_event_metrics.csv"
    records = []
    if args.resume and metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        validate_semantic_event_metrics(existing)
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
        valids = []
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
            valids.append(
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
        result = rollout_semantic_policy(
            model,
            checkpoint,
            np.stack(histories),
            np.stack(valids),
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
            candidates=args.candidates,
            response_samples=args.response_samples,
            response_aware=args.response_aware,
            stochastic_selection=not args.deterministic_selection,
            temperature=args.temperature,
            device=device,
            exogenous_future=future_np,
            exogenous_mask=np.stack(masks),
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
            error = trajectory_errors(
                result.states[local, :, :, response : response + 1],
                future_np[local, :, response : response + 1],
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
                    "fair_energy_score": fair_energy_score(
                        generated[:, :, None],
                        target[:, None],
                        np.ones((len(POINTS), 1), bool),
                        scales,
                    ),
                    **error,
                    **calibration,
                    **behavior,
                    "requested_futures": args.futures,
                    "completed_futures": args.futures,
                    "nn_evaluations": int(
                        args.futures
                        * 15
                        * args.denoising_steps
                        * args.candidates
                        * (1 + (args.response_samples if args.response_aware else 0))
                    ),
                    "proxy_steps": int(
                        args.futures
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
                    "failure_type": "",
                }
            )
        if begin and begin % (args.batch_size * 5) == 0:
            print(f"completed {begin}/{len(events)}", flush=True)
        if args.resume and (
            (begin // args.batch_size + 1) % 5 == 0
            or begin + len(batch_events) >= len(events)
        ):
            frame = pd.DataFrame(records)
            temporary = metrics_path.with_suffix(".csv.tmp")
            frame.to_csv(temporary, index=False)
            temporary.replace(metrics_path)
    frame = pd.DataFrame(records)
    validate_semantic_event_metrics(frame, expected_events=len(selected_events))
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
        "eligible_events": int(
            len(
                all_events[
                    (all_events.split_index == split_value)
                    & (all_events.response_agent_index > 0)
                ]
            )
        ),
        "is_partial": bool(args.limit),
        "completion_rate": len(frame) / max(len(selected_events), 1),
        "futures_per_event": args.futures,
        "candidates": args.candidates,
        "response_samples": args.response_samples,
        "nn_evaluations": int(frame["nn_evaluations"].sum()),
        "proxy_steps": int(frame["proxy_steps"].sum()),
        "event_macro_fair_energy_score": float(family["mean"].mean()),
        "event_families": {
            x.event_type: {"events": int(x.count), "fair_energy_score": float(x.mean)}
            for x in family.itertuples(index=False)
        },
        "calibration_metrics": {
            name: float(frame[name].mean())
            for name in frame.columns
            if name.startswith(("fair_CRPS_", "coverage_90_", "interval_width_90_"))
        },
        "behavior_metrics": {
            name: float(frame[name].mean())
            for name in (
                "lane_behavior_brier",
                "braking_behavior_brier",
                "sample_lane_change_probability",
                "sample_braking_probability",
            )
        },
        "wall_time_seconds": time.time() - started,
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
    p.add_argument("--batch-size", type=int, default=4)
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
