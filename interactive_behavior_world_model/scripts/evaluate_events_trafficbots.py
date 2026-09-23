"""Evaluate TrafficBots on strict-causal all-background logged T2a events."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from interactive_behavior_world_model.data.causal_cache import load_causal_cache
from interactive_behavior_world_model.data.manifest import load_benchmark_config
from interactive_behavior_world_model.evaluation.metrics import (
    ensemble_channel_metrics,
    fair_energy_score,
    trajectory_errors,
)
from interactive_behavior_world_model.scripts.evaluate_events import (
    POINTS,
    _behavior_brier,
    _event_scales,
    _relevant_neighbor,
    _response_features,
)
from interactive_behavior_world_model.scripts.evaluate_trafficbots import _batch
from reproduction.models.trafficbots.config import (
    load_config as load_trafficbots_config,
)
from reproduction.models.trafficbots.evaluation import load_checkpoint
from reproduction.models.trafficbots.rollout import TrafficBotsHighDRollout


def _seed(benchmark_id: str, training_seed: int, begin: int, rollout_id: int) -> int:
    value = hashlib.sha256(
        f"{benchmark_id}|trafficbots_t2a|{training_seed}|{begin}|{rollout_id}".encode()
    ).digest()
    return int.from_bytes(value[:8], "little") % (2**31 - 1)


def _trafficbots_sequence(
    initial: np.ndarray,
    future: np.ndarray,
    initial_valid: np.ndarray,
    future_valid: np.ndarray,
):
    """Make the fixed 150-state adapter tensor; only the first 76 are executed."""
    batch = len(initial)
    states = np.zeros((batch, 150, 7, 6), np.float32)
    valid = np.zeros((batch, 150, 7), bool)
    states[:, 0], valid[:, 0] = initial, initial_valid
    states[:, 1:76], valid[:, 1:76] = future, future_valid
    states[:, 76:] = future[:, -1:, :, :]
    valid[:, 76:] = future_valid[:, -1:, :]
    return states, valid


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    config, _ = load_benchmark_config(args.config)
    root = Path(config["paths"]["output_dir"])
    arrays, metadata, _ = load_causal_cache(args.config)
    all_events = pd.read_csv(root / "event_manifest.csv")
    split_value = 1 if args.split == "validation" else 2
    events = all_events[
        (all_events.split_index == split_value) & (all_events.response_agent_index > 0)
    ].copy()
    if args.limit:
        events = events.iloc[: args.limit]
    tb_config = load_trafficbots_config(args.trafficbots_config)
    training_seed = int(
        args.training_seed
        if args.training_seed is not None
        else tb_config["training"].get("seed", tb_config["experiment"]["seed"])
    )
    fit_seed = int(args.fit_seed)
    method_id = str(args.method_id)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_checkpoint(tb_config, args.checkpoint).to(device).eval()
    runner = TrafficBotsHighDRollout(
        model, require_follower_excluded=False, common_backend=True
    )
    scales = _event_scales(arrays, metadata, all_events, root / "metric_schema.json")
    output = root / f"methods/{method_id}/{fit_seed}" / args.split / "t2a_events"
    output.mkdir(parents=True, exist_ok=True)
    metric_path = output / "per_event_metrics.csv"
    records = []
    started = time.time()
    if args.resume and metric_path.is_file():
        existing = pd.read_csv(metric_path)
        required = {"event_id", "requested_futures", "training_seed"}
        if not required.issubset(existing):
            raise ValueError(
                f"cannot resume incomplete TrafficBots T2a schema at {metric_path}"
            )
        if (existing["requested_futures"] != args.futures).any() or (
            existing["training_seed"] != training_seed
        ).any():
            raise ValueError(
                "TrafficBots T2a resume contract differs from existing rows"
            )
        if existing["event_id"].duplicated().any():
            raise ValueError("TrafficBots T2a resume file contains duplicate event IDs")
        records = existing.to_dict("records")
    completed = {str(record["event_id"]) for record in records}

    def checkpoint_rows() -> None:
        frame = pd.DataFrame(records)
        if frame["event_id"].duplicated().any():
            raise RuntimeError(
                "duplicate event written during TrafficBots T2a evaluation"
            )
        temporary = metric_path.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(metric_path)

    for begin in range(0, len(events), args.batch_size):
        batch_events = events.iloc[begin : begin + args.batch_size]
        batch_events = batch_events[~batch_events.event_id.astype(str).isin(completed)]
        if batch_events.empty:
            continue
        initial = []
        initial_valid = []
        future = []
        future_valid = []
        masks = []
        maps = []
        map_valid = []
        for event in batch_events.itertuples(index=False):
            row, onset = int(event.scenario_row), int(event.local_onset_frame)
            initial.append(np.asarray(arrays["agent_states"][row, onset]))
            initial_valid.append(np.asarray(arrays["agent_valid"][row, onset]))
            future.append(
                np.asarray(arrays["agent_states"][row, onset + 1 : onset + 76])
            )
            future_valid.append(
                np.asarray(arrays["agent_valid"][row, onset + 1 : onset + 76])
            )
            mask = np.zeros(7, bool)
            mask[0] = True
            if int(event.stimulus_agent_index) != int(event.response_agent_index):
                mask[int(event.stimulus_agent_index)] = True
            masks.append(mask)
            maps.append(np.asarray(arrays["map_polylines"][row]))
            map_valid.append(np.asarray(arrays["map_polyline_valid"][row]))
        initial_np, initial_valid_np, future_np, future_valid_np = map(
            np.stack, (initial, initial_valid, future, future_valid)
        )
        states150, valid150 = _trafficbots_sequence(
            initial_np, future_np, initial_valid_np, future_valid_np
        )
        batch = _batch(states150, valid150, np.stack(maps), np.stack(map_valid))
        generated = []
        for rollout_id in range(args.futures):
            seed = _seed(config["benchmark"]["id"], training_seed, begin, rollout_id)
            np.random.seed(seed)
            torch.manual_seed(seed)
            result = runner.run(
                batch,
                deterministic=False,
                exogenous_states=torch.from_numpy(future_np),
                exogenous_mask=torch.from_numpy(np.stack(masks)),
                steps=75,
            )
            generated.append(result.states.detach().cpu().numpy())
        generated_np = np.stack(generated, 1)
        for local, event in enumerate(batch_events.itertuples(index=False)):
            row, onset = int(event.scenario_row), int(event.local_onset_frame)
            response, stimulus = int(event.response_agent_index), int(
                event.stimulus_agent_index
            )
            relevant = _relevant_neighbor(
                initial_np[local], initial_valid_np[local], response
            )
            generated_features = _response_features(
                generated_np[local],
                initial_np[local],
                event.event_type,
                stimulus,
                response,
                relevant,
                metadata["lengths_m"][row],
            )
            target_features = _response_features(
                future_np[local : local + 1],
                initial_np[local],
                event.event_type,
                stimulus,
                response,
                relevant,
                metadata["lengths_m"][row],
            )[0]
            error = trajectory_errors(
                generated_np[local, :, :, response : response + 1],
                future_np[local, :, response : response + 1],
                np.ones((75, 1), bool),
            )
            calibration = ensemble_channel_metrics(
                generated_features,
                target_features,
                np.ones(len(POINTS), bool),
                ("dx_m", "dy_m", "vx_mps", "vy_mps", "gap_m"),
            )
            behavior = _behavior_brier(
                generated_np[local],
                future_np[local],
                response,
                initial_np[local, response],
            )
            records.append(
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "recording_id": int(event.recording_id),
                    "method_id": method_id,
                    "fit_seed": fit_seed,
                    "training_seed": training_seed,
                    "requested_futures": args.futures,
                    "completed_futures": args.futures,
                    "fair_energy_score": fair_energy_score(
                        generated_features[:, :, None],
                        target_features[:, None],
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
            completed.add(str(event.event_id))
        checkpoint_rows()
    frame = pd.DataFrame(records)
    if len(frame) != len(events):
        raise RuntimeError(
            f"TrafficBots T2a completion mismatch: {len(frame)}/{len(events)}"
        )
    expected_ids = set(events.event_id.astype(str))
    observed_ids = set(frame["event_id"].astype(str))
    if observed_ids != expected_ids:
        raise RuntimeError(
            f"TrafficBots T2a event-ID mismatch: missing={len(expected_ids - observed_ids)}, "
            f"unexpected={len(observed_ids - expected_ids)}"
        )
    numeric = frame.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        raise RuntimeError("TrafficBots T2a produced non-finite formal metrics")
    family = (
        frame.groupby("event_type")["fair_energy_score"]
        .agg(["count", "mean"])
        .reset_index()
    )
    summary = {
        "benchmark_id": config["benchmark"]["id"],
        "method_id": method_id,
        "fit_seed": fit_seed,
        "training_seed": training_seed,
        "split": args.split,
        "events": len(frame),
        "completion_rate": len(frame) / max(len(events), 1),
        "futures_per_event": args.futures,
        "event_macro_fair_energy_score": float(family["mean"].mean()),
        "event_families": {
            x.event_type: {"events": int(x.count), "fair_energy_score": float(x.mean)}
            for x in family.itertuples(index=False)
        },
        "wall_time_seconds": time.time() - started,
    }
    summary["calibration_metrics"] = {
        name: float(frame[name].mean())
        for name in frame
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/benchmark_v1.yaml",
    )
    p.add_argument(
        "--trafficbots-config",
        type=Path,
        default=Path("reproduction/models/trafficbots/config/highd.yaml"),
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/baselines/trafficbots_highd/checkpoints/best.ckpt"),
    )
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--futures", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--fit-seed", type=int, default=0)
    p.add_argument("--training-seed", type=int)
    p.add_argument("--method-id", default="trafficbots_v15_all_background")
    p.add_argument(
        "--resume",
        action="store_true",
        help="atomically resume a matching partial per-event file",
    )
    args = p.parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
