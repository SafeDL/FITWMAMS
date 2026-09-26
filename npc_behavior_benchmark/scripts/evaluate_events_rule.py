"""Run the deterministic IDM lane-keeping baseline on T2a."""

from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from npc_behavior_benchmark.data.causal_cache import load_causal_cache
from npc_behavior_benchmark.data.manifest import load_benchmark_config
from npc_behavior_benchmark.evaluation.metrics import (
    ensemble_channel_metrics,
    fair_energy_score,
    trajectory_errors,
)
from npc_behavior_benchmark.evaluation.rule_rollout import rollout_idm_lane_keep
from npc_behavior_benchmark.scripts.evaluate_events import (
    POINTS,
    _behavior_brier,
    _event_scales,
    _relevant_neighbor,
    _response_features,
)


def evaluate(args):
    config, _ = load_benchmark_config(args.config)
    root = Path(config["paths"]["output_dir"])
    arrays, metadata, _ = load_causal_cache(args.config)
    all_events = pd.read_csv(root / "event_manifest.csv")
    events = all_events[
        (all_events.split_index == {"validation": 1, "test": 2}[args.split])
        & (all_events.response_agent_index > 0)
    ].copy()
    if args.limit:
        events = events.iloc[: args.limit]
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    scales = _event_scales(arrays, metadata, all_events, root / "metric_schema.json")
    output = root / "methods/idm_lane_keep/0" / args.split / "t2a_events"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.time()
    for begin in range(0, len(events), args.batch_size):
        batch = events.iloc[begin : begin + args.batch_size]
        histories = []
        valids = []
        future = []
        masks = []
        lengths = []
        widths = []
        maps = []
        map_valid = []
        for event in batch.itertuples(index=False):
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
            lengths.append(metadata["lengths_m"][row])
            widths.append(metadata["widths_m"][row])
            maps.append(np.asarray(arrays["map_polylines"][row]))
            map_valid.append(np.asarray(arrays["map_polyline_valid"][row]))
        future_np = np.stack(future)
        result = rollout_idm_lane_keep(
            np.stack(histories),
            np.stack(valids),
            future_np[:, :, 0],
            np.stack(lengths),
            np.stack(widths),
            np.stack(maps),
            np.stack(map_valid),
            futures=args.futures,
            device=device,
            exogenous_future=future_np,
            exogenous_mask=np.stack(masks),
        )
        for local, event in enumerate(batch.itertuples(index=False)):
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
                    "method_id": "idm_lane_keep",
                    "fit_seed": 0,
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
                    "failure_type": "",
                }
            )
    frame = pd.DataFrame(records)
    frame.to_csv(output / "per_event_metrics.csv", index=False)
    family = (
        frame.groupby("event_type")["fair_energy_score"]
        .agg(["count", "mean"])
        .reset_index()
    )
    summary = {
        "benchmark_id": config["benchmark"]["id"],
        "method_id": "idm_lane_keep",
        "fit_seed": 0,
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
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/benchmark_v1.yaml",
    )
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--futures", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    print(json.dumps(evaluate(a), indent=2))


if __name__ == "__main__":
    main()
