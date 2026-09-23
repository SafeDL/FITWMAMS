"""Run TrafficBots under the common fixed-PNC T3 closed-loop protocol."""

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
from interactive_behavior_world_model.evaluation.geometry import collision_matrix, straight_road_offroad
from interactive_behavior_world_model.evaluation.pnc import PNC_IDS
from interactive_behavior_world_model.evaluation.trafficbots_pnc_rollout import TrafficBotsPNCRollout
from interactive_behavior_world_model.policies.learned_pnc import StructuredBCPNC
from interactive_behavior_world_model.scripts.evaluate_pnc import validate_pnc_metrics
from interactive_behavior_world_model.scripts.evaluate_lateral_probes_trafficbots import (
    _per_scene_prior_samples,
)
from interactive_behavior_world_model.scripts.evaluate_trafficbots import _batch
from reproduction.models.trafficbots.config import (
    load_config as load_trafficbots_config,
)
from reproduction.models.trafficbots.evaluation import load_checkpoint


def _seed(benchmark_id: str, training_seed: int, identifier: str, rollout: int) -> int:
    value = hashlib.sha256(
        f"{benchmark_id}|trafficbots_t3|{training_seed}|{identifier}|{rollout}".encode()
    ).digest()
    return int.from_bytes(value[:8], "little") % (2**31 - 1)


def _prior_sequence(
    initial: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return np.repeat(initial[:, None], 150, axis=1).astype(
        np.float32, copy=False
    ), np.repeat(valid[:, None], 150, axis=1).astype(bool, copy=False)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    config, _ = load_benchmark_config(args.config)
    root = Path(config["paths"]["output_dir"])
    arrays, metadata, _ = load_causal_cache(args.config)
    split_value = {"validation": 1, "test": 2}[args.split]
    events = None
    if args.cohort == "d1_events":
        all_events = pd.read_csv(root / "event_manifest.csv")
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
    tb_config = load_trafficbots_config(args.trafficbots_config)
    training_seed = int(
        args.training_seed
        if args.training_seed is not None
        else tb_config["training"].get("seed", tb_config["experiment"]["seed"])
    )
    fit_seed, method_id = int(args.fit_seed), str(args.method_id)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    module = load_checkpoint(tb_config, args.checkpoint).to(device).eval()
    runner = TrafficBotsPNCRollout(
        module, require_follower_excluded=False, common_backend=True
    )
    pnc_model = pnc_checkpoint = None
    if args.pnc in {"structured_bc", "structured_bc_lane_stable"}:
        if args.pnc_checkpoint is None:
            raise ValueError("--pnc-checkpoint is required for a structured PNC")
        pnc_checkpoint = torch.load(args.pnc_checkpoint, map_location="cpu")
        pnc_model = StructuredBCPNC(**pnc_checkpoint["model_config"])
        pnc_model.load_state_dict(pnc_checkpoint["model_state"])
        pnc_model.to(device).eval()

    suffix = f"_partial{args.limit}" if args.limit else ""
    cohort_suffix = "_d1_events" if events is not None else ""
    output = (
        root
        / f"methods/{method_id}/{fit_seed}/{args.split}/t3_fixed_pnc{cohort_suffix}_{args.pnc}{suffix}"
    )
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "per_scene_metrics.csv"
    records: list[dict] = []
    if args.resume and metrics_path.is_file():
        existing = pd.read_csv(metrics_path)
        validate_pnc_metrics(existing, event_cohort=events is not None)
        required = {
            "training_seed",
            "requested_futures",
            "method_id",
            "fit_seed",
            "ego_policy_id",
        }
        if (
            not required.issubset(existing)
            or (existing.training_seed != training_seed).any()
            or (existing.requested_futures != args.futures).any()
        ):
            raise ValueError(
                "TrafficBots T3 resume contract differs from existing rows"
            )
        if (
            existing.method_id.astype(str).ne(method_id).any()
            or (existing.fit_seed != fit_seed).any()
            or existing.ego_policy_id.astype(str).ne(args.pnc).any()
        ):
            raise ValueError(
                "TrafficBots T3 resume method contract differs from existing rows"
            )
        records = existing.to_dict("records")
        if events is None:
            done = set(existing.scenario_id.astype(str))
            rows = np.asarray(
                [r for r in rows if str(metadata["sequence_id"][r]) not in done],
                np.int64,
            )
        else:
            done = set(existing.event_id.astype(str))
            rows = np.asarray(
                [r for r in rows if str(events.iloc[r].event_id) not in done], np.int64
            )
        print(
            f"resuming with {len(records)} completed; {len(rows)} remaining", flush=True
        )

    def checkpoint_rows() -> None:
        frame = pd.DataFrame(records)
        validate_pnc_metrics(frame, event_cohort=events is not None)
        temporary = metrics_path.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(metrics_path)

    started = time.time()
    for begin in range(0, len(rows), args.batch_size):
        take = rows[begin : begin + args.batch_size]
        if events is None:
            source_rows = take
            onsets = np.full(len(take), 24, np.int64)
            identifiers = metadata["sequence_id"][source_rows]
        else:
            selected = events.iloc[take]
            source_rows = selected.scenario_row.to_numpy(np.int64)
            onsets = selected.local_onset_frame.to_numpy(np.int64)
            identifiers = selected.event_id.astype(str).to_numpy()
        histories = np.stack(
            [
                np.asarray(arrays["agent_states"][row, onset - 24 : onset + 1])
                for row, onset in zip(source_rows, onsets)
            ]
        ).astype(np.float32)
        history_valid = np.stack(
            [
                np.asarray(arrays["agent_valid"][row, onset - 24 : onset + 1])
                for row, onset in zip(source_rows, onsets)
            ]
        ).astype(bool)
        initial, initial_valid = histories[:, -1], history_valid[:, -1]
        prior_states, prior_valid = _prior_sequence(initial, initial_valid)
        batch = _batch(
            prior_states,
            prior_valid,
            np.asarray(arrays["map_polylines"][source_rows]),
            np.asarray(arrays["map_polyline_valid"][source_rows]),
        )
        generated, requested, applied = [], [], []
        for rollout_id in range(args.futures):
            seeds = [
                _seed(
                    config["benchmark"]["id"],
                    training_seed,
                    str(identifier),
                    rollout_id,
                )
                for identifier in identifiers
            ]
            latent_sample, destination_sample = _per_scene_prior_samples(
                module, batch, seeds, device
            )
            result = runner.run_pnc(
                batch,
                deterministic=False,
                pnc_id=args.pnc,
                lengths=torch.from_numpy(metadata["lengths_m"][source_rows]),
                widths=torch.from_numpy(metadata["widths_m"][source_rows]),
                map_polylines=torch.from_numpy(
                    np.asarray(arrays["map_polylines"][source_rows])
                ),
                map_valid=torch.from_numpy(
                    np.asarray(arrays["map_polyline_valid"][source_rows])
                ),
                pnc_history=torch.from_numpy(histories),
                pnc_history_valid=torch.from_numpy(history_valid),
                pnc_model=pnc_model,
                pnc_checkpoint=pnc_checkpoint,
                latent_sample=latent_sample,
                destination_sample=destination_sample,
                steps=horizon,
            )
            generated.append(result.states.cpu().numpy())
            requested.append(result.reference_actions.cpu().numpy())
            applied.append(result.background_actions.cpu().numpy())
        generated = np.stack(generated, 1)
        requested = np.stack(requested, 1)[:, :, ::5]
        applied = np.stack(applied, 1)[:, :, ::5]
        target_valid = np.stack(
            [
                np.asarray(arrays["agent_valid"][row, onset + 1 : onset + 1 + horizon])
                for row, onset in zip(source_rows, onsets)
            ]
        )
        for local, row in enumerate(source_rows):
            collisions = collision_matrix(
                generated[local],
                np.broadcast_to(target_valid[local], generated[local].shape[:-1]),
                metadata["lengths_m"][row],
                metadata["widths_m"][row],
            )
            initial_collisions = collision_matrix(
                initial[local],
                initial_valid[local],
                metadata["lengths_m"][row],
                metadata["widths_m"][row],
            )
            new_collision = collisions & ~initial_collisions[None, None]
            valid_future = np.broadcast_to(
                target_valid[local], generated[local].shape[:-1]
            )
            offroad = straight_road_offroad(
                generated[local],
                valid_future,
                metadata["lengths_m"][row],
                metadata["widths_m"][row],
                np.asarray(arrays["map_polylines"][row]),
                np.asarray(arrays["map_polyline_valid"][row]),
            )
            collision_time = new_collision.any((-1, -2))
            has_collision = collision_time.any(-1)
            stop = np.where(has_collision, collision_time.argmax(-1), horizon - 1)
            time_mask = np.arange(horizon)[None] <= stop[:, None]
            member = np.arange(args.futures)
            progress = generated[local, member, stop, 0, 0] - initial[local, 0, 0]
            initial_speed = np.linalg.norm(initial[local, 0, 2:4])
            final_speed = np.linalg.norm(
                generated[local, member, stop, 0, 2:4], axis=-1
            )
            jerk = np.diff(applied[local, ..., 0], axis=1) / 0.2
            active_actions = target_valid[local, 0, 1:]
            jerk_time = (np.arange(jerk.shape[1]) + 1)[None] <= (stop[:, None] // 5)
            jerk_valid = jerk_time[..., None] & active_actions[None, None]
            record = {
                "benchmark_id": config["benchmark"]["id"],
                "method_id": method_id,
                "fit_seed": fit_seed,
                "training_seed": training_seed,
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
                    float(
                        (np.abs(requested[local] - applied[local]) > 1e-6)
                        .any(-1)[..., active_actions]
                        .mean()
                    )
                    if active_actions.any()
                    else 0.0
                ),
                "decision_clock_hz": 5,
                "failure_type": "",
                "fallback_used": False,
            }
            if events is not None:
                event = events.iloc[int(take[local])]
                record.update(
                    {
                        "event_id": str(event.event_id),
                        "event_type": str(event.event_type),
                    }
                )
            records.append(record)
        checkpoint_rows()
        if begin and begin % (args.batch_size * 10) == 0:
            print(f"completed {begin}/{len(rows)}", flush=True)

    frame = pd.DataFrame(records)
    validate_pnc_metrics(
        frame, event_cohort=events is not None, expected_scenarios=selected_scenarios
    )
    names = [
        name
        for name in frame.select_dtypes("number")
        if name
        not in {
            "fit_seed",
            "training_seed",
            "recording_id",
            "requested_futures",
            "completed_futures",
            "decision_clock_hz",
            "failure_type",
        }
    ]
    summary = {
        "benchmark_id": config["benchmark"]["id"],
        "method_id": method_id,
        "fit_seed": fit_seed,
        "training_seed": training_seed,
        "split": args.split,
        "suite": "t3_fixed_pnc",
        "cohort": args.cohort,
        "ego_policy_id": args.pnc,
        "scenarios": len(frame),
        "eligible_scenarios": eligible_scenarios,
        "is_partial": bool(args.limit),
        "completion_rate": len(frame) / max(selected_scenarios, 1),
        "futures_per_scenario": args.futures,
        "backend": "shared_kinematic_traffic_dynamics",
        "background_scope": "all_six",
        "decision_clock_hz": 5,
        "termination": "metrics_truncated_at_first_new_collision",
        "horizon_frames": horizon,
        "pnc_checkpoint": str(args.pnc_checkpoint) if args.pnc_checkpoint else None,
        "wall_time_seconds": time.time() - started,
        "metrics": {name: float(frame[name].mean()) for name in names},
    }
    frame.to_csv(metrics_path, index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--pnc", choices=PNC_IDS, default="cruise")
    parser.add_argument("--cohort", choices=("d0", "d1_events"), default="d0")
    parser.add_argument("--pnc-checkpoint", type=Path)
    parser.add_argument("--futures", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fit-seed", type=int, default=0)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--method-id", default="trafficbots_v15_all_background")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
