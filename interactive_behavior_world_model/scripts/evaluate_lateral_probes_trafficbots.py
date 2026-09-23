"""Evaluate all-background TrafficBots on the paired D3 lateral suite.

The two branches share the sampled TrafficBots latent and destination for
each (probe, dose, rollout) member.  The evaluator owns the public lateral
stimulus and never supplies a logged future to the TrafficBots prior.
"""

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
from interactive_behavior_world_model.evaluation.stimuli import (
    lateral_stimulus_trajectories,
    sustained_response_latency,
)
from interactive_behavior_world_model.scripts.evaluate_lateral_probes_action import (
    _expanded,
    validate_lateral_probe_metrics,
)
from interactive_behavior_world_model.scripts.evaluate_trafficbots import _batch
from reproduction.models.trafficbots.config import (
    load_config as load_trafficbots_config,
)
from reproduction.models.trafficbots.evaluation import load_checkpoint
from reproduction.models.trafficbots.rollout import TrafficBotsHighDRollout


def _seed(
    benchmark_id: str, training_seed: int, probe_id: str, dose: float, rollout: int
) -> int:
    payload = f"{benchmark_id}|trafficbots_t2b|{training_seed}|{probe_id}|{dose:.6g}|{rollout}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31 - 1)


def _prior_sequence(
    initial: np.ndarray, initial_valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build the fixed 150-step adapter input from S0 only (no future leak)."""
    states = np.repeat(initial[:, None], 150, axis=1).astype(np.float32, copy=False)
    valid = np.repeat(initial_valid[:, None], 150, axis=1).astype(bool, copy=False)
    return states, valid


@torch.no_grad()
def _per_scene_prior_samples(
    module, batch: dict, seeds: list[int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample each member with its immutable scenario key before batching.

    This makes a resumed file bitwise independent of which already-complete
    rows were removed from its next execution batch.
    """
    latents, destinations = [], []
    for local, seed in enumerate(seeds):
        one = {name: value[local : local + 1] for name, value in batch.items()}
        one = module._move(one, device)
        mp_tokens, tl_tokens = module._tokens(one)
        latent_distribution = module._latent(one, mp_tokens, tl_tokens, posterior=False)
        torch.manual_seed(int(seed))
        latents.append(latent_distribution.sample(False))
        destination_distribution = module._dest_distribution(one, mp_tokens)
        destinations.append(destination_distribution.sample(False))
    return torch.cat(latents, 0), torch.cat(destinations, 0)


def _response_metrics(
    acceleration_delta: np.ndarray,
    yaw_delta: np.ndarray,
    family: str,
    avoidance_sign: float,
):
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


@torch.no_grad()
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
    runner = TrafficBotsHighDRollout(
        module, require_follower_excluded=False, common_backend=True
    )

    suffix = f"_partial{args.limit}" if args.limit else ""
    output = (
        root / f"methods/{method_id}/{fit_seed}/{args.split}/t2b_lateral_probes{suffix}"
    )
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "per_probe_metrics.csv"
    records: list[dict] = []
    if args.resume and metrics_path.is_file():
        existing = pd.read_csv(metrics_path)
        validate_lateral_probe_metrics(existing)
        required = {"training_seed", "requested_pairs", "method_id", "fit_seed"}
        if (
            not required.issubset(existing)
            or (existing.training_seed != training_seed).any()
            or (existing.requested_pairs != args.futures).any()
        ):
            raise ValueError(
                "TrafficBots T2b resume contract differs from existing rows"
            )
        if (
            existing.method_id.astype(str).ne(method_id).any()
            or (existing.fit_seed != fit_seed).any()
        ):
            raise ValueError(
                "TrafficBots T2b resume method contract differs from existing rows"
            )
        records = existing.to_dict("records")
        done = {
            (str(row.probe_id), float(row.controller_rate_rps))
            for row in existing.itertuples(index=False)
        }
        probes = probes[
            [
                (str(row.probe_id), float(row.controller_rate_rps)) not in done
                for row in probes.itertuples(index=False)
            ]
        ].copy()
        print(
            f"resuming with {len(records)} completed; {len(probes)} remaining",
            flush=True,
        )

    def checkpoint_rows() -> None:
        frame = pd.DataFrame(records)
        validate_lateral_probe_metrics(frame)
        temporary = metrics_path.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(metrics_path)

    started = time.time()
    for begin in range(0, len(probes), args.batch_size):
        batch_frame = probes.iloc[begin : begin + args.batch_size]
        rows = batch_frame.scenario_row.to_numpy(np.int64)
        initial = np.asarray(arrays["agent_states"][rows, 24], np.float32)
        valid = np.asarray(arrays["agent_valid"][rows, 24], bool)
        stimulus = batch_frame.stimulus_agent_index.to_numpy(np.int64)
        natural_external, changed_external = lateral_stimulus_trajectories(
            initial,
            valid,
            stimulus,
            batch_frame.target_lane_center_y_m.to_numpy(np.float32),
            batch_frame.controller_rate_rps.to_numpy(np.float32),
        )
        external_mask = np.zeros((len(batch_frame), 7), bool)
        external_mask[np.arange(len(batch_frame)), stimulus] = True
        prior_states, prior_valid = _prior_sequence(initial, valid)
        batch = _batch(
            prior_states,
            prior_valid,
            np.asarray(arrays["map_polylines"][rows]),
            np.asarray(arrays["map_polyline_valid"][rows]),
        )
        natural_states, changed_states, natural_actions, changed_actions = (
            [],
            [],
            [],
            [],
        )
        for rollout_id in range(args.futures):
            seeds = [
                _seed(
                    config["benchmark"]["id"],
                    training_seed,
                    str(item.probe_id),
                    float(item.controller_rate_rps),
                    rollout_id,
                )
                for item in batch_frame.itertuples(index=False)
            ]
            latent_sample, destination_sample = _per_scene_prior_samples(
                module, batch, seeds, device
            )
            natural = runner.run(
                batch,
                deterministic=False,
                exogenous_states=torch.from_numpy(natural_external),
                exogenous_mask=torch.from_numpy(external_mask),
                latent_sample=latent_sample,
                destination_sample=destination_sample,
                steps=75,
            )
            changed = runner.run(
                batch,
                deterministic=False,
                latent_sample=latent_sample,
                destination_sample=destination_sample,
                exogenous_states=torch.from_numpy(changed_external),
                exogenous_mask=torch.from_numpy(external_mask),
                steps=75,
            )
            natural_states.append(natural.states.cpu().numpy())
            changed_states.append(changed.states.cpu().numpy())
            natural_actions.append(natural.background_actions.cpu().numpy())
            changed_actions.append(changed.background_actions.cpu().numpy())
        natural_states = np.stack(natural_states, 1)
        changed_states = np.stack(changed_states, 1)
        # Formal response onset is reported at the benchmark's common 5 Hz
        # decision clock although TrafficBots internally acts at 25 Hz.
        natural_actions = np.stack(natural_actions, 1)[:, :, ::5]
        changed_actions = np.stack(changed_actions, 1)[:, :, ::5]
        for local, probe in enumerate(batch_frame.itertuples(index=False)):
            response_index, stimulus_index = int(probe.response_agent_index), int(
                probe.stimulus_agent_index
            )
            action_slot = response_index - 1
            acceleration_delta = (
                changed_actions[local, :, :, action_slot, 0]
                - natural_actions[local, :, :, action_slot, 0]
            )
            yaw_delta = (
                changed_actions[local, :, :, action_slot, 1]
                - natural_actions[local, :, :, action_slot, 1]
            )
            side = np.sign(
                initial[local, stimulus_index, 1] - initial[local, response_index, 1]
            )
            avoidance_sign = float(-side if side else 1.0)
            responded, latency, longitudinal, lateral = _response_metrics(
                acceleration_delta,
                yaw_delta,
                str(probe.stimulus_family),
                avoidance_sign,
            )
            response_shift = np.linalg.norm(
                changed_states[local, :, :, response_index, :2]
                - natural_states[local, :, :, response_index, :2],
                axis=-1,
            )
            unrelated = valid[local].copy()
            unrelated[[0, stimulus_index, response_index]] = False
            unrelated_shift = (
                float(
                    np.linalg.norm(
                        changed_states[local, :, :, unrelated, :2]
                        - natural_states[local, :, :, unrelated, :2],
                        axis=-1,
                    ).mean()
                )
                if unrelated.any()
                else 0.0
            )
            future_valid = np.broadcast_to(
                valid[local], natural_states[local].shape[:-1]
            )
            lengths, widths = (
                metadata["lengths_m"][int(probe.scenario_row)],
                metadata["widths_m"][int(probe.scenario_row)],
            )
            initial_collision = collision_matrix(
                initial[local], valid[local], lengths, widths
            )
            natural_collision = (
                collision_matrix(natural_states[local], future_valid, lengths, widths)
                & ~initial_collision[None, None]
            )
            changed_collision = (
                collision_matrix(changed_states[local], future_valid, lengths, widths)
                & ~initial_collision[None, None]
            )
            natural_offroad = straight_road_offroad(
                natural_states[local],
                future_valid,
                lengths,
                widths,
                np.asarray(arrays["map_polylines"][int(probe.scenario_row)]),
                np.asarray(arrays["map_polyline_valid"][int(probe.scenario_row)]),
            )
            changed_offroad = straight_road_offroad(
                changed_states[local],
                future_valid,
                lengths,
                widths,
                np.asarray(arrays["map_polylines"][int(probe.scenario_row)]),
                np.asarray(arrays["map_polyline_valid"][int(probe.scenario_row)]),
            )
            records.append(
                {
                    "probe_id": probe.probe_id,
                    "scenario_id": probe.scenario_id,
                    "recording_id": int(probe.recording_id),
                    "probe_manifest_version": "npc_interaction_d3_lateral_v1",
                    "method_id": method_id,
                    "fit_seed": fit_seed,
                    "training_seed": training_seed,
                    "stimulus_family": probe.stimulus_family,
                    "controller_rate_rps": float(probe.controller_rate_rps),
                    "requested_pairs": args.futures,
                    "completed_pairs": args.futures,
                    "response_probability": float(responded.mean()),
                    "right_censored_latency_s": float(
                        np.where(responded, latency, 3.0).mean()
                    ),
                    "longitudinal_response_probability": float(longitudinal.mean()),
                    "lateral_avoidance_probability": float(lateral.mean()),
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
                        abs(
                            changed_external[local, -1, stimulus_index, 1]
                            - probe.target_lane_center_y_m
                        )
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
                    "decision_clock_hz": 5,
                    "failure_type": "",
                    "fallback_used": False,
                }
            )
        checkpoint_rows()
        if begin and begin % (args.batch_size * 10) == 0:
            print(f"completed {begin}/{len(probes)}", flush=True)

    frame = pd.DataFrame(records)
    validate_lateral_probe_metrics(frame, expected_conditions=len(selected_probes))
    numeric = [
        name
        for name in frame.select_dtypes("number")
        if name
        not in {
            "fit_seed",
            "training_seed",
            "recording_id",
            "requested_pairs",
            "completed_pairs",
            "controller_rate_rps",
            "decision_clock_hz",
            "failure_type",
        }
    ]
    conditions = {
        f"{family}_{dose:g}": {
            "probes": len(group),
            **{name: float(group[name].mean()) for name in numeric},
        }
        for (family, dose), group in frame.groupby(
            ["stimulus_family", "controller_rate_rps"]
        )
    }
    summary = {
        "benchmark_id": config["benchmark"]["id"],
        "probe_manifest_version": "npc_interaction_d3_lateral_v1",
        "method_id": method_id,
        "fit_seed": fit_seed,
        "training_seed": training_seed,
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
        "backend": "shared_kinematic_traffic_dynamics",
        "background_scope": "all_six",
        "decision_clock_hz": 5,
        "condition_metrics": conditions,
        "overall_metrics": {name: float(frame[name].mean()) for name in numeric},
        "wall_time_seconds": time.time() - started,
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
    parser.add_argument("--futures", type=int, default=8)
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
