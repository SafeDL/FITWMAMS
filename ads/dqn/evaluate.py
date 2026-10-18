#!/usr/bin/env python3
"""Fixed-event, per-event ADS evaluation for paired comparisons."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import pandas as pd
import numpy as np
from stable_baselines3 import DQN

from .env import ADSHighwayEnv
from .events import HighDEventStore
from .provenance import artifact_record
from .runtime import DEFAULT_WORLD_CONFIG, FrozenWorldFactory


def _publish_evaluation_artifacts(
    output: Path, results: pd.DataFrame, metadata: dict[str, object]
) -> None:
    """Publish an immutable CSV/JSON pair, with JSON as the commit marker."""
    output = Path(output)
    metadata_path = output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or metadata_path.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing evaluation artifact pair: {output}"
        )
    temporary_paths: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            csv_temporary = Path(stream.name)
            temporary_paths.append(csv_temporary)
            results.to_csv(stream, index=False, lineterminator="\n")
            stream.flush()
            os.fsync(stream.fileno())
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{metadata_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            metadata_temporary = Path(stream.name)
            temporary_paths.append(metadata_temporary)
            stream.write(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Same-directory hard links make each publication atomic and reject a
        # concurrent writer. The JSON file is published last as the pair's
        # completion marker; an interrupted partial pair fails closed.
        os.link(csv_temporary, output)
        os.link(metadata_temporary, metadata_path)
        directory_fd = os.open(
            output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


class RandomPolicy:
    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(int(seed))

    def predict(self, observation, deterministic: bool = True):
        del observation, deterministic
        return int(self.rng.integers(5)), None


def evaluate_policy(model, env: ADSHighwayEnv) -> pd.DataFrame:
    rows = []
    for index in range(len(env.events)):
        observation, reset_info = env.reset(options={"event_index": index})
        total_reward = 0.0
        decisions = 0
        minimum_distance_m = float("inf")
        minimum_ttc_s = float("inf")
        acceleration_magnitude = []
        action_counts = np.zeros(5, np.int64)
        action_switches = 0
        previous_action = None
        terminated = truncated = False
        info = dict(reset_info)
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            action = int(action)
            action_counts[action] += 1
            action_switches += int(
                previous_action is not None and action != previous_action
            )
            previous_action = action
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            decisions += 1
            acceleration_magnitude.append(abs(float(env._last_ego_action[0])))
            states = env._states()
            valid = np.asarray(env.event.valid, bool)
            if valid[1:].any():
                relative = states[1:][valid[1:], :2] - states[0, :2]
                dimensions = (
                    np.broadcast_to(np.asarray((4.8, 1.8), np.float64), (7, 2))
                    if env.event.vehicle_dimensions is None
                    else np.asarray(env.event.vehicle_dimensions, np.float64)
                )
                npc_dimensions = dimensions[1:][valid[1:]]
                longitudinal = np.abs(relative[:, 0]) - 0.5 * (
                    dimensions[0, 0] + npc_dimensions[:, 0]
                )
                lateral = np.abs(relative[:, 1]) - 0.5 * (
                    dimensions[0, 1] + npc_dimensions[:, 1]
                )
                clearance = np.hypot(
                    np.maximum(longitudinal, 0.0),
                    np.maximum(lateral, 0.0),
                ) + np.minimum(np.maximum(longitudinal, lateral), 0.0)
                minimum_distance_m = min(
                    minimum_distance_m,
                    float(clearance.min()),
                )
                dx = relative[:, 0]
                dy = relative[:, 1]
                closing = states[0, 2] - states[1:][valid[1:], 2]
                bumper_gap = dx - 0.5 * (dimensions[0, 0] + npc_dimensions[:, 0])
                lateral_overlap = np.abs(dy) < 0.5 * (
                    dimensions[0, 1] + npc_dimensions[:, 1]
                )
                candidate = (bumper_gap > 0.0) & lateral_overlap & (closing > 0.05)
                if candidate.any():
                    minimum_ttc_s = min(
                        minimum_ttc_s,
                        float((bumper_gap[candidate] / closing[candidate]).min()),
                    )
        rows.append(
            {
                "event_id": info["event_id"],
                "source_event_id": info["source_event_id"],
                "failure": bool(
                    info["collision"] or info["offroad"] or info.get("stalled", False)
                ),
                "collision": bool(info["collision"]),
                "any_vehicle_collision": bool(
                    info.get("any_vehicle_collision", info["collision"])
                ),
                "npc_collision": bool(info.get("npc_collision", False)),
                "offroad": bool(info["offroad"]),
                "stalled": bool(info.get("stalled", False)),
                "progress_m": float(info["progress_m"]),
                "speed_mps": float(info["speed_mps"]),
                "reward": total_reward,
                "decisions": decisions,
                "minimum_distance_m": minimum_distance_m,
                "minimum_clearance_m": minimum_distance_m,
                "minimum_ttc_s": minimum_ttc_s,
                "mean_abs_acceleration_mps2": float(np.mean(acceleration_magnitude)),
                "action_switches": action_switches,
                "npc_track_count": int(info["npc_track_count"]),
                "npc_longitudinal_count": int(info["npc_longitudinal_count"]),
                "npc_maneuver_count": int(info["npc_maneuver_count"]),
                "npc_mean_abs_delta_ax": float(info["npc_mean_abs_delta_ax"]),
                "npc_mean_abs_delta_yaw": float(info["npc_mean_abs_delta_yaw"]),
                **{
                    f"action_{action_index}_count": int(count)
                    for action_index, count in enumerate(action_counts)
                },
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--algorithm", choices=("dqn", "random"), default="dqn")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--sequence-cache", type=Path, required=True)
    parser.add_argument("--world-config", type=Path, default=DEFAULT_WORLD_CONFIG)
    parser.add_argument(
        "--behavior-randomization", choices=("W0", "W1", "W2"), default="W2"
    )
    parser.add_argument("--mobil-mode", choices=("M0", "M1", "M2", "M3"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--world-seed", type=int, default=0)
    parser.add_argument("--policy-seed", type=int, default=0)
    args = parser.parse_args()
    controller_kwargs = {}
    behavior_randomization = args.behavior_randomization
    if args.mobil_mode == "M0":
        controller_kwargs["mobil_minimum_incentive_mps2"] = 1.0e6
        behavior_randomization = "W1"
    elif args.mobil_mode == "M1":
        controller_kwargs.update(
            mobil_minimum_incentive_mps2=0.0,
            mobil_plan_keep_bias_mps2=0.0,
        )
        behavior_randomization = "W1"
    elif args.mobil_mode == "M2":
        behavior_randomization = "W1"
    elif args.mobil_mode == "M3":
        behavior_randomization = "W2"
    factory = FrozenWorldFactory(
        args.world_config,
        device=args.device,
        behavior_randomization=behavior_randomization,
        controller_kwargs=controller_kwargs,
    )
    events = HighDEventStore(
        args.events,
        args.sequence_cache,
        world_seed=args.world_seed,
    )
    env = ADSHighwayEnv(events, factory, random_events=False)
    if args.algorithm == "random":
        model = RandomPolicy(args.policy_seed)
    else:
        if args.policy is None:
            parser.error("--policy is required for DQN evaluation")
        model = DQN.load(args.policy, device=args.device)
    started = time.perf_counter()
    results = evaluate_policy(model, env)
    elapsed = time.perf_counter() - started
    finite_distance = results.loc[
        np.isfinite(results["minimum_distance_m"]), "minimum_distance_m"
    ]
    finite_ttc = results.loc[np.isfinite(results["minimum_ttc_s"]), "minimum_ttc_s"]
    summary = {
        "algorithm": args.algorithm,
        "events": int(len(results)),
        "failures": int(results["failure"].sum()),
        "failure_rate": float(results["failure"].mean()),
        "mean_progress_m": float(results["progress_m"].mean()),
        "mean_reward": float(results["reward"].mean()),
        "mean_minimum_distance_m": (
            None if finite_distance.empty else float(finite_distance.mean())
        ),
        "distance_metric": "axis_aligned_vehicle_surface_clearance_m",
        "minimum_distance_event_count": int(len(finite_distance)),
        "median_minimum_ttc_s": (
            None if finite_ttc.empty else float(finite_ttc.median())
        ),
        "finite_ttc_event_count": int(len(finite_ttc)),
        "npc_collision_rate": float(results["npc_collision"].mean()),
        "any_vehicle_collision_rate": float(results["any_vehicle_collision"].mean()),
        "mean_abs_acceleration_mps2": float(
            results["mean_abs_acceleration_mps2"].mean()
        ),
        "npc_response_mode_counts": {
            "track": int(results["npc_track_count"].sum()),
            "longitudinal": int(results["npc_longitudinal_count"].sum()),
            "maneuver": int(results["npc_maneuver_count"].sum()),
        },
        "npc_mean_abs_delta_ax": float(results["npc_mean_abs_delta_ax"].mean()),
        "npc_mean_abs_delta_yaw": float(results["npc_mean_abs_delta_yaw"].mean()),
        "elapsed_s": elapsed,
        "events_per_s": float(len(results) / elapsed),
        "world_seed": args.world_seed,
        "policy_seed": args.policy_seed,
        "behavior_randomization": behavior_randomization,
        "mobil_mode": args.mobil_mode,
        "input_artifacts": {
            "policy": artifact_record(args.policy),
            "events": artifact_record(args.events),
            "world_config": artifact_record(args.world_config),
            "sequence_cache": {
                name: artifact_record(args.sequence_cache / name)
                for name in (
                    "sequence_id.npy",
                    "agent_states.npy",
                    "agent_valid.npy",
                    "map_polylines.npy",
                    "map_polyline_valid.npy",
                    "vehicle_dimensions.npy",
                )
            },
        },
    }
    _publish_evaluation_artifacts(args.output, results, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
