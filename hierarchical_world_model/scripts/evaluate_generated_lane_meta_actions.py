#!/usr/bin/env python3
"""Matched Flow-world evaluation of complete left/right ADS lane meta-actions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.scripts.ads_test_policies import MaintainEntrySpeedADS  # noqa: E402
from hierarchical_world_model.src.ads_interventions import (  # noqa: E402
    OnlineAccelerationWindowPolicy,
    OnlineSemanticLaneChangePolicy,
)
from hierarchical_world_model.src.collision_attribution import collision_events  # noqa: E402
from hierarchical_world_model.src.composition import HierarchicalWorldSampler  # noqa: E402
from hierarchical_world_model.src.execution import _rollout_sample, hold_current_ego_action  # noqa: E402
from hierarchical_world_model.src.provenance import online_runtime_sha256  # noqa: E402
from traffic_components.src.core.utils import file_sha256, save_json  # noqa: E402


def _rear_in_target_lane(
    states_at_entry: np.ndarray,
    valid: np.ndarray,
    target_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find the nearest rear NPC already occupying the selected target lane."""
    dx = states_at_entry[:, 1:, 0] - states_at_entry[:, :1, 0]
    dy = states_at_entry[:, 1:, 1] - target_y[:, None]
    eligible = valid[:, 1:] & (dx < -4.8) & (np.abs(dy) < 1.8)
    receiver = np.where(eligible, dx, -np.inf).argmax(axis=1)
    has_receiver = eligible.any(axis=1)
    row = np.arange(len(states_at_entry))
    selected_gap = states_at_entry[row, 0, 0] - states_at_entry[row, receiver + 1, 0] - 4.8
    ego_speed = np.linalg.norm(states_at_entry[:, 0, 2:4], axis=-1)
    rear_speed = np.linalg.norm(states_at_entry[row, receiver + 1, 2:4], axis=-1)
    selected_closing = rear_speed - ego_speed
    return has_receiver, receiver, selected_gap, selected_closing


def _lane_policy(direction: str, *, braking: bool) -> OnlineSemanticLaneChangePolicy:
    base = MaintainEntrySpeedADS(start_frame=25)
    if braking:
        base = OnlineAccelerationWindowPolicy(
            base, acceleration_mps2=-4.0, start_frame=25, stop_frame=50
        )
    return OnlineSemanticLaneChangePolicy(base, direction=direction)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/hierarchical_world_model/evaluation/lane_meta_actions.json",
    )
    args = parser.parse_args()
    if args.scenes < 1 or args.batch_size < 1:
        raise ValueError("scenes and batch-size must be positive")

    sampler = HierarchicalWorldSampler(
        flow_checkpoint=ROOT / "results/highd_natural_driving_flow/checkpoints/best_scenario_condition_flow.pt",
        flow_output_dir=ROOT / "results/highd_natural_driving_flow",
        diffusion_checkpoint=ROOT / "results/background_diffusion/checkpoints/best_background_diffusion.pt",
        diffusion_contract=ROOT / "results/background_diffusion/dataset_contract.json",
        response_checkpoint=ROOT / "results/hierarchical_world_model/factual_hiqr/checkpoints/final_world_model.pt",
        repo_root=ROOT,
        device=args.device,
    )
    policy_keys = ("left", "right", "brake_left", "brake_right", "hold")
    totals = {
        key: {
            "numerically_finite": 0,
            "collision_scenes": 0,
            "offroad_scenes": 0,
            "raw_geometric_overlap_scenes": 0,
            "ads_cutin_exposure_scenes": 0,
        }
        for key in policy_keys
    }
    lane_stats = {
        key: {"completed": 0, "scenes": 0, "absolute_terminal_error_m": [],
              "incomplete_examples": []}
        for key in ("left", "right", "brake_left", "brake_right")
    }
    target_rear = {
        key: {
            "scenes": 0,
            "braking_response_scenes": 0,
            "changed_action_scenes": 0,
            "mean_effect_mps2": [],
            "near_20m_scenes": 0,
            "near_20m_closing_scenes": 0,
            "near_20m_braking_response_scenes": 0,
            "near_20m_effect_mps2": [],
        }
        for key in ("left", "right", "brake_left", "brake_right")
    }

    for batch_index, begin in enumerate(range(0, args.scenes, args.batch_size)):
        size = min(args.batch_size, args.scenes - begin)
        exogenous = sampler.sample_world_exogenous(
            size, seed=args.seed + batch_index, response_steps=149
        )
        sample = sampler.compose_exogenous(exogenous)
        lane_policies = {
            "left": _lane_policy("left", braking=False),
            "right": _lane_policy("right", braking=False),
            "brake_left": _lane_policy("left", braking=True),
            "brake_right": _lane_policy("right", braking=True),
        }
        branches = {
            key: _rollout_sample(
                sampler, sample, policy, reaction_controller="auto"
            )
            for key, policy in lane_policies.items()
        }
        branches["hold"] = _rollout_sample(
            sampler, sample, hold_current_ego_action, reaction_controller="auto"
        )
        for key, branch in branches.items():
            totals[key]["numerically_finite"] += int(branch.numerical_valid.sum())
            totals[key]["collision_scenes"] += int(
                branch.collision_pairs.any(axis=(1, 2, 3)).sum()
            )
            totals[key]["offroad_scenes"] += int(
                (branch.offroad & sample.initial_valid[:, None]).any(axis=(1, 2)).sum()
            )
            events = collision_events(branch.states, sample.initial_valid[:, 1:])
            totals[key]["raw_geometric_overlap_scenes"] += int(events["raw"].sum())
            totals[key]["ads_cutin_exposure_scenes"] += int(
                events["ads_cutin_exposure"].sum()
            )

            if key == "hold":
                continue
            policy = lane_policies[key]
            if policy.target_y is None:
                raise RuntimeError(f"{key} ADS policy did not choose a mapped lane")
            target_y = policy.target_y.detach().cpu().numpy()
            terminal = branch.states[:, -1, 0]
            terminal_error = terminal[:, 1] - target_y
            terminal_heading = np.arctan2(terminal[:, 3], terminal[:, 2])
            complete = (np.abs(terminal_error) <= 0.15) & (np.abs(terminal_heading) <= 0.02)
            lane_stats[key]["completed"] += int(complete.sum())
            lane_stats[key]["scenes"] += size
            lane_stats[key]["absolute_terminal_error_m"].extend(
                np.abs(terminal_error).tolist()
            )
            for local in np.flatnonzero(~complete):
                lane_stats[key]["incomplete_examples"].append({
                    "generated_scene_index": int(begin + local),
                    "target_lane_y_m": float(target_y[local]),
                    "terminal_lane_y_m": float(terminal[local, 1]),
                    "terminal_heading_rad": float(terminal_heading[local]),
                    "collision": bool(branch.collision_pairs[local].any()),
                    "offroad": bool((branch.offroad[local]
                                     & sample.initial_valid[local, None]).any()),
                })

            eligible, receiver, gap_m, closing_speed = _rear_in_target_lane(
                branch.states[:, 25], sample.initial_valid, target_y
            )
            rows = np.flatnonzero(eligible)
            if len(rows):
                action = branch.background_actions[rows, 25:100, receiver[rows], 0]
                baseline = branches["hold"].background_actions[
                    rows, 25:100, receiver[rows], 0
                ]
                effect = (action - baseline).mean(axis=1)
                maximum_effect = np.max(np.abs(action - baseline), axis=1)
                target_rear[key]["scenes"] += len(rows)
                target_rear[key]["braking_response_scenes"] += int((effect < -0.05).sum())
                target_rear[key]["changed_action_scenes"] += int((maximum_effect >= 0.1).sum())
                target_rear[key]["mean_effect_mps2"].extend(effect.tolist())
                near = gap_m[rows] <= 20.0
                near_rows = rows[near]
                if len(near_rows):
                    near_effect = effect[near]
                    near_closing = closing_speed[near_rows] > 0.5
                    target_rear[key]["near_20m_scenes"] += len(near_rows)
                    target_rear[key]["near_20m_closing_scenes"] += int(near_closing.sum())
                    target_rear[key]["near_20m_braking_response_scenes"] += int(
                        ((near_effect < -0.05) & near_closing).sum()
                    )
                    target_rear[key]["near_20m_effect_mps2"].extend(near_effect.tolist())
        print(f"completed {begin + size}/{args.scenes} matched generated worlds", flush=True)

    for key in totals:
        totals[key]["collision_rate"] = totals[key]["collision_scenes"] / args.scenes
        totals[key]["offroad_rate"] = totals[key]["offroad_scenes"] / args.scenes
        totals[key]["raw_geometric_overlap_rate"] = (
            totals[key]["raw_geometric_overlap_scenes"] / args.scenes
        )
    for key, values in lane_stats.items():
        error = values.pop("absolute_terminal_error_m")
        values["completion_rate"] = values["completed"] / values["scenes"]
        values["terminal_error_p95_m"] = float(np.quantile(error, 0.95))
        values["terminal_error_max_m"] = float(np.max(error))
    for key, values in target_rear.items():
        effects = values.pop("mean_effect_mps2")
        near_effects = values.pop("near_20m_effect_mps2")
        values["braking_response_rate"] = (
            values["braking_response_scenes"] / values["scenes"]
            if values["scenes"] else None
        )
        values["mean_effect_mps2"] = float(np.mean(effects)) if effects else None
        values["median_effect_mps2"] = float(np.median(effects)) if effects else None
        values["changed_action_rate"] = (
            values["changed_action_scenes"] / values["scenes"]
            if values["scenes"] else None
        )
        values["near_20m_braking_response_rate_among_closing"] = (
            values["near_20m_braking_response_scenes"] / values["near_20m_closing_scenes"]
            if values["near_20m_closing_scenes"] else None
        )
        values["near_20m_mean_effect_mps2"] = (
            float(np.mean(near_effects)) if near_effects else None
        )

    report = {
        "schema": "generated_lane_meta_actions_v1",
        "scenes": args.scenes,
        "batch_size": args.batch_size,
        "first_batch_seed": args.seed,
        "actions": {
            "left": "complete map-centred left lane change; entry-speed hold",
            "right": "complete map-centred right lane change; entry-speed hold",
            "brake_left": "-4 m/s^2 for frames 25-49 plus complete left lane change",
            "brake_right": "-4 m/s^2 for frames 25-49 plus complete right lane change",
            "hold": "matched exogenous reference; used only for after-the-fact NPC effect comparison",
        },
        "worlds": "Each branch starts from the same Flow/Diffusion exogenous sample and independently runs one causal closed loop with the online NPC controller.",
        "lane_completion": lane_stats,
        "target_lane_rear_npc_response": target_rear,
        "outcomes": totals,
        "collision_policy": "Collisions and off-road events are reported outcomes with ADS cut-in exposure tags; neither is a zero-required pass gate.",
        "online_runtime_sha256": online_runtime_sha256(),
        "online_controller_sha256": file_sha256(ROOT / "hierarchical_world_model/src/stochastic_drivers/online.py"),
        "limitations": "Generated-world response audit; it supplements but does not replace held-out highD Test reconstruction or observed natural collision distributions.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, args.output)
    print(report)


if __name__ == "__main__":
    main()
