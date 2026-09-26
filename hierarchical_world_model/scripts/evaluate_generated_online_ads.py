#!/usr/bin/env python3
"""Multi-ADS smoke audit of the actual Flow-generated HighwayEnv world."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.ads_interventions import (  # noqa: E402
    OnlineAccelerationWindowPolicy, OnlineSemanticLaneChangePolicy,
)
from hierarchical_world_model.src.composition import HierarchicalWorldSampler  # noqa: E402
from hierarchical_world_model.src.collision_attribution import collision_events  # noqa: E402
from hierarchical_world_model.src.execution import _rollout_sample, hold_current_ego_action  # noqa: E402
from hierarchical_world_model.src.provenance import online_runtime_sha256  # noqa: E402
from hierarchical_world_model.scripts.ads_test_policies import MaintainEntrySpeedADS  # noqa: E402
from traffic_components.src.core.utils import file_sha256, save_json  # noqa: E402


DOSES = (-8.0, -6.0, -4.0, -2.0, 2.0, 4.0)


def _rear_receiver(initial: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx = initial[:, 1:, 0] - initial[:, :1, 0]
    dy = initial[:, 1:, 1] - initial[:, :1, 1]
    eligible = valid[:, 1:] & (dx < -4.8) & (np.abs(dy) < 1.8)
    receiver = np.where(eligible, dx, -np.inf).argmax(axis=1)
    return eligible.any(axis=1), receiver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/hierarchical_world_model/evaluation/generated_ads_sweep.json",
    )
    args = parser.parse_args()
    controller_hash = file_sha256(
        ROOT / "hierarchical_world_model/src/stochastic_drivers/online.py"
    )
    if args.scenes < 1 or args.batch_size < 1:
        raise ValueError("scenes and batch-size must be positive")
    sampler = HierarchicalWorldSampler(
        flow_checkpoint=ROOT / "results/highd_natural_driving_flow/checkpoints/best_scenario_condition_flow.pt",
        flow_output_dir=ROOT / "results/highd_natural_driving_flow",
        diffusion_checkpoint=ROOT / "results/background_diffusion/checkpoints/best_background_diffusion.pt",
        diffusion_contract=ROOT / "results/background_diffusion/dataset_contract.json",
        response_checkpoint=ROOT / "results/hierarchical_world_model/factual_hiqr/checkpoints/final_world_model.pt",
        repo_root=ROOT, device=args.device,
    )
    policies = ("-8.0", "-6.0", "-4.0", "-2.0", "2.0", "4.0", "left", "hold")
    totals = {
        key: {
            "finite": 0, "collision": 0, "offroad": 0,
            "geometric_raw_overlap": 0, "geometric_adjusted_overlap": 0,
            "ads_pursuit_exposure": 0, "ads_cutin_exposure": 0,
            "rear_correct": 0, "rear_count": 0,
            "rear_same_route_correct": 0, "rear_same_route_count": 0,
            "rear_route_switched_count": 0,
            "rear_incorrect_examples": [],
        }
        for key in policies
    }
    lane_complete = 0
    lane_error = []
    lane_incomplete_examples: list[dict[str, object]] = []
    for batch_index, begin in enumerate(range(0, args.scenes, args.batch_size)):
        size = min(args.batch_size, args.scenes - begin)
        exogenous = sampler.sample_world_exogenous(
            size, seed=args.seed + batch_index, response_steps=149
        )
        sample = sampler.compose_exogenous(exogenous)
        near_rear, receiver = _rear_receiver(sample.initial_states, sample.initial_valid)
        branches = {}
        for dose in DOSES:
            branches[str(dose)] = _rollout_sample(
                sampler, sample,
                OnlineAccelerationWindowPolicy(hold_current_ego_action, dose),
                reaction_controller="auto",
            )
        left_policy = OnlineSemanticLaneChangePolicy(
            MaintainEntrySpeedADS(start_frame=25), direction="left"
        )
        branches["left"] = _rollout_sample(
            sampler, sample, left_policy, reaction_controller="auto"
        )
        # Reference is last and is never supplied to an intervention branch.
        branches["hold"] = _rollout_sample(
            sampler, sample, hold_current_ego_action, reaction_controller="auto"
        )
        baseline = branches["hold"]
        for key, branch in branches.items():
            totals[key]["finite"] += int(branch.numerical_valid.sum())
            totals[key]["collision"] += int(branch.collision_pairs.any(axis=(1, 2, 3)).sum())
            totals[key]["offroad"] += int(
                (branch.offroad & sample.initial_valid[:, None]).any(axis=(1, 2)).sum()
            )
            events = collision_events(
                branch.states, sample.initial_valid[:, 1:],
                ads_acceleration_setpoint_mps2=(
                    float(key) if key not in {"left", "hold"} else None
                ),
            )
            totals[key]["geometric_raw_overlap"] += int(events["raw"].sum())
            totals[key]["geometric_adjusted_overlap"] += int(events["adjusted"].sum())
            totals[key]["ads_pursuit_exposure"] += int(events["ads_pursuit"].sum())
            totals[key]["ads_cutin_exposure"] += int(events["ads_cutin_exposure"].sum())
            if key in {"left", "hold"} or not near_rear.any():
                continue
            row = np.flatnonzero(near_rear)
            action = branch.background_actions[row, :, receiver[row], 0]
            baseline_action = baseline.background_actions[row, :, receiver[row], 0]
            effect = (action[:, 25:100] - baseline_action[:, 25:100]).mean(axis=1)
            correct = effect < 0 if float(key) < 0 else effect > 0
            totals[key]["rear_correct"] += int(correct.sum())
            totals[key]["rear_count"] += len(row)
            final_y = branch.states[row, -1, receiver[row] + 1, 1]
            baseline_final_y = baseline.states[row, -1, receiver[row] + 1, 1]
            route_switched = np.abs(final_y - baseline_final_y) > 1.8
            totals[key]["rear_same_route_correct"] += int((correct & ~route_switched).sum())
            totals[key]["rear_same_route_count"] += int((~route_switched).sum())
            totals[key]["rear_route_switched_count"] += int(route_switched.sum())
            for local in np.flatnonzero(~correct):
                slot = int(receiver[row[local]] + 1)
                totals[key]["rear_incorrect_examples"].append({
                    "generated_scene_index": int(begin + row[local]),
                    "npc_slot": slot,
                    "mean_action_effect_mps2": float(effect[local]),
                    "anchor_y_m": float(sample.initial_states[row[local], slot, 1]),
                    "final_y_m": float(branch.states[row[local], -1, slot, 1]),
                    "hold_final_y_m": float(baseline.states[row[local], -1, slot, 1]),
                    "route_switched": bool(route_switched[local]),
                })
        target = left_policy.target_y
        if target is None:
            raise RuntimeError("left-lane ADS policy did not initialize")
        final = branches["left"].states[:, -1, 0]
        error = final[:, 1] - target.cpu().numpy()
        heading = np.arctan2(final[:, 3], final[:, 2])
        complete = (np.abs(error) < 0.15) & (np.abs(heading) < 0.02)
        lane_complete += int(complete.sum())
        lane_error.extend(np.abs(error).tolist())
        for local in np.flatnonzero(~complete):
            lane_incomplete_examples.append({
                "generated_scene_index": int(begin + local),
                "target_lane_y_m": float(target.cpu().numpy()[local]),
                "final_y_m": float(final[local, 1]),
                "final_lateral_error_m": float(error[local]),
                "final_heading_rad": float(heading[local]),
                "collision": bool(branches["left"].collision_pairs[local].any()),
                "offroad": bool((branches["left"].offroad[local]
                                  & sample.initial_valid[local, None]).any()),
            })
        print(f"completed {begin + size}/{args.scenes} generated worlds", flush=True)
    report = {
        "schema": "generated_online_ads_sweep_v1",
        "scenes": args.scenes,
        "batch_size": args.batch_size,
        "seed_first_batch": args.seed,
        "protocol": "Flow-generated independent single-pass HighwayEnv worlds; hold-current ADS run used only for after-the-fact effect comparison",
        "policies": totals,
        "left_lane_completion_rate": lane_complete / args.scenes,
        "left_lane_max_absolute_final_error_m": max(lane_error),
        "left_lane_incomplete_examples": lane_incomplete_examples,
        "online_controller_sha256": controller_hash,
        "online_runtime_sha256": online_runtime_sha256(),
        "limitations": (
            f"This {args.scenes}-scene generated runtime audit is separate from held-out highD Test; "
            "longer-horizon replanning and wider ADS policy coverage remain open."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, args.output)
    print(report)


if __name__ == "__main__":
    main()
