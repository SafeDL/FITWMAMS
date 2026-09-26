#!/usr/bin/env python3
"""Replay the generated-scene left-lane ADS policy and report terminal failures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.ads_interventions import OnlineSemanticLaneChangePolicy  # noqa: E402
from hierarchical_world_model.src.composition import HierarchicalWorldSampler  # noqa: E402
from hierarchical_world_model.src.collision_attribution import collision_events  # noqa: E402
from hierarchical_world_model.src.execution import _rollout_sample  # noqa: E402
from hierarchical_world_model.src.provenance import online_runtime_sha256  # noqa: E402
from hierarchical_world_model.scripts.ads_test_policies import MaintainEntrySpeedADS  # noqa: E402
from traffic_components.src.core.utils import save_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/hierarchical_world_model/evaluation/generated_left_lane_audit.json",
    )
    args = parser.parse_args()
    sampler = HierarchicalWorldSampler(
        flow_checkpoint=ROOT / "results/highd_natural_driving_flow/checkpoints/best_scenario_condition_flow.pt",
        flow_output_dir=ROOT / "results/highd_natural_driving_flow",
        diffusion_checkpoint=ROOT / "results/background_diffusion/checkpoints/best_background_diffusion.pt",
        diffusion_contract=ROOT / "results/background_diffusion/dataset_contract.json",
        response_checkpoint=ROOT / "results/hierarchical_world_model/factual_hiqr/checkpoints/final_world_model.pt",
        repo_root=ROOT, device=args.device,
    )
    completed = 0
    failures: list[dict[str, object]] = []
    collision_scenes = 0
    offroad_scenes = 0
    finite_scenes = 0
    raw_overlap_scenes = 0
    adjusted_overlap_scenes = 0
    cutin_exposure_scenes = 0
    maximum_error = 0.0
    for batch_index, begin in enumerate(range(0, args.scenes, args.batch_size)):
        size = min(args.batch_size, args.scenes - begin)
        exogenous = sampler.sample_world_exogenous(
            size, seed=args.seed + batch_index, response_steps=149
        )
        sample = sampler.compose_exogenous(exogenous)
        policy = OnlineSemanticLaneChangePolicy(
            MaintainEntrySpeedADS(start_frame=25), direction="left"
        )
        branch = _rollout_sample(
            sampler, sample, policy, reaction_controller="auto"
        )
        if policy.target_y is None:
            raise RuntimeError("left-lane ADS policy did not initialize")
        target_y = policy.target_y.cpu().numpy()
        final = branch.states[:, -1, 0]
        error = final[:, 1] - target_y
        heading = np.arctan2(final[:, 3], final[:, 2])
        done = (np.abs(error) < 0.15) & (np.abs(heading) < 0.02)
        collision = branch.collision_pairs.any(axis=(1, 2, 3))
        offroad = (branch.offroad & sample.initial_valid[:, None]).any(axis=(1, 2))
        completed += int(done.sum())
        collision_scenes += int(collision.sum())
        offroad_scenes += int(offroad.sum())
        finite_scenes += int(branch.numerical_valid.sum())
        geometry = collision_events(branch.states, sample.initial_valid[:, 1:])
        raw_overlap_scenes += int(geometry["raw"].sum())
        adjusted_overlap_scenes += int(geometry["adjusted"].sum())
        cutin_exposure_scenes += int(geometry["ads_cutin_exposure"].sum())
        maximum_error = max(maximum_error, float(np.max(np.abs(error))))
        for local in np.flatnonzero(~done):
            initial = sample.initial_states[local, 0]
            failures.append({
                "generated_scene_index": int(begin + local),
                "target_lane_y_m": float(target_y[local]),
                "initial_x_m": float(initial[0]),
                "initial_y_m": float(initial[1]),
                "final_x_m": float(final[local, 0]),
                "final_y_m": float(final[local, 1]),
                "final_lateral_error_m": float(error[local]),
                "final_heading_rad": float(heading[local]),
                "collision": bool(collision[local]),
                "offroad": bool(offroad[local]),
            })
        print(f"audited {begin + size}/{args.scenes} generated worlds", flush=True)
    report = {
        "schema": "generated_left_lane_completion_audit_v1",
        "scenes": args.scenes,
        "batch_size": args.batch_size,
        "seed_first_batch": args.seed,
        "completion_criteria": "mapped target lane centre error <0.15 m and absolute heading <0.02 rad",
        "completion_rate": completed / args.scenes,
        "finite_scenes": finite_scenes,
        "collision_scenes": collision_scenes,
        "offroad_scenes": offroad_scenes,
        "raw_overlap_scenes": raw_overlap_scenes,
        "adjusted_overlap_scenes": adjusted_overlap_scenes,
        "ads_cutin_exposure_scenes": cutin_exposure_scenes,
        "maximum_absolute_final_lane_error_m": maximum_error,
        "incomplete_examples": failures,
        "online_runtime_sha256": online_runtime_sha256(),
    }
    save_json(report, args.output)
    print(report)


if __name__ == "__main__":
    main()
