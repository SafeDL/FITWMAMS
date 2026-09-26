#!/usr/bin/env python3
"""Replay selected deterministic Flow worlds with full ego/NPC traces."""

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
from hierarchical_world_model.src.execution import _rollout_sample, hold_current_ego_action  # noqa: E402
from hierarchical_world_model.src.provenance import online_runtime_sha256  # noqa: E402
from hierarchical_world_model.scripts.ads_test_policies import MaintainEntrySpeedADS  # noqa: E402
from traffic_components.src.core.utils import save_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maintain-speed", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/hierarchical_world_model/evaluation/generated_lane_change_replays.json",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or any(index < 0 for index in args.indices):
        raise ValueError("batch size must be positive and scene indices nonnegative")
    sampler = HierarchicalWorldSampler(
        flow_checkpoint=ROOT / "results/highd_natural_driving_flow/checkpoints/best_scenario_condition_flow.pt",
        flow_output_dir=ROOT / "results/highd_natural_driving_flow",
        diffusion_checkpoint=ROOT / "results/background_diffusion/checkpoints/best_background_diffusion.pt",
        diffusion_contract=ROOT / "results/background_diffusion/dataset_contract.json",
        response_checkpoint=ROOT / "results/hierarchical_world_model/factual_hiqr/checkpoints/final_world_model.pt",
        repo_root=ROOT, device=args.device,
    )
    by_batch: dict[int, list[int]] = {}
    for index in sorted(set(args.indices)):
        by_batch.setdefault(index // args.batch_size, []).append(index)
    records: list[dict[str, object]] = []
    for batch_index, indices in by_batch.items():
        begin = batch_index * args.batch_size
        exogenous = sampler.sample_world_exogenous(
            args.batch_size, seed=args.seed + batch_index, response_steps=149
        )
        sample = sampler.compose_exogenous(exogenous)
        ads_base = (
            MaintainEntrySpeedADS(start_frame=25)
            if args.maintain_speed else hold_current_ego_action
        )
        policy = OnlineSemanticLaneChangePolicy(ads_base, direction="left")
        branch = _rollout_sample(
            sampler, sample, policy, reaction_controller="auto"
        )
        if policy.target_y is None:
            raise RuntimeError("left-lane ADS policy did not initialize")
        target_y = policy.target_y.cpu().numpy()
        for index in indices:
            local = index - begin
            states = branch.states[local]
            actions = branch.ego_actions[local]
            final = states[-1, 0]
            error = float(final[1] - target_y[local])
            heading = float(np.arctan2(final[3], final[2]))
            records.append({
                "generated_scene_index": index,
                "batch_seed": args.seed + batch_index,
                "target_lane_y_m": float(target_y[local]),
                "initial_speed_mps": float(np.linalg.norm(states[0, 0, 2:4])),
                "final_speed_mps": float(np.linalg.norm(final[2:4])),
                "final_lateral_error_m": error,
                "final_heading_rad": heading,
                "collision": bool(branch.collision_pairs[local].any()),
                "offroad": bool(branch.offroad[local].any()),
                "states": states.tolist(),
                "ego_actions": actions.tolist(),
            })
    report = {
        "schema": "generated_lane_change_replays_v1",
        "batch_size": args.batch_size,
        "seed_first_batch": args.seed,
        "completion_criteria": "mapped target lane centre error <0.15 m and absolute heading <0.02 rad",
        "online_runtime_sha256": online_runtime_sha256(),
        "records": records,
    }
    save_json(report, args.output)
    print({key: value for key, value in report.items() if key != "records"})
    for record in records:
        print({key: value for key, value in record.items() if key not in {"states", "ego_actions"}})


if __name__ == "__main__":
    main()
