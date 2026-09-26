#!/usr/bin/env python3
"""Stress the same online NPC controller against realized NPC leader braking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_model_baselines.models.bayesian_ma_idm.src.model import (  # noqa: E402
    load_posterior, sample_driver_joint,
)
from hierarchical_world_model.scripts.render_online_npc_to_npc_demo import run_probe  # noqa: E402
from hierarchical_world_model.src.provenance import online_runtime_sha256  # noqa: E402
from hierarchical_world_model.src.stochastic_drivers.online import DEFAULT_MA_IDM_POSTERIOR  # noqa: E402
from traffic_components.src.core.utils import file_sha256, save_json  # noqa: E402


GAPS_M = (20.0, 30.0, 45.0)
SPEED_PAIRS_MPS = ((20.0, 20.0), (25.0, 20.0), (20.0, 15.0))
BRAKES_MPS2 = (-2.0, -4.0, -6.0, -8.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/hierarchical_world_model/evaluation/npc_to_npc_sweep.json",
    )
    args = parser.parse_args()
    posterior = load_posterior(DEFAULT_MA_IDM_POSTERIOR)
    rng = np.random.default_rng(17)
    theta = np.asarray(
        [sample_driver_joint(posterior, rng)[:5] for _ in range(6)], np.float32
    )
    records = []
    for gap in GAPS_M:
        for follower_speed, leader_speed in SPEED_PAIRS_MPS:
            control = run_probe(
                leader_brakes=False, theta=theta,
                initial_net_gap_m=gap, follower_speed_mps=follower_speed,
                leader_speed_mps=leader_speed,
            )
            for brake_setpoint in BRAKES_MPS2:
                brake = run_probe(
                    leader_brakes=True, theta=theta,
                    initial_net_gap_m=gap, follower_speed_mps=follower_speed,
                    leader_speed_mps=leader_speed,
                    leader_brake_mps2=brake_setpoint,
                )
                difference = brake["follower_actions"] - control["follower_actions"]
                closing_speed = max(follower_speed - leader_speed, 0.0)
                record = {
                    "initial_net_gap_m": gap,
                    "follower_speed_mps": follower_speed,
                    "leader_speed_mps": leader_speed,
                    "leader_brake_mps2": brake_setpoint,
                    "initial_ttc_s": None if closing_speed == 0.0 else gap / closing_speed,
                    "first_second_follower_action_effect_mps2": float(difference[1]),
                    "mean_first_second_follower_action_effect_mps2": float(difference[:25].mean()),
                    "first_second_direction_correct": bool(difference[1] < -0.05),
                    "mean_first_second_direction_correct": bool(difference[:25].mean() < -0.05),
                    "follower_response_frames": int(brake["follower_responses"].sum()),
                    "minimum_net_gap_m": float(brake["net_gap_m"].min()),
                    "npc_npc_overlap": bool(brake["npc_npc_overlap"]),
                    "control_npc_npc_overlap": bool(control["npc_npc_overlap"]),
                    "offroad": bool(brake["offroad"]),
                    "finite": bool(
                        np.isfinite(brake["states"]).all()
                        and np.isfinite(brake["follower_actions"]).all()
                    ),
                    "maximum_follower_jerk_mps3": float(
                        np.abs(np.diff(brake["follower_actions"]) / 0.04).max()
                    ),
                }
                records.append(record)
    ordered_groups = 0
    for gap in GAPS_M:
        for follower_speed, leader_speed in SPEED_PAIRS_MPS:
            group = sorted(
                (
                    item for item in records
                    if item["initial_net_gap_m"] == gap
                    and item["follower_speed_mps"] == follower_speed
                    and item["leader_speed_mps"] == leader_speed
                ),
                key=lambda item: item["leader_brake_mps2"],
            )
            effects = [item["mean_first_second_follower_action_effect_mps2"] for item in group]
            ordered_groups += int(np.all(np.diff(effects) >= -1.0e-4))
    report = {
        "schema": "online_npc_to_npc_highway_sweep_v1",
        "scope": "controlled HighwayEnv NPC-leader perturbations; same controller and posterior across independent worlds",
        "initial_net_gaps_m": list(GAPS_M),
        "speed_pairs_mps": [list(pair) for pair in SPEED_PAIRS_MPS],
        "leader_brake_setpoints_mps2": list(BRAKES_MPS2),
        "scenario_count": len(records),
        "finite_scenarios": sum(item["finite"] for item in records),
        "first_second_correct_scenarios": sum(
            item["first_second_direction_correct"] for item in records
        ),
        "mean_first_second_correct_scenarios": sum(
            item["mean_first_second_direction_correct"] for item in records
        ),
        "near_30m_scenarios": sum(item["initial_net_gap_m"] <= 30.0 for item in records),
        "near_30m_first_second_correct_scenarios": sum(
            item["initial_net_gap_m"] <= 30.0 and item["first_second_direction_correct"]
            for item in records
        ),
        "dose_ordered_gap_speed_groups": ordered_groups,
        "gap_speed_groups": len(GAPS_M) * len(SPEED_PAIRS_MPS),
        "npc_npc_overlap_scenarios": sum(item["npc_npc_overlap"] for item in records),
        "offroad_scenarios": sum(item["offroad"] for item in records),
        "maximum_follower_jerk_mps3": max(
            item["maximum_follower_jerk_mps3"] for item in records
        ),
        "records": records,
        "online_controller_sha256": file_sha256(
            ROOT / "hierarchical_world_model/src/stochastic_drivers/online.py"
        ),
        "online_runtime_sha256": online_runtime_sha256(),
    }
    save_json(report, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
