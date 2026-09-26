#!/usr/bin/env python3
"""Inspect mapped NPC lane changes that used spatial-path progress."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import ANCHOR_INDEX  # noqa: E402
from hierarchical_world_model.src.ads_interventions import AbsoluteAccelerationPolicy  # noqa: E402
from hierarchical_world_model.scripts.evaluate_online_factual import _theta_for_rows  # noqa: E402
from hierarchical_world_model.src.collision_attribution import collision_events  # noqa: E402
from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.evaluation import _logged_ego_actions, rollout  # noqa: E402
from hierarchical_world_model.src.planner import frozen_diffusion_plans  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from hierarchical_world_model.src.provenance import online_runtime_sha256  # noqa: E402
from hierarchical_world_model.src.stochastic_drivers.online import (  # noqa: E402
    DEFAULT_MA_IDM_POSTERIOR, OnlineMAIDMController, planned_adjacent_lane_intent,
)
from hierarchical_world_model.src.train import load_checkpoint  # noqa: E402
from traffic_components.src.core.utils import file_sha256, save_json, select_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ads-acceleration-mps2", type=float)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/hierarchical_world_model/evaluation/spatial_path_scene_audit.json",
    )
    args = parser.parse_args()
    controller_hash = file_sha256(
        ROOT / "hierarchical_world_model/src/stochastic_drivers/online.py"
    )
    config = load_protocol_config(ROOT / "hierarchical_world_model/config/world_model.yaml")
    experiment = prepare_experiment_data(config, ROOT)
    test_rows = np.asarray(experiment.test_rows, np.int64)
    arrays = experiment.bundle.arrays
    states = np.asarray(arrays["agent_states"][test_rows], np.float32)
    valid = np.asarray(arrays["agent_valid"][test_rows], bool)
    maps = np.asarray(arrays["map_polylines"][test_rows], np.float32)
    map_valid = np.asarray(arrays["map_polyline_valid"][test_rows], bool)
    device = select_device("cuda")
    plans = frozen_diffusion_plans(
        experiment.bundle, test_rows, checkpoint=config["paths"]["diffusion_checkpoint"],
        output_dir=ROOT / "results/hierarchical_world_model/evaluation/plans_full",
        device=device, batch_size=64, ddim_steps=20, experiment_scope="full",
    )
    intent, _, target_y = planned_adjacent_lane_intent(
        torch.from_numpy(states[:, ANCHOR_INDEX, 1:, 1]),
        torch.from_numpy(plans[:, -1, :, 1]),
        torch.from_numpy(maps), torch.from_numpy(map_valid),
    )
    planned = intent.numpy() & valid[:, ANCHOR_INDEX, 1:]
    selected = np.flatnonzero(planned.any(axis=1))
    theta = _theta_for_rows(test_rows[selected], DEFAULT_MA_IDM_POSTERIOR)
    model, _ = load_checkpoint(Path(config["paths"]["evaluation_checkpoint"]), device=device)
    model.eval()
    records = []
    planned_lane_outcomes = []
    raw_overlap_scenes = 0
    npc_npc_overlap_scenes = 0
    for begin in range(0, len(selected), 64):
        end = min(begin + 64, len(selected))
        part = selected[begin:end]
        ads_policy = (
            None if args.ads_acceleration_mps2 is None else
            AbsoluteAccelerationPolicy(
                _logged_ego_actions(states[part], valid[part]),
                args.ads_acceleration_mps2,
            )
        )
        result = rollout(
            model, states[part], valid[part], plans[part], maps[part], map_valid[part],
            device=device, history_frames=25, motion_seed=None,
            ads_policy=ads_policy,
            controller=OnlineMAIDMController(theta[begin:end]).to(device),
            record_policy_trace=False,
        )
        spatial = np.asarray(result.controller_diagnostics["spatial_path_active"], bool)
        events = collision_events(
            np.concatenate((states[part, ANCHOR_INDEX:ANCHOR_INDEX + 1], result.states), axis=1),
            valid[part, ANCHOR_INDEX, 1:],
            ads_acceleration_setpoint_mps2=args.ads_acceleration_mps2,
        )
        raw_overlap_scenes += int(events["raw"].sum())
        npc_npc_overlap_scenes += int(events["npc_npc"].sum())
        for local, slot in np.argwhere(planned[part]):
            terminal = result.states[local, -1, slot + 1]
            logged = states[part[local], ANCHOR_INDEX + result.states.shape[1], slot + 1]
            target = float(target_y[part[local], slot])
            online_error = float(terminal[1] - target)
            logged_error = float(logged[1] - target)
            online_heading = float(np.arctan2(terminal[3], terminal[2]))
            logged_heading = float(np.arctan2(logged[3], logged[2]))
            online_completed = abs(online_error) < 0.75 and abs(online_heading) < 0.05
            logged_completed = abs(logged_error) < 0.75 and abs(logged_heading) < 0.05
            planned_lane_outcomes.append({
                "test_row": int(test_rows[part[local]]),
                "npc_slot": int(slot + 1),
                "online_terminal_error_m": online_error,
                "logged_terminal_error_m": logged_error,
                "online_terminal_heading_rad": online_heading,
                "logged_terminal_heading_rad": logged_heading,
                "online_completed": online_completed,
                "logged_completed": logged_completed,
                "spatial_path_active_frames": int(spatial[local, :, slot].sum()),
            })
        for local, slot in np.argwhere(spatial.any(axis=1)):
            terminal = result.states[local, -1, slot + 1]
            error = float(terminal[1] - target_y[part[local], slot])
            heading = float(np.arctan2(terminal[3], terminal[2]))
            last = result.states[local, -1]
            target_neighbours = [
                {
                    "slot": int(other),
                    "relative_x_m": float(last[other, 0] - terminal[0]),
                    "lateral_offset_to_target_m": float(last[other, 1] - target_y[part[local], slot]),
                }
                for other in range(7)
                if other != slot + 1
                and valid[part[local], ANCHOR_INDEX, other]
                and abs(last[other, 1] - target_y[part[local], slot]) < 1.8
                and -12.0 < last[other, 0] - terminal[0] < 12.0
            ]
            records.append({
                "test_row": int(test_rows[part[local]]),
                "npc_slot": int(slot + 1),
                "spatial_path_active_frames": int(spatial[local, :, slot].sum()),
                "planned_lane_change": bool(planned[part[local], slot]),
                "target_lane_y_m": float(target_y[part[local], slot]),
                "terminal_error_m": error,
                "terminal_heading_rad": heading,
                "terminal_target_lane_neighbours": target_neighbours,
                "npc_pair_overlap": bool(
                    events["raw_pairs"][local, slot + 1].any()
                    or events["raw_pairs"][local, :, slot + 1].any()
                ),
                "completed": abs(error) < 0.75 and abs(heading) < 0.05,
            })
    report = {
        "schema": "online_spatial_path_factual_audit_v1",
        "test_split_size": len(test_rows),
        "planned_lane_change_count": int(planned.sum()),
        "spatial_path_npc_count": len(records),
        "spatial_path_npcs": records,
        "raw_overlap_scenes": raw_overlap_scenes,
        "npc_npc_overlap_scenes": npc_npc_overlap_scenes,
        "planned_lane_online_completed_count": sum(
            item["online_completed"] for item in planned_lane_outcomes
        ),
        "planned_lane_logged_completed_count": sum(
            item["logged_completed"] for item in planned_lane_outcomes
        ),
        "planned_lane_online_misses": [
            item for item in planned_lane_outcomes if not item["online_completed"]
        ],
        "planned_lane_logged_misses": [
            item for item in planned_lane_outcomes if not item["logged_completed"]
        ],
        "ads_acceleration_setpoint_mps2": args.ads_acceleration_mps2,
        "online_controller_sha256": controller_hash,
        "online_runtime_sha256": online_runtime_sha256(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, args.output)
    print(report)


if __name__ == "__main__":
    main()
