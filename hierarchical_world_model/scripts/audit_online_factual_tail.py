#!/usr/bin/env python3
"""Attribute the worst held-out factual errors to plan, HiQR, or NPC response."""

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
from hierarchical_world_model.scripts.evaluate_online_factual import _theta_for_rows  # noqa: E402
from hierarchical_world_model.src.collision_attribution import collision_events  # noqa: E402
from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.evaluation import rollout  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from hierarchical_world_model.src.provenance import online_runtime_sha256  # noqa: E402
from hierarchical_world_model.src.stochastic_drivers.online import (  # noqa: E402
    DEFAULT_MA_IDM_POSTERIOR,
    OnlineMAIDMController,
    nearest_leader_observation,
    planned_adjacent_lane_intent,
)
from hierarchical_world_model.src.train import load_checkpoint  # noqa: E402
from traffic_components.src.core.utils import file_sha256, load_json, save_json, select_device  # noqa: E402


def main() -> None:
    directory = ROOT / "results/hierarchical_world_model/evaluation"
    parser = argparse.ArgumentParser()
    parser.add_argument("--factual-report", type=Path, default=directory / "factual_test.json")
    parser.add_argument("--rows", type=int, nargs="+")
    parser.add_argument("--output", type=Path, default=directory / "factual_tail_audit.json")
    args = parser.parse_args()
    factual = None if args.rows else load_json(args.factual_report)
    cache = directory / "plans_full/frozen_diffusion_test_plans.npz"
    manifest = load_json(directory / "plans_full/frozen_diffusion_test_plans.json")
    config = load_protocol_config(ROOT / "hierarchical_world_model/config/world_model.yaml")
    experiment = prepare_experiment_data(config, ROOT)
    test_rows = np.asarray(experiment.test_rows, np.int64)
    with np.load(cache) as saved:
        cached_rows = np.asarray(saved["row_index"], np.int64)
        if not np.array_equal(cached_rows, test_rows):
            raise ValueError("frozen plans cache does not match the complete test split")
        wanted = np.asarray(
            args.rows if args.rows else [item["test_row"] for item in factual["worst_scene_ADE"]],
            np.int64,
        )
        row_to_position = {int(row): index for index, row in enumerate(cached_rows)}
        plans = np.asarray(saved["positions"][[row_to_position[int(row)] for row in wanted]], np.float32)
    if manifest["sequences"] != len(test_rows):
        raise ValueError("frozen plan manifest has the wrong test size")
    controller_hash = file_sha256(
        ROOT / "hierarchical_world_model/src/stochastic_drivers/online.py"
    )
    if factual is not None and factual["online_controller_sha256"] != controller_hash:
        raise ValueError("factual report and controller source disagree")
    if factual is not None and factual["online_runtime_sha256"] != online_runtime_sha256():
        raise ValueError("factual report and runtime source disagree")
    arrays = experiment.bundle.arrays
    states = np.asarray(arrays["agent_states"][wanted], np.float32)
    valid = np.asarray(arrays["agent_valid"][wanted], bool)
    maps = np.asarray(arrays["map_polylines"][wanted], np.float32)
    map_valid = np.asarray(arrays["map_polyline_valid"][wanted], bool)
    sequence_ids = np.asarray(arrays["sequence_id"])[wanted]
    device = select_device("cuda")
    model, _ = load_checkpoint(Path(config["paths"]["evaluation_checkpoint"]), device=device)
    model.eval()
    theta = _theta_for_rows(wanted, DEFAULT_MA_IDM_POSTERIOR)
    hiqr = rollout(
        model, states, valid, plans, maps, map_valid,
        device=device, history_frames=25, motion_seed=None,
        controller=None, record_policy_trace=False,
    )
    online = rollout(
        model, states, valid, plans, maps, map_valid,
        device=device, history_frames=25, motion_seed=None,
        controller=OnlineMAIDMController(theta).to(device),
        record_policy_trace=False,
    )
    target = states[:, ANCHOR_INDEX + 1:174, 1:, :2]
    active = valid[:, ANCHOR_INDEX, 1:]
    trace = online.controller_diagnostics or {}
    delta = np.asarray(trace.get("delta_ax", np.zeros((len(wanted), 149, 6))), np.float32)
    lane_repair = np.asarray(trace.get("policy_active", np.zeros_like(delta)), bool)
    spatial_path = np.asarray(trace.get("spatial_path_active", np.zeros_like(delta)), bool)
    records = []
    for scene, row in enumerate(wanted):
        observed = np.concatenate(
            (states[scene:scene + 1, ANCHOR_INDEX:ANCHOR_INDEX + 1], online.states[scene:scene + 1, :-1]),
            axis=1,
        )[0]
        gap, follower_speed, leader_speed, leader_index = nearest_leader_observation(
            torch.from_numpy(observed),
            torch.from_numpy(np.broadcast_to(valid[scene, ANCHOR_INDEX], (149, 7)).copy()),
            prediction_horizon_s=2.0,
        )
        hiqr_overlap = collision_events(
            np.concatenate((states[scene:scene + 1, ANCHOR_INDEX:ANCHOR_INDEX + 1], hiqr.states[scene:scene + 1]), axis=1),
            active[scene:scene + 1],
        )["raw"][0]
        online_overlap = collision_events(
            np.concatenate((states[scene:scene + 1, ANCHOR_INDEX:ANCHOR_INDEX + 1], online.states[scene:scene + 1]), axis=1),
            active[scene:scene + 1],
        )["raw"][0]
        lane_intent, _, target_lane_y = planned_adjacent_lane_intent(
            torch.from_numpy(states[scene:scene + 1, ANCHOR_INDEX, 1:, 1]),
            torch.from_numpy(plans[scene:scene + 1, -1, :, 1]),
            torch.from_numpy(maps[scene:scene + 1]),
            torch.from_numpy(map_valid[scene:scene + 1]),
        )
        agents = []
        for slot in np.flatnonzero(active[scene]):
            reference = target[scene, :, slot]
            errors = {
                "plan": np.linalg.norm(plans[scene, :, slot] - reference, axis=-1),
                "hiqr": np.linalg.norm(hiqr.states[scene, :, slot + 1, :2] - reference, axis=-1),
                "online": np.linalg.norm(online.states[scene, :, slot + 1, :2] - reference, axis=-1),
            }
            worst_frame = int(np.argmax(errors["online"]))
            correction_frames = np.flatnonzero(np.abs(delta[scene, :, slot]) > 1.0e-4)
            first_correction = None
            if len(correction_frames):
                first = int(correction_frames[0])
                leader = int(leader_index[first, slot])
                times = np.arange(1, 51, dtype=np.float32) * 0.04
                future_indices = np.minimum(first + np.arange(1, 51), 148)
                lateral_future = plans[scene, future_indices, slot, 1] - (
                    observed[first, leader, 1] + observed[first, leader, 3] * times
                )
                gap_future = (
                    float(gap[first, slot])
                    + (float(leader_speed[first, slot]) - float(follower_speed[first, slot])) * times
                    + 0.5 * (
                        float(observed[first, leader, 4])
                        - float(hiqr.background_actions[scene, first, slot, 0])
                    ) * times * times
                )
                clearance = max(2.0, 0.1 * float(follower_speed[first, slot]))
                joint_risk = (gap_future < clearance) & (np.abs(lateral_future) < 1.8)
                other = observed[first]
                delta_x = other[:, 0] - observed[first, slot + 1, 0]
                future_delta_x = delta_x + 2.0 * (
                    other[:, 2] - observed[first, slot + 1, 2]
                )
                target_y = float(target_lane_y[0, slot])
                target_occupied = (
                    valid[scene, ANCHOR_INDEX]
                    & (np.arange(7) != slot + 1)
                    & (np.abs(other[:, 1] - target_y) < 1.8)
                    & (np.minimum(delta_x, future_delta_x) < 6.8)
                    & (np.maximum(delta_x, future_delta_x) > -8.0)
                )
                first_correction = {
                    "frame": first,
                    "leader_slot": leader,
                    "gap_m": float(gap[first, slot]),
                    "follower_speed_mps": float(follower_speed[first, slot]),
                    "leader_speed_mps": float(leader_speed[first, slot]),
                    "leader_lateral_offset_m": float(
                        observed[first, leader, 1] - observed[first, slot + 1, 1]
                    ),
                    "planned_lateral_offset_at_2s_m": float(
                        plans[scene, min(first + 50, 148), slot, 1]
                        - (observed[first, leader, 1] + 2.0 * observed[first, leader, 3])
                    ),
                    "planned_lane_intent": bool(lane_intent[0, slot]),
                    "on_plan_longitudinal_error_m": float(
                        observed[first, slot + 1, 0] - plans[scene, first, slot, 0]
                    ),
                    "target_lane_occupied_slots": np.flatnonzero(target_occupied).tolist(),
                    "target_lane_occupied_detail": [
                        {
                            "slot": int(neighbour),
                            "relative_x_m": float(delta_x[neighbour]),
                            "relative_x_at_2s_m": float(future_delta_x[neighbour]),
                            "lateral_offset_to_target_m": float(other[neighbour, 1] - target_y),
                            "lateral_velocity_mps": float(other[neighbour, 3]),
                        }
                        for neighbour in np.flatnonzero(target_occupied)
                    ],
                    "joint_gap_and_lateral_risk_frames": int(joint_risk.sum()),
                    "first_joint_risk_frame": (
                        None if not joint_risk.any() else first + int(np.flatnonzero(joint_risk)[0]) + 1
                    ),
                    "leader_ax_mps2": float(observed[first, leader, 4]),
                    "leader_vy_mps": float(observed[first, leader, 3]),
                    "follower_vy_mps": float(observed[first, slot + 1, 3]),
                    "hiqr_action_ax_mps2": float(hiqr.background_actions[scene, first, slot, 0]),
                    "online_action_ax_mps2": float(online.background_actions[scene, first, slot, 0]),
                }
            samples = []
            for frame in (0, 25, 50, 75, 100, 125, 148):
                samples.append({
                    "frame": frame,
                    "target_x_m": float(reference[frame, 0]),
                    "plan_x_m": float(plans[scene, frame, slot, 0]),
                    "hiqr_x_m": float(hiqr.states[scene, frame, slot + 1, 0]),
                    "online_x_m": float(online.states[scene, frame, slot + 1, 0]),
                    "target_vx_mps": float(states[scene, ANCHOR_INDEX + frame + 1, slot + 1, 2]),
                    "hiqr_vx_mps": float(hiqr.states[scene, frame, slot + 1, 2]),
                    "online_vx_mps": float(online.states[scene, frame, slot + 1, 2]),
                    "target_ax_mps2": float(states[scene, ANCHOR_INDEX + frame + 1, slot + 1, 4]),
                    "hiqr_ax_mps2": float(hiqr.background_actions[scene, frame, slot, 0]),
                    "online_ax_mps2": float(online.background_actions[scene, frame, slot, 0]),
                    "target_y_m": float(reference[frame, 1]),
                    "plan_y_m": float(plans[scene, frame, slot, 1]),
                    "hiqr_y_m": float(hiqr.states[scene, frame, slot + 1, 1]),
                    "online_y_m": float(online.states[scene, frame, slot + 1, 1]),
                    "online_yaw_rate_rps": float(online.background_actions[scene, frame, slot, 1]),
                    "lane_repair_active": bool(lane_repair[scene, frame, slot]),
                    "spatial_path_active": bool(spatial_path[scene, frame, slot]),
                })
            agents.append({
                "npc_slot": int(slot + 1),
                "plan_ADE_m": float(errors["plan"].mean()),
                "hiqr_ADE_m": float(errors["hiqr"].mean()),
                "online_ADE_m": float(errors["online"].mean()),
                "online_max_error_m": float(errors["online"][worst_frame]),
                "online_worst_frame": worst_frame,
                "target_xy_at_worst_m": reference[worst_frame].tolist(),
                "plan_xy_at_worst_m": plans[scene, worst_frame, slot].tolist(),
                "hiqr_xy_at_worst_m": hiqr.states[scene, worst_frame, slot + 1, :2].tolist(),
                "online_xy_at_worst_m": online.states[scene, worst_frame, slot + 1, :2].tolist(),
                "online_correction_frames": int((np.abs(delta[scene, :, slot]) > 1.0e-4).sum()),
                "first_correction": first_correction,
                "lane_repair_frames": int(lane_repair[scene, :, slot].sum()),
                "spatial_path_frames": int(spatial_path[scene, :, slot].sum()),
                "longitudinal_samples": samples,
            })
        records.append({
            "test_row": int(row),
            "sequence_id": str(sequence_ids[scene]),
            "hiqr_only_overlap": bool(hiqr_overlap),
            "online_overlap": bool(online_overlap),
            "agents": sorted(agents, key=lambda item: item["online_ADE_m"], reverse=True),
        })
    output = {
        "schema": "online_factual_tail_attribution_v1",
        "test_split_size": int(len(test_rows)),
        "audited_rows": wanted.tolist(),
        "audit_rows_selected_explicitly": bool(args.rows),
        "source_factual_report": None if factual is None else str(args.factual_report),
        "source_factual_controller_sha256": controller_hash,
        "source_factual_runtime_sha256": online_runtime_sha256(),
        "online_controller_sha256": controller_hash,
        "online_runtime_sha256": online_runtime_sha256(),
        "frozen_diffusion_plan_manifest": str(directory / "plans_full/frozen_diffusion_test_plans.json"),
        "scenes": records,
    }
    save_json(output, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
