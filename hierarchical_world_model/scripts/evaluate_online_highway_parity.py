#!/usr/bin/env python3
"""Check that the released HighwayEnv plant agrees with highD offline evaluation."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import ANCHOR_INDEX  # noqa: E402
from hierarchical_world_model.scripts.evaluate_online_factual import _theta_for_rows  # noqa: E402
from hierarchical_world_model.src.ads_interventions import AbsoluteAccelerationPolicy  # noqa: E402
from hierarchical_world_model.src.data import ego_controls, prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.evaluation import _logged_ego_actions, rollout  # noqa: E402
from hierarchical_world_model.src.highway import HighwayEnvClosedLoopWorld  # noqa: E402
from hierarchical_world_model.src.planner import frozen_diffusion_plans  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from hierarchical_world_model.src.provenance import online_runtime_sha256  # noqa: E402
from hierarchical_world_model.src.randomness import WorldExogenousState  # noqa: E402
from hierarchical_world_model.src.stochastic_drivers.online import (  # noqa: E402
    DEFAULT_MA_IDM_POSTERIOR, OnlineMAIDMController,
)
from hierarchical_world_model.src.train import load_checkpoint  # noqa: E402
from traffic_components.src.core.utils import file_sha256, save_json, select_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--include-test-row", type=int, default=14241)
    parser.add_argument("--ads-acceleration-mps2", type=float)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/hierarchical_world_model/evaluation/highway_parity.json",
    )
    args = parser.parse_args()
    if args.rows < 1:
        raise ValueError("--rows must be positive")
    controller_hash = file_sha256(
        ROOT / "hierarchical_world_model/src/stochastic_drivers/online.py"
    )
    config = load_protocol_config(ROOT / "hierarchical_world_model/config/world_model.yaml")
    experiment = prepare_experiment_data(config, ROOT)
    test_rows = np.asarray(experiment.test_rows, np.int64)
    if args.rows > len(test_rows):
        raise ValueError("--rows exceeds the test split")
    include = np.flatnonzero(test_rows == args.include_test_row)
    if len(include) != 1:
        raise ValueError("--include-test-row is not in the test split")
    indices = np.arange(args.rows)
    if int(include[0]) not in indices:
        indices[-1] = int(include[0])
    rows = test_rows[indices]
    arrays = experiment.bundle.arrays
    states = np.asarray(arrays["agent_states"][rows], np.float32)
    valid = np.asarray(arrays["agent_valid"][rows], bool)
    maps = np.asarray(arrays["map_polylines"][rows], np.float32)
    map_valid = np.asarray(arrays["map_polyline_valid"][rows], bool)
    device = select_device("cuda")
    plans = frozen_diffusion_plans(
        experiment.bundle, test_rows, checkpoint=config["paths"]["diffusion_checkpoint"],
        output_dir=ROOT / "results/hierarchical_world_model/evaluation/plans_full",
        device=device, batch_size=64, ddim_steps=20, experiment_scope="full",
    )[indices]
    theta = _theta_for_rows(rows, DEFAULT_MA_IDM_POSTERIOR)
    model, _ = load_checkpoint(Path(config["paths"]["evaluation_checkpoint"]), device=device)
    model.eval()
    controls = _logged_ego_actions(states, valid)
    ads_policy = (
        None if args.ads_acceleration_mps2 is None else
        AbsoluteAccelerationPolicy(controls, args.ads_acceleration_mps2)
    )
    offline = rollout(
        model, states, valid, plans, maps, map_valid,
        device=device, history_frames=25, motion_seed=None,
        ads_policy=ads_policy,
        controller=OnlineMAIDMController(theta).to(device),
        record_policy_trace=False,
    )
    history = states[:, ANCHOR_INDEX - 24:ANCHOR_INDEX + 1]
    history_valid = valid[:, ANCHOR_INDEX - 24:ANCHOR_INDEX + 1]
    historical_ego = ego_controls(
        states[:, ANCHOR_INDEX - 24:ANCHOR_INDEX, 0],
        states[:, ANCHOR_INDEX - 23:ANCHOR_INDEX + 1, 0], 0.04,
    )
    historical_ego[~(
        valid[:, ANCHOR_INDEX - 24:ANCHOR_INDEX, 0]
        & valid[:, ANCHOR_INDEX - 23:ANCHOR_INDEX + 1, 0]
    )] = 0.0
    random_state = WorldExogenousState.sample(
        len(rows), seed=31, response_steps=149,
        scene_refresh_responses=model.cfg.scene_refresh_responses,
        scene_dim=model.cfg.scene_latent_dim, agent_dim=model.cfg.agent_latent_dim,
    )
    random_state = replace(
        random_state,
        scene_innovations=np.zeros_like(random_state.scene_innovations),
        agent_response_innovations=np.zeros_like(random_state.agent_response_innovations),
        policy_response_innovations=np.zeros_like(random_state.policy_response_innovations),
        policy_calibration_innovations=np.zeros_like(random_state.policy_calibration_innovations),
    )
    world = HighwayEnvClosedLoopWorld(
        model, device=device, controller=OnlineMAIDMController(theta).to(device)
    )
    initial = world.reset(
        torch.from_numpy(states[:, ANCHOR_INDEX]),
        torch.from_numpy(valid[:, ANCHOR_INDEX]),
        torch.from_numpy(plans), torch.from_numpy(maps), torch.from_numpy(map_valid),
        exogenous_state=random_state,
        initial_history=torch.from_numpy(history),
        initial_history_valid=torch.from_numpy(history_valid),
        committed_ego_controls=torch.from_numpy(historical_ego),
        deterministic_response=True,
    )
    initial_difference = np.linalg.norm(
        initial["agent_states"].cpu().numpy()[:, :, :2] - states[:, ANCHOR_INDEX, :, :2],
        axis=-1,
    )
    realized = []
    spatial_activity = []
    collision = np.zeros(len(rows), bool)
    offroad = np.zeros(len(rows), bool)
    for frame in range(149):
        action = (
            torch.from_numpy(controls[:, frame])
            if ads_policy is None else ads_policy(world.observe()).cpu()
        )
        step = world.advance_response(action[:, None])
        realized.append(step["agent_states"].cpu().numpy())
        spatial_activity.append(step["controller_spatial_path_active"].cpu().numpy())
        collision |= step["collision"].cpu().numpy()
        offroad |= step["offroad"].cpu().numpy().any(axis=1)
    highway = np.stack(realized, axis=1)
    highway_spatial = np.stack(spatial_activity, axis=1).astype(bool)
    offline_spatial = np.asarray(
        offline.controller_diagnostics["spatial_path_active"], bool
    )
    mask = np.broadcast_to(valid[:, ANCHOR_INDEX, 1:][:, None], highway[:, :, 1:, :2].shape[:-1])
    plant_distance = np.linalg.norm(highway[..., 1:, :2] - offline.states[..., 1:, :2], axis=-1)
    factual_distance = np.linalg.norm(highway[..., 1:, :2] - states[:, ANCHOR_INDEX + 1:174, 1:, :2], axis=-1)
    controller_active = np.asarray(offline.controller_diagnostics["active"], bool)
    report = {
        "schema": "online_highway_offline_parity_v1",
        "test_split_size": int(len(test_rows)),
        "evaluated_rows": int(len(rows)),
        "included_test_row": args.include_test_row,
        "ads_acceleration_setpoint_mps2": args.ads_acceleration_mps2,
        "lane_control_active_scene_count": int(
            np.asarray(offline.controller_diagnostics["policy_active"], bool).any(axis=(1, 2)).sum()
        ),
        "spatial_path_active_frame_count_offline": int(offline_spatial.sum()),
        "spatial_path_active_frame_count_highway": int(highway_spatial.sum()),
        "spatial_path_activity_agreement_rate": float(
            (offline_spatial == highway_spatial).mean()
        ),
        "offline_comparison_only_not_full_test": True,
        "online_controller_sha256": controller_hash,
        "online_runtime_sha256": online_runtime_sha256(),
        "initial_position_max_error_m": float(initial_difference.max()),
        "offline_to_highway_ADE_m": float(plant_distance[mask].mean()),
        "offline_to_highway_P95_m": float(np.quantile(plant_distance[mask], 0.95)),
        "highway_factual_ADE_m": float(factual_distance[mask].mean()),
        "per_scene": [
            {
                "test_row": int(row),
                "offline_to_highway_ADE_m": float(plant_distance[index][mask[index]].mean()),
                "highway_factual_ADE_m": float(factual_distance[index][mask[index]].mean()),
                "controller_active_frames": int(controller_active[index].sum()),
                "spatial_path_active_frames": int(offline_spatial[index].sum()),
            }
            for index, row in enumerate(rows)
        ],
        "highway_collision_scene_count": int(collision.sum()),
        "highway_offroad_scene_count": int(offroad.sum()),
    }
    save_json(report, args.output)
    print(report)


if __name__ == "__main__":
    main()
