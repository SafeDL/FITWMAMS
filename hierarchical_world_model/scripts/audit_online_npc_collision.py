#!/usr/bin/env python3
"""Locate geometric overlaps in a screened ADS intervention cohort."""

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
from hierarchical_world_model.scripts.evaluate_online_ads import _screen  # noqa: E402
from hierarchical_world_model.scripts.evaluate_online_factual import _theta_for_rows  # noqa: E402
from hierarchical_world_model.src.ads_interventions import AbsoluteAccelerationPolicy  # noqa: E402
from hierarchical_world_model.src.collision_attribution import collision_events  # noqa: E402
from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.evaluation import _logged_ego_actions, rollout  # noqa: E402
from hierarchical_world_model.src.planner import frozen_diffusion_plans  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from hierarchical_world_model.src.stochastic_drivers.online import (  # noqa: E402
    DEFAULT_MA_IDM_POSTERIOR, OnlineMAIDMController, nearest_leader_observation,
)
from hierarchical_world_model.src.train import load_checkpoint  # noqa: E402
from traffic_components.src.core.utils import select_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", type=int)
    parser.add_argument("--dose", type=float, default=-8.0)
    parser.add_argument("--kind", choices=("npc_npc", "ego_npc", "raw"), default="npc_npc")
    args = parser.parse_args()
    config = load_protocol_config(ROOT / "hierarchical_world_model/config/world_model.yaml")
    experiment = prepare_experiment_data(config, ROOT)
    test_rows = np.asarray(experiment.test_rows, np.int64)
    arrays = experiment.bundle.arrays
    all_states = np.asarray(arrays["agent_states"][test_rows], np.float32)
    all_valid = np.asarray(arrays["agent_valid"][test_rows], bool)
    selected, _ = _screen(all_states, all_valid, "same")
    positions = np.flatnonzero(selected)
    if args.row is not None:
        positions = positions[test_rows[positions] == args.row]
        if len(positions) != 1:
            raise ValueError("row is not a unique eligible test scene")
    rows = test_rows[positions]
    states, valid = all_states[positions], all_valid[positions]
    maps = np.asarray(arrays["map_polylines"][rows], np.float32)
    map_valid = np.asarray(arrays["map_polyline_valid"][rows], bool)
    device = select_device("cuda")
    full_plans = frozen_diffusion_plans(
        experiment.bundle, test_rows, checkpoint=config["paths"]["diffusion_checkpoint"],
        output_dir=ROOT / "results/hierarchical_world_model/evaluation/plans_full",
        device=device, batch_size=64, ddim_steps=20, experiment_scope="full",
    )
    plans = full_plans[positions]
    theta = _theta_for_rows(test_rows, DEFAULT_MA_IDM_POSTERIOR)[positions]
    model, _ = load_checkpoint(Path(config["paths"]["evaluation_checkpoint"]), device=device)
    model.eval()
    for begin in range(0, len(rows), 64):
        end = min(begin + 64, len(rows))
        part = slice(begin, end)
        policy = AbsoluteAccelerationPolicy(
            _logged_ego_actions(states[part], valid[part]), args.dose
        )
        result = rollout(
            model, states[part], valid[part], plans[part], maps[part], map_valid[part],
            device=device, history_frames=25, motion_seed=None, ads_policy=policy,
            controller=OnlineMAIDMController(theta[part]).to(device),
            record_policy_trace=False,
        )
        active = valid[part, ANCHOR_INDEX, 1:]
        trajectory = np.concatenate(
            (states[part, ANCHOR_INDEX:ANCHOR_INDEX + 1], result.states), axis=1
        )
        events = collision_events(trajectory, active)
        for local in np.flatnonzero(events[args.kind]):
            pair = np.argwhere(events["raw_pairs"][local])
            if args.kind == "npc_npc":
                pair = pair[(pair[:, 0] > 0) & (pair[:, 1] > 0)]
            elif args.kind == "ego_npc":
                pair = pair[(pair[:, 0] == 0) & (pair[:, 1] > 0)]
            repairs = np.asarray(result.controller_diagnostics["policy_active"][local], bool)
            first_pair = pair[0]
            overlap = (
                (np.abs(trajectory[local, :, first_pair[0], 0] - trajectory[local, :, first_pair[1], 0]) < 4.8)
                & (np.abs(trajectory[local, :, first_pair[0], 1] - trajectory[local, :, first_pair[1], 1]) < 1.8)
            )
            collision_frame = int(np.flatnonzero(overlap)[0])
            leader_gap, _, _, leader_slot = nearest_leader_observation(
                torch.as_tensor(trajectory[local]),
                torch.as_tensor(
                    np.broadcast_to(valid[begin + local, ANCHOR_INDEX],
                                    (trajectory.shape[1], 7)).copy()
                ),
                prediction_horizon_s=2.0,
            )
            print({
                "test_row": int(rows[begin + local]),
                "dose_mps2": args.dose,
                "collision_kind": args.kind,
                "cohort_position": begin + int(local),
                "pairs": pair.tolist(),
                "repaired_npc_slots": np.flatnonzero(repairs.any(axis=0)).tolist(),
                "repair_frame_count": int(repairs.sum()),
                "first_repair_frame": (
                    int(np.flatnonzero(repairs.any(axis=1))[0]) if repairs.any() else None
                ),
                "first_overlap_frame": collision_frame,
                "npc_states_at_overlap": trajectory[local, collision_frame, first_pair].tolist(),
                "npc_start_y": states[begin + local, ANCHOR_INDEX, 1:, 1].tolist(),
                "plan_terminal_y": plans[begin + local, -1, :, 1].tolist(),
                "agents_at_frames": {
                    frame: trajectory[local, frame, :, :5].tolist()
                    for frame in (25, 50, 66)
                } if args.row is not None else None,
                "pair_history": [
                    {
                        "frame": frame,
                        "ego": trajectory[local, frame, 0, :4].tolist(),
                        "other": trajectory[local, frame, first_pair[1], :4].tolist(),
                        "other_ax": float(result.background_actions[local, min(frame, 148), first_pair[1] - 1, 0]),
                        "other_base_ax": float(result.base_background_actions[local, min(frame, 148), first_pair[1] - 1, 0]),
                        "other_leader_slot": int(leader_slot[frame, first_pair[1] - 1]),
                        "other_leader_gap_m": float(leader_gap[frame, first_pair[1] - 1]),
                        "ego_ax": float(trajectory[local, frame, 0, 4]),
                        "other_ax_state": float(trajectory[local, frame, first_pair[1], 4]),
                    }
                    for frame in (25, 50, 75, 100, 115, 125, collision_frame)
                ] if args.row is not None and first_pair[0] == 0 else None,
            })


if __name__ == "__main__":
    main()
