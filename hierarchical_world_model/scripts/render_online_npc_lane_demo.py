#!/usr/bin/env python3
"""Render map-grounded NPC lane-delay semantics in the existing traffic GIF style."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import ANCHOR_INDEX  # noqa: E402
from hierarchical_world_model.scripts.evaluate_online_factual import _theta_for_rows  # noqa: E402
from hierarchical_world_model.scripts.playback_helpers import _draw_world  # noqa: E402
from hierarchical_world_model.src.ads_interventions import AbsoluteAccelerationPolicy  # noqa: E402
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
from tools.plot_style import get_pyplot  # noqa: E402
from traffic_components.src.core.utils import file_sha256, save_json, select_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-row", type=int, default=14241)
    parser.add_argument("--npc-slot", type=int, default=1, help="NPC slot in [1, 6]")
    parser.add_argument("--blocker-slot", type=int, default=3, help="nearby car slot in [0, 6]")
    parser.add_argument("--ads-acceleration-mps2", type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/hierarchical_world_model/evaluation/npc_lane_demo.gif",
    )
    args = parser.parse_args()
    if not 1 <= args.npc_slot <= 6 or not 0 <= args.blocker_slot <= 6:
        raise ValueError("invalid NPC or blocker slot")
    controller_hash = file_sha256(
        ROOT / "hierarchical_world_model/src/stochastic_drivers/online.py"
    )
    config = load_protocol_config(ROOT / "hierarchical_world_model/config/world_model.yaml")
    experiment = prepare_experiment_data(config, ROOT)
    test_rows = np.asarray(experiment.test_rows, np.int64)
    found = np.flatnonzero(test_rows == args.test_row)
    if len(found) != 1:
        raise ValueError("--test-row is not in the highD test split")
    index = int(found[0])
    arrays = experiment.bundle.arrays
    states = np.asarray(arrays["agent_states"][[args.test_row]], np.float32)
    valid = np.asarray(arrays["agent_valid"][[args.test_row]], bool)
    maps = np.asarray(arrays["map_polylines"][[args.test_row]], np.float32)
    map_valid = np.asarray(arrays["map_polyline_valid"][[args.test_row]], bool)
    device = select_device(args.device)
    all_plans = frozen_diffusion_plans(
        experiment.bundle, test_rows, checkpoint=config["paths"]["diffusion_checkpoint"],
        output_dir=ROOT / "results/hierarchical_world_model/evaluation/plans_full",
        device=device, batch_size=64, ddim_steps=20, experiment_scope="full",
    )
    plan = all_plans[index:index + 1]
    lane_intent, source_y, target_y = planned_adjacent_lane_intent(
        torch.from_numpy(states[:, ANCHOR_INDEX, 1:, 1]),
        torch.from_numpy(plan[:, -1, :, 1]),
        torch.from_numpy(maps), torch.from_numpy(map_valid),
    )
    focus = args.npc_slot - 1
    if not bool(lane_intent[0, focus]):
        raise ValueError("selected NPC does not have a mapped adjacent-lane plan")
    theta = _theta_for_rows(np.asarray([args.test_row]), DEFAULT_MA_IDM_POSTERIOR)
    controller = OnlineMAIDMController(theta).to(device)
    model, _ = load_checkpoint(Path(config["paths"]["evaluation_checkpoint"]), device=device)
    model.eval()
    ads_policy = (
        None if args.ads_acceleration_mps2 is None else
        AbsoluteAccelerationPolicy(
            _logged_ego_actions(states, valid), args.ads_acceleration_mps2
        )
    )

    def execute(controller=None):
        return rollout(
            model, states, valid, plan, maps, map_valid,
            device=device, history_frames=25, motion_seed=None,
            ads_policy=ads_policy, controller=controller,
            record_policy_trace=False,
        )

    # The online world executes first; the same-ADS HiQR-only world is an
    # after-the-fact diagnostic and never provides its future to the first.
    online = execute(controller)
    hiqr = execute()
    anchor = states[:, ANCHOR_INDEX:ANCHOR_INDEX + 1]
    online_states = np.concatenate((anchor, online.states), axis=1)[0]
    hiqr_states = np.concatenate((anchor, hiqr.states), axis=1)[0]
    present = valid[0, ANCHOR_INDEX]
    active = np.asarray(online.controller_diagnostics["policy_active"][0, :, focus], bool)
    spatial = np.asarray(online.controller_diagnostics["spatial_path_active"][0, :, focus], bool)
    time = np.arange(150) * 0.04
    online_y = online_states[:, args.npc_slot, 1]
    hiqr_y = hiqr_states[:, args.npc_slot, 1]
    plan_y = np.concatenate(([states[0, ANCHOR_INDEX, args.npc_slot, 1]], plan[0, :, focus, 1]))
    online_gap = online_states[:, args.blocker_slot, 0] - online_states[:, args.npc_slot, 0]
    hiqr_gap = hiqr_states[:, args.blocker_slot, 0] - hiqr_states[:, args.npc_slot, 0]
    front_threshold = controller.vehicle_length_m + controller.minimum_clearance_m
    plt = get_pyplot()
    figure, axes = plt.subplots(2, 2, figsize=(12, 6.8), constrained_layout=True)
    frames = []
    for frame in range(0, 150, 3):
        _draw_world(
            axes[0, 0], states=online_states, valid=present, frame=frame,
            title=f"Online NPC response | t={time[frame]:.2f} s",
            focus_slot=focus, source_slot=(args.blocker_slot - 1 if args.blocker_slot else None),
            focus_role="lane-changing NPC", source_role="target-lane neighbour",
        )
        axes[0, 0].set_xlim(
            online_states[frame, args.npc_slot, 0] - 36.0,
            online_states[frame, args.npc_slot, 0] + 36.0,
        )
        _draw_world(
            axes[0, 1], states=hiqr_states, valid=present, frame=frame,
            title=f"Same ADS, HiQR-only diagnostic | t={time[frame]:.2f} s",
            focus_slot=focus, source_slot=(args.blocker_slot - 1 if args.blocker_slot else None),
            focus_role="lane-changing NPC", source_role="target-lane neighbour",
        )
        axes[0, 1].set_xlim(
            hiqr_states[frame, args.npc_slot, 0] - 36.0,
            hiqr_states[frame, args.npc_slot, 0] + 36.0,
        )
        chart = axes[1, 0]
        chart.clear()
        chart.plot(time, online_y, color="#7e22ce", label="online NPC y")
        chart.plot(time, hiqr_y, color="#8c8c8c", label="HiQR-only NPC y")
        chart.plot(time, plan_y, color="#1f78b4", alpha=0.65, label="Diffusion plan y")
        chart.axhline(float(source_y[0, focus]), color="#e69f00", linestyle=":", label="source lane")
        chart.axhline(float(target_y[0, focus]), color="#2ca02c", linestyle="--", label="target lane")
        chart.axvline(time[frame], color="black", linewidth=0.9)
        chart.set(xlim=(0, 5.96), xlabel="t [s]", ylabel="lateral position y [m]",
                  title="Map-grounded lane intention")
        chart.legend(loc="lower right", fontsize=6, frameon=False)
        gap_chart = axes[1, 1]
        gap_chart.clear()
        gap_chart.plot(time, online_gap, color="#7e22ce", label="online relative x")
        gap_chart.plot(time, hiqr_gap, color="#8c8c8c", label="HiQR-only relative x")
        gap_chart.axhline(front_threshold, color="#d62728", linestyle="--", label="front-gap threshold")
        gap_chart.fill_between(time[1:], 0, 1, where=active, color="#7e22ce",
                               alpha=0.13, transform=gap_chart.get_xaxis_transform(),
                               label="NPC lane control active")
        gap_chart.axvline(time[frame], color="black", linewidth=0.9)
        gap_chart.set(xlim=(0, 5.96), xlabel="t [s]", ylabel="neighbour relative x [m]",
                      title="Realized target-lane gap")
        gap_chart.legend(loc="lower right", fontsize=6, frameon=False)
        figure.canvas.draw()
        frames.append(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output, frames, duration=0.12, loop=0)
    plt.close(figure)
    branch_events = {}
    for label, trajectory in (("online", online_states), ("hiqr_only", hiqr_states)):
        events = collision_events(trajectory[None], present[None, 1:])
        branch_events[label] = {
            "raw_overlap": bool(events["raw"][0]),
            "npc_npc_overlap": bool(events["npc_npc"][0]),
        }
    save_json({
        "schema": "online_npc_mapped_lane_gif_v1",
        "test_row": args.test_row,
        "npc_slot": args.npc_slot,
        "target_lane_neighbour_slot": args.blocker_slot,
        "ads_acceleration_setpoint_mps2": args.ads_acceleration_mps2,
        "source_lane_y_m": float(source_y[0, focus]),
        "target_lane_y_m": float(target_y[0, focus]),
        "front_gap_threshold_m": front_threshold,
        "npc_lane_control_active_frames": int(active.sum()),
        "npc_spatial_path_active_frames": int(spatial.sum()),
        "online_final_target_lane_error_m": float(online_y[-1] - target_y[0, focus]),
        "hiqr_final_target_lane_error_m": float(hiqr_y[-1] - target_y[0, focus]),
        "branch_overlap": branch_events,
        "online_controller_sha256": controller_hash,
        "online_runtime_sha256": online_runtime_sha256(),
        "gif": args.output.name,
    }, args.output.with_name(f"{args.output.stem}_manifest.json"))
    print(args.output)


if __name__ == "__main__":
    main()
