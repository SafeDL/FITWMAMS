#!/usr/bin/env python3
"""Render one highD Test NPC-to-NPC perturbation in the project GIF style."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import ANCHOR_INDEX  # noqa: E402
from hierarchical_world_model.scripts.evaluate_online_factual import _theta_for_rows  # noqa: E402
from hierarchical_world_model.scripts.evaluate_online_npc_to_npc_highd import (  # noqa: E402
    FORCE, _ForcedNPCBrakeController, _screen,
)
from hierarchical_world_model.scripts.playback_helpers import _draw_world  # noqa: E402
from hierarchical_world_model.src.collision_attribution import collision_events  # noqa: E402
from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.evaluation import rollout  # noqa: E402
from hierarchical_world_model.src.planner import frozen_diffusion_plans  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from hierarchical_world_model.src.provenance import online_runtime_sha256  # noqa: E402
from hierarchical_world_model.src.stochastic_drivers.online import (  # noqa: E402
    DEFAULT_MA_IDM_POSTERIOR, OnlineMAIDMController,
)
from hierarchical_world_model.src.train import load_checkpoint  # noqa: E402
from tools.plot_style import get_pyplot  # noqa: E402
from traffic_components.src.core.utils import file_sha256, save_json, select_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-row", type=int, default=12989)
    parser.add_argument("--leader-brake-mps2", type=float, default=-8.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/hierarchical_world_model/evaluation/npc_to_npc_highd_demo.gif",
    )
    args = parser.parse_args()
    if args.leader_brake_mps2 not in (-8.0, -6.0, -4.0, -2.0):
        raise ValueError("leader brake must be one of the evaluated highD doses")
    config = load_protocol_config(ROOT / "hierarchical_world_model/config/world_model.yaml")
    experiment = prepare_experiment_data(config, ROOT)
    test_rows = np.asarray(experiment.test_rows, np.int64)
    found = np.flatnonzero(test_rows == args.test_row)
    if len(found) != 1:
        raise ValueError("--test-row is not in the highD Test split")
    index = int(found[0])
    arrays = experiment.bundle.arrays
    states = np.asarray(arrays["agent_states"][[args.test_row]], np.float32)
    valid = np.asarray(arrays["agent_valid"][[args.test_row]], bool)
    chosen, follower_slot, leader_slot = _screen(states, valid)
    if not bool(chosen[0]):
        raise ValueError("--test-row does not meet the NPC-following selection")
    follower, leader = int(follower_slot[0]), int(leader_slot[0])
    maps = np.asarray(arrays["map_polylines"][[args.test_row]], np.float32)
    map_valid = np.asarray(arrays["map_polyline_valid"][[args.test_row]], bool)
    device = select_device(args.device)
    plans = frozen_diffusion_plans(
        experiment.bundle, test_rows,
        checkpoint=config["paths"]["diffusion_checkpoint"],
        output_dir=ROOT / "results/hierarchical_world_model/evaluation/plans_full",
        device=device, batch_size=64, ddim_steps=20, experiment_scope="full",
    )[index:index + 1]
    theta = _theta_for_rows(np.asarray([args.test_row]), DEFAULT_MA_IDM_POSTERIOR)
    model, _ = load_checkpoint(Path(config["paths"]["evaluation_checkpoint"]), device=device)
    model.eval()

    def execute(controller):
        return rollout(
            model, states, valid, plans, maps, map_valid,
            device=device, history_frames=25, motion_seed=None,
            controller=controller.to(device), record_policy_trace=False,
        )

    # The baseline is computed afterward; it never supplies future states to
    # the perturbed online branch. Only the selected NPC leader is forced.
    brake_controller = _ForcedNPCBrakeController(
        theta, np.asarray([leader]), args.leader_brake_mps2,
    )
    brake = execute(brake_controller)
    baseline = execute(OnlineMAIDMController(theta))
    anchor = states[:, ANCHOR_INDEX:ANCHOR_INDEX + 1]
    brake_states = np.concatenate((anchor, brake.states), axis=1)[0]
    baseline_states = np.concatenate((anchor, baseline.states), axis=1)[0]
    present = valid[0, ANCHOR_INDEX]
    brake_events = collision_events(brake_states[None], present[None, 1:])
    baseline_events = collision_events(baseline_states[None], present[None, 1:])
    if brake_events["npc_npc"][0]:
        raise RuntimeError("chosen example has an NPC-to-NPC overlap")
    brake_actions = brake.background_actions[0, :, follower, 0]
    baseline_actions = baseline.background_actions[0, :, follower, 0]
    leader_actions = brake.background_actions[0, :, leader, 0]
    first_second = slice(FORCE.stop - 24, FORCE.stop)
    effect = float((brake_actions[first_second] - baseline_actions[first_second]).mean())
    if effect >= -0.05:
        raise RuntimeError("chosen example lacks a first-second brake response")
    time = np.arange(150) * 0.04
    brake_gap = brake_states[:, leader + 1, 0] - brake_states[:, follower + 1, 0] - 4.8
    baseline_gap = baseline_states[:, leader + 1, 0] - baseline_states[:, follower + 1, 0] - 4.8
    autonomous_lane_frames = int(np.asarray(
        (brake.controller_diagnostics or {}).get(
            "autonomous_lane_active", np.zeros((1, 149, 6), bool),
        ), bool,
    )[0, :, follower].sum())
    autonomous_target_y = (
        float(brake_controller._autonomous_lane_target_y[0, follower])
        if autonomous_lane_frames else None
    )
    plt = get_pyplot()
    figure, axes = plt.subplots(2, 2, figsize=(12, 6.8), constrained_layout=True)
    frames = []
    for frame in range(0, 150, 3):
        for axis, trajectory, title in (
            (axes[0, 0], brake_states, "NPC leader brakes | online world"),
            (axes[0, 1], baseline_states, "Logged ADS | no forced NPC brake"),
        ):
            _draw_world(
                axis, states=trajectory, valid=present, frame=frame,
                title=f"{title} | t={time[frame]:.2f} s",
                focus_slot=follower, source_slot=leader,
                focus_role="following NPC", source_role="leader NPC",
            )
            axis.set_xlim(
                trajectory[frame, follower + 1, 0] - 36,
                trajectory[frame, follower + 1, 0] + 72,
            )
        action_chart = axes[1, 0]
        action_chart.clear()
        action_chart.plot(time[1:], brake_actions, color="#7e22ce", label="follower: brake world")
        action_chart.plot(time[1:], baseline_actions, color="#8c8c8c", label="follower: baseline")
        action_chart.plot(time[1:], leader_actions, color="#e69f00", label="forced leader")
        action_chart.axvspan(time[FORCE.start + 1], time[FORCE.stop], color="#e69f00", alpha=0.13)
        action_chart.axvline(time[frame], color="black", linewidth=0.9)
        action_chart.set(
            xlim=(0, 5.96), ylim=(-8.5, 4.5), xlabel="t [s]",
            ylabel="acceleration [m/s²]", title="Executed actions; shaded: NPC leader brake",
        )
        action_chart.legend(loc="lower right", fontsize=6, frameon=False)
        detail_chart = axes[1, 1]
        detail_chart.clear()
        if autonomous_lane_frames:
            detail_chart.plot(time, brake_states[:, follower + 1, 1],
                              color="#7e22ce", label="brake world")
            detail_chart.plot(time, baseline_states[:, follower + 1, 1],
                              color="#8c8c8c", label="logged-ADS world")
            detail_chart.axhline(
                autonomous_target_y, color="#2ca02c", linestyle="--",
                label="mapped target centre",
            )
            detail_chart.set(xlim=(0, 5.96), xlabel="t [s]", ylabel="lateral y [m]",
                             title="Autonomous NPC lane decision and completion")
        else:
            detail_chart.plot(time, brake_gap, color="#7e22ce", label="brake-world gap")
            detail_chart.plot(time, baseline_gap, color="#8c8c8c", label="baseline gap")
            detail_chart.axhline(0, color="#d62728", linestyle="--", label="overlap boundary")
            detail_chart.set(xlim=(0, 5.96), xlabel="t [s]", ylabel="net gap [m]",
                             title="Realized NPC-to-NPC separation")
        detail_chart.axvline(time[frame], color="black", linewidth=0.9)
        detail_chart.legend(loc="lower left", fontsize=6, frameon=False)
        figure.canvas.draw()
        frames.append(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output, frames, duration=0.12, loop=0)
    plt.close(figure)
    save_json({
        "schema": "online_npc_to_npc_highd_gif_v1",
        "scope": "selected highD Test diagnostic; forced NPC leader action",
        "test_row": args.test_row,
        "follower_slot": follower + 1,
        "leader_slot": leader + 1,
        "leader_brake_setpoint_mps2": args.leader_brake_mps2,
        "forced_window_frames": [FORCE.start, FORCE.stop],
        "first_second_follower_effect_mps2": effect,
        "forced_leader_max_action_error_mps2": float(np.max(np.abs(
            leader_actions[FORCE] - args.leader_brake_mps2
        ))),
        "brake_raw_overlap": bool(brake_events["raw"][0]),
        "brake_npc_npc_overlap": bool(brake_events["npc_npc"][0]),
        "baseline_raw_overlap": bool(baseline_events["raw"][0]),
        "follower_autonomous_lane_active_frames": autonomous_lane_frames,
        "follower_autonomous_lane_target_y_m": autonomous_target_y,
        "follower_autonomous_lane_final_error_m": (
            None if autonomous_target_y is None else
            float(brake_states[-1, follower + 1, 1] - autonomous_target_y)
        ),
        "online_controller_sha256": file_sha256(
            ROOT / "hierarchical_world_model/src/stochastic_drivers/online.py"
        ),
        "online_runtime_sha256": online_runtime_sha256(),
        "gif": args.output.name,
    }, args.output.with_name(f"{args.output.stem}_manifest.json"))
    print(args.output)


if __name__ == "__main__":
    main()
