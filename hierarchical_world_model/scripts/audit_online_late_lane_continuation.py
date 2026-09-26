#!/usr/bin/env python3
"""Audit late autonomous lane changes past the fixed highD evaluation window.

The first 149 frames use the released frozen Diffusion plan unchanged. Beyond
that window the plan is explicitly extrapolated, so the tail is a continuation
diagnostic, not another held-out highD factual accuracy measurement.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import ANCHOR_INDEX  # noqa: E402
from hierarchical_world_model.scripts.evaluate_online_factual import _theta_for_rows  # noqa: E402
from hierarchical_world_model.src.data import ego_controls, prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.continuation import append_future_reference  # noqa: E402
from hierarchical_world_model.src.evaluation import _logged_ego_actions, rollout  # noqa: E402
from hierarchical_world_model.src.highway import HighwayEnvClosedLoopWorld  # noqa: E402
from hierarchical_world_model.scripts.playback_helpers import _draw_world  # noqa: E402
from hierarchical_world_model.src.planner import frozen_diffusion_plans  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from hierarchical_world_model.src.provenance import online_runtime_sha256  # noqa: E402
from hierarchical_world_model.src.randomness import WorldExogenousState  # noqa: E402
from hierarchical_world_model.src.stochastic_drivers.online import (  # noqa: E402
    DEFAULT_MA_IDM_POSTERIOR, OnlineMAIDMController,
)
from hierarchical_world_model.src.train import load_checkpoint  # noqa: E402
from tools.plot_style import get_pyplot  # noqa: E402
from traffic_components.src.core.utils import file_sha256, load_json, save_json, select_device  # noqa: E402


SOURCE = ROOT / "results/hierarchical_world_model/evaluation/npc_to_npc_highd_test.json"
OUTPUT = ROOT / "results/hierarchical_world_model/evaluation/late_autonomous_lane_continuation.json"
GIF = OUTPUT.with_name("late_autonomous_lane_continuation.gif")
TOTAL_FRAMES = 224
PLAN_FRAMES = 149


class ForcedNPCBrakeController(OnlineMAIDMController):
    """Vary the realized leading-NPC action, leaving all others online."""

    def __init__(self, theta: np.ndarray, leader_slots: np.ndarray, doses: np.ndarray):
        super().__init__(theta)
        self.leader_slots = torch.as_tensor(leader_slots, dtype=torch.long)
        self.doses = torch.as_tensor(doses, dtype=torch.float32)

    def forward(self, context, *, deterministic=False):
        output = super().forward(context, deterministic=deterministic)
        if not 25 <= context.response_index < 50:
            return output
        actions = output.actions.clone()
        row = torch.arange(len(actions), device=actions.device)
        actions[row, 0, self.leader_slots.to(actions.device), 0] = self.doses.to(actions.device)
        return replace(output, actions=actions)


def _tail_plan(plan: np.ndarray, frames: int) -> np.ndarray:
    """Continue terminal lane and longitudinal plan velocity, not logged future."""
    if plan.shape[1] != PLAN_FRAMES or frames <= 0:
        raise ValueError("expected a frozen 149-frame plan and a positive tail")
    delta_x = np.clip(plan[:, -1, :, 0] - plan[:, -2, :, 0], 0.0, 3.0)
    ticks = np.arange(1, frames + 1, dtype=np.float32)[None, :, None]
    tail = np.repeat(plan[:, -1:, :, :], frames, axis=1)
    tail[..., 0] += ticks * delta_x[:, None]
    return tail


def _render_example(
    trajectory: np.ndarray,
    actions: np.ndarray,
    present: np.ndarray,
    *,
    test_row: int,
    follower_slot: int,
    leader_slot: int,
    target_y: float,
    dose: float,
) -> None:
    """Use the project's existing road/vehicle drawing style for the tail."""
    time = np.arange(TOTAL_FRAMES + 1, dtype=np.float32) * 0.04
    follower = follower_slot - 1
    leader = leader_slot - 1
    plt = get_pyplot()
    figure, axes = plt.subplots(2, 2, figsize=(12, 6.8), constrained_layout=True)
    frames = []
    sample_frames = list(range(0, TOTAL_FRAMES + 1, 4))
    if sample_frames[-1] != TOTAL_FRAMES:
        sample_frames.append(TOTAL_FRAMES)
    gap = trajectory[:, leader + 1, 0] - trajectory[:, follower + 1, 0] - 4.8
    for frame in sample_frames:
        road = axes[0, 0]
        _draw_world(
            road, states=trajectory, valid=present, frame=frame,
            title=(f"Test row {test_row} | t={time[frame]:.2f} s | "
                   + ("frozen plan" if frame <= PLAN_FRAMES else "extrapolated tail")),
            focus_slot=follower, source_slot=leader,
            focus_role="following NPC", source_role="leader NPC",
        )
        road.set_xlim(
            trajectory[frame, follower + 1, 0] - 36,
            trajectory[frame, follower + 1, 0] + 72,
        )
        separation = axes[0, 1]
        separation.clear()
        separation.plot(time, gap, color="#2563eb", label="NPC leader–follower gap")
        separation.axhline(0, color="#d62728", linestyle="--", linewidth=0.9)
        separation.axvline(time[PLAN_FRAMES], color="#555", linestyle=":", label="plan ends")
        separation.axvline(time[frame], color="black", linewidth=0.9)
        separation.set(xlim=(0, time[-1]), xlabel="t [s]", ylabel="net gap [m]",
                       title="Realized longitudinal separation")
        separation.legend(loc="upper right", fontsize=6, frameon=False)
        action = axes[1, 0]
        action.clear()
        action.plot(time[1:], actions[:, follower, 0], color="#7e22ce", label="follower")
        action.plot(time[1:], actions[:, leader, 0], color="#e69f00", label="forced leader")
        action.axhline(dose, color="#e69f00", linestyle="--", linewidth=0.8)
        action.axvspan(time[26], time[50], color="#e69f00", alpha=0.13)
        action.axvline(time[PLAN_FRAMES], color="#555", linestyle=":")
        action.axvline(time[frame], color="black", linewidth=0.9)
        action.set(xlim=(0, time[-1]), ylim=(-8.5, 4.5), xlabel="t [s]",
                   ylabel="acceleration [m/s²]", title="Executed NPC actions")
        action.legend(loc="lower right", fontsize=6, frameon=False)
        lateral = axes[1, 1]
        lateral.clear()
        lateral.plot(time, trajectory[:, follower + 1, 1],
                     color="#7e22ce", label="follower lateral position")
        lateral.axhline(target_y, color="#2ca02c", linestyle="--",
                        label="mapped target centre")
        lateral.axvline(time[PLAN_FRAMES], color="#555", linestyle=":", label="plan ends")
        lateral.axvline(time[frame], color="black", linewidth=0.9)
        lateral.set(xlim=(0, time[-1]), xlabel="t [s]", ylabel="lateral y [m]",
                    title="Autonomous lane completion across plan boundary")
        lateral.legend(loc="lower left", fontsize=6, frameon=False)
        figure.canvas.draw()
        frames.append(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())
    GIF.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(GIF, frames, duration=0.12, loop=0)
    plt.close(figure)


def main() -> None:
    source = load_json(SOURCE)
    selected = sorted(
        (
            item for item in source["records"]
            if item["follower_autonomous_lane"]
            and not item["follower_autonomous_lane_complete"]
        ),
        key=lambda item: (item["test_row"], item["leader_brake_mps2"]),
    )
    if not selected:
        raise ValueError("no incomplete autonomous lane changes in the source report")
    config = load_protocol_config(ROOT / "hierarchical_world_model/config/world_model.yaml")
    experiment = prepare_experiment_data(config, ROOT)
    test_rows = np.asarray(experiment.test_rows, np.int64)
    position = {int(row): index for index, row in enumerate(test_rows)}
    indices = np.asarray([position[item["test_row"]] for item in selected], np.int64)
    rows = test_rows[indices]
    arrays = experiment.bundle.arrays
    states = np.asarray(arrays["agent_states"][rows], np.float32)
    valid = np.asarray(arrays["agent_valid"][rows], bool)
    maps = np.asarray(arrays["map_polylines"][rows], np.float32)
    map_valid = np.asarray(arrays["map_polyline_valid"][rows], bool)
    follower = np.asarray([item["follower_slot"] - 1 for item in selected], np.int64)
    leader = np.asarray([item["leader_slot"] - 1 for item in selected], np.int64)
    doses = np.asarray([item["leader_brake_mps2"] for item in selected], np.float32)
    row_index = np.arange(len(rows))
    device = select_device("cuda")
    all_plans = frozen_diffusion_plans(
        experiment.bundle, test_rows,
        checkpoint=config["paths"]["diffusion_checkpoint"],
        output_dir=ROOT / "results/hierarchical_world_model/evaluation/plans_full",
        device=device, batch_size=64, ddim_steps=20, experiment_scope="full",
    )
    plans = np.asarray(all_plans[indices], np.float32)
    theta = _theta_for_rows(rows, DEFAULT_MA_IDM_POSTERIOR)
    model, _ = load_checkpoint(Path(config["paths"]["evaluation_checkpoint"]), device=device)
    model.eval()
    offline = rollout(
        model, states, valid, plans, maps, map_valid,
        device=device, history_frames=25, motion_seed=None,
        controller=ForcedNPCBrakeController(theta, leader, doses).to(device),
        record_policy_trace=False,
    )
    exogenous = WorldExogenousState.sample(
        len(rows), seed=31, response_steps=TOTAL_FRAMES,
        scene_refresh_responses=model.cfg.scene_refresh_responses,
        scene_dim=model.cfg.scene_latent_dim,
        agent_dim=model.cfg.agent_latent_dim,
    )
    exogenous = replace(
        exogenous,
        scene_innovations=np.zeros_like(exogenous.scene_innovations),
        agent_response_innovations=np.zeros_like(exogenous.agent_response_innovations),
        policy_response_innovations=np.zeros_like(exogenous.policy_response_innovations),
        policy_calibration_innovations=np.zeros_like(exogenous.policy_calibration_innovations),
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
    controller = ForcedNPCBrakeController(theta, leader, doses).to(device)
    world = HighwayEnvClosedLoopWorld(model, device=device, controller=controller)
    world.reset(
        torch.from_numpy(states[:, ANCHOR_INDEX]),
        torch.from_numpy(valid[:, ANCHOR_INDEX]),
        torch.from_numpy(plans), torch.from_numpy(maps), torch.from_numpy(map_valid),
        exogenous_state=exogenous,
        initial_history=torch.from_numpy(history),
        initial_history_valid=torch.from_numpy(history_valid),
        committed_ego_controls=torch.from_numpy(historical_ego),
        deterministic_response=True,
    )
    logged_ego = _logged_ego_actions(states, valid)
    tail_plan = _tail_plan(plans, TOTAL_FRAMES - PLAN_FRAMES)
    positions = []
    executed_background = []
    start = np.full(len(rows), -1, np.int64)
    finish = np.full(len(rows), -1, np.int64)
    collision = np.zeros(len(rows), bool)
    npc_npc_collision = np.zeros(len(rows), bool)
    follower_offroad = np.zeros(len(rows), bool)
    follower_completed_at_149 = np.zeros(len(rows), bool)
    for frame in range(TOTAL_FRAMES):
        if frame == PLAN_FRAMES:
            # An explicit diagnostic plan update at the existing boundary.
            # No frozen Diffusion or realized first-window state is replaced.
            append_future_reference(world, torch.from_numpy(tail_plan))
        ego = (
            torch.from_numpy(logged_ego[:, frame]).to(device)
            if frame < PLAN_FRAMES else
            torch.zeros(len(rows), 2, device=device)
        )
        step = world.advance_response(ego[:, None])
        state = step["agent_states"].cpu().numpy()
        positions.append(state)
        executed_background.append(step["background_actions"].cpu().numpy()[:, 0])
        active = controller._autonomous_lane_active.cpu().numpy()[row_index, follower]
        target = controller._autonomous_lane_target_y.cpu().numpy()[row_index, follower]
        heading = np.arctan2(state[row_index, follower + 1, 3], state[row_index, follower + 1, 2])
        completed = active & (np.abs(state[row_index, follower + 1, 1] - target) < 0.75) & (np.abs(heading) < 0.05)
        start[(start < 0) & active] = frame
        finish[(finish < 0) & completed] = frame
        pair = step["collision_pairs"].cpu().numpy().astype(bool)
        collision |= pair.any(axis=(1, 2))
        npc_npc_collision |= pair[:, 1:, 1:].any(axis=(1, 2))
        follower_offroad |= step["offroad"].cpu().numpy()[row_index, follower + 1]
        if frame == PLAN_FRAMES - 1:
            follower_completed_at_149 = completed.copy()
    highway = np.stack(positions, axis=1)
    actions = np.stack(executed_background, axis=1)
    prefix_error = np.linalg.norm(
        highway[:, :PLAN_FRAMES, :, :2] - offline.states[..., :2], axis=-1
    )
    present = np.broadcast_to(valid[:, ANCHOR_INDEX, None, :], prefix_error.shape)
    final = highway[:, -1]
    target = controller._autonomous_lane_target_y.cpu().numpy()[row_index, follower]
    target_error = final[row_index, follower + 1, 1] - target
    final_heading = np.arctan2(
        final[row_index, follower + 1, 3], final[row_index, follower + 1, 2]
    )
    records = [
        {
            "test_row": int(rows[index]),
            "leader_brake_mps2": float(doses[index]),
            "follower_slot": int(follower[index] + 1),
            "reported_start_frame": int(selected[index]["follower_autonomous_lane_start_frame"]),
            "highway_start_frame": int(start[index]),
            "completion_frame": None if finish[index] < 0 else int(finish[index]),
            "completed_at_149": bool(follower_completed_at_149[index]),
            "completed_by_224": bool(finish[index] >= 0),
            "final_target_error_m": float(target_error[index]),
            "final_heading_rad": float(final_heading[index]),
            "raw_collision": bool(collision[index]),
            "npc_npc_collision": bool(npc_npc_collision[index]),
            "follower_offroad": bool(follower_offroad[index]),
            "prefix_position_ADE_m": float(prefix_error[index][present[index]].mean()),
        }
        for index in range(len(rows))
    ]
    example_index = int(np.argmax(finish))
    example = records[example_index]
    trace = np.concatenate(
        (states[example_index:example_index + 1, ANCHOR_INDEX],
         highway[example_index]),
        axis=0,
    )
    _render_example(
        trace, actions[example_index], valid[example_index, ANCHOR_INDEX],
        test_row=example["test_row"], follower_slot=example["follower_slot"],
        leader_slot=int(leader[example_index] + 1),
        target_y=float(target[example_index]), dose=float(doses[example_index]),
    )
    report = {
        "schema": "late_autonomous_lane_continuation_diagnostic_v1",
        "test_split_size": int(len(test_rows)),
        "selected_late_incomplete_scenes": len(rows),
        "frozen_diffusion_frames": PLAN_FRAMES,
        "total_frames": TOTAL_FRAMES,
        "tail_plan": "terminal plan longitudinal velocity extrapolation; terminal lane fixed",
        "tail_ads_policy": "zero acceleration and zero yaw rate after logged ADS frame 149",
        "not_a_highd_factual_or_diffusion_tail_metric": True,
        "online_controller_sha256": file_sha256(
            ROOT / "hierarchical_world_model/src/stochastic_drivers/online.py"
        ),
        "online_runtime_sha256": online_runtime_sha256(),
        "continuation_api_sha256": file_sha256(
            ROOT / "hierarchical_world_model/src/continuation.py"
        ),
        "prefix_position_ADE_m": float(prefix_error[present].mean()),
        "prefix_position_P95_m": float(np.quantile(prefix_error[present], 0.95)),
        "prefix_max_start_frame_difference": int(np.max(np.abs(
            start - np.asarray([item["follower_autonomous_lane_start_frame"] for item in selected])
        ))),
        "completed_at_149": int(follower_completed_at_149.sum()),
        "completed_by_224": int((finish >= 0).sum()),
        "raw_collision_scenes": int(collision.sum()),
        "npc_npc_collision_scenes": int(npc_npc_collision.sum()),
        "follower_offroad_scenes": int(follower_offroad.sum()),
        "gif_example": {
            "gif": GIF.name,
            "gif_sha256": file_sha256(GIF),
            "test_row": example["test_row"],
            "leader_brake_mps2": example["leader_brake_mps2"],
            "completion_frame": example["completion_frame"],
            "final_target_error_m": example["final_target_error_m"],
        },
        "records": records,
    }
    save_json(report, OUTPUT)
    print({key: value for key, value in report.items() if key != "records"})


if __name__ == "__main__":
    main()
