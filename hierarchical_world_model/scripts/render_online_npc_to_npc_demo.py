#!/usr/bin/env python3
"""HighwayEnv probe of causal NPC-to-NPC braking, with project-style playback."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_model_baselines.models.bayesian_ma_idm.src.model import (  # noqa: E402
    load_posterior, sample_driver_joint,
)
from hierarchical_world_model.scripts.playback_helpers import _draw_world  # noqa: E402
from hierarchical_world_model.src.highway import HighwayEnvTraffic  # noqa: E402
from hierarchical_world_model.src.provenance import online_runtime_sha256  # noqa: E402
from hierarchical_world_model.src.reaction_controller import ReactionControllerContext  # noqa: E402
from hierarchical_world_model.src.stochastic_drivers.online import (  # noqa: E402
    DEFAULT_MA_IDM_POSTERIOR, OnlineMAIDMController,
)
from tools.plot_style import get_pyplot  # noqa: E402
from traffic_components.src.core.utils import file_sha256, save_json  # noqa: E402


def run_probe(
    *, leader_brakes: bool, theta: np.ndarray,
    initial_net_gap_m: float = 25.2,
    follower_speed_mps: float = 20.0,
    leader_speed_mps: float = 20.0,
    leader_brake_mps2: float = -4.0,
    brake_frames: int = 25,
) -> dict[str, np.ndarray | float | bool]:
    """Run one world; the comparison world never supplies actions or states."""
    if initial_net_gap_m <= 0.0 or min(follower_speed_mps, leader_speed_mps) < 0.0:
        raise ValueError("initial gap and speeds must be physically valid")
    if not -8.0 <= leader_brake_mps2 < 0.0 or not 0 < brake_frames <= 149:
        raise ValueError("brake setpoint or duration is outside the probe range")
    initial = np.zeros((7, 6), np.float32)
    # Keep ADS visible in the adjacent lane, without making it the follower's
    # leader; this isolates the NPC-to-NPC causal edge in the road view.
    initial[0, [0, 1, 2]] = [20.0, 3.6, 20.0]
    initial[1, [0, 2]] = [0.0, follower_speed_mps]
    initial[2, [0, 2]] = [initial_net_gap_m + 4.8, leader_speed_mps]
    valid = np.zeros(7, bool)
    valid[:3] = True
    traffic = HighwayEnvTraffic(seed=17)
    traffic.reset(initial, valid)
    controller = OnlineMAIDMController(theta)
    current = torch.from_numpy(initial[None])
    history = current[:, None].expand(-1, 25, -1, -1).clone()
    history_valid = torch.from_numpy(valid[None, None]).expand(-1, 25, -1)
    present = torch.from_numpy(valid[None])
    base = torch.zeros(1, 1, 6, 2)
    previous: torch.Tensor | None = None
    frames = [initial.copy()]
    follower_actions = []
    leader_actions = []
    follower_responses = []
    overlap = []
    offroad = []
    for frame in range(149):
        context = ReactionControllerContext(
            history=history, history_valid=history_valid,
            current=current, current_valid=present,
            committed_ego_controls=torch.zeros(1, 1, 2),
            base_actions=base, reference_actions=base,
            intervention_trigger=torch.zeros(1),
            intervention_memory=torch.zeros(1),
            lateral_intervention_memory=torch.zeros(1),
            agent_style_state=torch.zeros(1, 7, 1),
            response_field_gain=None, response_sensitivity_bounds=None,
            adapter_gain=None, reaction_enabled=None, cfg=SimpleNamespace(),
            previous_background_actions=previous,
        )
        output = controller(context)
        command = output.actions[0, 0].detach().numpy().copy()
        if leader_brakes and frame < brake_frames:
            command[1, 0] = leader_brake_mps2
        step = traffic.step(command, ego_action=np.zeros(2, np.float32))
        frames.append(step.states.copy())
        follower_actions.append(float(step.background_actions[0, 0]))
        leader_actions.append(float(step.background_actions[1, 0]))
        follower_responses.append(bool(output.active[0, 0]))
        overlap.append(bool(step.collision_pairs[1, 2]))
        offroad.append(bool(step.offroad.any()))
        current = torch.from_numpy(step.states[None].copy())
        history = torch.cat((history[:, 1:], current[:, None]), dim=1)
        previous = torch.from_numpy(step.background_actions[None].copy())
    positions = np.stack(frames)
    return {
        "states": positions,
        "follower_actions": np.asarray(follower_actions, np.float32),
        "leader_actions": np.asarray(leader_actions, np.float32),
        "follower_responses": np.asarray(follower_responses, bool),
        "net_gap_m": positions[:, 2, 0] - positions[:, 1, 0] - 4.8,
        "npc_npc_overlap": bool(any(overlap)),
        "offroad": bool(any(offroad)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/hierarchical_world_model/evaluation/npc_to_npc_brake_demo.gif",
    )
    output = parser.parse_args().output
    posterior = load_posterior(DEFAULT_MA_IDM_POSTERIOR)
    rng = np.random.default_rng(17)
    theta = np.asarray(
        [sample_driver_joint(posterior, rng)[:5] for _ in range(6)], np.float32
    )
    brake = run_probe(leader_brakes=True, theta=theta)
    control = run_probe(leader_brakes=False, theta=theta)
    if brake["npc_npc_overlap"] or brake["offroad"]:
        raise RuntimeError("NPC-to-NPC probe collided or left the road")
    if brake["follower_actions"][1] >= control["follower_actions"][1]:
        raise RuntimeError("follower did not respond to the realized NPC brake")
    maximum_follower_jerk = float(
        np.abs(np.diff(brake["follower_actions"]) / 0.04).max()
    )
    if maximum_follower_jerk > 12.01:
        raise RuntimeError("NPC response exceeded its 12 m/s³ jerk-release limit")
    time = np.arange(150) * 0.04
    plt = get_pyplot()
    figure, axes = plt.subplots(2, 2, figsize=(12, 6.8), constrained_layout=True)
    frames = []
    for frame in range(0, 150, 3):
        for axis, result, title in (
            (axes[0, 0], brake, "NPC leader brake | online response"),
            (axes[0, 1], control, "Smooth NPC leader | same controller"),
        ):
            _draw_world(
                axis, states=result["states"], valid=np.array([True, True, True, False, False, False, False]),
                frame=frame, title=f"{title} | t={time[frame]:.2f} s",
                focus_slot=0, source_slot=1,
                focus_role="following NPC", source_role="leader NPC",
            )
            axis.set_xlim(result["states"][frame, 1, 0] - 36, result["states"][frame, 1, 0] + 72)
        chart = axes[1, 0]
        chart.clear()
        chart.plot(time[1:], brake["follower_actions"], color="#7e22ce", label="brake-world follower")
        chart.plot(time[1:], control["follower_actions"], color="#8c8c8c", label="smooth-world follower")
        chart.plot(time[1:], brake["leader_actions"], color="#e69f00", label="leader NPC")
        chart.axvline(time[frame], color="black", linewidth=0.9)
        chart.set(xlim=(0, 5.96), ylim=(-8.5, 4.5), xlabel="t [s]", ylabel="acceleration [m/s²]",
                  title="Executed actions (no future-state input)")
        chart.legend(loc="lower right", fontsize=6, frameon=False)
        gap_chart = axes[1, 1]
        gap_chart.clear()
        gap_chart.plot(time, brake["net_gap_m"], color="#7e22ce", label="brake-world net gap")
        gap_chart.plot(time, control["net_gap_m"], color="#8c8c8c", label="smooth-world net gap")
        gap_chart.axhline(0, color="#d62728", linestyle="--", label="overlap boundary")
        gap_chart.axvline(time[frame], color="black", linewidth=0.9)
        gap_chart.set(xlim=(0, 5.96), xlabel="t [s]", ylabel="net gap [m]",
                      title="Realized NPC-to-NPC separation")
        gap_chart.legend(loc="lower left", fontsize=6, frameon=False)
        figure.canvas.draw()
        frames.append(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, duration=0.12, loop=0)
    plt.close(figure)
    save_json({
        "schema": "online_npc_to_npc_highway_probe_v1",
        "scope": "synthetic controlled probe, not a highD test metric",
        "plant": "HighwayEnvTraffic",
        "leader_brake_duration_s": 1.0,
        "leader_brake_setpoint_mps2": -4.0,
        "follower_action_at_second_frame_brake_mps2": float(brake["follower_actions"][1]),
        "follower_action_at_second_frame_control_mps2": float(control["follower_actions"][1]),
        "follower_max_abs_jerk_mps3": maximum_follower_jerk,
        "follower_response_frames": int(brake["follower_responses"].sum()),
        "minimum_net_gap_brake_m": float(np.min(brake["net_gap_m"])),
        "npc_npc_overlap": brake["npc_npc_overlap"],
        "offroad": brake["offroad"],
        "online_controller_sha256": file_sha256(
            ROOT / "hierarchical_world_model/src/stochastic_drivers/online.py"
        ),
        "online_runtime_sha256": online_runtime_sha256(),
        "gif": output.name,
    }, output.with_name(f"{output.stem}_manifest.json"))
    print(output)


if __name__ == "__main__":
    main()
