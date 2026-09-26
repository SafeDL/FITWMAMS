#!/usr/bin/env python3
"""Show when a clear NPC pass should keep HiQR instead of a cut-in correction."""

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
from hierarchical_world_model.src.ads_interventions import SemanticLaneChangePolicy  # noqa: E402
from hierarchical_world_model.src.collision_attribution import collision_events  # noqa: E402
from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.evaluation import _logged_ego_actions, rollout  # noqa: E402
from hierarchical_world_model.src.planner import frozen_diffusion_plans  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from hierarchical_world_model.src.provenance import online_runtime_sha256  # noqa: E402
from hierarchical_world_model.src.stochastic_drivers.online import (  # noqa: E402
    DEFAULT_MA_IDM_POSTERIOR, OnlineMAIDMController,
)
from hierarchical_world_model.src.train import load_checkpoint  # noqa: E402
from tools.plot_style import get_pyplot  # noqa: E402
from traffic_components.src.core.utils import file_sha256, save_json, select_device  # noqa: E402


class _NoPassGateDiagnostic(OnlineMAIDMController):
    """Same current online controller with only the passing gate disabled."""

    def _passing_cutin_clearance(self, context):
        return torch.zeros_like(context.current[:, 1:, 0], dtype=torch.bool)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-row", type=int, default=12462)
    parser.add_argument("--npc-slot", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/hierarchical_world_model/evaluation/ads_cutin_pass_demo.gif",
    )
    args = parser.parse_args()
    if not 1 <= args.npc_slot <= 6:
        raise ValueError("NPC slot must be in [1, 6]")
    config = load_protocol_config(ROOT / "hierarchical_world_model/config/world_model.yaml")
    experiment = prepare_experiment_data(config, ROOT)
    test_rows = np.asarray(experiment.test_rows, np.int64)
    found = np.flatnonzero(test_rows == args.test_row)
    if len(found) != 1:
        raise ValueError("--test-row is not in the highD Test split")
    arrays = experiment.bundle.arrays
    states = np.asarray(arrays["agent_states"][[args.test_row]], np.float32)
    valid = np.asarray(arrays["agent_valid"][[args.test_row]], bool)
    maps = np.asarray(arrays["map_polylines"][[args.test_row]], np.float32)
    map_valid = np.asarray(arrays["map_polyline_valid"][[args.test_row]], bool)
    if not valid[0, ANCHOR_INDEX, args.npc_slot]:
        raise ValueError("selected NPC is absent at the anchor")
    device = select_device(args.device)
    plan = frozen_diffusion_plans(
        experiment.bundle, test_rows,
        checkpoint=config["paths"]["diffusion_checkpoint"],
        output_dir=ROOT / "results/hierarchical_world_model/evaluation/plans_full",
        device=device, batch_size=64, ddim_steps=20, experiment_scope="full",
    )[int(found[0]):int(found[0]) + 1]
    theta = _theta_for_rows(np.asarray([args.test_row]), DEFAULT_MA_IDM_POSTERIOR)
    model, _ = load_checkpoint(Path(config["paths"]["evaluation_checkpoint"]), device=device)
    model.eval()
    controls = _logged_ego_actions(states, valid)

    def execute(controller):
        return rollout(
            model, states, valid, plan, maps, map_valid,
            device=device, history_frames=25, motion_seed=None,
            ads_policy=SemanticLaneChangePolicy(
                controls, states[:, ANCHOR_INDEX],
                map_polylines=maps, map_polyline_valid=map_valid,
                direction="left",
            ),
            controller=controller.to(device), record_policy_trace=False,
        )

    online = execute(OnlineMAIDMController(theta))
    no_gate = execute(_NoPassGateDiagnostic(theta))
    anchor = states[:, ANCHOR_INDEX:ANCHOR_INDEX + 1]
    online_states = np.concatenate((anchor, online.states), axis=1)[0]
    no_gate_states = np.concatenate((anchor, no_gate.states), axis=1)[0]
    present = valid[0, ANCHOR_INDEX]
    online_events = collision_events(online_states[None], present[None, 1:])
    no_gate_events = collision_events(no_gate_states[None], present[None, 1:])
    if online_events["raw"][0] or not no_gate_events["raw"][0]:
        raise RuntimeError("selected example does not show the expected gate effect")
    focus = args.npc_slot - 1
    online_ax = online.background_actions[0, :, focus, 0]
    no_gate_ax = no_gate.background_actions[0, :, focus, 0]
    online_dx = online_states[:, 0, 0] - online_states[:, args.npc_slot, 0]
    no_gate_dx = no_gate_states[:, 0, 0] - no_gate_states[:, args.npc_slot, 0]
    no_gate_dy = no_gate_states[:, 0, 1] - no_gate_states[:, args.npc_slot, 1]
    time = np.arange(150) * 0.04
    plt = get_pyplot()
    figure, axes = plt.subplots(2, 2, figsize=(12, 6.8), constrained_layout=True)
    frames = []
    diagnostic_label = "pass gate disabled"
    for frame in range(0, 150, 3):
        for axis, trajectory, title in (
            (axes[0, 0], online_states, "Online NPC | mapped pass commitment"),
            (axes[0, 1], no_gate_states, f"Same ADS | {diagnostic_label}"),
        ):
            _draw_world(
                axis, states=trajectory, valid=present, frame=frame,
                title=f"{title} | t={time[frame]:.2f} s",
                focus_slot=focus, focus_role="target-lane NPC",
            )
            axis.set_xlim(
                trajectory[frame, args.npc_slot, 0] - 36,
                trajectory[frame, args.npc_slot, 0] + 72,
            )
        action_chart = axes[1, 0]
        action_chart.clear()
        action_chart.plot(time[1:], online_ax, color="#7e22ce", label="online NPC")
        action_chart.plot(time[1:], no_gate_ax, color="#8c8c8c", label=diagnostic_label)
        action_chart.axvline(time[frame], color="black", linewidth=0.9)
        action_chart.set(xlim=(0, 5.96), xlabel="t [s]", ylabel="acceleration [m/s²]",
                         title="Executed NPC longitudinal action")
        action_chart.legend(loc="lower right", fontsize=6, frameon=False)
        gap_chart = axes[1, 1]
        gap_chart.clear()
        gap_chart.plot(time, online_dx, color="#7e22ce", label="ADS − NPC x: online")
        gap_chart.plot(time, no_gate_dx, color="#8c8c8c", label=f"ADS − NPC x: {diagnostic_label}")
        gap_chart.axhspan(-4.8, 4.8, color="#d62728", alpha=0.08,
                          label="longitudinal overlap range")
        gap_chart.fill_between(
            time, 0, 1, where=np.abs(no_gate_dy) < 1.8,
            color="#e69f00", alpha=0.12, transform=gap_chart.get_xaxis_transform(),
            label="lateral overlap: gate disabled",
        )
        gap_chart.axvline(time[frame], color="black", linewidth=0.9)
        gap_chart.set(xlim=(0, 5.96), xlabel="t [s]", ylabel="ADS − NPC x [m]",
                      title="Same ADS lane-change trajectory")
        gap_chart.legend(loc="upper right", fontsize=6, frameon=False)
        figure.canvas.draw()
        frames.append(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output, frames, duration=0.12, loop=0)
    plt.close(figure)
    save_json({
        "schema": "online_ads_cutin_pass_gif_v1",
        "test_row": args.test_row,
        "npc_slot": args.npc_slot,
        "ads_intervention": "complete left lane change",
        "online_raw_overlap": bool(online_events["raw"][0]),
        "diagnostic_mode": "no_gate",
        "diagnostic_raw_overlap": bool(no_gate_events["raw"][0]),
        "no_pass_gate_raw_overlap": bool(no_gate_events["raw"][0]),
        "online_npc_npc_overlap": bool(online_events["npc_npc"][0]),
        "no_pass_gate_npc_npc_overlap": bool(no_gate_events["npc_npc"][0]),
        "online_controller_sha256": file_sha256(
            ROOT / "hierarchical_world_model/src/stochastic_drivers/online.py"
        ),
        "online_runtime_sha256": online_runtime_sha256(),
        "gif": args.output.name,
    }, args.output.with_name(f"{args.output.stem}_manifest.json"))
    print(args.output)


if __name__ == "__main__":
    main()
