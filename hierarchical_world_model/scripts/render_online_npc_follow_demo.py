#!/usr/bin/env python3
"""Render a planned-trajectory pursuit correction in the project GIF style."""

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
from hierarchical_world_model.scripts.playback_helpers import _draw_world  # noqa: E402
from hierarchical_world_model.src.ads_interventions import AbsoluteAccelerationPolicy  # noqa: E402
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-row", type=int, default=32028)
    parser.add_argument("--npc-slot", type=int, default=2)
    parser.add_argument("--ads-acceleration-mps2", type=float, default=-8.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/hierarchical_world_model/evaluation/npc_plan_pursuit_demo.gif",
    )
    args = parser.parse_args()
    if not 1 <= args.npc_slot <= 6:
        raise ValueError("NPC slot must be in [1, 6]")

    config = load_protocol_config(ROOT / "hierarchical_world_model/config/world_model.yaml")
    experiment = prepare_experiment_data(config, ROOT)
    test_rows = np.asarray(experiment.test_rows, np.int64)
    found = np.flatnonzero(test_rows == args.test_row)
    if len(found) != 1:
        raise ValueError("--test-row is not in the highD test split")
    arrays = experiment.bundle.arrays
    states = np.asarray(arrays["agent_states"][[args.test_row]], np.float32)
    valid = np.asarray(arrays["agent_valid"][[args.test_row]], bool)
    maps = np.asarray(arrays["map_polylines"][[args.test_row]], np.float32)
    map_valid = np.asarray(arrays["map_polyline_valid"][[args.test_row]], bool)
    if not valid[0, ANCHOR_INDEX, args.npc_slot]:
        raise ValueError("selected NPC is not present at the anchor")
    device = select_device(args.device)
    plans = frozen_diffusion_plans(
        experiment.bundle, test_rows,
        checkpoint=config["paths"]["diffusion_checkpoint"],
        output_dir=ROOT / "results/hierarchical_world_model/evaluation/plans_full",
        device=device, batch_size=64, ddim_steps=20, experiment_scope="full",
    )
    plan = plans[found[0]:found[0] + 1]
    theta = _theta_for_rows(np.asarray([args.test_row]), DEFAULT_MA_IDM_POSTERIOR)
    model, _ = load_checkpoint(Path(config["paths"]["evaluation_checkpoint"]), device=device)
    model.eval()
    policy = AbsoluteAccelerationPolicy(
        _logged_ego_actions(states, valid), args.ads_acceleration_mps2,
    )

    def execute(controller=None):
        return rollout(
            model, states, valid, plan, maps, map_valid,
            device=device, history_frames=25, motion_seed=None,
            ads_policy=policy, controller=controller, record_policy_trace=False,
        )

    # The HiQR-only rollout is only a post-hoc comparison, never an input to
    # the online controller.
    online = execute(OnlineMAIDMController(theta).to(device))
    hiqr = execute()
    anchor = states[:, ANCHOR_INDEX:ANCHOR_INDEX + 1]
    online_states = np.concatenate((anchor, online.states), axis=1)[0]
    hiqr_states = np.concatenate((anchor, hiqr.states), axis=1)[0]
    present = valid[0, ANCHOR_INDEX]
    focus = args.npc_slot - 1
    time = np.arange(150) * 0.04
    plan_x = np.concatenate(([anchor[0, 0, args.npc_slot, 0]], plan[0, :, focus, 0]))
    online_lag = plan_x - online_states[:, args.npc_slot, 0]
    hiqr_lag = plan_x - hiqr_states[:, args.npc_slot, 0]
    online_ax = online.background_actions[0, :, focus, 0]
    hiqr_base_ax = online.base_background_actions[0, :, focus, 0]
    online_active = np.asarray(online.controller_diagnostics["active"][0, :, focus], bool)
    suppressed = (hiqr_base_ax > 0.0) & (online_ax <= 0.0)
    plt = get_pyplot()
    figure, axes = plt.subplots(2, 2, figsize=(12, 6.8), constrained_layout=True)
    frames = []
    for frame in range(0, 150, 3):
        for column, trajectory, title in (
            (0, online_states, "Online NPC response"),
            (1, hiqr_states, "Same ADS, HiQR-only diagnostic"),
        ):
            _draw_world(
                axes[0, column], states=trajectory, valid=present, frame=frame,
                title=f"{title} | t={time[frame]:.2f} s", focus_slot=focus,
                source_slot=None, focus_role="following NPC", source_role="ADS leader",
            )
            axes[0, column].set_xlim(
                trajectory[frame, args.npc_slot, 0] - 36.0,
                trajectory[frame, args.npc_slot, 0] + 36.0,
            )
        lag_chart = axes[1, 0]
        lag_chart.clear()
        lag_chart.plot(time, online_lag, color="#7e22ce", label="online plan lag")
        lag_chart.plot(time, hiqr_lag, color="#8c8c8c", label="HiQR-only plan lag")
        lag_chart.axhline(2.0, color="#d62728", linestyle="--", label="lag trigger")
        lag_chart.axvline(time[frame], color="black", linewidth=0.9)
        lag_chart.set(xlim=(0, 5.96), xlabel="t [s]", ylabel="plan x - NPC x [m]",
                      title="Realized lag behind frozen plan")
        lag_chart.legend(loc="upper left", fontsize=6, frameon=False)
        action_chart = axes[1, 1]
        action_chart.clear()
        action_chart.plot(time[1:], hiqr_base_ax, color="#8c8c8c", label="HiQR base ax")
        action_chart.plot(time[1:], online_ax, color="#7e22ce", label="online NPC ax")
        action_chart.axhline(0.0, color="#d62728", linestyle="--")
        action_chart.fill_between(
            time[1:], 0, 1, where=online_active, color="#7e22ce", alpha=0.13,
            transform=action_chart.get_xaxis_transform(), label="online correction active",
        )
        action_chart.axvline(time[frame], color="black", linewidth=0.9)
        action_chart.set(xlim=(0, 5.96), xlabel="t [s]", ylabel="NPC ax [m/s²]",
                         title="Suppressed plan-pursuit acceleration")
        action_chart.legend(loc="lower right", fontsize=6, frameon=False)
        figure.canvas.draw()
        frames.append(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output, frames, duration=0.12, loop=0)
    plt.close(figure)

    overlaps = {}
    for name, trajectory in (("online", online_states), ("hiqr_only", hiqr_states)):
        events = collision_events(trajectory[None], present[None, 1:])
        overlaps[name] = bool(events["raw"][0])
    effect = slice(25, 100)
    save_json({
        "schema": "online_npc_plan_pursuit_gif_v1",
        "test_row": args.test_row,
        "npc_slot": args.npc_slot,
        "ads_acceleration_setpoint_mps2": args.ads_acceleration_mps2,
        "online_mean_ax_effect_window_mps2": float(online_ax[effect].mean()),
        "hiqr_base_mean_ax_effect_window_mps2": float(hiqr_base_ax[effect].mean()),
        "online_mean_plan_lag_effect_window_m": float(online_lag[1:][effect].mean()),
        "positive_hiqr_suppressed_frames": int(suppressed.sum()),
        "branch_raw_overlap": overlaps,
        "online_controller_sha256": file_sha256(
            ROOT / "hierarchical_world_model/src/stochastic_drivers/online.py"
        ),
        "online_runtime_sha256": online_runtime_sha256(),
        "gif": args.output.name,
    }, args.output.with_name(f"{args.output.stem}_manifest.json"))
    print(args.output)


if __name__ == "__main__":
    main()
