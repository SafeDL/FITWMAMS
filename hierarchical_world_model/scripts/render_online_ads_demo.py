#!/usr/bin/env python3
"""Render a reproducible single-pass ADS/NPC closed-loop GIF.

All three panels are separate online worlds with common exogenous randomness;
none is computed to initialize another branch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.scripts.playback_helpers import _draw_world  # noqa: E402
from hierarchical_world_model.src.ads_interventions import (  # noqa: E402
    OnlineAccelerationWindowPolicy,
    OnlineSemanticLaneChangePolicy,
)
from hierarchical_world_model.src.composition import HierarchicalWorldSampler  # noqa: E402
from hierarchical_world_model.src.execution import hold_current_ego_action, rollout_world  # noqa: E402
from hierarchical_world_model.src.provenance import online_runtime_sha256  # noqa: E402
from hierarchical_world_model.scripts.ads_test_policies import MaintainEntrySpeedADS  # noqa: E402
from tools.plot_style import get_pyplot  # noqa: E402
from traffic_components.src.core.utils import file_sha256, save_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--scene-index", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/hierarchical_world_model/evaluation/ads_demo.gif",
    )
    args = parser.parse_args()
    controller_hash = file_sha256(
        ROOT / "hierarchical_world_model/src/stochastic_drivers/online.py"
    )
    if not 0 <= args.scene_index < args.batch_size:
        raise ValueError("scene-index must be inside the sampled batch")
    sampler = HierarchicalWorldSampler(
        flow_checkpoint=ROOT / "results/highd_natural_driving_flow/checkpoints/best_scenario_condition_flow.pt",
        flow_output_dir=ROOT / "results/highd_natural_driving_flow",
        diffusion_checkpoint=ROOT / "results/background_diffusion/checkpoints/best_background_diffusion.pt",
        diffusion_contract=ROOT / "results/background_diffusion/dataset_contract.json",
        response_checkpoint=ROOT / "results/hierarchical_world_model/factual_hiqr/checkpoints/final_world_model.pt",
        repo_root=ROOT, device=args.device,
    )
    exogenous = sampler.sample_world_exogenous(args.batch_size, seed=args.seed)
    sample = sampler.compose_exogenous(exogenous)
    policies = (
        hold_current_ego_action,
        OnlineAccelerationWindowPolicy(hold_current_ego_action, -8.0),
        OnlineSemanticLaneChangePolicy(
            MaintainEntrySpeedADS(start_frame=25), direction="left"
        ),
    )
    rollouts = [rollout_world(sampler, exogenous, policy) for policy in policies]
    index = args.scene_index
    if not all(bool(rollout.numerical_valid[index]) for rollout in rollouts):
        raise RuntimeError("selected scene produced non-finite state")
    valid = sample.initial_valid[index]
    states = [rollout.states[index] for rollout in rollouts]
    focus = 1  # selected seed/scene has a same-lane rear NPC in slot b2
    plt = get_pyplot()
    figure, axes = plt.subplots(2, 2, figsize=(12, 6.8), constrained_layout=True)
    titles = ("ADS holds current action", "ADS brakes at -8 m/s²", "ADS semantic left lane change")
    frames = []
    horizon = len(states[0])
    for frame in range(0, horizon, 3):
        for axis, trajectory, title in zip((axes[0, 0], axes[0, 1], axes[1, 0]), states, titles):
            _draw_world(
                axis, states=trajectory, valid=valid, frame=frame,
                title=f"{title} | t={frame * 0.04:.2f} s",
                focus_slot=focus, focus_role="rear NPC",
            )
        chart = axes[1, 1]
        chart.clear()
        t = np.arange(149) * 0.04
        chart.plot(t, rollouts[0].background_actions[index, :, focus, 0],
                   color="#1f78b4", label="rear NPC: hold-current ADS")
        chart.plot(t, rollouts[1].background_actions[index, :, focus, 0],
                   color="#7e22ce", label="rear NPC: ADS brake")
        chart.plot(t, rollouts[1].ego_actions[index, :, 0],
                   color="#d62728", alpha=0.65, label="ADS: brake branch")
        chart.axvline(frame * 0.04, color="black", linewidth=0.9)
        chart.set(xlim=(0, 5.96), ylim=(-8.5, 4.5), xlabel="t [s]",
                  ylabel="longitudinal acceleration [m/s²]",
                  title="Executed controls (single-pass branches)")
        chart.legend(loc="lower right", fontsize=7, frameon=False)
        figure.canvas.draw()
        rgba = np.asarray(figure.canvas.buffer_rgba())
        frames.append(rgba[..., :3].copy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output, frames, duration=0.12, loop=0)
    plt.close(figure)
    target = policies[2].target_y
    save_json(
        {
            "schema": "online_ads_generated_gif_v1",
            "seed": args.seed,
            "batch_size": args.batch_size,
            "scene_index": index,
            "focus_npc_slot_zero_based": focus,
            "policies": [
                "hold_current_ego_action",
                "absolute_-8_mps2_frames_25_to_49",
                "semantic_left_lane_change_from_frame_25_with_entry_speed_hold",
            ],
            "all_branches_numerically_valid": True,
            "collision_frames_by_policy": [
                int(rollout.collision_pairs[index].any(axis=(1, 2)).sum())
                for rollout in rollouts
            ],
            "offroad_agent_frames_by_policy": [
                int((rollout.offroad[index] & valid[None]).sum())
                for rollout in rollouts
            ],
            "left_lane_final_lateral_error_m": (
                None if target is None else float(states[2][-1, 0, 1] - target[index].item())
            ),
            "online_controller_sha256": controller_hash,
            "online_runtime_sha256": online_runtime_sha256(),
            "gif": args.output.name,
        },
        args.output.with_name(f"{args.output.stem}_manifest.json"),
    )
    print(args.output)


if __name__ == "__main__":
    main()
