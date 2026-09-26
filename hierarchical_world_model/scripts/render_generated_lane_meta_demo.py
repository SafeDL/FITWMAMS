#!/usr/bin/env python3
"""Animate matched left/right lane-change ADS actions and online NPC response."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.scripts.evaluate_generated_lane_meta_actions import (  # noqa: E402
    _lane_policy,
    _rear_in_target_lane,
)
from hierarchical_world_model.scripts.playback_helpers import _draw_world  # noqa: E402
from hierarchical_world_model.src.composition import HierarchicalWorldSampler  # noqa: E402
from hierarchical_world_model.src.execution import _rollout_sample, hold_current_ego_action  # noqa: E402
from hierarchical_world_model.src.provenance import online_runtime_sha256  # noqa: E402
from tools.plot_style import get_pyplot  # noqa: E402
from traffic_components.src.core.utils import file_sha256, save_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--scene-index", type=int, default=-1,
                        help="negative selects a scene with a rear NPC in the right target lane")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/hierarchical_world_model/evaluation/right_lane_meta_demo.gif",
    )
    args = parser.parse_args()
    sampler = HierarchicalWorldSampler(
        flow_checkpoint=ROOT / "results/highd_natural_driving_flow/checkpoints/best_scenario_condition_flow.pt",
        flow_output_dir=ROOT / "results/highd_natural_driving_flow",
        diffusion_checkpoint=ROOT / "results/background_diffusion/checkpoints/best_background_diffusion.pt",
        diffusion_contract=ROOT / "results/background_diffusion/dataset_contract.json",
        response_checkpoint=ROOT / "results/hierarchical_world_model/factual_hiqr/checkpoints/final_world_model.pt",
        repo_root=ROOT,
        device=args.device,
    )
    exogenous = sampler.sample_world_exogenous(args.batch_size, seed=args.seed, response_steps=149)
    sample = sampler.compose_exogenous(exogenous)
    policies = {
        "hold": hold_current_ego_action,
        "left": _lane_policy("left", braking=False),
        "right": _lane_policy("right", braking=False),
        "brake_right": _lane_policy("right", braking=True),
    }
    branches = {
        key: _rollout_sample(sampler, sample, policy, reaction_controller="auto")
        for key, policy in policies.items()
    }
    for key, branch in branches.items():
        if not bool(branch.numerical_valid.all()):
            raise RuntimeError(f"{key} branch produced non-finite state")

    right_y = policies["right"].target_y.detach().cpu().numpy()
    eligible, receiver, _gap_m, _closing_speed = _rear_in_target_lane(
        branches["right"].states[:, 25], sample.initial_valid, right_y
    )
    if args.scene_index >= 0:
        index = args.scene_index
        if index >= args.batch_size:
            raise ValueError("scene-index must be inside the sampled batch")
    else:
        right_collision = branches["right"].collision_pairs.any(axis=(1, 2, 3))
        selected = np.flatnonzero(eligible & right_collision)
        if not len(selected):
            selected = np.flatnonzero(eligible)
        index = int(selected[0]) if len(selected) else 0
    focus = int(receiver[index] + 1) if eligible[index] else 1
    focus_action_index = focus - 1
    valid = sample.initial_valid[index]
    keys = ("hold", "right", "brake_right")
    titles = (
        "ADS holds current action",
        "ADS completes a semantic right lane change",
        "ADS brakes (-4 m/s²) while changing right",
    )
    states = {key: branches[key].states[index] for key in keys}
    plt = get_pyplot()
    figure, axes = plt.subplots(2, 2, figsize=(12, 6.8), constrained_layout=True)
    frames = []
    for frame in range(0, len(states["hold"]), 3):
        for axis, key, title in zip(
            (axes[0, 0], axes[0, 1], axes[1, 0]), keys, titles
        ):
            _draw_world(
                axis,
                states=states[key],
                valid=valid,
                frame=frame,
                title=f"{title} | t={frame * 0.04:.2f} s",
                focus_slot=focus,
                focus_role="target-lane rear NPC" if eligible[index] else "NPC",
            )
        chart = axes[1, 1]
        chart.clear()
        t = np.arange(149) * 0.04
        for key, color, label in (
            ("hold", "#1f78b4", "rear NPC: hold-current ADS"),
            ("right", "#e31a1c", "rear NPC: right lane change"),
            ("brake_right", "#7e22ce", "rear NPC: brake + right lane change"),
        ):
            chart.plot(t, branches[key].background_actions[index, :, focus_action_index, 0],
                       color=color, label=label)
        chart.axvline(frame * 0.04, color="black", linewidth=0.9)
        chart.set(xlim=(0, 5.96), ylim=(-8.5, 4.5), xlabel="t [s]",
                  ylabel="longitudinal acceleration [m/s²]",
                  title="Matched online NPC controls")
        chart.legend(loc="lower right", fontsize=7, frameon=False)
        figure.canvas.draw()
        frames.append(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output, frames, duration=0.12, loop=0)
    plt.close(figure)
    save_json({
        "schema": "generated_lane_meta_demo_v1",
        "seed": args.seed,
        "batch_size": args.batch_size,
        "scene_index": index,
        "focus_npc_slot_zero_based": focus,
        "focus_is_right_target_lane_rear": bool(eligible[index]),
        "policies": [
            "matched hold-current ADS",
            "semantic right lane change to mapped lane centre with entry-speed hold",
            "-4 m/s^2 for frames 25-49 combined with semantic right lane change",
        ],
        "right_lane_terminal_error_m": float(
            states["right"][-1, 0, 1] - right_y[index]
        ),
        "collision_scenes_by_policy": {
            key: int(branches[key].collision_pairs[index].any()) for key in keys
        },
        "offroad_agent_frames_by_policy": {
            key: int((branches[key].offroad[index] & valid[None]).sum()) for key in keys
        },
        "online_runtime_sha256": online_runtime_sha256(),
        "online_controller_sha256": file_sha256(ROOT / "hierarchical_world_model/src/stochastic_drivers/online.py"),
        "gif": args.output.name,
    }, args.output.with_name(f"{args.output.stem}_manifest.json"))
    print(args.output)


if __name__ == "__main__":
    main()
