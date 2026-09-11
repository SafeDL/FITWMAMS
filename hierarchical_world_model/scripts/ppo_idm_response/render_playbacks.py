#!/usr/bin/env python3
"""Render PPO--IDM response playbacks for one highD event.

The natural replay keeps the logged ego control and overlays highD reference
trajectories.  The braking probe applies a fixed, documented ego braking
offset and contrasts the learned controller with frozen HiQR under
identical prefix samples and exogenous randomness.  It is a diagnostic, not a
safety or closed-loop validation claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from diffusion.src.data import ANCHOR_INDEX
from hierarchical_world_model.src.composition import HierarchicalWorldSampler
from hierarchical_world_model.src.data import ego_controls, prepare_experiment_data
from hierarchical_world_model.src.highway import HighwayEnvClosedLoopWorld
from hierarchical_world_model.src.nominal_reference import build_nominal_reference
from hierarchical_world_model.src.prefix_reference import prefix_only_sample, row_prefix_inputs
from hierarchical_world_model.src.randomness import WorldExogenousState
from hierarchical_world_model.src.reaction_controller import IDMResidualReactionController, NoReactionController
from hierarchical_world_model.src.reaction_evidence import ReactionEventReference
from hierarchical_world_model.src.rule_models import RuleModelBundle
from hierarchical_world_model.src.visualization import (
    DIFFUSION_COLOR,
    EGO_COLOR,
    LOGGED_REFERENCE_COLOR,
    ROAD_COLOR,
    _draw_lane_markings,
    _draw_vehicle,
)
from tools.plot_style import get_pyplot

from common import ROOT, event_directory, load_response_config, result_directory


IDM = {
    "a_max": 1.0,
    "b_comfort": 2.0,
    "desired_speed": 30.0,
    "time_headway": 1.5,
    "minimum_gap": 2.0,
}
DT_S = 0.04
COMMON_COLOR = "#2ca25f"
def _load_sampler(base: dict) -> HierarchicalWorldSampler:
    return HierarchicalWorldSampler(
        flow_checkpoint=base["paths"]["flow_checkpoint"],
        flow_output_dir=base["paths"]["flow_output_dir"],
        diffusion_checkpoint=base["paths"]["diffusion_checkpoint"],
        diffusion_contract=base["paths"]["diffusion_contract"],
        response_checkpoint=base["paths"]["evaluation_checkpoint"],
        repo_root=ROOT,
        device="cuda",
        ddim_steps=20,
        excluded_slots=(),
    )


@torch.no_grad()
def _simulate(
    *,
    sampler: HierarchicalWorldSampler,
    controller: torch.nn.Module,
    bundle,
    reference: ReactionEventReference,
    event_index: int,
    ego_actions: np.ndarray,
) -> dict[str, np.ndarray]:
    events = reference.events
    row = int(events.row_index[event_index])
    onset = int(events.local_onset_frame[event_index])
    states = np.asarray(bundle.arrays["agent_states"])[row]
    valid = np.asarray(bundle.arrays["agent_valid"])[row]
    maps = np.asarray(bundle.arrays["map_polylines"])[row]
    map_valid = np.asarray(bundle.arrays["map_polyline_valid"])[row]
    c0, mask = row_prefix_inputs(bundle, np.asarray([row]))
    exogenous = WorldExogenousState.sample(
        1,
        seed=20260907 + event_index,
        response_steps=149,
        scene_dim=sampler.response.cfg.scene_latent_dim,
        agent_dim=sampler.response.cfg.agent_latent_dim,
    )
    sample = prefix_only_sample(sampler, c0=c0, slot_mask=mask, exogenous=exogenous)
    nominal = build_nominal_reference(sampler, sample, idm_config=IDM)
    offset = max(0, onset - ANCHOR_INDEX)
    plan = np.asarray(sample.soft_plan[:, offset:])
    if len(plan[0]) < len(ego_actions):
        plan = np.concatenate(
            (plan, np.repeat(plan[:, -1:], len(ego_actions) - len(plan[0]), axis=1)),
            axis=1,
        )
    initial_state = np.array(states[onset][None], copy=True)
    initial_valid = np.array(valid[onset][None], copy=True)
    history = np.array(states[onset - 24 : onset + 1][None], copy=True)
    history_valid = np.array(valid[onset - 24 : onset + 1][None], copy=True)
    committed = ego_controls(
        states[onset - 24 : onset, 0], states[onset - 23 : onset + 1, 0], DT_S
    )[None]
    world = HighwayEnvClosedLoopWorld(
        sampler.response, device="cuda", controller=controller, idm_config=IDM
    )
    world.reset(
        torch.as_tensor(initial_state),
        torch.as_tensor(initial_valid),
        torch.as_tensor(plan[:, : len(ego_actions)]),
        torch.as_tensor(np.array(maps[None], copy=True)),
        torch.as_tensor(np.array(map_valid[None], copy=True)),
        exogenous_state=exogenous,
        initial_history=torch.as_tensor(history),
        initial_history_valid=torch.as_tensor(history_valid),
        committed_ego_controls=torch.as_tensor(committed),
        nominal_reference_states=nominal.states,
        nominal_reference_actions=nominal.background_actions,
        nominal_initial_states=nominal.initial_states,
    )
    realized, actions, active, collisions = [], [], [], []
    for action in ego_actions:
        transition = world.advance_response(torch.as_tensor(action[None], device="cuda"))
        realized.append(transition["agent_state_frames"][0, 0].cpu().numpy())
        actions.append(transition["background_actions"][0, 0].cpu().numpy())
        active.append(transition["controller_active"][0].cpu().numpy())
        collisions.append(bool(transition["collision"][0].item()))
    return {
        "states": np.asarray(realized),
        "actions": np.asarray(actions),
        "active": np.asarray(active),
        "collisions": np.asarray(collisions),
        "row": np.asarray(row),
        "onset": np.asarray(onset),
    }


def _logged_actions(states: np.ndarray, onset: int, steps: int) -> np.ndarray:
    available = min(steps, len(states) - onset - 1)
    actions = ego_controls(
        states[onset : onset + available, 0],
        states[onset + 1 : onset + available + 1, 0],
        DT_S,
    )
    if available < steps:
        actions = np.concatenate((actions, np.repeat(actions[-1:], steps - available, axis=0)))
    return actions


def _frame_axis(axis, *, center_x: float, title: str) -> None:
    axis.clear()
    axis.set_facecolor(ROAD_COLOR)
    _draw_lane_markings(axis)
    axis.set(
        xlim=(center_x - 80.0, center_x + 80.0), ylim=(-8.2, 8.2),
        title=title, aspect="equal", xlabel="x [m]", ylabel="y [m]",
    )


def _draw_natural_frame(
    axis, *, rollout: dict[str, np.ndarray], reference: np.ndarray,
    valid: np.ndarray, focus_slot: int, frame: int, event_label: str, candidate_label: str,
) -> None:
    states = rollout["states"]
    center_x = float(0.5 * (states[frame, 0, 0] + reference[frame, focus_slot + 1, 0]))
    _frame_axis(axis, center_x=center_x, title=f"{event_label} | t={(frame + 1) * DT_S:.2f}s | natural replay")
    trail = slice(max(0, frame - 45), frame + 1)
    axis.plot(reference[trail, 0, 0], reference[trail, 0, 1], color=EGO_COLOR, linewidth=1.8, alpha=0.78, label="ego (logged replay)")
    for slot in np.flatnonzero(valid[1:]):
        axis.plot(reference[trail, slot + 1, 0], reference[trail, slot + 1, 1], color="#d9d9d9", linestyle=":", linewidth=1.2, alpha=0.9)
        axis.plot(states[trail, slot + 1, 0], states[trail, slot + 1, 1], color=DIFFUSION_COLOR, linewidth=1.5, alpha=0.86)
        _draw_vehicle(axis, reference[frame, slot + 1], color=LOGGED_REFERENCE_COLOR, label=f"b{slot + 1}" if slot == focus_slot else None, filled=False, alpha=0.9)
        _draw_vehicle(axis, states[frame, slot + 1], color=DIFFUSION_COLOR, label=f"{candidate_label} b{slot + 1}" if slot == focus_slot else None, filled=True, alpha=0.52)
    _draw_vehicle(axis, reference[frame, 0], color=EGO_COLOR, label="ego (logged)", filled=True, alpha=0.9)
    axis.text(0.01, 0.02, f"red: logged ego replay | blue: {candidate_label} background | white outline/dotted: highD reference", transform=axis.transAxes, fontsize=7.5, va="bottom", ha="left", bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.2})
    axis.text(0.01, 0.94, f"learned controller authority: {int(rollout['active'][frame].sum())}/6 slots", transform=axis.transAxes, fontsize=7.5, va="top", ha="left", color="#1d4ed8", fontweight="bold", bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.2})
    axis.legend(frameon=False, fontsize=7, loc="upper right")
    axis.tick_params(labelsize=8)


def _draw_probe_world(
    axis, *, rollout: dict[str, np.ndarray], valid: np.ndarray, follower: int,
    frame: int, label: str, affected_color: str, status: str,
) -> None:
    states = rollout["states"]
    center_x = float(states[frame, 0, 0])
    _frame_axis(axis, center_x=center_x, title=f"{label}\nt={(frame + 1) * DT_S:.2f}s")
    trail = slice(max(0, frame - 45), frame + 1)
    for slot in np.flatnonzero(valid[1:]):
        color = affected_color if slot == follower - 1 else "#d9d9d9"
        axis.plot(states[trail, slot + 1, 0], states[trail, slot + 1, 1], color=color, linewidth=1.3, alpha=0.9)
        _draw_vehicle(
            axis, states[frame, slot + 1], color=color,
            label=("affected same-rear NPC" if slot == follower - 1
                   else "unaffected NPC" if slot == np.flatnonzero(valid[1:])[0] else None),
            filled=slot == follower - 1, alpha=0.88,
        )
    _draw_vehicle(axis, states[frame, 0], color=EGO_COLOR, label="ADS ego", filled=True, alpha=0.92)
    axis.text(
        0.01, 0.94, status, transform=axis.transAxes, fontsize=7.5,
        va="top", ha="left", bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.2},
    )
    axis.legend(frameon=False, fontsize=7, loc="lower right")
    axis.tick_params(labelsize=8)


def _write_probe_gif(
    path: Path, *, candidate: dict[str, np.ndarray], common: dict[str, np.ndarray],
    valid: np.ndarray, follower: int, event_label: str, frame_stride: int,
    candidate_label: str, baseline_label: str,
) -> int:
    plt = get_pyplot()
    steps = len(candidate["states"])
    frames = np.arange(0, steps, frame_stride)
    if frames[-1] != steps - 1:
        frames = np.append(frames, steps - 1)
    time_s = np.arange(steps) * DT_S
    follower_index = follower - 1
    candidate_ax = candidate["actions"][:, follower_index, 0]
    common_ax = common["actions"][:, follower_index, 0]
    candidate_gap = candidate["states"][:, 0, 0] - candidate["states"][:, follower, 0] - 4.8
    common_gap = common["states"][:, 0, 0] - common["states"][:, follower, 0] - 4.8
    figure, axes = plt.subplots(2, 2, figsize=(16.0, 9.0), dpi=100)
    figure.subplots_adjust(left=0.05, right=0.985, bottom=0.08, top=0.90, hspace=0.42, wspace=0.10)
    figure.suptitle(
        f"{event_label} | same prefix sample and exogenous seed | counterfactual: no highD future target",
        fontsize=10,
    )
    with imageio.get_writer(path, mode="I", duration=DT_S * frame_stride * 1000.0, loop=0) as writer:
        for frame in frames:
            _draw_probe_world(
                axes[0, 0], rollout=common, valid=valid, follower=follower, frame=int(frame),
                label=baseline_label, affected_color="#ffbf00",
                status=f"forced ego brake: -8 m/s² at 1.00–2.00 s | rear ax={common_ax[frame]:+.2f} m/s²",
            )
            _draw_probe_world(
                axes[0, 1], rollout=candidate, valid=valid, follower=follower, frame=int(frame),
                label=candidate_label, affected_color=COMMON_COLOR,
                status=(f"learned authority: {int(candidate['active'][frame].sum())}/6 | "
                        f"rear ax={candidate_ax[frame]:+.2f} m/s²"),
            )
            axes[1, 0].clear()
            axes[1, 0].plot(time_s[: frame + 1], common_ax[: frame + 1], color="#ffbf00", label=f"{baseline_label} rear ax")
            axes[1, 0].plot(time_s[: frame + 1], candidate_ax[: frame + 1], color=COMMON_COLOR, label=f"{candidate_label} rear ax")
            axes[1, 0].axvspan(1.0, 2.0, color="#ef9a9a", alpha=0.3, label="ADS brake window")
            axes[1, 0].set(title="Affected rear NPC longitudinal action", xlabel="time [s]", ylabel="ax [m/s²]", xlim=(0.0, time_s[-1]), ylim=(-8.2, 4.2))
            axes[1, 0].legend(frameon=True, fontsize=7, loc="lower left")
            axes[1, 0].grid(alpha=0.3)
            axes[1, 1].clear()
            axes[1, 1].plot(time_s[: frame + 1], common_gap[: frame + 1], color="#ffbf00", label=f"{baseline_label} gap")
            axes[1, 1].plot(time_s[: frame + 1], candidate_gap[: frame + 1], color=COMMON_COLOR, label=f"{candidate_label} gap")
            axes[1, 1].axvspan(1.0, 2.0, color="#ef9a9a", alpha=0.3, label="ADS brake window")
            axes[1, 1].set(title="Ego → affected rear-NPC longitudinal gap", xlabel="time [s]", ylabel="gap [m]", xlim=(0.0, time_s[-1]))
            axes[1, 1].legend(frameon=True, fontsize=7, loc="upper right")
            axes[1, 1].grid(alpha=0.3)
            figure.canvas.draw()
            writer.append_data(np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(figure)
    return int(len(frames))


def _write_gif(path: Path, draw_frame, steps: int, frame_stride: int) -> int:
    plt = get_pyplot()
    frames = np.arange(0, steps, frame_stride)
    if frames[-1] != steps - 1:
        frames = np.append(frames, steps - 1)
    figure, axis = plt.subplots(figsize=(12.0, 4.8), dpi=100)
    figure.subplots_adjust(left=0.065, right=0.965, bottom=0.18, top=0.83)
    with imageio.get_writer(path, mode="I", duration=DT_S * frame_stride * 1000.0, loop=0) as writer:
        for frame in frames:
            draw_frame(axis, int(frame))
            figure.canvas.draw()
            writer.append_data(np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(figure)
    return int(len(frames))


def _rollout_summary(rollout: dict[str, np.ndarray]) -> dict[str, float | bool]:
    return {
        "controller_active_rate": float(rollout["active"].mean()),
        "any_collision": bool(rollout["collisions"].any()),
        "minimum_background_acceleration_mps2": float(rollout["actions"][..., 0].min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--event-index", type=int, default=901)
    parser.add_argument("--steps", type=int, default=125)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 2 or args.steps > 149:
        raise ValueError("--steps must be in 2..149")

    response, base = load_response_config()
    output = args.output or result_directory(response) / "evaluation/playbacks"
    checkpoint = args.checkpoint or ROOT / response["paths"]["checkpoint"]
    candidate_label, baseline_label = "PPO + IDM response", "frozen HiQR baseline"
    output.mkdir(parents=True, exist_ok=True)
    data = prepare_experiment_data(base, ROOT)
    event_reference = ReactionEventReference.load(event_directory(response) / args.split)
    if args.event_index < 0 or args.event_index >= len(event_reference.events.row_index):
        raise ValueError(f"event index {args.event_index} is outside split {args.split}")
    event = event_reference.events
    row, onset = int(event.row_index[args.event_index]), int(event.local_onset_frame[args.event_index])
    arrays = data.bundle.arrays
    available_steps = int(len(np.asarray(arrays["agent_states"])[row]) - onset - 1)
    steps = min(args.steps, available_steps)
    if steps < 2:
        raise ValueError("selected event has fewer than two observable post-onset frames")
    logged = np.asarray(arrays["agent_states"])[row, onset + 1 : onset + steps + 1]
    valid = np.asarray(arrays["agent_valid"])[row, onset]
    logged_ego = _logged_actions(np.asarray(arrays["agent_states"])[row], onset, steps)
    probe_ego = logged_ego.copy()
    probe_ego[25:50, 0] -= 8.0

    sampler = _load_sampler(base)
    rule = RuleModelBundle.load(ROOT / response["paths"]["rule_model"])
    payload = torch.load(checkpoint, map_location="cuda", weights_only=False)
    if payload.get("controller_mode") != response["model"]["controller_mode"]:
        raise ValueError("--checkpoint is not a PPO + IDM response controller")
    candidate = IDMResidualReactionController(rule).to("cuda")
    candidate.load_state_dict(payload["state_dict"], strict=True)
    common = NoReactionController().to("cuda").eval()
    candidate.eval()
    natural = _simulate(
        sampler=sampler, controller=candidate, bundle=data.bundle,
        reference=event_reference, event_index=args.event_index, ego_actions=logged_ego,
    )
    candidate_probe = _simulate(
        sampler=sampler, controller=candidate, bundle=data.bundle,
        reference=event_reference, event_index=args.event_index, ego_actions=probe_ego,
    )
    common_probe = _simulate(
        sampler=sampler, controller=common, bundle=data.bundle,
        reference=event_reference, event_index=args.event_index, ego_actions=probe_ego,
    )
    slots = np.flatnonzero(valid[1:]).astype(int)
    if not len(slots):
        raise ValueError("selected event has no active background slots")
    natural_error = np.linalg.norm(natural["states"][:, 1:, :2] - logged[:, 1:, :2], axis=-1)
    natural_focus = int(slots[np.argmax(natural_error[:, slots].mean(axis=0))])
    follower = int(event.follower_slot[args.event_index])
    if follower < 1 or follower > 6 or not valid[follower]:
        raise ValueError("selected event has no valid same-rear follower slot")
    event_label = f"highD {args.split} event {args.event_index} (row {row})"

    natural_path = output / "natural_replay.gif"
    natural_frames = _write_gif(
        natural_path,
        lambda axis, frame: _draw_natural_frame(
            axis, rollout=natural, reference=logged, valid=valid,
            focus_slot=natural_focus, frame=frame, event_label=event_label, candidate_label=candidate_label,
        ),
        steps, args.frame_stride,
    )
    probe_path = output / "braking_probe.gif"
    probe_frames = _write_probe_gif(
        probe_path, candidate=candidate_probe, common=common_probe, valid=valid,
        follower=follower, event_label=event_label, frame_stride=args.frame_stride,
        candidate_label=candidate_label, baseline_label=baseline_label,
    )
    manifest = {
        "role": "hierarchical response-controller visual diagnostics",
        "candidate": candidate_label,
        "baseline": baseline_label,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "event": {
            "split": args.split,
            "event_index": args.event_index,
            "row_index": row,
            "recording": int(event.recording_id[args.event_index]),
            "local_onset_frame": onset,
            "requested_steps": args.steps,
            "rendered_steps": steps,
        },
        "common_randomness": "same prefix sample and exogenous seed per controller comparison",
        "natural_replay": {
            "gif": natural_path.name,
            "frames": natural_frames,
            "ego_command": "logged highD ego controls",
            "reference_overlay": "dotted highD trajectories",
            "focus_slot": natural_focus + 1,
        },
        "braking_probe": {
            "gif": probe_path.name,
            "frames": probe_frames,
            "ego_command": "logged highD controls plus -8 m/s² from 1.00 s to 2.00 s",
            "comparison": f"{candidate_label} versus {baseline_label}",
            "affected_same_rear_slot": follower,
            "ppo_idm_response": _rollout_summary(candidate_probe),
            "frozen_hiqr": _rollout_summary(common_probe),
            "mean_absolute_background_action_difference": float(
                np.abs(candidate_probe["actions"] - common_probe["actions"]).mean()
            ),
            "affected_rear": {
                "ppo_idm_minimum_acceleration_mps2": float(candidate_probe["actions"][:, follower - 1, 0].min()),
                "frozen_hiqr_minimum_acceleration_mps2": float(common_probe["actions"][:, follower - 1, 0].min()),
                "ppo_idm_mean_acceleration_in_brake_window_mps2": float(candidate_probe["actions"][25:50, follower - 1, 0].mean()),
                "frozen_hiqr_mean_acceleration_in_brake_window_mps2": float(common_probe["actions"][25:50, follower - 1, 0].mean()),
            },
        },
        "warning": "These GIFs diagnose one selected event; they do not establish safety, risk reduction, or aggregate performance.",
    }
    (output / "playback_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
