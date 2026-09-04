#!/usr/bin/env python3
"""Render factual and four-arm causal HighwayEnv playbacks with one contract.

The two animations deliberately answer different questions:
1. factual: can frozen Flow--Diffusion--HiQR (A0) stay close to the observed
   highD future under logged ego replay?
2. counterfactual: after the same *executed* ADS brake, do A1/A2/A3 modify
   only the affected rear NPC, and do they sustain that response after the
   command window ends?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import ANCHOR_INDEX
from hierarchical_world_model.src.data import prepare_experiment_data
from hierarchical_world_model.src.human_prior import HumanActionPrior
from hierarchical_world_model.src.influence_graph import dynamic_candidate_scene_mask
from hierarchical_world_model.src.planner import complete_missing_background_plans, frozen_diffusion_plans
from hierarchical_world_model.src.protocol import load_protocol_config
from hierarchical_world_model.src.reaction_controller import IDMResidualReactionController, RLResidualReactionController
from hierarchical_world_model.src.reaction_ppo import highway_controller_rollout
from hierarchical_world_model.src.rule_models import RuleModelBundle
from hierarchical_world_model.src.train import load_checkpoint
from world_model.src.core.utils import ensure_dir, save_json, select_device


DEFAULT = ROOT / "hierarchical_world_model/config/reaction_naturalistic.yaml"
# Vehicles are initialized on 3.6 m lane *centres*.  Draw the road boundaries
# half a lane away, never dashed lines through vehicle centres.
ROAD_EDGES = (-9.0, 9.0)
LANE_BOUNDARIES = (-5.4, -1.8, 1.8, 5.4)
COLORS = {"A0": "#3274a1", "A1": "#6a3d9a", "A2": "#2ca02c", "A3": "#d95f02"}


def _controller(arm: str, config: dict, device: torch.device, root: Path | None = None,
                checkpoint: Path | None = None, human_prior: Path | None = None):
    if arm == "A0":
        return "none"
    root = Path(config["paths"]["output_dir"]) if root is None else root
    payload_path = checkpoint if checkpoint is not None else root / "controllers" / {"A1": "rl_residual", "A2": "rl_residual_idm", "A3": "rl_residual_gail"}[arm] / "reaction_ppo.pt"
    payload = torch.load(payload_path, map_location=device, weights_only=False)
    expected = "reaction_residual_ppo_dynamic_v3" if arm == "A3" else "reaction_residual_ppo_dynamic_v2"
    if payload.get("schema") != expected:
        raise ValueError(f"{arm} requires a dynamic-scope controller checkpoint")
    if arm == "A1":
        value = RLResidualReactionController().to(device)
    else:
        rule = RuleModelBundle.load(config["paths"]["rule_model"])
        prior = None
        if arm == "A3":
            prior_path = human_prior if human_prior is not None else Path(config["paths"]["human_prior"])
            checkpoint = torch.load(prior_path, map_location=device, weights_only=False)
            if checkpoint.get("schema") != "longitudinal_gail_human_prior_v4":
                raise ValueError("A3 playback requires HumanActionPriorV4")
            prior = HumanActionPrior().to(device); prior.load_state_dict(checkpoint["state_dict"]); prior.eval()
        value = IDMResidualReactionController(rule, prior).to(device)
    value.load_state_dict(payload["state_dict"], strict=(arm != "A2")); value.eval()
    return value


def _authority(config: dict) -> dict:
    train = config["training"]
    return {
        "reaction_min_frames": int(train["reaction_min_frames"]),
        "reaction_max_frames": int(train["reaction_max_frames"]),
        "reaction_recovery_frames": int(train["reaction_recovery_frames"]),
        "reaction_safety_ttc_s": float(train["safety_ttc_s"]),
        "reaction_release_ttc_s": float(train.get("reaction_release_ttc_s", 4.0)),
        "influence_radius_m": float(train.get("influence_radius_m", 50.0)),
        "influence_secondary_radius_m": float(train.get("influence_secondary_radius_m", 35.0)),
        "influence_prediction_horizon_s": float(train.get("influence_prediction_horizon_s", 1.5)),
        "influence_stable_release_frames": int(train.get("influence_stable_release_frames", 13)),
    }


def _draw(axis, state: np.ndarray, valid: np.ndarray, frame: int, title: str, *, diagnostic=None, influenced=None) -> None:
    axis.clear(); axis.set_facecolor("#595d62")
    ego = state[0]
    for edge in ROAD_EDGES:
        axis.axhline(edge, color="white", alpha=.62, linewidth=1.0, linestyle="-")
    for boundary in LANE_BOUNDARIES:
        axis.axhline(boundary, color="white", alpha=.52, linewidth=.8, linestyle="--")
    for slot in np.flatnonzero(valid):
        vehicle = state[slot]
        x, y = vehicle[0] - ego[0], vehicle[1] - ego[1]
        if slot == 0:
            color, label, alpha = "#d62728", "ADS ego", .95
        elif influenced is not None and bool(influenced[slot - 1]):
            color, label, alpha = "#2ca02c", "dynamically influenced NPC", .95
        else:
            color, label, alpha = "#3274a1", "unaffected NPC", .55
        yaw = np.degrees(np.arctan2(vehicle[3], vehicle[2])) if abs(vehicle[2]) + abs(vehicle[3]) > 1.e-4 else 0.
        axis.add_patch(Rectangle((x - 2.25, y - .9), 4.5, 1.8, angle=yaw, facecolor=color,
                                 edgecolor="#202020", linewidth=.6, alpha=alpha,
                                 label=label if slot in (0, 2, 1) else None))
    axis.set(xlim=(-55., 55.), ylim=(-10.8, 10.8), aspect="equal", xlabel="ego-relative x [m]",
             ylabel="ego-relative y [m]", title=f"{title}\nt={frame * .04:.2f}s")
    if diagnostic:
        axis.text(.01, .98, diagnostic, transform=axis.transAxes, va="top", fontsize=7.2,
                  bbox={"facecolor": "white", "alpha": .82, "edgecolor": "none"})
    axis.tick_params(labelsize=7)


def _frames(writer, figure, stride: int):
    for frame in range(0, 149, stride):
        yield frame
    if 148 % stride:
        yield 148


def _factual_gif(path: Path, highd: np.ndarray, highd_valid: np.ndarray, a0, *, stride: int, focus_slot: int) -> dict[str, float]:
    compare = np.linalg.norm(a0.states[:, :, 1:, :2] - highd[None, :, 1:, :2], axis=-1)
    mask = highd_valid[None, :, 1:]
    background_ade = float(compare[mask].mean()) if mask.any() else float("nan")
    focus_mask = highd_valid[None, :, focus_slot + 1]
    focus_ade = float(compare[:, :, focus_slot][focus_mask].mean()) if focus_mask.any() else float("nan")
    figure = plt.figure(figsize=(13.4, 6.9), dpi=100, constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.5, 1.0))
    ref_axis, model_axis = figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])
    gap_axis, error_axis = figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])
    with imageio.get_writer(path, mode="I", duration=.08, loop=0) as writer:
        for frame in _frames(writer, figure, stride):
            _draw(ref_axis, highd[frame], highd_valid[frame], frame, "Observed highD factual future")
            _draw(model_axis, a0.states[0, frame], highd_valid[frame], frame,
                  "Frozen Flow–Diffusion–HiQR + HighwayEnv (A0)",
                  diagnostic=(f"factual replay\nall-background ADE={background_ade:.2f} m\n"
                              f"focus NPC ADE={focus_ade:.2f} m"))
            time = (np.arange(frame + 1) + 1) * .04
            gap_axis.clear(); error_axis.clear()
            gap_axis.plot(time, highd[:frame + 1, 0, 0] - highd[:frame + 1, focus_slot + 1, 0], color="#202020", label="highD ego–focus Δx")
            gap_axis.plot(time, a0.states[0, :frame + 1, 0, 0] - a0.states[0, :frame + 1, focus_slot + 1, 0], color=COLORS["A0"], label="A0 ego–focus Δx")
            error_axis.plot(time, compare[0, :frame + 1, focus_slot], color="#2ca02c", label="focus NPC position error")
            for axis in (gap_axis, error_axis):
                axis.grid(alpha=.25); axis.legend(fontsize=7); axis.set(xlim=(0., 6.0)); axis.tick_params(labelsize=7)
            gap_axis.set(title="Factual ego–focus longitudinal separation", xlabel="time [s]", ylabel="Δx [m]")
            error_axis.set(title="Closed-loop reconstruction error", xlabel="time [s]", ylabel="position error [m]")
            figure.canvas.draw(); writer.append_data(np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(figure)
    return {"all_background_position_ADE_m": background_ade, "focus_npc_position_ADE_m": focus_ade}


def _counterfactual_gif(path: Path, rollouts: dict, visible: np.ndarray, *, stride: int, dose: float, duration: int) -> dict:
    # Two road panels per row preserve the readable road-scale visual
    # contract used by the retained IDM-subset playbacks; four panels in one
    # row make the 3--4 lane geometry too thin to inspect.
    figure = plt.figure(figsize=(16.0, 10.4), dpi=100, constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=(1.05, 1.05, 1.0))
    roads = [figure.add_subplot(grid[index // 2, index % 2]) for index in range(4)]
    action_axis, gap_axis = figure.add_subplot(grid[2, 0]), figure.add_subplot(grid[2, 1])
    label = {"A0": "A0 frozen HiQR", "A1": "A1 pure Residual PPO", "A2": "A2 PPO + IDM", "A3": "A3 PPO + IDM + GAIL"}
    focus = int(np.argmax(rollouts["A3"].controller_diagnostics["influence_authority"][0].sum(0)))
    figure.suptitle("Same highD initial state, frozen soft plan and random stream. After the red window, no highD future is a counterfactual ground truth.", fontsize=9)
    with imageio.get_writer(path, mode="I", duration=.08, loop=0) as writer:
        for frame in _frames(writer, figure, stride):
            for axis, arm in zip(roads, ("A0", "A1", "A2", "A3")):
                item = rollouts[arm]; diag = item.controller_diagnostics
                influenced = diag["influence_authority"][0, frame] > 0.
                slots = (np.flatnonzero(influenced) + 1).tolist()
                message = "counterfactual: forced ADS brake" if arm == "A0" else (
                    f"influenced NPC slots={slots}\n"
                    f"focus role={int(diag['influence_role'][0, frame, focus])}, phase={int(diag['phase'][0, frame, focus])}\n"
                    f"α={diag['alpha'][0, frame, focus]:.2f}; ax "
                    f"{item.base_background_actions[0, frame, focus, 0]:+.2f}→{item.background_actions[0, frame, focus, 0]:+.2f}"
                )
                _draw(axis, item.states[0, frame], visible, frame, label[arm], diagnostic=message, influenced=influenced)
            time = (np.arange(frame + 1) + 1) * .04
            action_axis.clear(); gap_axis.clear()
            for arm, item in rollouts.items():
                action_axis.plot(time, item.background_actions[0, :frame + 1, focus, 0], color=COLORS[arm], linewidth=1.7, label=f"{arm} focus ax")
                desired = item.controller_diagnostics.get("desired_action_ax")
                if arm != "A0" and desired is not None:
                    action_axis.plot(time, desired[0, :frame + 1, focus], color=COLORS[arm], linewidth=1.0,
                                     linestyle=":", alpha=.85, label=f"{arm} desired")
                parent = np.maximum(item.controller_diagnostics["influence_parent"][0, :frame + 1, focus], 0)
                child_x = item.states[0, :frame + 1, focus + 1, 0]
                parent_x = item.states[0, np.arange(frame + 1), parent, 0]
                gap_axis.plot(time, parent_x - child_x - 4.8, color=COLORS[arm], linewidth=1.7, label=f"{arm} parent–focus gap")
            for axis in (action_axis, gap_axis):
                axis.axvspan(1.0, 1.0 + duration * .04, color="#d62728", alpha=.12, label="ADS brake command")
                axis.axvline(time[-1], color="#202020", linewidth=.7); axis.grid(alpha=.25); axis.legend(ncol=2, fontsize=7); axis.set(xlim=(0., 6.0)); axis.tick_params(labelsize=7)
            action_axis.set(title="Focused influenced NPC acceleration (solid=executed, dotted=target)", xlabel="time [s]", ylabel="ax [m/s²]", ylim=(-8.4, 4.4))
            gap_axis.set(title="Causal parent → focused NPC gap", xlabel="time [s]", ylabel="gap [m]")
            figure.canvas.draw(); writer.append_data(np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(figure)
    result = {}
    for arm, item in rollouts.items():
        influence = item.controller_diagnostics["influence_authority"][0] > 0.
        gaps = item.controller_diagnostics["influence_predicted_min_gap_m"][0][influence]
        result[arm] = {
            "influenced_npc_collision": bool((item.crashed[0, :, 1:] & (item.controller_diagnostics["influence_authority"][0] > 0.)).any()),
            "minimum_predicted_gap_m": float(np.min(gaps)) if len(gaps) else None,
            "active_frames": int(item.controller_diagnostics["active"][0].sum()),
            "focus_slot": int(focus + 1),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Render factual and A0–A3 causal comparison GIFs.")
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--row", type=int, default=64622)
    parser.add_argument("--dose", type=float, default=8.0)
    parser.add_argument("--duration-frames", type=int, default=25)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--artifact-dir", type=Path, default=None,
                        help="candidate PPO artifact root")
    parser.add_argument("--human-prior", type=Path, default=None)
    args = parser.parse_args()
    config = load_protocol_config(args.config.resolve()); base = load_protocol_config(ROOT / config.get("base_config", "hierarchical_world_model/config/release.yaml"))
    if args.human_prior is not None:
        config["paths"]["human_prior"] = str(args.human_prior)
    device = select_device(config["training"].get("device", "auto")); model, _ = load_checkpoint(base["paths"]["evaluation_checkpoint"], device=device)
    experiment = prepare_experiment_data(base, ROOT); arrays = experiment.bundle.arrays
    if int(args.row) not in set(experiment.test_rows.tolist()):
        raise ValueError("row must be a record-isolated highD test row")
    index = int(np.flatnonzero(experiment.test_rows == int(args.row))[0]); row = experiment.test_rows[index:index + 1]
    if not dynamic_candidate_scene_mask(arrays["agent_states"], arrays["agent_valid"], rows=row)[0]:
        raise ValueError("row is not in the dynamic causal-influence test subset")
    artifact_root = Path(config["paths"]["output_dir"]) if args.artifact_dir is None else args.artifact_dir
    output = ensure_dir(artifact_root / "visualization" / "comparative_playbacks")
    plans = frozen_diffusion_plans(experiment.bundle, row, checkpoint=base["paths"]["diffusion_checkpoint"], output_dir=output / "frozen_plans", device=device, batch_size=int(base["training"]["validation_batch_size"]), ddim_steps=int(config["training"]["diffusion_ddim_steps"]), experiment_scope=base["training"].get("experiment_scope", "full"))
    states, valid = arrays["agent_states"][row], arrays["agent_valid"][row]
    plans = complete_missing_background_plans(plans, states, valid); maps, map_valid = arrays["map_polylines"][row], arrays["map_polyline_valid"][row]
    authority = _authority(config)
    factual = highway_controller_rollout(model, states=states, valid=valid, soft_plans=plans, maps=maps, map_valid=map_valid, controller="none", device=device, motion_seed=20260902, **authority)
    rollouts = {arm: highway_controller_rollout(model, states=states, valid=valid, soft_plans=plans, maps=maps, map_valid=map_valid, controller=_controller(arm, config, device, artifact_root), device=device, motion_seed=20260902, intervention="brake", dose=float(args.dose), intervention_duration_frames=int(args.duration_frames), **authority) for arm in ("A0", "A1", "A2", "A3")}
    highd, highd_valid = states[0, ANCHOR_INDEX + 1:ANCHOR_INDEX + 150], valid[0, ANCHOR_INDEX + 1:ANCHOR_INDEX + 150]
    anchor_state = states[0, ANCHOR_INDEX]
    anchor_valid = valid[0, ANCHOR_INDEX, 1:]
    focus_slot = int(np.argmin(np.where(anchor_valid, np.linalg.norm(anchor_state[1:, :2] - anchor_state[:1, :2], axis=-1), np.inf)))
    factual_path = output / f"test_row_{int(args.row)}_factual_highd_vs_a0.gif"
    counterfactual_path = output / f"test_row_{int(args.row)}_counterfactual_a0_a3_brake_{int(args.dose)}.gif"
    factual_ade = _factual_gif(factual_path, highd, highd_valid, factual, stride=max(1, int(args.frame_stride)), focus_slot=focus_slot)
    counterfactual = _counterfactual_gif(counterfactual_path, rollouts, valid[0, ANCHOR_INDEX], stride=max(1, int(args.frame_stride)), dose=float(args.dose), duration=int(args.duration_frames))
    manifest = {
        "schema": "highwayenv_dynamic_factual_and_counterfactual_playbacks_v2", "row": int(args.row), "split": "highD record-isolated test", "backend": "HighwayEnvClosedLoopWorld", "common_random_numbers": True,
        "selection": "row belongs to the complete dynamic-eligible subset of the 10,151-sequence highD record-isolated test split", "factual_gif": str(factual_path), "counterfactual_gif": str(counterfactual_path),
        "factual_position_error_m": factual_ade, "intervention": {"start_frame": 25, "dose_mps2": float(args.dose), "duration_frames": int(args.duration_frames)}, "counterfactual_summary": counterfactual,
        "interpretation": "The factual GIF compares highD with A0 under logged ego replay. The four-arm GIF is a counterfactual: after forced ego braking, highD future is not a target; compare causal response, locality, safety and smoothness across arms instead.",
    }
    save_json(manifest, output / f"test_row_{int(args.row)}_playback_manifest.json")
    print(factual_path); print(counterfactual_path)


if __name__ == "__main__":
    main()
