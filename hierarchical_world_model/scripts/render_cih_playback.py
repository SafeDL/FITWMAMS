#!/usr/bin/env python3
"""Render one matched highD / factual-transition / CIH-WM response playback.

The GIF is a qualitative diagnostic.  It does not constitute a factual or
causal acceptance result; the selected CIH-WM checkpoint is intentionally
labelled as unaccepted until the complete validation protocol passes.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import ANCHOR_INDEX  # noqa: E402
from hierarchical_world_model.src.cih_model import (  # noqa: E402
    CausalInfluenceHierarchicalWorldModel,
    load_cih_method_config,
)
from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.evaluation import rollout  # noqa: E402
from hierarchical_world_model.src.planner import (  # noqa: E402
    complete_endogenous_response_plans,
    frozen_diffusion_plans,
)
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from hierarchical_world_model.src.reaction_evidence import ReactionEventReference  # noqa: E402
from hierarchical_world_model.src.visualization import (  # noqa: E402
    DIFFUSION_COLOR,
    EGO_COLOR,
    LOGGED_REFERENCE_COLOR,
    ROAD_COLOR,
    _draw_lane_markings,
    _draw_vehicle,
)
from tools.plot_style import get_pyplot  # noqa: E402
from world_model.src.core.utils import select_device  # noqa: E402

DT_S = 0.04
CIH_COLOR = "#7e22ce"


def _event_row(reference: ReactionEventReference, requested: int | None) -> int:
    events = reference.events
    eligible = events.indices(reference.supported_cells)
    eligible = eligible[events.leader_slot[eligible] == 0]
    if not len(eligible):
        raise RuntimeError("no supported ego-led follower response event is available")
    if requested is not None:
        matches = eligible[events.row_index[eligible] == requested]
        if not len(matches):
            raise ValueError("--row is not a supported ego-led response event in this split")
        return int(requested)
    # Evidence is deterministically ordered by recording/event; choosing its
    # median avoids a manually selected showcase while remaining reproducible.
    return int(events.row_index[eligible[len(eligible) // 2]])


def _receiver_slot(active: np.ndarray, factual: np.ndarray, cih: np.ndarray, valid: np.ndarray) -> int:
    active_count = active.sum(axis=0)
    valid_slots = np.flatnonzero(valid[1:])
    if len(valid_slots) == 0:
        raise RuntimeError("selected row has no valid background traffic slots")
    if active_count.max() > 0:
        return int(np.argmax(active_count))
    delta = np.linalg.norm(cih[:, 1:, :2] - factual[:, 1:, :2], axis=-1).sum(axis=0)
    delta[~valid[1:]] = -np.inf
    return int(np.argmax(delta))


def _draw_panel(
    axis, *, states: np.ndarray, logged: np.ndarray, valid: np.ndarray, frame: int,
    color: str, label: str, receiver: int, active_count: int | None,
) -> None:
    axis.clear()
    axis.set_facecolor(ROAD_COLOR)
    _draw_lane_markings(axis)
    center = float(0.5 * (logged[frame, 0, 0] + states[frame, receiver + 1, 0]))
    axis.set(xlim=(center - 65.0, center + 65.0), ylim=(-8.2, 8.2), aspect="equal")
    trail = slice(max(0, frame - 45), frame + 1)
    axis.plot(logged[trail, 0, 0], logged[trail, 0, 1], color=EGO_COLOR, linewidth=1.8)
    for slot in np.flatnonzero(valid[1:]):
        agent = int(slot) + 1
        axis.plot(logged[trail, agent, 0], logged[trail, agent, 1], color="#d9d9d9", linestyle=":", linewidth=1.0)
        axis.plot(states[trail, agent, 0], states[trail, agent, 1], color=color, linewidth=1.35, alpha=.9)
        _draw_vehicle(axis, logged[frame, agent], color=LOGGED_REFERENCE_COLOR,
                      label="highD reference" if slot == receiver else None, filled=False, alpha=.9)
        _draw_vehicle(axis, states[frame, agent], color=color,
                      label=f"{label} receiver" if slot == receiver else None, filled=True, alpha=.58)
    _draw_vehicle(axis, logged[frame, 0], color=EGO_COLOR, label="logged ego", filled=True, alpha=.9)
    status = label if active_count is None else f"{label} | controller-active slots: {active_count}/6"
    axis.set_title(status, fontsize=9)
    axis.text(.01, .02, "red: logged ego | white: highD | coloured: model rollout",
              transform=axis.transAxes, fontsize=7, va="bottom", ha="left",
              bbox={"facecolor": "white", "alpha": .78, "edgecolor": "none", "pad": 1.2})
    axis.tick_params(labelsize=7)


def _draw_counterfactual_world(
    axis, *, states: np.ndarray, valid: np.ndarray, frame: int, receiver: int,
    color: str, label: str, active_count: int | None,
) -> None:
    """Draw one synthetic-braking world; no highD future is overlaid."""
    axis.clear()
    axis.set_facecolor(ROAD_COLOR)
    _draw_lane_markings(axis)
    center = float(states[frame, 0, 0])
    axis.set(xlim=(center - 65.0, center + 65.0), ylim=(-8.2, 8.2), aspect="equal")
    trail = slice(max(0, frame - 45), frame + 1)
    for slot in np.flatnonzero(valid[1:]):
        agent = int(slot) + 1
        slot_color = color if slot == receiver else "#d9d9d9"
        axis.plot(states[trail, agent, 0], states[trail, agent, 1], color=slot_color, linewidth=1.35)
        _draw_vehicle(axis, states[frame, agent], color=slot_color,
                      label="affected follower" if slot == receiver else None,
                      filled=slot == receiver, alpha=.72)
    _draw_vehicle(axis, states[frame, 0], color=EGO_COLOR, label="ego: added braking", filled=True, alpha=.9)
    status = label if active_count is None else f"{label} | controller-active slots: {active_count}/6"
    axis.set_title(status, fontsize=9)
    axis.text(.01, .02, "counterfactual rollout: no highD future target after the added brake",
              transform=axis.transAxes, fontsize=7, va="bottom", ha="left",
              bbox={"facecolor": "white", "alpha": .78, "edgecolor": "none", "pad": 1.2})
    axis.tick_params(labelsize=7)


def _write_braking_probe(
    output: Path, *, baseline, cih, valid: np.ndarray, receiver: int, row: int,
    dose: float, frame_stride: int,
) -> tuple[int, dict[str, float | int]]:
    """Render the historical A0/A3-style comparison under current CIH-WM dose."""
    baseline_full = np.concatenate((baseline.states[0, :1], baseline.states[0]), axis=0)
    cih_full = np.concatenate((cih.states[0, :1], cih.states[0]), axis=0)
    active = np.concatenate((np.zeros((1, 6), bool), cih.controller_diagnostics["active"][0].astype(bool)))
    baseline_action = baseline.background_actions[0, :, receiver, 0]
    cih_action = cih.background_actions[0, :, receiver, 0]
    baseline_gap = baseline.states[0, :, 0, 0] - baseline.states[0, :, receiver + 1, 0] - 4.8
    cih_gap = cih.states[0, :, 0, 0] - cih.states[0, :, receiver + 1, 0] - 4.8
    frames = np.arange(0, len(baseline.states[0]), frame_stride)
    if frames[-1] != len(baseline.states[0]) - 1:
        frames = np.append(frames, len(baseline.states[0]) - 1)
    time_s = np.arange(len(baseline.states[0])) * DT_S
    plt = get_pyplot()
    with imageio.get_writer(output, mode="I", duration=DT_S * frame_stride * 1000, loop=0) as writer:
        figure, axes = plt.subplots(2, 2, figsize=(15, 8), dpi=100)
        figure.subplots_adjust(left=.045, right=.99, bottom=.08, top=.87, hspace=.36, wspace=.08)
        for frame in frames:
            figure.suptitle(
                f"CIH-WM counterfactual braking probe | validation row {row} | t={frame * DT_S:.2f}s\n"
                f"Logged ego control + {dose:g} m/s² braking during 1.00–2.00 s; matched plan and initial state.",
                fontsize=10, fontweight="bold",
            )
            _draw_counterfactual_world(axes[0, 0], states=baseline_full, valid=valid,
                                       frame=int(frame), receiver=receiver, color="#e69f00",
                                       label="Frozen factual transition (no CIH controller)", active_count=None)
            _draw_counterfactual_world(axes[0, 1], states=cih_full, valid=valid,
                                       frame=int(frame), receiver=receiver, color=CIH_COLOR,
                                       label="CIH-WM candidate", active_count=int(active[frame].sum()))
            axes[1, 0].clear()
            axes[1, 0].plot(time_s[:frame + 1], baseline_action[:frame + 1], color="#e69f00", label="frozen factual transition")
            axes[1, 0].plot(time_s[:frame + 1], cih_action[:frame + 1], color=CIH_COLOR, label="CIH-WM")
            axes[1, 0].axvspan(1., 2., color="#ef9a9a", alpha=.3, label="ego brake window")
            axes[1, 0].set(title="Affected follower longitudinal action", xlabel="time [s]", ylabel="acceleration [m/s²]", xlim=(0., time_s[-1]))
            axes[1, 0].grid(alpha=.3); axes[1, 0].legend(fontsize=7, loc="lower left")
            axes[1, 1].clear()
            axes[1, 1].plot(time_s[:frame + 1], baseline_gap[:frame + 1], color="#e69f00", label="frozen factual transition")
            axes[1, 1].plot(time_s[:frame + 1], cih_gap[:frame + 1], color=CIH_COLOR, label="CIH-WM")
            axes[1, 1].axvspan(1., 2., color="#ef9a9a", alpha=.3, label="ego brake window")
            axes[1, 1].set(title="Ego → affected follower gap", xlabel="time [s]", ylabel="gap [m]", xlim=(0., time_s[-1]))
            axes[1, 1].grid(alpha=.3); axes[1, 1].legend(fontsize=7, loc="best")
            figure.canvas.draw()
            writer.append_data(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())
        plt.close(figure)
    summary = {
        "receiver_slot": receiver + 1,
        "controller_active_frames": int(active[:, receiver].sum()),
        "minimum_gap_factual_transition_m": float(baseline_gap.min()),
        "minimum_gap_cih_wm_m": float(cih_gap.min()),
        "mean_abs_receiver_action_difference_mps2": float(np.abs(cih_action - baseline_action).mean()),
    }
    return int(len(frames)), summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--row", type=int, default=None, help="supported ego-led event row; default is deterministic median")
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--probe-dose", type=float, default=3.0,
                        help="additional ego braking magnitude in m/s²; 3.0 is the formal strong dose")
    parser.add_argument("--candidate", type=Path,
                        default=ROOT / "results/hierarchical_world_model/cih_wm/candidate_unaccepted/response_policy.pt")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be positive")

    world = load_protocol_config(ROOT / "hierarchical_world_model/config/world_model.yaml")
    cih_config = load_cih_method_config(ROOT / "hierarchical_world_model/config/cih_world_model.yaml")
    device = select_device(world["training"].get("device", "auto"))
    experiment = prepare_experiment_data(world, ROOT)
    reference = ReactionEventReference.load(
        ROOT / cih_config["paths"]["event_reference"] / args.split
    )
    row = _event_row(reference, args.row)
    full_states = np.asarray(experiment.bundle.arrays["agent_states"][row:row + 1], np.float32)
    full_valid = np.asarray(experiment.bundle.arrays["agent_valid"][row:row + 1], bool)
    maps = np.asarray(experiment.bundle.arrays["map_polylines"][row:row + 1], np.float32)
    map_valid = np.asarray(experiment.bundle.arrays["map_polyline_valid"][row:row + 1], bool)
    method, _ = CausalInfluenceHierarchicalWorldModel.load(
        cih_config, root=ROOT, device=device, response_checkpoint=args.candidate
    )
    with tempfile.TemporaryDirectory(prefix="cih_playback_") as cache:
        plans = frozen_diffusion_plans(
            experiment.bundle, np.asarray([row]), checkpoint=world["paths"]["diffusion_checkpoint"],
            output_dir=cache, device=device, batch_size=1, ddim_steps=20,
            experiment_scope=str(world["training"].get("experiment_scope", "full")),
        )
        response_plans = complete_endogenous_response_plans(plans, full_states, full_valid)
        factual = rollout(method.factual_dynamics, full_states, full_valid, plans, maps, map_valid,
                          device=device, history_frames=25, motion_seed=None, excluded_slots=())
        cih = rollout(method.factual_dynamics, full_states, full_valid, response_plans, maps, map_valid,
                      device=device, history_frames=25, motion_seed=None, controller=method.response_policy,
                      controller_deterministic=True, excluded_slots=(),
                      influence_graph_config=method.influence_graph_config)
    if cih.controller_diagnostics is None or "active" not in cih.controller_diagnostics:
        raise RuntimeError("CIH rollout did not return controller activation diagnostics")
    logged = full_states[0, ANCHOR_INDEX:174]
    factual_full = np.concatenate((logged[:1], factual.states[0]), axis=0)
    cih_full = np.concatenate((logged[:1], cih.states[0]), axis=0)
    active = np.concatenate((np.zeros((1, 6), bool), cih.controller_diagnostics["active"][0].astype(bool)))
    receiver = _receiver_slot(active, factual_full, cih_full, full_valid[0, ANCHOR_INDEX])
    output = args.output or (ROOT / "results/hierarchical_world_model/cih_wm/playbacks" / f"cih_response_{args.split}_row_{row}.gif")
    output = output.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    plt = get_pyplot()
    frames = np.arange(0, len(logged), args.frame_stride)
    if frames[-1] != len(logged) - 1:
        frames = np.append(frames, len(logged) - 1)
    with imageio.get_writer(output, mode="I", duration=DT_S * args.frame_stride * 1000, loop=0) as writer:
        figure, axes = plt.subplots(1, 2, figsize=(14, 4.8), dpi=100)
        figure.subplots_adjust(left=.045, right=.99, bottom=.12, top=.78, wspace=.08)
        for frame in frames:
            figure.suptitle(
                f"CIH-WM response diagnostic | validation event row {row} | t={frame * DT_S:.2f}s\n"
                "Candidate is unaccepted: visual comparison only, not a factual/causal acceptance claim.",
                fontsize=10, fontweight="bold",
            )
            _draw_panel(axes[0], states=factual_full, logged=logged, valid=full_valid[0, ANCHOR_INDEX],
                        frame=int(frame), color=DIFFUSION_COLOR, label="Frozen factual transition", receiver=receiver,
                        active_count=None)
            _draw_panel(axes[1], states=cih_full, logged=logged, valid=full_valid[0, ANCHOR_INDEX],
                        frame=int(frame), color=CIH_COLOR, label="CIH-WM candidate", receiver=receiver,
                        active_count=int(active[frame].sum()))
            figure.canvas.draw()
            writer.append_data(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())
        plt.close(figure)
    probe = rollout(method.factual_dynamics, full_states, full_valid, response_plans, maps, map_valid,
                    device=device, history_frames=25, motion_seed=None, intervention="brake", dose=args.probe_dose,
                    excluded_slots=())
    cih_probe = rollout(method.factual_dynamics, full_states, full_valid, response_plans, maps, map_valid,
                        device=device, history_frames=25, motion_seed=None, intervention="brake", dose=args.probe_dose,
                        controller=method.response_policy, controller_deterministic=True, excluded_slots=(),
                        influence_graph_config=method.influence_graph_config)
    if cih_probe.controller_diagnostics is None or "active" not in cih_probe.controller_diagnostics:
        raise RuntimeError("CIH braking probe did not return controller diagnostics")
    probe_receiver = _receiver_slot(
        cih_probe.controller_diagnostics["active"][0].astype(bool), probe.states[0], cih_probe.states[0],
        full_valid[0, ANCHOR_INDEX],
    )
    probe_path = output.with_name(output.stem + "_counterfactual_brake.gif")
    probe_frames, probe_summary = _write_braking_probe(
        probe_path, baseline=probe, cih=cih_probe, valid=full_valid[0, ANCHOR_INDEX], receiver=probe_receiver,
        row=row, dose=args.probe_dose, frame_stride=args.frame_stride,
    )
    manifest = {
        "schema": "cih_wm_response_playback_v1", "row": row, "split": args.split,
        "receiver_slot": receiver + 1, "frames": int(len(frames)), "frame_stride": args.frame_stride,
        "candidate": str(args.candidate.resolve()), "gif": str(output),
        "controller_active_frames": int(active[:, receiver].sum()),
        "factual_interpretation": "Matched logged-ego replay: highD outlines are reference; blue is the frozen factual transition; purple is the unaccepted CIH-WM candidate.",
        "counterfactual_braking_probe": {
            "gif": str(probe_path), "frames": probe_frames, "dose_mps2": args.probe_dose,
            "window_frames": [25, 50], "summary": probe_summary,
            "interpretation": "No highD future target after the added ego brake; compare only the matched frozen factual-transition and CIH-WM response trajectories and actions.",
        },
    }
    manifest_path = output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
