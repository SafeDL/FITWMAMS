#!/usr/bin/env python3
"""Evaluate realized NPC-leader brake propagation on the complete highD Test split."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import ANCHOR_INDEX  # noqa: E402
from hierarchical_world_model.scripts.evaluate_online_factual import _theta_for_rows  # noqa: E402
from hierarchical_world_model.src.collision_attribution import collision_events  # noqa: E402
from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.evaluation import rollout  # noqa: E402
from hierarchical_world_model.src.planner import frozen_diffusion_plans  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from hierarchical_world_model.src.provenance import online_runtime_sha256  # noqa: E402
from hierarchical_world_model.src.stochastic_drivers.online import (  # noqa: E402
    DEFAULT_MA_IDM_POSTERIOR, OnlineMAIDMController, nearest_leader_observation,
)
from hierarchical_world_model.src.train import load_checkpoint  # noqa: E402
from traffic_components.src.core.utils import file_sha256, save_json, select_device  # noqa: E402


DOSES = (-8.0, -6.0, -4.0, -2.0)
FORCE = slice(25, 50)
EFFECT = slice(26, 100)


class _ForcedNPCBrakeController(OnlineMAIDMController):
    """Inject a realized NPC action after normal control, for diagnostics only."""

    def __init__(self, theta: np.ndarray, leader_slots: np.ndarray, dose: float):
        super().__init__(theta)
        self.leader_slots = torch.as_tensor(leader_slots, dtype=torch.long)
        self.dose = float(dose)

    def forward(self, context, *, deterministic=False):
        output = super().forward(context, deterministic=deterministic)
        if not FORCE.start <= context.response_index < FORCE.stop:
            return output
        actions = output.actions.clone()
        rows = torch.arange(actions.shape[0], device=actions.device)
        actions[rows, 0, self.leader_slots.to(actions.device), 0] = self.dose
        return replace(output, actions=actions)


def _screen(states: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the nearest well-aligned actual NPC leader for one follower per scene."""
    anchor = states[:, ANCHOR_INDEX]
    present = valid[:, ANCHOR_INDEX]
    gap, _, _, leader = nearest_leader_observation(
        torch.from_numpy(anchor), torch.from_numpy(present),
        prediction_horizon_s=2.0,
    )
    net_gap = gap.numpy()
    leader_index = leader.numpy()
    safe_index = np.maximum(leader_index, 0)
    leader_y = np.take_along_axis(anchor[:, :, 1], safe_index, axis=1)
    aligned = np.abs(leader_y - anchor[:, 1:, 1]) < 0.8
    eligible = (
        present[:, 1:] & (leader_index > 0) & aligned
        & (net_gap >= 8.0) & (net_gap <= 45.0)
    )
    selected_gap = np.where(eligible, net_gap, np.inf)
    follower_slots = selected_gap.argmin(axis=1)
    chosen = eligible.any(axis=1)
    row = np.arange(len(states))
    return chosen, follower_slots, leader_index[row, follower_slots] - 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/hierarchical_world_model/evaluation/npc_to_npc_highd_test.json",
    )
    args = parser.parse_args()
    config = load_protocol_config(ROOT / "hierarchical_world_model/config/world_model.yaml")
    experiment = prepare_experiment_data(config, ROOT)
    test_rows = np.asarray(experiment.test_rows, np.int64)
    arrays = experiment.bundle.arrays
    all_states = np.asarray(arrays["agent_states"][test_rows], np.float32)
    all_valid = np.asarray(arrays["agent_valid"][test_rows], bool)
    chosen, follower_slot, leader_slot = _screen(all_states, all_valid)
    selected = np.flatnonzero(chosen)
    if len(selected) == 0:
        raise ValueError("no NPC-NPC following scenes on the held-out test split")
    rows = test_rows[selected]
    states, valid = all_states[selected], all_valid[selected]
    follower_slot, leader_slot = follower_slot[selected], leader_slot[selected]
    maps = np.asarray(arrays["map_polylines"][rows], np.float32)
    map_valid = np.asarray(arrays["map_polyline_valid"][rows], bool)
    device = select_device("cuda")
    full_plans = frozen_diffusion_plans(
        experiment.bundle, test_rows,
        checkpoint=config["paths"]["diffusion_checkpoint"],
        output_dir=ROOT / "results/hierarchical_world_model/evaluation/plans_full",
        device=device, batch_size=args.batch_size, ddim_steps=20, experiment_scope="full",
    )
    plans = full_plans[selected]
    theta = _theta_for_rows(test_rows, DEFAULT_MA_IDM_POSTERIOR)[selected]
    model, _ = load_checkpoint(Path(config["paths"]["evaluation_checkpoint"]), device=device)
    model.eval()
    records = []
    for begin in range(0, len(rows), args.batch_size):
        end = min(begin + args.batch_size, len(rows))
        part = slice(begin, end)

        def execute(controller):
            result = rollout(
                model, states[part], valid[part], plans[part], maps[part], map_valid[part],
                device=device, history_frames=25, motion_seed=None,
                controller=controller, record_policy_trace=False,
            )
            return result, controller

        # Independent perturbations execute before the logged-ADS factual
        # comparison, which is never supplied to either online branch.
        branches = {
            dose: execute(_ForcedNPCBrakeController(
                theta[part], leader_slot[part], dose,
            ).to(device))
            for dose in DOSES
        }
        reference, _ = execute(OnlineMAIDMController(theta[part]).to(device))
        for local in range(end - begin):
            row = begin + local
            follower = int(follower_slot[row])
            leader = int(leader_slot[row])
            baseline = reference.background_actions[local, :, follower, 0]
            anchor = states[row, ANCHOR_INDEX]
            base_record = {
                "test_row": int(rows[row]),
                "follower_slot": follower + 1,
                "leader_slot": leader + 1,
                "anchor_net_gap_m": float(anchor[leader + 1, 0] - anchor[follower + 1, 0] - 4.8),
            }
            for dose, (branch, controller) in branches.items():
                action = branch.background_actions[local, :, follower, 0]
                effect = action - baseline
                before = np.concatenate((
                    states[row, ANCHOR_INDEX:ANCHOR_INDEX + 1],
                    branch.states[local, :-1],
                ), axis=0)
                present = np.broadcast_to(
                    valid[row, ANCHOR_INDEX], before.shape[:2],
                ).copy()
                gap_t, _, _, leader_t = nearest_leader_observation(
                    torch.from_numpy(before), torch.from_numpy(present),
                    prediction_horizon_s=2.0,
                )
                direct_leader = (
                    (leader_t[:, follower].numpy() == leader + 1)
                    & (np.abs(before[:, leader + 1, 1] - before[:, follower + 1, 1]) < 0.8)
                    & (gap_t[:, follower].numpy() < 45.0)
                )
                first = np.flatnonzero(effect[EFFECT] < -0.05)
                events = collision_events(
                    np.concatenate((
                        states[row:row + 1, ANCHOR_INDEX:ANCHOR_INDEX + 1],
                        branch.states[local:local + 1],
                    ), axis=1),
                    valid[row:row + 1, ANCHOR_INDEX, 1:],
                )
                other = np.abs(
                    branch.background_actions[local, EFFECT, :, 0]
                    - reference.background_actions[local, EFFECT, :, 0]
                ) > 0.05
                other[:, (follower, leader)] = False
                forced = branch.background_actions[local, FORCE, leader, 0]
                autonomous_lane = np.asarray(
                    (branch.controller_diagnostics or {}).get(
                        "autonomous_lane_active",
                        np.zeros((end - begin, 149, 6), bool),
                    ),
                    bool,
                )[local]
                follower_lane = bool(autonomous_lane[:, follower].any())
                target_y = (
                    float(controller._autonomous_lane_target_y[local, follower])
                    if follower_lane else None
                )
                final_follower = branch.states[local, -1, follower + 1]
                final_heading = float(np.arctan2(final_follower[3], final_follower[2]))
                map_centers = maps[row, map_valid[row].any(axis=-1), 0, 1]
                follower_y = branch.states[local, :, follower + 1, 1]
                records.append({
                    **base_record,
                    "leader_brake_mps2": dose,
                    "forced_leader_action_max_error_mps2": float(np.max(np.abs(forced - dose))),
                    "follower_mean_action_effect_mps2": float(effect[EFFECT].mean()),
                    "follower_mean_first_second_effect_mps2": float(
                        effect[FORCE.stop - 24:FORCE.stop].mean()
                    ),
                    "follower_first_second_brake_direction": bool(
                        effect[FORCE.stop - 24:FORCE.stop].mean() < -0.05
                    ),
                    "follower_brake_direction": bool(effect[EFFECT].mean() < -0.05),
                    "follower_first_response_latency_s": (
                        None if len(first) == 0 else float((first[0] + EFFECT.start - FORCE.start) * 0.04)
                    ),
                    "other_npc_action_changed": bool(other.any()),
                    "forced_npc_direct_leader_fraction_first_second": float(
                        direct_leader[FORCE.stop - 24:FORCE.stop].mean()
                    ),
                    "forced_leader_realized_ax_first_frame_mps2": float(
                        branch.states[local, FORCE.start, leader + 1, 4]
                    ),
                    "raw_overlap": bool(events["raw"][0]),
                    "overlap_pairs": [
                        pair.tolist() for pair in np.argwhere(events["raw_pairs"][0])
                    ],
                    "npc_npc_overlap": bool(events["npc_npc"][0]),
                    "autonomous_lane_slots": (
                        np.flatnonzero(autonomous_lane.any(axis=0)) + 1
                    ).tolist(),
                    "follower_autonomous_lane": follower_lane,
                    "follower_autonomous_lane_start_frame": (
                        int(np.flatnonzero(autonomous_lane[:, follower])[0])
                        if follower_lane else None
                    ),
                    "follower_autonomous_lane_target_y_m": target_y,
                    "follower_autonomous_lane_complete": bool(
                        follower_lane and abs(float(final_follower[1]) - target_y) < 0.75
                        and abs(final_heading) < 0.05
                    ),
                    "follower_offroad": bool(
                        (follower_y < map_centers.min() - 1.8).any()
                        or (follower_y > map_centers.max() + 1.8).any()
                    ),
                    "finite": bool(np.isfinite(branch.states[local]).all()),
                })
        print(f"completed {end}/{len(rows)} NPC-led Test scenes", flush=True)
    dose_summary = {}
    for dose in DOSES:
        cohort = [item for item in records if item["leader_brake_mps2"] == dose]
        near = [item for item in cohort if item["anchor_net_gap_m"] < 30.0]
        dose_summary[str(dose)] = {
            "scenes": len(cohort),
            "near_30m_scenes": len(near),
            "first_second_brake_direction_scenes": sum(
                item["follower_first_second_brake_direction"] for item in cohort
            ),
            "near_30m_first_second_brake_direction_scenes": sum(
                item["follower_first_second_brake_direction"] for item in near
            ),
            "effect_window_brake_direction_scenes": sum(
                item["follower_brake_direction"] for item in cohort
            ),
            "mean_first_second_action_effect_mps2": float(np.mean([
                item["follower_mean_first_second_effect_mps2"] for item in cohort
            ])),
            "raw_overlap_scenes": sum(item["raw_overlap"] for item in cohort),
            "npc_npc_overlap_scenes": sum(item["npc_npc_overlap"] for item in cohort),
            "autonomous_lane_scenes": sum(bool(item["autonomous_lane_slots"]) for item in cohort),
            "follower_autonomous_lane_scenes": sum(
                item["follower_autonomous_lane"] for item in cohort
            ),
            "follower_autonomous_lane_completed_scenes": sum(
                item["follower_autonomous_lane_complete"] for item in cohort
            ),
            "follower_offroad_scenes": sum(item["follower_offroad"] for item in cohort),
            "finite_scenes": sum(item["finite"] for item in cohort),
            "forced_leader_max_action_error_mps2": max(
                item["forced_leader_action_max_error_mps2"] for item in cohort
            ),
        }
    strictly_ordered = 0
    ordered = 0
    worst_inversion = {"test_row": None, "magnitude_mps2": 0.0, "effects_mps2": []}
    for index, row in enumerate(rows):
        effects = np.asarray([
            records[index * len(DOSES) + dose_index]["follower_mean_first_second_effect_mps2"]
            for dose_index in range(len(DOSES))
        ], np.float32)
        inversion = np.maximum(-np.diff(effects), 0.0)
        strictly_ordered += int((inversion <= 1.0e-5).all())
        ordered += int((inversion <= 0.05).all())
        magnitude = float(inversion.max())
        if magnitude > worst_inversion["magnitude_mps2"]:
            worst_inversion = {
                "test_row": int(row),
                "magnitude_mps2": magnitude,
                "effects_mps2": effects.tolist(),
            }
    report = {
        "schema": "online_npc_to_npc_highd_test_v1",
        "test_split_size": len(test_rows),
        "screened_full_test": True,
        "selected_scenes": len(rows),
        "selection": "one closest aligned NPC leader per Test scene, anchor net gap 8–45 m",
        "brake_doses_mps2": list(DOSES),
        "forced_window_frames": [FORCE.start, FORCE.stop],
        "effect_window_frames": [EFFECT.start, EFFECT.stop],
        "dose_summary": dose_summary,
        "first_second_strict_dose_ordered_scenes": strictly_ordered,
        "first_second_dose_ordered_with_0p05_tolerance_scenes": ordered,
        "worst_first_second_dose_inversion": worst_inversion,
        "records": records,
        "online_controller_sha256": file_sha256(
            ROOT / "hierarchical_world_model/src/stochastic_drivers/online.py"
        ),
        "online_runtime_sha256": online_runtime_sha256(),
    }
    save_json(report, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
