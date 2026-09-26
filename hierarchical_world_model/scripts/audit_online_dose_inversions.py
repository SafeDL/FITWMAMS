#!/usr/bin/env python3
"""Inspect realized geometry behind the largest scene-level dose inversions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import ANCHOR_INDEX  # noqa: E402
from hierarchical_world_model.scripts.evaluate_online_ads import DOSES, EFFECT, _screen  # noqa: E402
from hierarchical_world_model.scripts.evaluate_online_factual import _theta_for_rows  # noqa: E402
from hierarchical_world_model.src.ads_interventions import AbsoluteAccelerationPolicy  # noqa: E402
from hierarchical_world_model.src.collision_attribution import collision_events  # noqa: E402
from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.evaluation import _logged_ego_actions, rollout  # noqa: E402
from hierarchical_world_model.src.planner import frozen_diffusion_plans  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from hierarchical_world_model.src.provenance import online_runtime_sha256  # noqa: E402
from hierarchical_world_model.src.stochastic_drivers.online import (  # noqa: E402
    DEFAULT_MA_IDM_POSTERIOR, OnlineMAIDMController, nearest_leader_observation,
)
from hierarchical_world_model.src.train import load_checkpoint  # noqa: E402
from traffic_components.src.core.utils import file_sha256, load_json, save_json, select_device  # noqa: E402


class _AuditedController(OnlineMAIDMController):
    """Collect gate decisions without changing the online control action."""

    def __init__(self, theta):
        super().__init__(theta)
        self.planned_escape_frames = []
        self.departing_leader_frames = []

    def _planned_lane_escape(self, *args):
        result = super()._planned_lane_escape(*args)
        self.planned_escape_frames.append(result.detach().cpu().numpy())
        return result

    def _departing_leader_clearance(self, *args):
        result = super()._departing_leader_clearance(*args)
        self.departing_leader_frames.append(result.detach().cpu().numpy())
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--dose-report", type=Path,
        default=ROOT / "results/hierarchical_world_model/evaluation/dose_order_test.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/hierarchical_world_model/evaluation/dose_inversion_audit.json",
    )
    args = parser.parse_args()
    if args.top < 1 or args.top > 20:
        raise ValueError("--top must be in [1, 20]")
    source = load_json(args.dose_report)
    wanted = [entry["test_row"] for entry in source["worst_inversion_scenes"][:args.top]]
    config = load_protocol_config(ROOT / "hierarchical_world_model/config/world_model.yaml")
    experiment = prepare_experiment_data(config, ROOT)
    test_rows = np.asarray(experiment.test_rows, np.int64)
    arrays = experiment.bundle.arrays
    all_states = np.asarray(arrays["agent_states"][test_rows], np.float32)
    all_valid = np.asarray(arrays["agent_valid"][test_rows], bool)
    selected, receiver = _screen(all_states, all_valid, "same")
    eligible_rows = test_rows[selected]
    if not set(wanted).issubset(set(eligible_rows.tolist())):
        raise ValueError("dose audit contains a row outside the screened test cohort")
    selected_indices = np.asarray([np.flatnonzero(test_rows == row)[0] for row in wanted])
    states, valid = all_states[selected_indices], all_valid[selected_indices]
    receiver = receiver[selected_indices]
    maps = np.asarray(arrays["map_polylines"][wanted], np.float32)
    map_valid = np.asarray(arrays["map_polyline_valid"][wanted], bool)
    device = select_device("cuda")
    full_plans = frozen_diffusion_plans(
        experiment.bundle, test_rows, checkpoint=config["paths"]["diffusion_checkpoint"],
        output_dir=ROOT / "results/hierarchical_world_model/evaluation/plans_full",
        device=device, batch_size=64, ddim_steps=20, experiment_scope="full",
    )
    plans = full_plans[selected_indices]
    theta = _theta_for_rows(test_rows, DEFAULT_MA_IDM_POSTERIOR)[selected_indices]
    model, _ = load_checkpoint(Path(config["paths"]["evaluation_checkpoint"]), device=device)
    model.eval()
    controls = _logged_ego_actions(states, valid)

    def execute(policy=None):
        controller = _AuditedController(theta).to(device)
        result = rollout(
            model, states, valid, plans, maps, map_valid,
            device=device, history_frames=25, motion_seed=None,
            ads_policy=policy, controller=controller,
            record_policy_trace=False,
        )
        return result, (
            np.stack(controller.planned_escape_frames),
            np.stack(controller.departing_leader_frames),
        )

    branch_runs = [execute(AbsoluteAccelerationPolicy(controls, dose)) for dose in DOSES]
    reference, _ = execute()
    reports = []
    for index, row in enumerate(wanted):
        slot = int(receiver[index])
        baseline = reference.background_actions[index, EFFECT, slot, 0]
        branch_reports = []
        for dose, (branch, (planned_escape, departing_leader)) in zip(DOSES, branch_runs):
            npc = branch.states[index, :, slot + 1]
            ego = branch.states[index, :, 0]
            before = np.concatenate((
                states[index, ANCHOR_INDEX:ANCHOR_INDEX + 1],
                branch.states[index, :-1],
            ), axis=0)
            before_valid = np.broadcast_to(
                valid[index, ANCHOR_INDEX], before.shape[:2],
            ).copy()
            gap_t, speed_t, leader_speed_t, leader_index_t = nearest_leader_observation(
                torch.from_numpy(before), torch.from_numpy(before_valid),
                prediction_horizon_s=2.0,
            )
            selected_gap = gap_t[:, slot].numpy()
            selected_speed = speed_t[:, slot].numpy()
            selected_leader_speed = leader_speed_t[:, slot].numpy()
            selected_leader = leader_index_t[:, slot].numpy()
            before_lag = plans[index, :, slot, 0] - before[:, slot + 1, 0]
            before_base_ax = branch.base_background_actions[index, :, slot, 0]
            direct_follow = (
                (selected_leader == 0)
                & (np.abs(before[:, 0, 1] - before[:, slot + 1, 1]) < 1.8)
                & (selected_gap < 45.0)
            )
            pursuit_candidate = (
                direct_follow & (selected_speed > selected_leader_speed)
                & (before_lag > 2.0) & (before_base_ax > 0.0)
            )
            escape_mask = planned_escape[:, index, slot]
            departing_mask = departing_leader[:, index, slot]
            pursuit_trigger = pursuit_candidate & ~escape_mask & ~departing_mask
            relative_x = ego[:, 0] - npc[:, 0]
            lateral_overlap = np.abs(ego[:, 1] - npc[:, 1]) < 1.8
            behind = (relative_x > 4.8) & lateral_overlap
            events = collision_events(
                np.concatenate((states[index:index + 1, ANCHOR_INDEX:ANCHOR_INDEX + 1],
                                branch.states[index:index + 1]), axis=1),
                valid[index:index + 1, ANCHOR_INDEX, 1:],
                ads_acceleration_setpoint_mps2=dose,
            )
            diagnostics = branch.controller_diagnostics or {}
            active = np.asarray(diagnostics.get("active", np.zeros((len(wanted), 149, 6))), bool)
            delta_ax = np.asarray(diagnostics["delta_ax"])[index, EFFECT, slot]
            base_ax = branch.base_background_actions[index, EFFECT, slot, 0]
            action_ax = branch.background_actions[index, EFFECT, slot, 0]
            plan_x_lag = plans[index, EFFECT, slot, 0] - npc[EFFECT, 0]
            branch_reports.append({
                "dose_mps2": float(dose),
                "effect_mps2": float((branch.background_actions[index, EFFECT, slot, 0] - baseline).mean()),
                "receiver_behind_ads_window_fraction": float(behind[EFFECT].mean()),
                "receiver_ahead_ads_window_fraction": float(((relative_x < -4.8) & lateral_overlap)[EFFECT].mean()),
                "minimum_longitudinal_gap_m": float(np.min(np.abs(relative_x) - 4.8)),
                "receiver_active_window_fraction": float(active[index, EFFECT, slot].mean()),
                "controller_delta_ax_window_mean_mps2": float(delta_ax.mean()),
                "controller_delta_ax_window_min_mps2": float(delta_ax.min()),
                "controller_delta_ax_window_max_mps2": float(delta_ax.max()),
                "hiqr_base_ax_window_mean_mps2": float(base_ax.mean()),
                "online_action_ax_window_mean_mps2": float(action_ax.mean()),
                "hiqr_base_over_2_mps2_window_fraction": float((base_ax > 2.0).mean()),
                "planned_x_lag_window_mean_m": float(plan_x_lag.mean()),
                "planned_x_lag_over_2m_window_fraction": float((plan_x_lag > 2.0).mean()),
                "pre_action_plan_lag_window_mean_m": float(before_lag[EFFECT].mean()),
                "selected_ads_leader_window_fraction": float((selected_leader[EFFECT] == 0).mean()),
                "direct_ads_follow_window_fraction": float(direct_follow[EFFECT].mean()),
                "faster_than_selected_leader_window_fraction": float(
                    (selected_speed[EFFECT] > selected_leader_speed[EFFECT]).mean()
                ),
                "positive_hiqr_with_pre_action_plan_lag_window_fraction": float(
                    ((before_base_ax[EFFECT] > 0.0) & (before_lag[EFFECT] > 2.0)).mean()
                ),
                "plan_pursuit_candidate_window_fraction_before_lane_exceptions": float(
                    pursuit_candidate[EFFECT].mean()
                ),
                "planned_escape_window_fraction": float(escape_mask[EFFECT].mean()),
                "departing_leader_window_fraction": float(departing_mask[EFFECT].mean()),
                "plan_pursuit_trigger_window_fraction": float(pursuit_trigger[EFFECT].mean()),
                "raw_overlap": bool(events["raw"][0]),
                "ego_npc_overlap": bool(events["ego_npc"][0]),
                "npc_npc_overlap": bool(events["npc_npc"][0]),
            })
        reports.append({"test_row": int(row), "receiver_slot": slot + 1, "branches": branch_reports})
    save_json({
        "schema": "online_dose_inversion_audit_v1",
        "source_dose_controller_sha256": source["online_controller_sha256"],
        "source_dose_runtime_sha256": source["online_runtime_sha256"],
        "online_controller_sha256": file_sha256(
            ROOT / "hierarchical_world_model/src/stochastic_drivers/online.py"
        ),
        "online_runtime_sha256": online_runtime_sha256(),
        "scenes": reports,
    }, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
