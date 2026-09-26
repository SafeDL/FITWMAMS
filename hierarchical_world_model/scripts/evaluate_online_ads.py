#!/usr/bin/env python3
"""Full held-out ADS interventions in independent online NPC worlds.

Every branch starts from the same logged anchor and diffusion plan. A logged-
ADS run is computed only for after-the-fact effect measurement; its future
states/actions are never supplied to an intervention branch or controller.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import ANCHOR_INDEX  # noqa: E402
from hierarchical_world_model.scripts.evaluate_online_factual import _theta_for_rows  # noqa: E402
from hierarchical_world_model.src.ads_interventions import (  # noqa: E402
    AbsoluteAccelerationPolicy,
    SemanticLaneChangePolicy,
)
from hierarchical_world_model.src.collision_attribution import (  # noqa: E402
    collision_events,
    paired_collision_summary,
)
from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.evaluation import _logged_ego_actions, rollout  # noqa: E402
from hierarchical_world_model.src.planner import frozen_diffusion_plans  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from hierarchical_world_model.src.provenance import online_runtime_sha256  # noqa: E402
from hierarchical_world_model.src.stochastic_drivers.online import (  # noqa: E402
    DEFAULT_MA_IDM_POSTERIOR,
    OnlineMAIDMController,
)
from hierarchical_world_model.src.train import load_checkpoint  # noqa: E402
from traffic_components.src.core.utils import file_sha256, save_json, select_device  # noqa: E402


DOSES = (-8.0, -6.0, -4.0, -2.0, 2.0, 4.0)
EFFECT = slice(25, 100)


def _screen(
    states: np.ndarray, valid: np.ndarray, lane: str,
    map_polylines: np.ndarray | None = None,
    map_polyline_valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    anchor = states[:, ANCHOR_INDEX]
    dx = anchor[:, 1:, 0] - anchor[:, :1, 0]
    dy = anchor[:, 1:, 1] - anchor[:, :1, 1]
    if lane == "same":
        lateral = np.abs(dy) < 1.8
    elif lane in {"left", "right"}:
        if map_polylines is None or map_polyline_valid is None:
            raise ValueError("adjacent-lane screening requires the scene map")
        lane_y = np.asarray(map_polylines)[..., 0, 1]
        lane_present = np.asarray(map_polyline_valid, bool).any(axis=-1)
        source_index = np.where(
            lane_present, np.abs(lane_y - anchor[:, None, 0, 1]), np.inf
        ).argmin(axis=1)
        source_y = lane_y[np.arange(len(states)), source_index]
        displacement = lane_y - source_y[:, None]
        direction = 1.0 if lane == "left" else -1.0
        directed = displacement * direction
        adjacent = lane_present & (directed > 1.8) & (directed < 5.4)
        target_index = np.where(adjacent, directed, np.inf).argmin(axis=1)
        target_y = lane_y[np.arange(len(states)), target_index]
        lateral = adjacent.any(axis=1)[:, None] & (
            np.abs(anchor[:, 1:, 1] - target_y[:, None]) < 1.8
        )
    else:
        raise ValueError(f"unknown lane cohort: {lane}")
    eligible = valid[:, ANCHOR_INDEX, 1:] & (dx < -4.8) & lateral
    receiver = np.where(eligible, dx, -np.inf).argmax(axis=1)
    return eligible.any(axis=1), receiver


def _run_cohort(
    model, states: np.ndarray, valid: np.ndarray, plans: np.ndarray,
    maps: np.ndarray, map_valid: np.ndarray, theta: np.ndarray,
    receiver: np.ndarray, *, lane: str, batch_size: int, device,
) -> dict:
    values: dict[str, list[float]] = {
        key: [] for key in ((str(dose) for dose in DOSES) if lane == "same" else (lane,))
    }
    hiqr_effects: dict[str, list[float]] = {key: [] for key in values}
    controller_effects: dict[str, list[float]] = {key: [] for key in values}
    controller_active_rates: dict[str, list[float]] = {key: [] for key in values}
    controller_abs_corrections: dict[str, list[float]] = {key: [] for key in values}
    collisions: dict[str, dict[str, int]] = {
        key: {field: 0 for field in ("raw", "adjusted", "ads_pursuit", "ego_npc", "npc_npc")}
        for key in values
    }
    completed: dict[str, int] = {key: 0 for key in values}
    lane_terminal_errors: dict[str, list[float]] = {key: [] for key in values}
    finite: dict[str, int] = {key: 0 for key in values}
    lane_repairs: dict[str, int] = {key: 0 for key in values}
    autonomous_lane_frames: dict[str, int] = {key: 0 for key in values}
    autonomous_lane_scenes: dict[str, int] = {key: 0 for key in values}
    receiver_autonomous_lane_scenes: dict[str, int] = {key: 0 for key in values}
    spatial_path_frames: dict[str, int] = {key: 0 for key in values}
    receiver_spatial_path_frames: dict[str, int] = {key: 0 for key in values}
    collision_ablation: dict[str, dict[str, int]] = {key: {} for key in values}
    count = len(states)
    row = np.arange(count)
    anchor = states[:, ANCHOR_INDEX]
    gap_at_anchor = (
        anchor[:, 0, 0] - anchor[row, receiver + 1, 0] - 4.8
    )
    within_response_radius = gap_at_anchor < 45.0
    for begin in range(0, count, batch_size):
        end = min(begin + batch_size, count)
        part = slice(begin, end)
        initial = states[part]
        present = valid[part]
        controls = _logged_ego_actions(initial, present)

        def execute(policy=None):
            return rollout(
                model, initial, present, plans[part], maps[part], map_valid[part],
                device=device, history_frames=25, motion_seed=None,
                ads_policy=policy,
                controller=OnlineMAIDMController(theta[part]).to(device),
                record_policy_trace=False,
            )

        # Interventions execute first. The reference below exists only for
        # retrospective metric comparison, never branch initialization.
        branches = {}
        policies = {}
        for key in values:
            policy = (
                AbsoluteAccelerationPolicy(controls, float(key))
                if lane == "same" else
                SemanticLaneChangePolicy(
                    controls, initial[:, ANCHOR_INDEX],
                    map_polylines=maps[part], map_polyline_valid=map_valid[part],
                    direction=lane,
                )
            )
            policies[key] = policy
            branches[key] = execute(policy)
        reference = execute()
        row = np.arange(end - begin)
        local_receiver = receiver[part]
        reference_action = reference.background_actions[row, :, local_receiver, 0]
        reference_base = reference.base_background_actions[row, :, local_receiver, 0]
        reference_correction = reference_action - reference_base
        active = present[:, ANCHOR_INDEX, 1:]
        for key, branch in branches.items():
            action = branch.background_actions[row, :, local_receiver, 0]
            branch_base = branch.base_background_actions[row, :, local_receiver, 0]
            correction = action - branch_base
            values[key].extend((action[:, EFFECT] - reference_action[:, EFFECT]).mean(axis=1).tolist())
            hiqr_effects[key].extend(
                (branch_base[:, EFFECT] - reference_base[:, EFFECT]).mean(axis=1).tolist()
            )
            controller_effects[key].extend(
                (correction[:, EFFECT] - reference_correction[:, EFFECT]).mean(axis=1).tolist()
            )
            controller_active_rates[key].extend(
                (np.abs(correction[:, EFFECT]) > 0.05).mean(axis=1).tolist()
            )
            controller_abs_corrections[key].extend(
                np.abs(correction[:, EFFECT]).mean(axis=1).tolist()
            )
            events = collision_events(
                np.concatenate((initial[:, ANCHOR_INDEX:ANCHOR_INDEX + 1], branch.states), axis=1),
                active,
                ads_acceleration_setpoint_mps2=(float(key) if lane == "same" else None),
            )
            for field in collisions[key]:
                collisions[key][field] += int(events[field].sum())
            # Same-ADS HiQR-only ablation is an after-the-fact diagnostic,
            # never a nominal scene supplied to the responsive branch.
            hiqr_only = rollout(
                model, initial, present, plans[part], maps[part], map_valid[part],
                device=device, history_frames=25, motion_seed=None,
                ads_policy=policies[key], controller=None,
                record_policy_trace=False,
            )
            hiqr_events = collision_events(
                np.concatenate((initial[:, ANCHOR_INDEX:ANCHOR_INDEX + 1], hiqr_only.states), axis=1),
                active,
                ads_acceleration_setpoint_mps2=(float(key) if lane == "same" else None),
            )
            paired = paired_collision_summary(hiqr_events, events)
            for field, value in paired.items():
                if field.endswith("_rate") or field == "scenes":
                    continue
                collision_ablation[key][field] = collision_ablation[key].get(field, 0) + int(value)
            finite[key] += int(np.isfinite(branch.states).all(axis=(1, 2, 3)).sum())
            diagnostics = branch.controller_diagnostics or {}
            autonomous_lane = np.asarray(
                diagnostics.get(
                    "autonomous_lane_active",
                    np.zeros((end - begin, 149, 6), bool),
                ), bool,
            )
            autonomous_lane_frames[key] += int(autonomous_lane.sum())
            autonomous_lane_scenes[key] += int(autonomous_lane.any(axis=(1, 2)).sum())
            receiver_autonomous_lane_scenes[key] += int(
                autonomous_lane[row, :, local_receiver].any(axis=1).sum()
            )
            if "policy_active" in diagnostics:
                lane_repairs[key] += int((
                    np.asarray(diagnostics["policy_active"], bool) & ~autonomous_lane
                ).sum())
            if "spatial_path_active" in diagnostics:
                spatial = np.asarray(diagnostics["spatial_path_active"], bool)
                spatial_path_frames[key] += int(spatial.sum())
                receiver_spatial_path_frames[key] += int(spatial[row, :, local_receiver].sum())
            if lane in {"left", "right"}:
                target_y = policies[key].target_y
                heading = np.arctan2(branch.states[:, -1, 0, 3], branch.states[:, -1, 0, 2])
                final_error = np.abs(branch.states[:, -1, 0, 1] - target_y)
                lane_terminal_errors[key].extend(final_error.tolist())
                completed[key] += int(((final_error < 0.15)
                                       & (np.abs(heading) < 0.02)).sum())
        print(f"{lane}: {end}/{count}", flush=True)
    result = {}
    for key, effects in values.items():
        array = np.asarray(effects)
        expected = array < 0 if lane in {"left", "right"} or float(key) < 0 else array > 0
        result[key] = {
            "scenes": count,
            "receiver_mean_acceleration_effect_mps2": float(array.mean()),
            "receiver_mean_hiqr_action_effect_mps2": float(np.mean(hiqr_effects[key])),
            "receiver_mean_online_correction_effect_mps2": float(np.mean(controller_effects[key])),
            "receiver_online_correction_active_frame_rate": float(np.mean(controller_active_rates[key])),
            "receiver_mean_abs_online_correction_mps2": float(np.mean(controller_abs_corrections[key])),
            "effect_decomposition_max_error_mps2": float(
                np.max(np.abs(array - np.asarray(hiqr_effects[key]) - np.asarray(controller_effects[key])))
            ),
            "expected_direction_rate": float(expected.mean()),
            "finite_scene_rate": finite[key] / count,
            "npc_lane_repair_frames": lane_repairs[key],
            "npc_autonomous_lane_frames": autonomous_lane_frames[key],
            "npc_autonomous_lane_scenes": autonomous_lane_scenes[key],
            "receiver_autonomous_lane_scenes": receiver_autonomous_lane_scenes[key],
            "npc_spatial_path_frames": spatial_path_frames[key],
            "receiver_spatial_path_frames": receiver_spatial_path_frames[key],
            "collision_scene_counts": collisions[key],
            "same_ads_hiqr_only_ablation_counts": collision_ablation[key],
        }
        if within_response_radius.any():
            result[key]["within_45m_receiver_scenes"] = int(within_response_radius.sum())
            result[key]["within_45m_mean_acceleration_effect_mps2"] = float(
                array[within_response_radius].mean()
            )
            result[key]["within_45m_expected_direction_rate"] = float(
                expected[within_response_radius].mean()
            )
        if lane in {"left", "right"}:
            result[key]["ads_lane_change_completion_rate"] = completed[key] / count
            result[key]["ads_final_mapped_lane_error_p95_m"] = float(
                np.quantile(lane_terminal_errors[key], 0.95)
            )
            result[key]["ads_final_mapped_lane_error_max_m"] = float(
                np.max(lane_terminal_errors[key])
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "hierarchical_world_model/config/world_model.yaml")
    parser.add_argument("--posterior", type=Path, default=DEFAULT_MA_IDM_POSTERIOR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--cohort", choices=("same", "left", "right", "both"), default="both")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/hierarchical_world_model/evaluation/ads_test.json",
    )
    args = parser.parse_args()
    controller_hash = file_sha256(
        ROOT / "hierarchical_world_model/src/stochastic_drivers/online.py"
    )
    config = load_protocol_config(args.config.resolve())
    experiment = prepare_experiment_data(config, ROOT)
    test_rows = np.asarray(experiment.test_rows, np.int64)
    rows = test_rows if args.max_rows is None else test_rows[:args.max_rows]
    device = select_device(args.device)
    arrays = experiment.bundle.arrays
    states = np.asarray(arrays["agent_states"][rows], np.float32)
    valid = np.asarray(arrays["agent_valid"][rows], bool)
    maps = np.asarray(arrays["map_polylines"][rows], np.float32)
    map_valid = np.asarray(arrays["map_polyline_valid"][rows], bool)
    plans = frozen_diffusion_plans(
        experiment.bundle, rows, checkpoint=config["paths"]["diffusion_checkpoint"],
        output_dir=args.output.resolve().parent / ("plans_full" if args.max_rows is None else f"plans_{len(rows)}"),
        device=device, batch_size=args.batch_size, ddim_steps=20,
        experiment_scope="full" if args.max_rows is None else f"prefix_{len(rows)}",
    )
    theta = _theta_for_rows(rows, args.posterior.resolve())
    model, _ = load_checkpoint(Path(config["paths"]["evaluation_checkpoint"]), device=device)
    model.eval()
    cohorts = {}
    selected_lanes = ("same", "left") if args.cohort == "both" else (args.cohort,)
    for lane in selected_lanes:
        selected, receiver = _screen(
            states, valid, lane,
            map_polylines=maps if lane in {"left", "right"} else None,
            map_polyline_valid=map_valid if lane in {"left", "right"} else None,
        )
        if not selected.any():
            raise RuntimeError(f"no {lane} rear receiver in selected test rows")
        cohorts[lane] = _run_cohort(
            model, states[selected], valid[selected], plans[selected], maps[selected],
            map_valid[selected], theta[selected], receiver[selected],
            lane=lane, batch_size=args.batch_size, device=device,
        )
    report = {
        "schema": "online_ma_idm_ads_test_v2",
        "online_controller_sha256": controller_hash,
        "online_runtime_sha256": online_runtime_sha256(),
        "test_split_size": len(test_rows),
        "screened_test_rows": len(rows),
        "full_test_screen": args.max_rows is None,
        "adjacent_cohort_map_rule": (
            "nearest mapped source; next mapped requested-side lane 1.8–5.4 m away; "
            "rear NPC within 1.8 m of target centre"
        ),
        "same_ads_hiqr_ablation_legacy_handcrafted_adapter_applied": False,
        "evaluated_cohorts": list(selected_lanes),
        "protocol": "independent single-pass online branches; logged-ADS and same-ADS HiQR-only ablations used only after branch execution for metrics",
        "same_lane_rear": cohorts.get("same", {}),
        "left_lane_rear": cohorts.get("left", {}),
        "right_lane_rear": cohorts.get("right", {}),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, args.output)
    print(report)


if __name__ == "__main__":
    main()
