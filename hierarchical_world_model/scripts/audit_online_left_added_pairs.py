#!/usr/bin/env python3
"""List same-ADS collision pairs newly exposed by online NPC response."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import ANCHOR_INDEX  # noqa: E402
from hierarchical_world_model.scripts.evaluate_online_ads import _screen  # noqa: E402
from hierarchical_world_model.scripts.evaluate_online_factual import _theta_for_rows  # noqa: E402
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
from traffic_components.src.core.utils import file_sha256, save_json, select_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/hierarchical_world_model/evaluation/left_added_pair_audit.json",
    )
    args = parser.parse_args()
    config = load_protocol_config(ROOT / "hierarchical_world_model/config/world_model.yaml")
    experiment = prepare_experiment_data(config, ROOT)
    test_rows = np.asarray(experiment.test_rows, np.int64)
    arrays = experiment.bundle.arrays
    all_states = np.asarray(arrays["agent_states"][test_rows], np.float32)
    all_valid = np.asarray(arrays["agent_valid"][test_rows], bool)
    all_maps = np.asarray(arrays["map_polylines"][test_rows], np.float32)
    all_map_valid = np.asarray(arrays["map_polyline_valid"][test_rows], bool)
    selected, receiver = _screen(
        all_states, all_valid, "left", all_maps, all_map_valid
    )
    rows = test_rows[selected]
    states, valid = all_states[selected], all_valid[selected]
    receiver = receiver[selected]
    maps = all_maps[selected]
    map_valid = all_map_valid[selected]
    device = select_device("cuda")
    full_plans = frozen_diffusion_plans(
        experiment.bundle, test_rows, checkpoint=config["paths"]["diffusion_checkpoint"],
        output_dir=ROOT / "results/hierarchical_world_model/evaluation/plans_full",
        device=device, batch_size=64, ddim_steps=20, experiment_scope="full",
    )
    plans = full_plans[selected]
    theta = _theta_for_rows(test_rows, DEFAULT_MA_IDM_POSTERIOR)[selected]
    model, _ = load_checkpoint(Path(config["paths"]["evaluation_checkpoint"]), device=device)
    model.eval()
    added_count = 0
    cutin_exposure_count = 0
    added_records = []
    overlap_records = []
    for begin in range(0, len(rows), 64):
        end = min(begin + 64, len(rows))
        part = slice(begin, end)
        policy = SemanticLaneChangePolicy(
            _logged_ego_actions(states[part], valid[part]),
            states[part, ANCHOR_INDEX],
            map_polylines=maps[part], map_polyline_valid=map_valid[part],
            direction="left",
        )
        common = dict(
            model=model, logged_states=states[part], logged_valid=valid[part],
            soft_plans=plans[part], map_polylines=maps[part],
            map_polyline_valid=map_valid[part], device=device,
            history_frames=25, motion_seed=None, ads_policy=policy,
            record_policy_trace=False,
        )
        responsive = rollout(
            **common, controller=OnlineMAIDMController(theta[part]).to(device)
        )
        hiqr_only = rollout(**common, controller=None)
        active = valid[part, ANCHOR_INDEX, 1:]
        initial = states[part, ANCHOR_INDEX:ANCHOR_INDEX + 1]
        responsive_states = np.concatenate((initial, responsive.states), axis=1)
        hiqr_states = np.concatenate((initial, hiqr_only.states), axis=1)
        responsive_events = collision_events(responsive_states, active)
        hiqr_events = collision_events(hiqr_states, active)
        cutin_exposure_count += int(responsive_events["ads_cutin_exposure"].sum())
        for local in np.flatnonzero(responsive_events["raw"]):
            scene = responsive_states[local]
            pairs = np.argwhere(responsive_events["raw_pairs"][local])
            ads_shift = np.abs(scene[:, 0, 1] - scene[24, 0, 1])
            moving = np.flatnonzero(ads_shift[25:] > 0.1)
            ads_lateral_onset = None if len(moving) == 0 else int(moving[0] + 25)
            for a, b in pairs:
                pair_mask = (
                    (np.abs(scene[:, a, 0] - scene[:, b, 0]) < 4.8)
                    & (np.abs(scene[:, a, 1] - scene[:, b, 1]) < 1.8)
                )
                first = int(np.flatnonzero(pair_mask)[0])
                npc = int(b if a == 0 else a)
                lateral_overlap = np.flatnonzero(
                    np.abs(scene[:, a, 1] - scene[:, b, 1]) < 1.8
                )
                first_lateral = int(lateral_overlap[0])
                correction = responsive.controller_diagnostics["delta_ax"][local, :, npc - 1]
                before = np.flatnonzero(correction[25:first] < -0.05)
                first_brake = None if len(before) == 0 else int(before[0] + 25)
                overlap_records.append({
                    "test_row": int(rows[begin + local]),
                    "pair": [int(a), int(b)],
                    "first_overlap_frame": first,
                    "ads_lateral_onset_frame": ads_lateral_onset,
                    "frames_from_ads_lateral_onset_to_overlap": (
                        None if ads_lateral_onset is None else first - ads_lateral_onset
                    ),
                    "first_npc_brake_correction_frame": first_brake,
                    "frames_from_npc_brake_to_overlap": (
                        None if first_brake is None else first - first_brake
                    ),
                    "anchor_ads_minus_npc_x_m": float(scene[24, 0, 0] - scene[24, npc, 0]),
                    "anchor_ads_minus_npc_y_m": float(scene[24, 0, 1] - scene[24, npc, 1]),
                    "first_lateral_overlap_frame": first_lateral,
                    "frames_from_first_lateral_overlap_to_collision": first - first_lateral,
                    "ads_minus_npc_x_at_lateral_overlap_m": float(
                        scene[first_lateral, 0, 0] - scene[first_lateral, npc, 0]
                    ),
                    "ads_minus_npc_x_at_collision_m": float(
                        scene[first, 0, 0] - scene[first, npc, 0]
                    ),
                    "ads_minus_npc_vx_at_lateral_overlap_mps": float(
                        scene[first_lateral, 0, 2] - scene[first_lateral, npc, 2]
                    ),
                    "npc_online_minus_hiqr_x_at_collision_m": float(
                        scene[first, npc, 0] - hiqr_states[local, first, npc, 0]
                    ),
                    "npc_delta_ax_at_overlap_mps2": float(
                        correction[max(0, first - 1)]
                    ),
                    "ads_cutin_exposure": bool(
                        a == 0 and responsive_events["ads_cutin_exposure_pairs"][local, b - 1]
                    ),
                    "same_pair_overlaps_hiqr_only": bool(hiqr_events["raw_pairs"][local, a, b]),
                })
        added = responsive_events["raw_pairs"] & ~hiqr_events["raw_pairs"]
        for local in np.flatnonzero(added.any(axis=(1, 2))):
            added_count += 1
            pairs = np.argwhere(added[local])
            for a, b in pairs:
                frame_mask = (
                    (np.abs(responsive_states[local, :, a, 0] - responsive_states[local, :, b, 0]) < 4.8)
                    & (np.abs(responsive_states[local, :, a, 1] - responsive_states[local, :, b, 1]) < 1.8)
                )
                first = int(np.flatnonzero(frame_mask)[0])
                record = {
                    "test_row": int(rows[begin + local]),
                    "cohort_position": begin + int(local),
                    "pair": [int(a), int(b)],
                    "ads_cutin_exposure": bool(
                        a == 0 and responsive_events["ads_cutin_exposure_pairs"][local, b - 1]
                    ),
                    "first_overlap_frame": first,
                    "receiver_slot_zero_based": int(receiver[begin + local]),
                    "pair_state": responsive_states[local, first, [a, b]].tolist(),
                    "controller_delta_at_overlap": (
                        None if first == 0 else
                        responsive.controller_diagnostics["delta_ax"][local, first - 1].tolist()
                    ),
                }
                added_records.append(record)
                print(record)
    report = {
        "schema": "online_left_added_pair_audit_v3",
        "screened_left_rear_scenes": len(rows),
        "responsive_ads_cutin_exposure_scenes": cutin_exposure_count,
        "same_ads_hiqr_only_to_online_added_pair_scenes": added_count,
        "added_pairs": added_records,
        "all_online_overlap_pairs": overlap_records,
        "overlap_timing_summary": {
            "pair_count": len(overlap_records),
            "ads_npc_pair_count": sum(item["pair"][0] == 0 for item in overlap_records),
            "ads_cutin_exposure_pair_count": sum(
                item["ads_cutin_exposure"] for item in overlap_records
            ),
            "same_pair_hiqr_only_overlap_count": sum(
                item["same_pair_overlaps_hiqr_only"] for item in overlap_records
            ),
            "ads_ahead_of_npc_at_anchor_count": sum(
                item["anchor_ads_minus_npc_x_m"] > 0 for item in overlap_records
            ),
            "ads_ahead_npc_braked_before_overlap_count": sum(
                item["anchor_ads_minus_npc_x_m"] > 0
                and item["first_npc_brake_correction_frame"] is not None
                for item in overlap_records
            ),
            "ads_behind_or_beside_at_anchor_count": sum(
                item["anchor_ads_minus_npc_x_m"] <= 0 for item in overlap_records
            ),
            "ads_behind_or_beside_no_npc_brake_count": sum(
                item["anchor_ads_minus_npc_x_m"] <= 0
                and item["first_npc_brake_correction_frame"] is None
                for item in overlap_records
            ),
            "ads_ahead_when_lateral_overlap_begins_count": sum(
                item["ads_minus_npc_x_at_lateral_overlap_m"] > 0
                for item in overlap_records
            ),
            "ads_behind_or_beside_when_lateral_overlap_begins_count": sum(
                item["ads_minus_npc_x_at_lateral_overlap_m"] <= 0
                for item in overlap_records
            ),
            "npc_online_braked_relative_to_hiqr_at_collision_count": sum(
                item["npc_online_minus_hiqr_x_at_collision_m"] < -0.1
                for item in overlap_records
            ),
        },
        "online_controller_sha256": file_sha256(
            ROOT / "hierarchical_world_model/src/stochastic_drivers/online.py"
        ),
        "online_runtime_sha256": online_runtime_sha256(),
        "interpretation": (
            f"ADS cut-in exposure is a geometric label; all {added_count} added overlaps "
            "remain in raw and adjusted rates."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, args.output)
    print({"added_pair_scenes": added_count, "screened_left_rear_scenes": len(rows),
           "ads_cutin_exposure_scenes": cutin_exposure_count})


if __name__ == "__main__":
    main()
