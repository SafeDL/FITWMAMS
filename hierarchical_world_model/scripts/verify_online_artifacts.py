#!/usr/bin/env python3
"""Verify that every maintained online-world report describes one controller build."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.model import HIGHD_INTEGRATION_SPEED_MAX_MPS  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from hierarchical_world_model.src.provenance import online_runtime_sha256  # noqa: E402
from traffic_components.src.core.utils import file_sha256, load_json  # noqa: E402


def _recording_ids(sequence_ids: np.ndarray) -> set[int]:
    identifiers = set()
    for value in sequence_ids:
        parts = str(value).split("_")
        if len(parts) < 2 or parts[0] != "nat":
            raise ValueError(f"unexpected highD sequence ID: {value!r}")
        identifiers.add(int(parts[1]))
    return identifiers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory", type=Path,
        default=ROOT / "results/hierarchical_world_model/evaluation",
        help="candidate artifact directory to verify before promotion",
    )
    directory = parser.parse_args().directory.resolve()
    filenames = {
        "factual": "factual_test.json",
        "factual_hiqr_only": "factual_hiqr_only_test.json",
        "factual_tail": "factual_tail_audit.json",
        "departing_audit": "departing_leader_audit.json",
        "lane_entry_audit": "target_lane_entry_audit.json",
        "ads": "ads_test.json",
        "ads_right": "ads_right_lane_test.json",
        "dose": "dose_order_test.json",
        "dose_audit": "dose_inversion_audit.json",
        "left_audit": "left_added_pair_audit.json",
        "left_cutin_regression": "left_added_pair_trace_audit.json",
        "generated": "generated_ads_sweep.json",
        "lane_meta_actions": "lane_meta_actions.json",
        "highway_parity": "highway_parity.json",
        "high_speed_parity": "high_speed_parity.json",
        "ads_highway_parity": "ads_highway_parity.json",
        "spatial_audit": "spatial_path_scene_audit.json",
        "ads_spatial_audit": "spatial_path_ads_m8_audit.json",
        "late_finish_open_parity": "late_finish_open_highway_parity.json",
        "late_finish_neighbor_parity": "late_finish_neighbor_highway_parity.json",
        "gif": "ads_demo_manifest.json",
        "npc_gif": "npc_lane_demo_manifest.json",
        "clear_lane_escape_gif": "npc_clear_lane_escape_demo_manifest.json",
        "ads_npc_crossing_gif": "ads_npc_crossing_demo_manifest.json",
        "opening_lane_gap_gif": "npc_opening_lane_gap_demo_manifest.json",
        "ads_brake_lane_gif": "npc_ads_brake_lane_demo_manifest.json",
        "late_finish_open_gif": "npc_spatial_delayed_open_demo_manifest.json",
        "late_finish_neighbor_gif": "npc_spatial_delayed_neighbor_demo_manifest.json",
        "plan_pursuit_gif": "npc_plan_pursuit_demo_manifest.json",
        "spatial_brake_gif": "npc_spatial_brake_demo_manifest.json",
        "spatial_logged_gif": "npc_spatial_logged_demo_manifest.json",
        "npc_to_npc_gif": "npc_to_npc_brake_demo_manifest.json",
        "npc_to_npc_sweep": "npc_to_npc_sweep.json",
        "npc_to_npc_highd": "npc_to_npc_highd_test.json",
        "late_lane_continuation": "late_autonomous_lane_continuation.json",
        "npc_to_npc_highd_gif": "npc_to_npc_highd_demo_manifest.json",
        "npc_autonomous_lane_gif": "npc_autonomous_lane_demo_manifest.json",
        "ads_cutin_pass_gif": "ads_cutin_pass_demo_manifest.json",
        "right_lane_meta_gif": "right_lane_meta_demo_manifest.json",
    }
    reports = {name: load_json(directory / filename) for name, filename in filenames.items()}
    departing_scene = next(
        scene for scene in reports["departing_audit"]["scenes"]
        if scene["test_row"] == 61899
    )
    departing_npc = next(
        agent for agent in departing_scene["agents"] if agent["npc_slot"] == 2
    )
    lane_entry_scenes = {
        scene["test_row"]: scene for scene in reports["lane_entry_audit"]["scenes"]
    }
    lane_entry_npcs = {
        row: next(agent for agent in lane_entry_scenes[row]["agents"] if agent["npc_slot"] == slot)
        for row, slot in ((76975, 6), (27342, 2))
    }
    natural_support = load_json(directory / "natural_leader_acceleration_support.json")
    asset_manifest = load_json(directory / "runtime_assets_manifest.json")
    plan_manifest = load_json(directory / "plans_full/frozen_diffusion_test_plans.json")
    controller_hash = file_sha256(
        ROOT / "hierarchical_world_model/src/stochastic_drivers/online.py"
    )
    ads_test_policy_hash = file_sha256(
        ROOT / "hierarchical_world_model/scripts/ads_test_policies.py"
    )
    runtime_hash = online_runtime_sha256()
    config = load_protocol_config(ROOT / "hierarchical_world_model/config/world_model.yaml")
    experiment = prepare_experiment_data(config, ROOT)
    plan_cache = (directory / "plans_full/frozen_diffusion_test_plans.npz").resolve()
    with np.load(plan_cache) as cached_plans:
        cached_rows = np.asarray(cached_plans["row_index"], np.int64)
    test_rows = np.asarray(experiment.test_rows, np.int64)
    required_assets = {
        str(Path(config["paths"][key]).resolve().relative_to(ROOT))
        for key in ("evaluation_checkpoint", "diffusion_checkpoint", "flow_checkpoint")
    }
    required_assets.add(str(Path(reports["factual"]["posterior"]).resolve().relative_to(ROOT)))
    required_assets.add(str(plan_cache.relative_to(ROOT)))
    test_recordings = _recording_ids(
        np.asarray(experiment.bundle.arrays["sequence_id"])[experiment.test_rows]
    )
    with np.load(reports["factual"]["posterior"]) as posterior:
        posterior_recordings = {int(value) for value in posterior["recording_id"]}
    checks = {
        "all_reports_match_controller": all(
            report["online_controller_sha256"] == controller_hash
            for report in reports.values()
        ),
        "all_reports_match_runtime": all(
            report["online_runtime_sha256"] == runtime_hash
            for report in reports.values()
        ),
        "frozen_runtime_assets_match": (
            asset_manifest["schema"] == "online_closed_loop_runtime_assets_v1"
            and set(asset_manifest["assets"]) == required_assets
            and all(
                file_sha256(ROOT / relative) == expected
                for relative, expected in asset_manifest["assets"].items()
            )
        ),
        "full_test_diffusion_plan_cache_matches": (
            plan_manifest["sequences"] == 10151
            and plan_manifest["experiment_scope"] == "full"
            and plan_manifest["checkpoint_sha256"]
            == asset_manifest["assets"][
                str(Path(config["paths"]["diffusion_checkpoint"]).resolve().relative_to(ROOT))
            ]
            and plan_manifest["row_digest"] == hashlib.sha256(test_rows.tobytes()).hexdigest()
            and np.array_equal(cached_rows, test_rows)
        ),
        "full_held_out_test": (
            reports["factual"]["full_test"]
            and reports["factual"]["evaluated_rows"] == 10151
            and reports["factual_hiqr_only"]["full_test"]
            and reports["factual_hiqr_only"]["evaluated_rows"] == 10151
            and reports["ads"]["test_split_size"] == 10151
            and reports["ads_right"]["test_split_size"] == 10151
            and reports["ads_right"]["screened_test_rows"] == 10151
            and reports["ads_right"]["full_test_screen"]
            and reports["dose"]["full_test_rows_screened"] == 10151
        ),
        "controller_free_ablation_uses_hiqr_without_legacy_adapter": (
            reports["factual_hiqr_only"].get("legacy_handcrafted_adapter_applied") is False
            and reports["ads"].get(
                "same_ads_hiqr_ablation_legacy_handcrafted_adapter_applied"
            ) is False
        ),
        "same_screened_cohorts": (
            reports["ads"]["schema"] == "online_ma_idm_ads_test_v2"
            and "mapped" in reports["ads"].get("left_cohort_map_rule", "")
            and reports["dose"]["same_lane_rear_scenes"] == 1851
            and all(value["scenes"] == 1851 for value in reports["ads"]["same_lane_rear"].values())
            and reports["ads"]["left_lane_rear"]["left"]["scenes"] == 1000
            and reports["left_audit"]["screened_left_rear_scenes"] == 1000
            and reports["ads_right"]["right_lane_rear"]["right"]["scenes"] == 1306
        ),
        "npc_effect_decomposition_exact": all(
            value.get("effect_decomposition_max_error_mps2", float("inf")) < 1.0e-4
            for value in (
                *reports["ads"]["same_lane_rear"].values(),
                reports["ads"]["left_lane_rear"]["left"],
                reports["ads_right"]["right_lane_rear"]["right"],
            )
        ),
        "right_lane_test_action_and_attribution": (
            reports["ads_right"]["evaluated_cohorts"] == ["right"]
            and reports["ads_right"]["right_lane_rear"]["right"]["finite_scene_rate"] == 1.0
            and reports["ads_right"]["right_lane_rear"]["right"]["ads_lane_change_completion_rate"] == 1.0
            and reports["ads_right"]["right_lane_rear"]["right"]["collision_scene_counts"]["raw"] == 449
            and reports["ads_right"]["right_lane_rear"]["right"]["same_ads_hiqr_only_ablation_counts"]["passive_raw_scene_count"] == 465
            and reports["ads_right"]["right_lane_rear"]["right"]["same_ads_hiqr_only_ablation_counts"]["raw_avoided_with_npc_response"] == 16
            and reports["ads_right"]["right_lane_rear"]["right"]["same_ads_hiqr_only_ablation_counts"]["raw_introduced_with_npc_response"] == 0
        ),
        "generated_lane_meta_actions_and_gif": (
            reports["lane_meta_actions"]["schema"] == "generated_lane_meta_actions_v1"
            and reports["lane_meta_actions"]["scenes"] == 256
            and reports["lane_meta_actions"]["lane_completion"]["right"]["completed"] == 256
            and reports["lane_meta_actions"]["lane_completion"]["brake_right"]["completed"] == 256
            and "neither is a zero-required pass gate" in reports["lane_meta_actions"]["collision_policy"]
            and reports["right_lane_meta_gif"]["schema"] == "generated_lane_meta_demo_v1"
            and reports["right_lane_meta_gif"]["focus_is_right_target_lane_rear"]
            and reports["right_lane_meta_gif"]["right_lane_terminal_error_m"] < 0.15
            and (directory / reports["right_lane_meta_gif"]["gif"]).is_file()
        ),
        "audit_matches_screened_results": (
            reports["dose_audit"]["source_dose_controller_sha256"] == controller_hash
            and reports["dose_audit"]["source_dose_runtime_sha256"] == runtime_hash
            and reports["left_audit"]["same_ads_hiqr_only_to_online_added_pair_scenes"]
            == reports["ads"]["left_lane_rear"]["left"]["same_ads_hiqr_only_ablation_counts"]["raw_responsive_overlap_npc_response_added_pair_scene_count"]
            and reports["left_audit"]["same_ads_hiqr_only_to_online_added_pair_scenes"] == 0
            and reports["left_audit"]["added_pairs"] == []
        ),
        "left_overlap_pair_timing_audited": (
            reports["left_audit"]["schema"] == "online_left_added_pair_audit_v3"
            and reports["left_audit"]["overlap_timing_summary"]["pair_count"]
            == len(reports["left_audit"]["all_online_overlap_pairs"])
            == reports["ads"]["left_lane_rear"]["left"]["collision_scene_counts"]["raw"]
            and reports["left_audit"]["overlap_timing_summary"]["ads_npc_pair_count"]
            == reports["left_audit"]["overlap_timing_summary"]["pair_count"]
            and reports["left_audit"]["overlap_timing_summary"]["same_pair_hiqr_only_overlap_count"]
            == reports["left_audit"]["overlap_timing_summary"]["pair_count"]
            and reports["left_audit"]["overlap_timing_summary"]["ads_ahead_of_npc_at_anchor_count"]
            + reports["left_audit"]["overlap_timing_summary"]["ads_behind_or_beside_at_anchor_count"]
            == reports["left_audit"]["overlap_timing_summary"]["pair_count"]
            and reports["left_audit"]["overlap_timing_summary"]["ads_ahead_npc_braked_before_overlap_count"]
            <= reports["left_audit"]["overlap_timing_summary"]["ads_ahead_of_npc_at_anchor_count"]
        ),
        "previous_left_cutin_added_pairs_resolved": (
            reports["left_cutin_regression"]["schema"] == "online_left_cutin_regressions_v3"
            and reports["left_cutin_regression"]["historical_added_pairs"] == 5
            and [item["test_row"] for item in reports["left_cutin_regression"]["records"]]
            == [58883, 634, 76098, 19594, 73169]
            and all(
                (not item["branches"]["online"]["raw_overlap"]
                 or item["branches"]["controller_free_hiqr"]["raw_overlap"])
                and not item["branches"]["online"]["npc_npc_overlap"]
                and abs(
                    item["branches"]["controller_free_hiqr"]["npc_x_at_75_m"]
                    - item["branches"]["hiqr_base_only"]["npc_x_at_75_m"]
                ) < 0.001
                for item in reports["left_cutin_regression"]["records"]
            )
        ),
        "generated_policy_coverage": (
            reports["generated"]["schema"] == "generated_online_ads_sweep_v2"
            and reports["generated"]["scenes"] == 1024
            and set(reports["generated"]["policies"])
            == {"hold", "left", "-8.0", "-6.0", "-4.0", "-2.0", "2.0", "4.0"}
            and reports["generated"]["evaluation_script_sha256"] == file_sha256(
                ROOT / "hierarchical_world_model/scripts/evaluate_generated_online_ads.py"
            )
            and reports["generated"]["ads_test_policy_sha256"] == ads_test_policy_hash
        ),
        "all_factual_finite": reports["factual"]["finite_scene_rate"] == 1.0,
        "speed_ceiling_covers_test": (
            reports["factual"]["evaluated_observed_max_speed_mps"]
            < HIGHD_INTEGRATION_SPEED_MAX_MPS
            and reports["factual"]["evaluated_observed_speed_samples_above_50_mps"] > 0
        ),
        "factual_tail_audited": (
            reports["factual_tail"]["source_factual_runtime_sha256"] == runtime_hash
            and not reports["factual_tail"]["audit_rows_selected_explicitly"]
            and reports["factual_tail"]["audited_rows"]
            == [item["test_row"] for item in reports["factual"]["worst_scene_ADE"]]
        ),
        "departing_leader_false_brake_removed": (
            reports["departing_audit"]["source_factual_runtime_sha256"]
            == runtime_hash
            and reports["departing_audit"]["audited_rows"] == [61899, 46684]
            and departing_npc["online_correction_frames"] == 0
            and departing_npc["online_ADE_m"] < 0.2
            and not departing_scene["online_overlap"]
        ),
        "opening_target_lane_gap_preserves_planned_changes": (
            reports["lane_entry_audit"]["source_factual_runtime_sha256"] == runtime_hash
            and reports["lane_entry_audit"]["audited_rows"] == [76975, 27342]
            and all(not lane_entry_scenes[row]["online_overlap"] for row in lane_entry_npcs)
            and all(agent["online_ADE_m"] < 0.2 for agent in lane_entry_npcs.values())
            and all(agent["online_correction_frames"] == 0 for agent in lane_entry_npcs.values())
        ),
        "factual_fidelity_within_five_percent_of_same_plant_hiqr": (
            reports["factual"]["ADE_m"]
            <= 1.05 * reports["factual_hiqr_only"]["ADE_m"]
            and reports["factual"]["FDE_m"]
            <= 1.05 * reports["factual_hiqr_only"]["FDE_m"]
        ),
        "npc_to_npc_factual_response_audited": (
            reports["factual"].get("npc_leader_correction_scenes", 0) > 0
            and reports["factual"].get("npc_leader_correction_frames", 0) > 0
            and reports["factual"].get("factual_raw_overlap_scenes", -1) == 0
            and reports["factual"].get("factual_npc_npc_overlap_scenes", -1) == 0
        ),
        "autonomous_lane_preserves_factual_reconstruction": (
            reports["factual"].get("npc_autonomous_lane_frames") == 0
            and reports["factual"].get("npc_autonomous_lane_scenes") == 0
            and reports["factual"]["factual_raw_overlap_scenes"] == 0
        ),
        "autonomous_lane_responds_to_ads_and_npc_braking": (
            reports["ads"]["same_lane_rear"]["-8.0"].get("receiver_autonomous_lane_scenes", 0) > 0
            and reports["ads"]["same_lane_rear"]["-8.0"]["collision_scene_counts"]["raw"] == 0
            and reports["dose"].get("autonomous_lane_scenes_by_dose", [None])[0]
            == reports["ads"]["same_lane_rear"]["-8.0"]["receiver_autonomous_lane_scenes"]
            and reports["npc_to_npc_highd"]["dose_summary"]["-8.0"].get(
                "follower_autonomous_lane_scenes", 0
            ) > 0
            and reports["npc_to_npc_highd"]["dose_summary"]["-8.0"].get(
                "follower_autonomous_lane_completed_scenes", 0
            ) >= 50
            and all(
                value.get("follower_offroad_scenes") == 0
                and value["npc_npc_overlap_scenes"] == 0
                for value in reports["npc_to_npc_highd"]["dose_summary"].values()
            )
        ),
        "ads_reaches_actual_mapped_target_lane": (
            reports["ads"]["left_lane_rear"]["left"]["ads_lane_change_completion_rate"] == 1.0
            and reports["ads"]["left_lane_rear"]["left"]["ads_final_mapped_lane_error_max_m"] < 0.15
        ),
        "posterior_recording_disjoint_from_test": not (
            posterior_recordings & test_recordings
        ),
        "natural_response_support_audited": (
            natural_support["full_test_rows"] == 10151
            and natural_support["same_lane_rear_scenes_within_45m"] == 1214
            and natural_support["stable_lane_scenes"] > 0
        ),
        "mapped_npc_lane_semantics_covered": (
            reports["factual"].get("npc_planned_lane_change_count", 0) > 0
            and reports["factual"].get("npc_planned_lane_completion_rate", 0.0) >= 0.9
            and reports["spatial_audit"]["planned_lane_online_completed_count"]
            == round(
                reports["factual"]["npc_planned_lane_change_count"]
                * reports["factual"]["npc_planned_lane_completion_rate"]
            )
            and reports["spatial_audit"]["planned_lane_online_completed_count"]
            >= reports["spatial_audit"]["planned_lane_logged_completed_count"]
            and sum(
                item["logged_completed"]
                for item in reports["spatial_audit"]["planned_lane_online_misses"]
            ) <= 2
        ),
        "spatial_path_factual_audit": (
            reports["spatial_audit"]["test_split_size"] == 10151
            and reports["spatial_audit"]["planned_lane_change_count"]
            == reports["factual"]["npc_planned_lane_change_count"]
            and (
                (reports["spatial_audit"]["spatial_path_npc_count"] > 0)
                == (reports["factual"]["scene_with_spatial_path_rate"] > 0.0)
            )
        ),
        "ads_braking_spatial_path_audited": (
            reports["ads_spatial_audit"]["test_split_size"] == 10151
            and reports["ads_spatial_audit"]["planned_lane_change_count"] == 535
            and reports["ads_spatial_audit"]["ads_acceleration_setpoint_mps2"] == -8.0
            and reports["ads_spatial_audit"]["spatial_path_npc_count"]
            == len(reports["ads_spatial_audit"]["spatial_path_npcs"])
            and reports["ads_spatial_audit"]["spatial_path_npc_count"] > 0
            and sum(
                item["completed"]
                for item in reports["ads_spatial_audit"]["spatial_path_npcs"]
            ) >= 21
            and reports["ads_spatial_audit"]["planned_lane_online_completed_count"] >= 497
            and reports["ads_spatial_audit"]["raw_overlap_scenes"] == 0
            and reports["ads_spatial_audit"]["npc_npc_overlap_scenes"] == 0
            and all(
                not item["npc_pair_overlap"]
                for item in reports["ads_spatial_audit"]["spatial_path_npcs"]
            )
        ),
        "no_npc_npc_overlap_in_screened_ads": (
            all(value["collision_scene_counts"]["npc_npc"] == 0
                for value in reports["ads"]["same_lane_rear"].values())
            and reports["ads"]["left_lane_rear"]["left"]["collision_scene_counts"]["npc_npc"] == 0
        ),
        "full_test_strong_ads_braking_has_no_overlap": all(
            reports["ads"]["same_lane_rear"][dose]["collision_scene_counts"]["raw"] == 0
            for dose in ("-8.0", "-6.0")
        ),
        "generated_finite_and_on_road": all(
            value["finite"] == reports["generated"]["scenes"] and value["offroad"] == 0
            for value in reports["generated"]["policies"].values()
        ),
        "generated_left_lane_semantics": (
            reports["generated"]["left_lane_completion_rate"] == 1.0
            and reports["generated"]["left_lane_max_absolute_final_error_m"] < 0.15
            and not reports["generated"]["left_lane_incomplete_examples"]
            and reports["generated"]["policies"]["left"]["geometric_raw_overlap"]
            == reports["generated"]["policies"]["left"]["ads_cutin_exposure"]
        ),
        "highway_plant_matches_offline_sample": (
            reports["highway_parity"]["evaluated_rows"] >= 16
            and reports["highway_parity"]["offline_to_highway_ADE_m"] < 0.001
        ),
        "high_speed_plant_matches_offline": (
            reports["high_speed_parity"]["included_test_row"] == 89300
            and reports["high_speed_parity"]["offline_to_highway_ADE_m"] < 0.001
            and reports["high_speed_parity"]["highway_collision_scene_count"] == 0
            and reports["high_speed_parity"]["highway_offroad_scene_count"] == 0
        ),
        "interventional_spatial_path_matches_highway": (
            reports["ads_highway_parity"]["ads_acceleration_setpoint_mps2"] == -8.0
            and reports["ads_highway_parity"]["spatial_path_active_frame_count_offline"] > 0
            and reports["ads_highway_parity"]["spatial_path_active_frame_count_offline"]
            == reports["ads_highway_parity"]["spatial_path_active_frame_count_highway"]
            and reports["ads_highway_parity"]["spatial_path_activity_agreement_rate"] == 1.0
            and reports["ads_highway_parity"]["offline_to_highway_ADE_m"] < 0.001
            and reports["ads_highway_parity"]["highway_collision_scene_count"] == 0
            and reports["ads_highway_parity"]["highway_offroad_scene_count"] == 0
        ),
        "spatial_brake_gif_shows_causal_lane_change": (
            reports["spatial_brake_gif"]["ads_acceleration_setpoint_mps2"] == -8.0
            and reports["spatial_brake_gif"]["test_row"] == 38725
            and reports["spatial_brake_gif"]["npc_slot"] == 6
            and reports["spatial_brake_gif"]["npc_spatial_path_active_frames"] > 0
            and reports["spatial_logged_gif"]["ads_acceleration_setpoint_mps2"] is None
            and reports["spatial_logged_gif"]["npc_spatial_path_active_frames"] == 0
            and reports["spatial_brake_gif"]["test_row"]
            == reports["spatial_logged_gif"]["test_row"]
            and abs(reports["spatial_brake_gif"]["online_final_target_lane_error_m"]) < 0.75
        ),
        "clear_planned_lane_exit_preserves_reconstruction": (
            reports["clear_lane_escape_gif"]["test_row"] == 46684
            and reports["clear_lane_escape_gif"]["npc_slot"] == 2
            and reports["clear_lane_escape_gif"]["npc_lane_control_active_frames"] == 0
            and abs(reports["clear_lane_escape_gif"]["online_final_target_lane_error_m"]) < 0.75
            and not reports["clear_lane_escape_gif"]["branch_overlap"]["online"]["raw_overlap"]
            and not reports["clear_lane_escape_gif"]["branch_overlap"]["online"]["npc_npc_overlap"]
        ),
        "ads_crossing_blocks_unsafe_lane_escape": (
            reports["ads_npc_crossing_gif"]["test_row"] == 6944
            and reports["ads_npc_crossing_gif"]["npc_slot"] == 2
            and reports["ads_npc_crossing_gif"]["ads_acceleration_setpoint_mps2"] == -8.0
            and reports["ads_npc_crossing_gif"]["npc_spatial_path_active_frames"] > 0
            and abs(reports["ads_npc_crossing_gif"]["online_final_target_lane_error_m"]) < 0.75
            and reports["ads_npc_crossing_gif"]["branch_overlap"]["hiqr_only"]["raw_overlap"]
            and not reports["ads_npc_crossing_gif"]["branch_overlap"]["online"]["raw_overlap"]
        ),
        "opening_lane_gap_gif_shows_safe_completed_change": (
            reports["opening_lane_gap_gif"]["test_row"] == 76975
            and reports["opening_lane_gap_gif"]["npc_slot"] == 6
            and reports["opening_lane_gap_gif"]["npc_lane_control_active_frames"] == 0
            and abs(reports["opening_lane_gap_gif"]["online_final_target_lane_error_m"]) < 0.75
            and not reports["opening_lane_gap_gif"]["branch_overlap"]["online"]["raw_overlap"]
        ),
        "hard_braking_ads_does_not_hide_behind_target_lane_leader": (
            reports["ads_brake_lane_gif"]["test_row"] == 27342
            and reports["ads_brake_lane_gif"]["npc_slot"] == 2
            and reports["ads_brake_lane_gif"]["ads_acceleration_setpoint_mps2"] == -8.0
            and reports["ads_brake_lane_gif"]["branch_overlap"]["hiqr_only"]["raw_overlap"]
            and not reports["ads_brake_lane_gif"]["branch_overlap"]["online"]["raw_overlap"]
            and abs(reports["ads_brake_lane_gif"]["online_final_target_lane_error_m"]) < 0.75
        ),
        "late_clear_lane_change_finishes_without_overlap": all(
            reports[gif]["ads_acceleration_setpoint_mps2"] == -8.0
            and reports[gif]["test_row"] == row
            and reports[gif]["npc_spatial_path_active_frames"] > 0
            and abs(reports[gif]["online_final_target_lane_error_m"]) < 0.75
            and reports[gif]["branch_overlap"]["hiqr_only"]["raw_overlap"]
            and not reports[gif]["branch_overlap"]["online"]["raw_overlap"]
            and reports[parity]["included_test_row"] == row
            and reports[parity]["ads_acceleration_setpoint_mps2"] == -8.0
            and reports[parity]["spatial_path_activity_agreement_rate"] == 1.0
            and reports[parity]["offline_to_highway_ADE_m"] < 0.001
            and reports[parity]["highway_collision_scene_count"] == 0
            and reports[parity]["highway_offroad_scene_count"] == 0
            for gif, parity, row in (
                ("late_finish_open_gif", "late_finish_open_parity", 6261),
                ("late_finish_neighbor_gif", "late_finish_neighbor_parity", 46684),
            )
        ),
        "npc_to_npc_highway_probe": (
            reports["npc_to_npc_gif"]["scope"]
            == "synthetic controlled probe, not a highD test metric"
            and reports["npc_to_npc_gif"]["follower_action_at_second_frame_brake_mps2"]
            < reports["npc_to_npc_gif"]["follower_action_at_second_frame_control_mps2"]
            and reports["npc_to_npc_gif"]["follower_response_frames"] > 0
            and reports["npc_to_npc_gif"]["follower_max_abs_jerk_mps3"] <= 12.01
            and reports["npc_to_npc_gif"]["minimum_net_gap_brake_m"] > 0.0
            and not reports["npc_to_npc_gif"]["npc_npc_overlap"]
            and not reports["npc_to_npc_gif"]["offroad"]
        ),
        "npc_to_npc_highway_sweep": (
            reports["npc_to_npc_sweep"]["scenario_count"] == 36
            and reports["npc_to_npc_sweep"]["finite_scenarios"] == 36
            and reports["npc_to_npc_sweep"]["near_30m_scenarios"] == 24
            and reports["npc_to_npc_sweep"]["near_30m_first_second_correct_scenarios"] == 24
            and reports["npc_to_npc_sweep"]["gap_speed_groups"] == 9
            and reports["npc_to_npc_sweep"]["dose_ordered_gap_speed_groups"] == 9
            and reports["npc_to_npc_sweep"]["npc_npc_overlap_scenarios"] == 0
            and reports["npc_to_npc_sweep"]["offroad_scenarios"] == 0
            and reports["npc_to_npc_sweep"]["maximum_follower_jerk_mps3"] <= 12.01
        ),
        "npc_to_npc_highd_full_test_screen_and_response": (
            reports["npc_to_npc_highd"]["test_split_size"] == 10151
            and reports["npc_to_npc_highd"]["screened_full_test"]
            and reports["npc_to_npc_highd"]["selected_scenes"] == 936
            and len(reports["npc_to_npc_highd"]["records"]) == 936 * 4
            and reports["npc_to_npc_highd"]["brake_doses_mps2"] == [-8.0, -6.0, -4.0, -2.0]
            and all(
                value["scenes"] == 936
                and value["finite_scenes"] == 936
                and value["forced_leader_max_action_error_mps2"] == 0
                and value["npc_npc_overlap_scenes"] == 0
                for value in reports["npc_to_npc_highd"]["dose_summary"].values()
            )
            and reports["npc_to_npc_highd"]["dose_summary"]["-8.0"]["first_second_brake_direction_scenes"] >= 850
            and reports["npc_to_npc_highd"]["dose_summary"]["-8.0"]["near_30m_first_second_brake_direction_scenes"] >= 465
            and reports["npc_to_npc_highd"]["first_second_strict_dose_ordered_scenes"] >= 890
            and reports["npc_to_npc_highd"]["first_second_dose_ordered_with_0p05_tolerance_scenes"] >= 935
            and all(
                all(pair[0] == 0 for pair in item["overlap_pairs"])
                for item in reports["npc_to_npc_highd"]["records"] if item["raw_overlap"]
            )
        ),
        "npc_to_npc_highd_gif_matches_test_report": (
            reports["npc_to_npc_highd_gif"]["test_row"] == 12989
            and reports["npc_to_npc_highd_gif"]["leader_brake_setpoint_mps2"] == -8.0
            and reports["npc_to_npc_highd_gif"]["first_second_follower_effect_mps2"] < -3.0
            and reports["npc_to_npc_highd_gif"]["forced_leader_max_action_error_mps2"] == 0
            and not reports["npc_to_npc_highd_gif"]["brake_raw_overlap"]
            and not reports["npc_to_npc_highd_gif"]["brake_npc_npc_overlap"]
            and any(
                item["test_row"] == reports["npc_to_npc_highd_gif"]["test_row"]
                and item["follower_slot"] == reports["npc_to_npc_highd_gif"]["follower_slot"]
                and item["leader_slot"] == reports["npc_to_npc_highd_gif"]["leader_slot"]
                and item["leader_brake_mps2"] == -8.0
                and abs(item["follower_mean_first_second_effect_mps2"]
                        - reports["npc_to_npc_highd_gif"]["first_second_follower_effect_mps2"]) < 1.0e-5
                for item in reports["npc_to_npc_highd"]["records"]
            )
        ),
        "npc_autonomous_lane_gif_matches_test_report": (
            reports["npc_autonomous_lane_gif"]["test_row"] == 83340
            and reports["npc_autonomous_lane_gif"]["leader_brake_setpoint_mps2"] == -8.0
            and reports["npc_autonomous_lane_gif"]["follower_autonomous_lane_active_frames"] > 0
            and abs(reports["npc_autonomous_lane_gif"][
                "follower_autonomous_lane_final_error_m"
            ]) < 0.75
            and not reports["npc_autonomous_lane_gif"]["brake_raw_overlap"]
            and not reports["npc_autonomous_lane_gif"]["brake_npc_npc_overlap"]
            and any(
                item["test_row"] == 83340
                and item["leader_brake_mps2"] == -8.0
                and item["follower_autonomous_lane_complete"]
                and abs(item["follower_autonomous_lane_target_y_m"]
                        - reports["npc_autonomous_lane_gif"][
                            "follower_autonomous_lane_target_y_m"] ) < 1.0e-4
                for item in reports["npc_to_npc_highd"]["records"]
            )
        ),
        "late_autonomous_lanes_complete_in_highway_continuation": (
            reports["late_lane_continuation"]["schema"]
            == "late_autonomous_lane_continuation_diagnostic_v1"
            and reports["late_lane_continuation"]["test_split_size"] == 10151
            and reports["late_lane_continuation"]["selected_late_incomplete_scenes"]
            == sum(
                item["follower_autonomous_lane"]
                and not item["follower_autonomous_lane_complete"]
                for item in reports["npc_to_npc_highd"]["records"]
            )
            and reports["late_lane_continuation"]["frozen_diffusion_frames"] == 149
            and reports["late_lane_continuation"]["total_frames"] == 224
            and reports["late_lane_continuation"]["not_a_highd_factual_or_diffusion_tail_metric"]
            and reports["late_lane_continuation"]["continuation_api_sha256"]
            == file_sha256(ROOT / "hierarchical_world_model/src/continuation.py")
            and reports["late_lane_continuation"]["prefix_position_ADE_m"] < 0.001
            and reports["late_lane_continuation"]["prefix_max_start_frame_difference"] == 0
            and reports["late_lane_continuation"]["completed_at_149"] == 0
            and reports["late_lane_continuation"]["completed_by_224"]
            == reports["late_lane_continuation"]["selected_late_incomplete_scenes"]
            and reports["late_lane_continuation"]["raw_collision_scenes"] == 0
            and reports["late_lane_continuation"]["npc_npc_collision_scenes"] == 0
            and reports["late_lane_continuation"]["follower_offroad_scenes"] == 0
            and {
                (item["test_row"], item["leader_brake_mps2"], item["follower_slot"])
                for item in reports["late_lane_continuation"]["records"]
            } == {
                (item["test_row"], item["leader_brake_mps2"], item["follower_slot"])
                for item in reports["npc_to_npc_highd"]["records"]
                if item["follower_autonomous_lane"]
                and not item["follower_autonomous_lane_complete"]
            }
            and all(
                item["completed_by_224"]
                and abs(item["final_target_error_m"]) < 0.75
                and abs(item["final_heading_rad"]) < 0.05
                and not item["raw_collision"]
                and not item["follower_offroad"]
                for item in reports["late_lane_continuation"]["records"]
            )
            and file_sha256(
                directory / reports["late_lane_continuation"]["gif_example"]["gif"]
            ) == reports["late_lane_continuation"]["gif_example"]["gif_sha256"]
            and any(
                item["test_row"] == reports["late_lane_continuation"]["gif_example"]["test_row"]
                and item["leader_brake_mps2"] == reports["late_lane_continuation"]["gif_example"]["leader_brake_mps2"]
                and item["completion_frame"] == reports["late_lane_continuation"]["gif_example"]["completion_frame"]
                for item in reports["late_lane_continuation"]["records"]
            )
        ),
        "ads_cutin_clear_pass_gif": (
            reports["ads_cutin_pass_gif"]["schema"] == "online_ads_cutin_pass_gif_v1"
            and reports["ads_cutin_pass_gif"]["test_row"] == 12462
            and reports["ads_cutin_pass_gif"]["npc_slot"] == 4
            and not reports["ads_cutin_pass_gif"]["online_raw_overlap"]
            and reports["ads_cutin_pass_gif"]["no_pass_gate_raw_overlap"]
            and not reports["ads_cutin_pass_gif"]["online_npc_npc_overlap"]
        ),
        "plan_pursuit_gif_shows_suppressed_chase": (
            reports["plan_pursuit_gif"]["test_row"] == 32028
            and reports["plan_pursuit_gif"]["npc_slot"] == 2
            and reports["plan_pursuit_gif"]["ads_acceleration_setpoint_mps2"] == -8.0
            and reports["plan_pursuit_gif"]["positive_hiqr_suppressed_frames"] > 0
            and reports["plan_pursuit_gif"]["online_mean_ax_effect_window_mps2"]
            < reports["plan_pursuit_gif"]["hiqr_base_mean_ax_effect_window_mps2"]
            and not reports["plan_pursuit_gif"]["branch_raw_overlap"]["online"]
        ),
        "gif_available": all(
            (directory / reports[name]["gif"]).is_file()
            for name in (
                "gif", "npc_gif", "spatial_brake_gif", "spatial_logged_gif",
                "npc_to_npc_gif", "clear_lane_escape_gif", "ads_npc_crossing_gif",
                "opening_lane_gap_gif", "ads_brake_lane_gif",
                "late_finish_open_gif", "late_finish_neighbor_gif",
                "plan_pursuit_gif",
                "npc_to_npc_highd_gif",
                "npc_autonomous_lane_gif",
                "ads_cutin_pass_gif",
                "right_lane_meta_gif",
            )
        ) and (directory / reports["late_lane_continuation"]["gif_example"]["gif"]).is_file(),
    }
    output = {
        "controller_sha256": controller_hash,
        "runtime_sha256": runtime_hash,
        "posterior_training_recordings": sorted(posterior_recordings),
        "highd_test_recordings": sorted(test_recordings),
        "checks": checks,
        "consistent": all(checks.values()),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    if not output["consistent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
