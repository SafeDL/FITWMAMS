#!/usr/bin/env python3
"""Compose the full ADS sweep with its deterministic full-cohort lane audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from traffic_components.src.core.utils import file_sha256, load_json, save_json


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/hierarchical_world_model/evaluation"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=RESULTS / "generated_ads_1024_candidate.json")
    parser.add_argument("--lane-audit", type=Path, default=RESULTS / "generated_left_lane_audit.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "generated_ads_sweep.json")
    args = parser.parse_args()
    base = load_json(args.base)
    lane = load_json(args.lane_audit)
    if base.get("schema") != "generated_online_ads_sweep_v1":
        raise ValueError("base report must be a generated_online_ads_sweep_v1 report")
    if lane.get("schema") != "generated_left_lane_completion_audit_v1":
        raise ValueError("lane audit has an unexpected schema")
    for key in ("scenes", "batch_size", "seed_first_batch", "online_runtime_sha256"):
        if base.get(key) != lane.get(key):
            raise ValueError(f"base report and lane audit disagree on {key}")
    scenes = int(base["scenes"])
    if lane.get("finite_scenes") != scenes or lane.get("offroad_scenes") != 0:
        raise ValueError("left-lane replay must be finite and on-road in every scene")
    if (
        lane.get("completion_rate") != 1.0
        or lane.get("maximum_absolute_final_lane_error_m", float("inf")) >= 0.15
        or lane.get("incomplete_examples")
    ):
        raise ValueError("left-lane semantic completion gate did not pass")

    result = dict(base)
    result["schema"] = "generated_online_ads_sweep_v2"
    policies = dict(base["policies"])
    left = dict(policies["left"])
    left.update({
        "finite": lane["finite_scenes"],
        "collision": lane["collision_scenes"],
        "offroad": lane["offroad_scenes"],
        "geometric_raw_overlap": lane["raw_overlap_scenes"],
        "geometric_adjusted_overlap": lane["adjusted_overlap_scenes"],
        "ads_pursuit_exposure": 0,
        "ads_cutin_exposure": lane["ads_cutin_exposure_scenes"],
    })
    policies["left"] = left
    result["policies"] = policies
    result["left_lane_completion_rate"] = lane["completion_rate"]
    result["left_lane_max_absolute_final_error_m"] = lane[
        "maximum_absolute_final_lane_error_m"
    ]
    result["left_lane_incomplete_examples"] = lane["incomplete_examples"]
    result["left_lane_ads_policy"] = (
        "map-target semantic lane change; feedback maintains the realized entry speed "
        "from frame 25 through the manoeuvre; this is not an acceleration-dose arm"
    )
    result["left_lane_audit_source"] = str(args.lane_audit.resolve().relative_to(ROOT))
    result["policy_results_composition"] = (
        "Seven acceleration/hold policies from the full single-pass sweep; the left-lane "
        "policy was replayed separately on the exact same exogenous worlds and seeds."
    )
    result["evaluation_script_sha256"] = file_sha256(
        ROOT / "hierarchical_world_model/scripts/evaluate_generated_online_ads.py"
    )
    result["ads_test_policy_sha256"] = file_sha256(
        ROOT / "hierarchical_world_model/scripts/ads_test_policies.py"
    )
    result["limitations"] = (
        f"This {scenes}-scene Flow-generated runtime audit is separate from held-out highD Test; "
        "longer-horizon replanning and wider ADS policy coverage remain open."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(result, args.output)
    print({
        "output": str(args.output),
        "scenes": scenes,
        "policies": list(policies),
        "left_lane_completion_rate": result["left_lane_completion_rate"],
        "left_lane_max_absolute_final_error_m": result[
            "left_lane_max_absolute_final_error_m"
        ],
        "left_lane_geometric_overlaps": left["geometric_raw_overlap"],
        "left_lane_offroad": left["offroad"],
    })


if __name__ == "__main__":
    main()
