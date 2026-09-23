"""Build the protocol-aware highD behavior-world-model comparison."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTERACTION_SUMMARY = (
    ROOT
    / "results"
    / "interactive_behavior_world_model"
    / "benchmark_v1"
    / "aggregate"
    / "test_seed_summary.csv"
)
OUTPUT_DIR = ROOT / "results" / "comparisons"

METHOD_LABELS = {
    "idm_lane_keep": "IDM lane keep",
    "response_a2": "Response A2",
    "rolling_action_b2": "B2",
    "rolling_action_b2r": "B2-R",
    "rolling_action_b2rl_s2": "B2-RL",
    "rolling_action_b2rlp_e3": "B2-RLP-e3",
    "semantic_c2": "Semantic C2",
    "trafficbots_v15_all_background": "TrafficBots V1.5-highD",
}

NATURAL_METRICS = (
    "fair_energy_score",
    "sample_mean_ADE_m",
    "sample_mean_FDE_m",
    "joint_min_ADE_m",
    "joint_min_corresponding_FDE_m",
    "mean_pairwise_trajectory_distance_m",
    "coverage_90_dx_m",
    "coverage_90_dy_m",
    "new_collision_probability",
)
EVENT_METRICS = (
    "fair_energy_score",
    "sample_mean_ADE_m",
    "sample_mean_FDE_m",
    "lane_behavior_brier",
    "braking_behavior_brier",
)


def load_json(relative_path: str) -> dict:
    with (ROOT / relative_path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_interaction_rows() -> list[dict[str, str]]:
    with INTERACTION_SUMMARY.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate_suite(
    rows: list[dict[str, str]], suite: str, metrics: tuple[str, ...]
) -> list[dict[str, object]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        if row["suite"] == suite and row["metric"] in metrics:
            indexed.setdefault(row["method_id"], {})[row["metric"]] = row

    output = []
    for method_id in METHOD_LABELS:
        method_rows = indexed.get(method_id)
        if not method_rows:
            continue
        record: dict[str, object] = {
            "method_id": method_id,
            "method": METHOD_LABELS[method_id],
            "fit_seeds": int(next(iter(method_rows.values()))["fit_seeds"]),
            "rows_per_seed": int(next(iter(method_rows.values()))["rows_per_seed"]),
        }
        for metric in metrics:
            if metric in method_rows:
                record[metric] = float(method_rows[metric]["seed_mean"])
        output.append(record)
    return output


def probe_record(
    rows: list[dict[str, str]], method_id: str, probe_name: str
) -> dict[str, object]:
    marker = f"/t2b_{probe_name}_"
    selected = [
        row
        for row in rows
        if row["method_id"] == method_id and marker in row["configuration_id"]
    ]
    if not selected:
        raise ValueError(f"missing {probe_name} results for {method_id}")
    return {
        "method_id": method_id,
        "method": METHOD_LABELS[method_id],
        "fit_seeds": int(selected[0]["fit_seeds"]),
        "rows_per_seed": int(selected[0]["rows_per_seed"]),
        **{row["metric"]: float(row["seed_mean"]) for row in selected},
    }


def closed_loop_record(
    rows: list[dict[str, str]], method_id: str, controller: str
) -> dict[str, object]:
    configuration = f"{method_id}/t3_fixed_pnc_{method_id}_{controller}_ddim8"
    selected = [row for row in rows if row["configuration_id"] == configuration]
    if not selected:
        raise ValueError(f"missing {controller} closed-loop results for {method_id}")
    wanted = {
        "ego_collision_probability",
        "npc_collision_probability",
        "npc_offroad_probability",
        "ego_progress_m",
    }
    values = {
        row["metric"]: float(row["seed_mean"])
        for row in selected
        if row["metric"] in wanted
    }
    return {
        "method_id": method_id,
        "method": METHOD_LABELS[method_id],
        "controller": controller,
        "fit_seeds": int(selected[0]["fit_seeds"]),
        "rows_per_seed": int(selected[0]["rows_per_seed"]),
        **values,
    }


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    fields = list(records[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def build_summary() -> dict:
    rows = load_interaction_rows()
    natural = aggregate_suite(rows, "t1_natural", NATURAL_METRICS)
    events = aggregate_suite(rows, "t2a_logged_events", EVENT_METRICS)
    interactive_methods = ("rolling_action_b2rl_s2", "rolling_action_b2rlp_e3")
    longitudinal = [
        probe_record(rows, method_id, "probes") for method_id in interactive_methods
    ]
    lateral = [
        probe_record(rows, method_id, "lateral_probes")
        for method_id in interactive_methods
    ]
    closed_loop = [
        closed_loop_record(rows, method_id, controller)
        for method_id in interactive_methods
        for controller in ("cruise", "trajectory_mpc")
    ]

    cache_manifest = load_json(
        "results/highd_shared_training_data/highd_sequence_cache/"
        "sequence_cache/manifest.json"
    )
    interaction_manifest = load_json(
        "results/interactive_behavior_world_model/benchmark_v1/dataset_manifest.json"
    )
    factual = load_json(
        "results/hierarchical_world_model/factual_all_slots/evaluation.json"
    )
    diffusion = load_json("results/background_diffusion/evaluation_summary.json")
    flow = load_json("results/highd_natural_driving_flow/evaluation_summary.json")
    drivers = load_json(
        "results/driver_reproduction/matched_highd/matched_metrics.json"
    )
    active_inference = load_json(
        "reproduction/models/active_inference_driver/artifacts/full_highd/"
        "highd_evaluation.json"
    )
    cih_decision = load_json(
        "results/hierarchical_world_model/cih_wm/continuation/decision.json"
    )

    driver_rows = [
        {"model": model, **metrics} for model, metrics in drivers["models"].items()
    ]

    return {
        "schema": "highd_behavior_world_model_comparison_v1",
        "comparison_rule": (
            "Rank only records within the same protocol block. Conditional factual "
            "reconstruction, prefix-only world models, scenario priors, and "
            "car-following drivers are not one leaderboard."
        ),
        "highd_coverage": {
            "recordings": interaction_manifest["recording_split_audit"]["recordings"],
            "canonical_sequences": cache_manifest["num_sequences"],
            "canonical_split": cache_manifest["split_summary"],
            "strict_causal_split": interaction_manifest["split_summary"],
            "recording_disjoint_split_passed": interaction_manifest[
                "recording_split_audit"
            ]["passed"],
        },
        "prefix_only_all_background": {
            "natural_test": natural,
            "logged_event_test": events,
            "longitudinal_intervention_test": longitudinal,
            "lateral_intervention_test": lateral,
            "closed_loop_test": closed_loop,
            "source": str(INTERACTION_SUMMARY.relative_to(ROOT)),
        },
        "conditional_factual_reference": {
            "conditions": (
                "Flow/diffusion future constraints are available; this is factual "
                "reconstruction, not prefix-only autonomous prediction."
            ),
            "test_sequences": factual["test_sequences"],
            "metrics": factual["factual_fidelity"]["diffusion_guided_hiqr"],
            "cih_response_candidate_status": cih_decision["status"],
            "cih_failed_gates": cih_decision["failed_gates"],
            "source": (
                "results/hierarchical_world_model/factual_all_slots/evaluation.json"
            ),
        },
        "oracle_knot_conditioned_diffusion": {
            "condition_disclosure": diffusion["condition_disclosure"],
            "metrics": diffusion["metrics"]["all"],
            "source": "results/background_diffusion/evaluation_summary.json",
        },
        "scenario_prior_flow": {
            "train": flow["num_train"],
            "validation": flow["num_validation"],
            "test": flow["num_test"],
            "held_out_nll": flow["held_out_nll"],
            "c0_mean_ks": flow["distribution"]["c0_mean_ks"],
            "k_mean_ks": flow["distribution"]["k"]["mean_ks"],
            "k_max_ks": max(
                item["ks"] for item in flow["distribution"]["k"]["per_feature"]
            ),
            "physical_validity": flow["physical_validity"],
            "source": "results/highd_natural_driving_flow/evaluation_summary.json",
        },
        "matched_car_following": {
            "cohort_pairs": drivers["cohort_pairs"],
            "training_pairs": drivers["training_pairs"],
            "test_pairs": drivers["test_pairs"],
            "training_recordings": drivers["training_recordings"],
            "test_recordings": drivers["test_recordings"],
            "models": driver_rows,
            "scope_warning": drivers["comparison_scope"],
            "source": "results/driver_reproduction/matched_highd/matched_metrics.json",
        },
        "active_inference_highd_adaptation": {
            "events": active_inference["events"],
            "metrics": active_inference["summary"],
            "no_steering_adaptation": active_inference["no_steering_adaptation"],
            "source": (
                "reproduction/models/active_inference_driver/artifacts/full_highd/"
                "highd_evaluation.json"
            ),
        },
    }


def main() -> None:
    summary = build_summary()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "highd_behavior_world_model_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    prefix_only = summary["prefix_only_all_background"]
    write_csv(OUTPUT_DIR / "highd_natural_test.csv", prefix_only["natural_test"])
    write_csv(
        OUTPUT_DIR / "highd_logged_event_test.csv", prefix_only["logged_event_test"]
    )
    write_csv(
        OUTPUT_DIR / "highd_matched_car_following.csv",
        summary["matched_car_following"]["models"],
    )


if __name__ == "__main__":
    main()
