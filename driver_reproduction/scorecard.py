"""Build a provenance-preserving scorecard from retained model evidence.

The scorecard intentionally does not rank natural-rollout results from different
data splits.  A number is directly comparable only within its ``benchmark_id``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MEDIUM_BRAKE_DOSE = "3.0"


def _read(relative_path: str, root: Path = ROOT) -> dict[str, Any]:
    path = root / relative_path
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _natural_row(
    *,
    model: str,
    benchmark_id: str,
    comparison_status: str,
    protocol: dict[str, Any],
    metrics: dict[str, Any],
    evidence: str,
) -> dict[str, Any]:
    return {
        "model": model,
        "benchmark_id": benchmark_id,
        "comparison_status": comparison_status,
        "protocol": protocol,
        "metrics": metrics,
        "evidence": evidence,
    }


def build_scorecard(root: Path = ROOT) -> dict[str, Any]:
    """Normalize final evidence without changing any model-specific result."""
    population = _read("bayesian_ma_idm/evidence/population/summary.json", root)
    dynamic = _read(
        "dynamic_ar_idm/artifacts/full_evaluation/"
        "natural_metrics_full_stable_map_rho.json",
        root,
    )
    multi_regime = _read(
        "multi_regime_bidm/evidence/heldout/recording_heldout_metrics.json", root
    )
    active = _read(
        "active_inference_driver/artifacts/full_highd/highd_evaluation.json", root
    )
    response = _read("counterfactual_response/artifacts/response_metrics.json", root)
    matched = _read(
        "results/driver_reproduction/matched_highd/matched_metrics.json", root
    )

    natural_rollouts = []
    for model_name, model_result in population["models"].items():
        natural_rollouts.append(
            _natural_row(
                model=model_name,
                benchmark_id="highd_oof_source_loader_25hz_v1",
                comparison_status="comparable_within_benchmark",
                protocol={
                    "data_split": population["protocol"],
                    "pairs": model_result["pairs"],
                    "anchors": model_result["anchors"],
                    "futures": model_result["futures"],
                    "native_fps": 25,
                    "decision_fps": 5,
                },
                metrics=model_result["pair_weighted_horizons"],
                evidence="bayesian_ma_idm/evidence/population/summary.json",
            )
        )

    natural_rollouts.append(
        _natural_row(
            model="dynamic_ar5",
            benchmark_id="highd_in_sample_driver_posterior_25hz_v1",
            comparison_status="not_comparable_to_oof_results",
            protocol={
                "data_split": dynamic["evaluation"],
                "pairs": dynamic["pairs"],
                "anchors": dynamic["anchors"],
                "futures": dynamic["futures"],
                "native_fps": dynamic["native_fps"],
                "decision_fps": dynamic["decision_fps"],
            },
            metrics=dynamic["horizons"],
            evidence=(
                "dynamic_ar_idm/artifacts/full_evaluation/"
                "natural_metrics_full_stable_map_rho.json"
            ),
        )
    )
    natural_rollouts.extend(
        [
            _natural_row(
                model="pooled_b_idm",
                benchmark_id="highd_recording_heldout_small_cohort_25hz_v1",
                comparison_status="comparable_within_benchmark",
                protocol={
                    "data_split": multi_regime["kind"],
                    "pairs": multi_regime["test_pairs"],
                    "test_recordings": multi_regime["test_recordings"],
                    "native_fps": 25,
                    "decision_fps": 5,
                },
                metrics=multi_regime["pooled_b_idm"],
                evidence="multi_regime_bidm/evidence/heldout/recording_heldout_metrics.json",
            ),
            _natural_row(
                model="multi_regime_b_idm",
                benchmark_id="highd_recording_heldout_small_cohort_25hz_v1",
                comparison_status="rejected_not_well_calibrated",
                protocol={
                    "data_split": multi_regime["kind"],
                    "pairs": multi_regime["test_pairs"],
                    "test_recordings": multi_regime["test_recordings"],
                    "native_fps": 25,
                    "decision_fps": 5,
                },
                metrics=multi_regime["filtered_hierarchical"],
                evidence="multi_regime_bidm/evidence/heldout/recording_heldout_metrics.json",
            ),
        ]
    )

    dose_response = []
    for model_name, model_result in response["models"].items():
        dose = model_result["dose_response"][MEDIUM_BRAKE_DOSE]
        dose_response.append(
            {
                "model": model_name,
                "benchmark_id": "highd_counterfactual_braking_anchor_55398_v1",
                "comparison_status": "comparable_response_mechanics_only",
                "metrics": {
                    "response_probability": dose["response_probability"],
                    "median_response_latency_s": dose["response_latency_s"]["median"],
                    "median_peak_additional_braking_mps2": (
                        dose["peak_additional_braking_mps2"]["median"]
                    ),
                    "median_terminal_gap_benefit_m": (
                        dose["terminal_gap_benefit_m"]["median"]
                    ),
                    "median_natural_gap_rmse_to_highd_m": (
                        dose["natural_gap_rmse_to_highd_m"]["median"]
                    ),
                },
                "evidence": "counterfactual_response/artifacts/response_metrics.json",
            }
        )

    return {
        "schema_version": 1,
        "contract": {
            "source_dataset": "highD",
            "native_fps": 25,
            "driver_decision_fps": 5,
            "plant": "25 Hz ballistic; action held for five native frames",
            "rule": (
                "Direct ranking is allowed only between rows with the same "
                "benchmark_id. Different benchmark_id values are evidence records, "
                "not a leaderboard."
            ),
        },
        "natural_rollouts": natural_rollouts,
        "active_inference_adaptation": {
            "benchmark_id": "highd_leader_brake_events_adapted_25hz_v1",
            "comparison_status": "separate_event_response_evaluation",
            "model": active["model"],
            "events": active["events"],
            "metrics": active["summary"],
            "evidence": "active_inference_driver/artifacts/full_highd/highd_evaluation.json",
        },
        "matched_highd_adaptation": {
            "benchmark_id": matched["benchmark_id"],
            "comparison_status": "directly_comparable_test_protocol_with_model_specific_training",
            "protocol": {
                key: matched[key]
                for key in (
                    "dataset", "cohort_pairs", "training_recordings", "training_pairs",
                    "test_recordings", "test_pairs", "prefix_s", "horizon_s", "futures",
                    "native_fps", "decision_fps", "all_eligible_test_pairs_evaluated",
                )
            },
            "models": matched["models"],
            "comparison_limit": matched["comparison_limit"],
            "evidence": "results/driver_reproduction/matched_highd/matched_metrics.json",
        },
        "counterfactual_braking": {
            "benchmark_id": "highd_counterfactual_braking_anchor_55398_v1",
            "dose_mps2": float(MEDIUM_BRAKE_DOSE),
            "futures": response["futures"],
            "scenario": response["scenario"],
            "rows": dose_response,
            "interpretation": (
                "This is a matched internal response-mechanics test. It does not "
                "identify real human counterfactual correctness."
            ),
        },
    }


def write_scorecard(output: Path, root: Path = ROOT) -> Path:
    """Write a stable, reviewable JSON scorecard and return its path."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(build_scorecard(root), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return output
