#!/usr/bin/env python3
"""Verify one registered world model's IDM subset and Monte-Carlo artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from IDM_subset.src.world_model_registry import get_world_model  # noqa: E402
from world_model.src.core.utils import load_yaml, save_json  # noqa: E402


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _intervals_overlap(left: dict, right: dict) -> bool:
    return bool(
        left["probability_ci95_lower"] <= right["probability_ci95_upper"]
        and right["probability_ci95_lower"] <= left["probability_ci95_upper"]
    )


def _replay_case(spec, evaluator, case: dict) -> bool:
    world = spec.load_exogenous(case["world_exogenous_state"])
    result = evaluator.evaluate(world)
    return bool(
        np.array_equal(result.evt_score, np.asarray([case["evt_score"]]))
        and np.array_equal(result.event_risk, np.asarray([case["event_risk"]]))
        and np.array_equal(result.collision, np.asarray([case["collision"]]))
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify one model's maintained IDM subset and Monte Carlo artifacts."
    )
    parser.add_argument("--model", default="hierarchical")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    spec = get_world_model(args.model)
    config_path = (
        args.config.resolve() if args.config is not None else spec.default_config
    )
    config = load_yaml(config_path)
    subset_dir = (config_path.parent / config["subset_simulation"]["output_dir"]).resolve()
    monte_carlo_dir = (config_path.parent / config["monte_carlo"]["output_dir"]).resolve()
    subset = _load(subset_dir / "world_subset_summary.json")
    monte_carlo = _load(monte_carlo_dir / "world_monte_carlo_summary.json")
    subset_cases = _load(subset_dir / "world_subset_top_cases.json")
    monte_carlo_cases = _load(monte_carlo_dir / "world_monte_carlo_top_cases.json")
    evaluator, provenance = spec.build_evaluator(config, config_path.parent)
    final_population = np.load(subset_dir / "world_subset_final_population.npz")
    declared_space = subset.get("evaluation_contract", {}).get("test_space", {})
    expected_blocks = (
        {
            "test_row_uniform",
            "diffusion_noise",
            "scene_innovations",
            "agent_response_innovations",
        }
        if declared_space.get("test_space") == "empirical_test_fixed_k_gt"
        else set(spec.random_blocks)
    )
    population_has_blocks = expected_blocks.issubset(final_population.files)
    checks = {
        "same_world_model": (
            subset.get("world_model_id") == spec.model_id
            and monte_carlo.get("world_model_id") == spec.model_id
        ),
        "same_failure_threshold": (
            subset["failure_event"] == monte_carlo["failure_event"]
        ),
        "same_formal_artifacts": all(
            subset["provenance"][key] == monte_carlo["provenance"][key]
            for key in spec.provenance_keys
        ),
        "subset_numerically_valid": all(
            row["numerical_valid_fraction"] == 1.0
            for row in subset["level_statistics"]
        ),
        "monte_carlo_numerically_valid": (
            monte_carlo["numerical_valid_fraction"] == 1.0
        ),
        "subset_population_has_all_randomness_blocks": population_has_blocks,
        "subset_case_replay_exact": _replay_case(
            spec, evaluator, subset_cases[0]
        ),
        "monte_carlo_case_replay_exact": _replay_case(
            spec, evaluator, monte_carlo_cases[0]
        ),
        "subset_and_monte_carlo_ci_overlap": _intervals_overlap(
            subset["uncertainty"],
            monte_carlo["uncertainty"],
        ),
        "run_provenance_matches_current_formal_artifacts": all(
            subset["provenance"][key] == provenance[key]
            for key in spec.provenance_keys
        ),
        "highway_env_execution": (
            subset["provenance"].get("execution_backend") == "local_highway_env"
            and monte_carlo["provenance"].get("execution_backend")
            == "local_highway_env"
        ),
    }
    report = {
        "schema": "highway_env_idm_world_model_acceptance_v2",
        "world_model_id": spec.model_id,
        "world_model": spec.display_name,
        "checks": checks,
        "all_passed": bool(all(checks.values())),
        "subset_probability": subset["probability"],
        "monte_carlo_probability": monte_carlo["probability"],
        "sources": {
            "subset": str(subset_dir / "world_subset_summary.json"),
            "monte_carlo": str(monte_carlo_dir / "world_monte_carlo_summary.json"),
        },
    }
    save_json(report, subset_dir.parent / "acceptance.json")
    if not report["all_passed"]:
        raise RuntimeError("current-world IDM acceptance failed")


if __name__ == "__main__":
    main()
