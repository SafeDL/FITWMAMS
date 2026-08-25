#!/usr/bin/env python3
"""Verify current-world IDM subset and Monte Carlo artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from IDM_subset.src.world_subset_runner import _build_evaluator  # noqa: E402
from hierarchical_world_model.src.randomness import (  # noqa: E402
    WorldExogenousState,
)
from world_model.src.core.utils import load_yaml, save_json  # noqa: E402

CONFIG = ROOT / "IDM_subset/configs/world_subset_idm.yaml"


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _intervals_overlap(left: dict, right: dict) -> bool:
    return bool(
        left["probability_ci95_lower"] <= right["probability_ci95_upper"]
        and right["probability_ci95_lower"] <= left["probability_ci95_upper"]
    )


def _replay_case(evaluator, case: dict) -> bool:
    world = WorldExogenousState.load(case["world_exogenous_state"])
    result = evaluator.evaluate(world)
    return bool(
        np.array_equal(result.evt_score, np.asarray([case["evt_score"]]))
        and np.array_equal(result.event_risk, np.asarray([case["event_risk"]]))
        and np.array_equal(result.collision, np.asarray([case["collision"]]))
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the maintained IDM subset and Monte Carlo artifacts."
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    subset_dir = (config_path.parent / config["subset_simulation"]["output_dir"]).resolve()
    monte_carlo_dir = (config_path.parent / config["monte_carlo"]["output_dir"]).resolve()
    subset = _load(subset_dir / "world_subset_summary.json")
    monte_carlo = _load(monte_carlo_dir / "world_monte_carlo_summary.json")
    subset_cases = _load(subset_dir / "world_subset_top_cases.json")
    monte_carlo_cases = _load(monte_carlo_dir / "world_monte_carlo_top_cases.json")
    evaluator, provenance = _build_evaluator(config, config_path.parent)
    final_population = np.load(subset_dir / "world_subset_final_population.npz")
    required_blocks = set(WorldExogenousState.__dataclass_fields__) - {
        "batch_size",
        "response_steps",
    }
    population_has_blocks = required_blocks.issubset(final_population.files)
    checks = {
        "same_failure_threshold": (
            subset["failure_event"] == monte_carlo["failure_event"]
        ),
        "same_formal_artifacts": all(
            subset["provenance"][key] == monte_carlo["provenance"][key]
            for key in (
                "flow_checkpoint_sha256",
                "diffusion_checkpoint_sha256",
                "response_checkpoint_sha256",
                "evt_model_sha256",
                "idm_ego_config_sha256",
                "execution_backend",
                "hiqr_vehicle_dynamics_contract",
            )
        ),
        "subset_numerically_valid": all(
            row["numerical_valid_fraction"] == 1.0
            for row in subset["level_statistics"]
        ),
        "monte_carlo_numerically_valid": (
            monte_carlo["numerical_valid_fraction"] == 1.0
        ),
        "subset_population_has_all_randomness_blocks": population_has_blocks,
        "subset_case_replay_exact": _replay_case(evaluator, subset_cases[0]),
        "monte_carlo_case_replay_exact": _replay_case(evaluator, monte_carlo_cases[0]),
        "subset_and_monte_carlo_ci_overlap": _intervals_overlap(
            subset["uncertainty"],
            monte_carlo["uncertainty"],
        ),
        "run_provenance_matches_current_formal_artifacts": all(
            subset["provenance"][key] == provenance[key]
            for key in (
                "flow_checkpoint_sha256",
                "diffusion_checkpoint_sha256",
                "response_checkpoint_sha256",
                "evt_model_sha256",
                "idm_ego_config_sha256",
                "execution_backend",
                "hiqr_vehicle_dynamics_contract",
            )
        ),
        "highway_env_execution": (
            subset["provenance"].get("execution_backend") == "local_highway_env"
            and monte_carlo["provenance"].get("execution_backend")
            == "local_highway_env"
        ),
    }
    report = {
        "schema": "highway_env_idm_acceptance_v3",
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
