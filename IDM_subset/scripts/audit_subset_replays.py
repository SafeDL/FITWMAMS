#!/usr/bin/env python3
"""Replay retained AMS cases and audit deterministic metric reproduction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from IDM_subset.src.world_model_registry import get_world_model  # noqa: E402
from world_model.src.core.utils import load_json, load_yaml, save_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit exact replay of one model's retained AMS top cases."
    )
    parser.add_argument("--model", default="hierarchical")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    spec = get_world_model(args.model)
    config_path = (
        args.config.resolve() if args.config is not None else spec.default_config
    )
    config = load_yaml(config_path)
    subset_dir = (
        config_path.parent / config["subset_simulation"]["output_dir"]
    ).resolve()
    cases = load_json(subset_dir / "world_subset_top_cases.json")
    evaluator, provenance = spec.build_evaluator(config, config_path.parent)
    records = []
    for case in cases:
        world = spec.load_exogenous(case["world_exogenous_state"])
        replay = evaluator.evaluate(world)
        score_error = abs(float(replay.evt_score[0]) - float(case["evt_score"]))
        risk_error = abs(float(replay.event_risk[0]) - float(case["event_risk"]))
        checks = {
            "evt_score_exact": bool(
                np.array_equal(replay.evt_score, np.asarray([case["evt_score"]]))
            ),
            "event_risk_exact": bool(
                np.array_equal(replay.event_risk, np.asarray([case["event_risk"]]))
            ),
            "collision_exact": bool(
                np.array_equal(replay.collision, np.asarray([case["collision"]]))
            ),
            "numerical_valid": bool(replay.numerical_valid[0]),
        }
        records.append(
            {
                "case_id": case["case_id"],
                "checks": checks,
                "all_passed": bool(all(checks.values())),
                "evt_score_abs_error": score_error,
                "event_risk_abs_error": risk_error,
            }
        )
    report = {
        "schema": "idm_subset_top_case_replay_audit_v1",
        "world_model_id": spec.model_id,
        "world_model": spec.display_name,
        "num_cases": len(records),
        "all_passed": bool(records and all(item["all_passed"] for item in records)),
        "source_summary": str(subset_dir / "world_subset_summary.json"),
        "provenance": provenance,
        "cases": records,
    }
    path = subset_dir / "subset_replay_audit.json"
    save_json(report, path)
    print(path)
    if not report["all_passed"]:
        raise RuntimeError("subset top-case replay audit failed")


if __name__ == "__main__":
    main()
