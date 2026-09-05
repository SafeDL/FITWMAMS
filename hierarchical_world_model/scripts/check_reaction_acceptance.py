#!/usr/bin/env python3
"""Apply frozen A3/A4 acceptance gates to one validation evaluation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _paired_bootstrap(candidate: list[bool], baseline: list[bool], *, seed: int = 17) -> dict[str, float]:
    left, right = np.asarray(candidate, np.float64), np.asarray(baseline, np.float64)
    if len(left) != len(right) or not len(left):
        raise ValueError("paired collision evidence is missing or misaligned")
    delta = left - right
    rng = np.random.default_rng(seed)
    draws = np.asarray([delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(2000)])
    return {
        "mean": float(delta.mean()), "ci95_low": float(np.quantile(draws, .025)),
        "ci95_high": float(np.quantile(draws, .975)),
    }


def _condition_gates(report: dict, candidate: str, baseline: str, *, response_floor: float) -> tuple[bool, dict]:
    gates: dict[str, dict] = {}
    passed = True
    for duration, doses in report.get("interventions", {}).items():
        for dose, arms in doses.items():
            if candidate not in arms or baseline not in arms:
                passed = False
                gates[f"{duration}/{dose}"] = {"passed": False, "reason": "missing arm"}
                continue
            base_dose = float(arms[baseline]["target_response_dose_mps2"])
            candidate_dose = float(arms[candidate]["target_response_dose_mps2"])
            ratio = candidate_dose / max(base_dose, 1.e-6)
            sequence = report.get("acceptance_sequences", {}).get(duration, {}).get(dose, {})
            try:
                collision = _paired_bootstrap(
                    sequence[candidate]["rear_collision"], sequence[baseline]["rear_collision"],
                )
                collision_passed = collision["mean"] <= .001 and collision["ci95_low"] <= 0.0
            except (KeyError, ValueError):
                collision = {"error": "missing paired rear-collision sequences"}
                collision_passed = False
            condition_passed = ratio >= response_floor and collision_passed
            gates[f"{duration}/{dose}"] = {
                "passed": condition_passed, "response_ratio": ratio,
                "response_floor": response_floor, "rear_collision": collision,
            }
            passed &= condition_passed
    return passed, gates


def evaluate(report: dict) -> dict:
    realism = report.get("supported_rollout_realism", {})
    a2, a3, a4 = (realism.get(name, {}) for name in (
        "A2_rl_residual_idm", "A3_rl_residual_gail", "A4_rl_residual_realism",
    ))
    result: dict[str, object] = {"schema": "reaction_a3_a4_acceptance_v1"}
    a3_kl_improvement = 1.0 - float(a3.get("natural_kl_mean", np.inf)) / max(float(a2.get("natural_kl_mean", np.inf)), 1.e-12)
    a3_synthetic, a3_conditions = _condition_gates(report, "A3_rl_residual_gail", "A2_rl_residual_idm", response_floor=.85)
    result["A3"] = {
        "supported_kl_improvement": a3_kl_improvement,
        "kl_passed": a3_kl_improvement >= .10,
        "synthetic_passed": a3_synthetic, "conditions": a3_conditions,
    }
    a4_synthetic, a4_conditions = _condition_gates(report, "A4_rl_residual_realism", "A3_rl_residual_gail", response_floor=.85)
    a4_composite = float(a4.get("composite_realism", 0.0)) / max(float(a3.get("composite_realism", 0.0)), 1.e-12) - 1.0
    a4_kl_change = float(a4.get("natural_kl_mean", np.inf)) / max(float(a3.get("natural_kl_mean", np.inf)), 1.e-12) - 1.0
    w1_a3, w1_a4 = a3.get("feature_w1", {}), a4.get("feature_w1", {})
    acceleration = float(w1_a4.get("final_ax_mps2", np.inf)) / max(float(w1_a3.get("final_ax_mps2", np.inf)), 1.e-12) - 1.0
    jerk = float(w1_a4.get("abs_jerk_mps3", np.inf)) / max(float(w1_a3.get("abs_jerk_mps3", np.inf)), 1.e-12) - 1.0
    remaining = [
        float(w1_a4.get(name, np.inf)) / max(float(w1_a3.get(name, np.inf)), 1.e-12) - 1.0
        for name in ("gap_m", "closing_mps", "ttc_s")
    ]
    result["A4"] = {
        "composite_improvement": a4_composite, "composite_passed": a4_composite >= .05,
        "acceleration_w1_change": acceleration, "jerk_w1_change": jerk,
        "dynamic_w1_changes": remaining,
        "w1_passed": acceleration <= -.10 and jerk <= -.10 and np.mean(remaining) <= -.05 and max(remaining) <= .05,
        "supported_kl_change": a4_kl_change, "kl_passed": a4_kl_change <= .05,
        "synthetic_passed": a4_synthetic, "conditions": a4_conditions,
    }
    result["A3"]["accepted"] = bool(result["A3"]["kl_passed"] and a3_synthetic)
    result["A4"]["accepted"] = bool(
        result["A4"]["composite_passed"] and result["A4"]["w1_passed"]
        and result["A4"]["kl_passed"] and a4_synthetic
    )
    result["selected_controller"] = "A4_rl_residual_realism" if result["A4"]["accepted"] else (
        "A3_rl_residual_gail" if result["A3"]["accepted"] else "A2_rl_residual_idm"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Check frozen A3/A4 validation gates; never promotes a failing candidate.")
    parser.add_argument("report", type=Path, help="validation four_arm_comparison_*.json")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = evaluate(json.loads(args.report.read_text()))
    output = args.output or args.report.with_name("a3_a4_acceptance.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"selected_controller": result["selected_controller"], "output": str(output)}))


if __name__ == "__main__":
    main()
