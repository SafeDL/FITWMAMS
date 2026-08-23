#!/usr/bin/env python3
"""Assess a Stochastic Causal HiQR candidate against declared acceptance gates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.core.utils import load_json, save_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-energy-improvement", type=float, default=0.02)
    parser.add_argument("--minimum-terminal-multiplier", type=float, default=2.0)
    args = parser.parse_args()
    baseline = load_json(args.baseline)
    candidate = load_json(args.candidate)
    base_factual = baseline["factual_fidelity"]["diffusion_guided_hiqr"]
    candidate_factual = candidate["factual_fidelity"]["diffusion_guided_hiqr"]
    base_random = baseline["distribution_stochasticity"]
    candidate_random = candidate["distribution_stochasticity"]
    effects = candidate["intervention_effectiveness"]
    energy_improvement = 1.0 - (
        candidate_random["energy_score_m"] / base_random["energy_score_m"]
    )
    terminal_multiplier = (
        candidate_random["terminal_pairwise_distance_m"]
        / base_random["terminal_pairwise_distance_m"]
    )
    factual = {
        "ADE_within_15pct": candidate_factual["ADE_m"] <= 1.15 * base_factual["ADE_m"],
        "FDE_within_20pct": candidate_factual["FDE_m"] <= 1.20 * base_factual["FDE_m"],
        "P95_within_20pct": candidate_factual["P95_displacement_error_m"]
        <= 1.20 * base_factual["P95_displacement_error_m"],
    }
    randomness = {
        "energy_improvement_gate": energy_improvement >= args.minimum_energy_improvement,
        "terminal_diversity_gate": terminal_multiplier >= args.minimum_terminal_multiplier,
    }
    interventions = {
        "brake_direction_at_least_0p95": effects["brake"]["direction_success_rate"] >= 0.95,
        "accelerate_direction_at_least_0p95": effects["accelerate"]["direction_success_rate"] >= 0.95,
        "brake_monotonicity_at_least_0p95": effects["brake"]["dose_monotonicity_rate"] >= 0.95,
        "accelerate_monotonicity_at_least_0p95": effects["accelerate"]["dose_monotonicity_rate"] >= 0.95,
        "lateral_direction_at_least_0p90": effects["left"]["separation_non_decrease_rate"] >= 0.90,
        "lateral_monotonicity_at_least_0p90": effects["left"]["strong_to_mild_response_ratio"] >= 0.90,
        "brake_natural_coverage_at_least_0p60": effects["brake"][
            "response_within_natural_p10_p90_rate"
        ] >= 0.60,
        "accelerate_natural_coverage_at_least_0p50": effects["accelerate"][
            "response_within_natural_p10_p90_rate"
        ] >= 0.50,
        "locality_below_0p15": all(
            effects[name]["locality_ratio_far_to_near"] < 0.15
            for name in ("brake", "accelerate", "left")
        ),
    }
    gates = {**factual, **randomness, **interventions}
    report = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "energy_improvement_fraction": energy_improvement,
        "terminal_diversity_multiplier": terminal_multiplier,
        "gates": gates,
        "promoted_to_full_training": all(gates.values()),
    }
    save_json(report, args.output)
    print(report)
    if not report["promoted_to_full_training"]:
        raise SystemExit("pilot rejected: full training is not authorized")


if __name__ == "__main__":
    main()
