#!/usr/bin/env python3
"""Apply frozen calibrated-residual acceptance gates to an evaluation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def evaluate(report: dict) -> dict:
    """Do not tune thresholds or promote a failed candidate.

    The evaluator supplies event-level paired Energy Scores and factual/OOD
    diagnostics.  This intentionally never uses collision rate as a proxy for
    human likeness.
    """
    event = report.get("held_out_events", {})
    factual = report.get("factual", {})
    ood = report.get("physical_ood", {})
    comparison = event.get("arms", {}).get("calibrated_residual", {})
    bootstrap = comparison.get("paired_energy_score", {})
    diagnostic_gates = {
        name: float(values.get("ci95_high", np.inf)) <= float(values.get("allowed_degradation", -np.inf))
        for name, values in event.get("paired_diagnostics", {}).items()
    }
    gates = {
        "human_energy_score": float(bootstrap.get("lcb95", -np.inf)) > 0.,
        "human_diagnostics": bool(diagnostic_gates) and all(diagnostic_gates.values()),
        "factual_noninferior": bool(factual.get("calibrated_residual", {}).get("noninferior", False)),
        "physical_valid": bool(ood.get("calibrated_residual", {}).get("valid", False)),
        "no_jerk_limiter_failure": not bool(ood.get("calibrated_residual", {}).get("jerk_limiter_failed", True)),
    }
    accepted = all(gates.values())
    return {
        "schema_name": "reaction_policy_acceptance", "schema_version": 1,
        "candidate": "calibrated_residual", "baseline": "a2_transfer",
        "gates": gates, "diagnostic_gates": diagnostic_gates, "accepted": accepted,
        "selected_controller": "calibrated_residual" if accepted else "a2_transfer",
        "reason": "all frozen human-evidence, factual, and physical gates passed" if accepted else "candidate retained as evidence; A2-transfer remains active",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(json.loads(args.report.read_text()))
    output = args.output or args.report.with_name("reaction_policy_acceptance.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"selected_controller": result["selected_controller"], "output": str(output)}))


if __name__ == "__main__":
    main()
