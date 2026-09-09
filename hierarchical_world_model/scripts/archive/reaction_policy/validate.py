#!/usr/bin/env python3
"""Apply frozen calibrated-residual acceptance gates to an evaluation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def evaluate(report: dict, *, stage: str) -> dict:
    """Do not tune thresholds or promote a failed candidate.

    The evaluator supplies event-level paired Energy Scores and factual/OOD
    diagnostics.  This intentionally never uses collision rate as a proxy for
    human likeness.
    """
    event = report.get("held_out_events", {})
    factual = report.get("factual", {})
    ood = report.get("physical_ood", {})
    pairs = event.get("paired_energy_score", {})
    a2_pair = pairs.get("a2_transfer_minus_calibrated_residual", {})
    supervised_pair = pairs.get("calibrated_supervised_minus_calibrated_residual", {})
    collision = event.get("paired_rear_collision", {}).get("a2_transfer", {})
    evidence_sufficient = (
        int(event.get("events", 0)) >= 100
        and int(event.get("recordings", 0)) >= 5
    )
    diagnostic_gates = {
        name: float(values.get("ci95_high", np.inf)) <= float(values.get("allowed_degradation", -np.inf))
        for name, values in event.get("paired_diagnostics", {}).items()
    }
    supervised_mean = float(
        event.get("arms", {}).get("calibrated_supervised", {}).get(
            "energy_score_mean", np.inf,
        )
    )
    stage_gate = (
        float(supervised_pair.get("lcb95", -np.inf)) >= -0.05 * supervised_mean
        if stage == "supervised"
        else float(supervised_pair.get("lcb95", -np.inf)) > 0.0
    )
    gates = {
        "human_evidence_sufficient": evidence_sufficient,
        "human_energy_score_vs_a2": float(a2_pair.get("lcb95", -np.inf)) > 0.0,
        "human_diagnostics": bool(diagnostic_gates) and all(diagnostic_gates.values()),
        "factual_noninferior": bool(factual.get("calibrated_residual", {}).get("noninferior", False)),
        "physical_valid": bool(ood.get("calibrated_residual", {}).get("valid", False)),
        "no_jerk_limiter_failure": not bool(ood.get("calibrated_residual", {}).get("jerk_limiter_failed", True)),
        "natural_collision_noninferior": float(collision.get("ci95_high", np.inf)) <= 0.0,
        "no_strict_causal_ood_regression": int(
            ood.get("failure_analysis", {}).get("strict_causal_regressions", -1)
        ) == 0,
        (
            "natural_noninferior_to_supervised_v1"
            if stage == "supervised" else "ppo_increment_over_supervised_v2"
        ): stage_gate,
    }
    accepted = all(gates.values())
    decision = "proceed_to_ppo" if accepted and stage == "supervised" else (
        "select_ppo" if accepted else "stop"
    )
    return {
        "schema_name": "reaction_policy_acceptance", "schema_version": 2,
        "stage": stage, "candidate": "calibrated_residual",
        "gates": gates, "diagnostic_gates": diagnostic_gates, "accepted": accepted,
        "decision": decision,
        "research_selection": (
            "calibrated_residual" if accepted else
            ("calibrated_supervised" if stage == "ppo" else "none")
        ),
        "release_controller": "a2_transfer",
        "reason": (
            "validation gates passed; test remains frozen"
            if accepted else "validation gate failed; PPO/release promotion stops"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--stage", choices=("supervised", "ppo"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    result = evaluate(report, stage=args.stage)
    output = args.output or args.report.with_name("reaction_policy_acceptance.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    decision_path = (
        output.parent.parent / "decision.md"
        if output.parent.name == "evaluation" else output.with_name("decision.md")
    )
    failed = [name for name, passed in result["gates"].items() if not passed]
    event = report.get("held_out_events", {})
    failures = report.get("physical_ood", {}).get("failure_analysis", {})
    decision_path.write_text(
        f"# Reaction PPO decision\n\n"
        f"Decision: **{result['decision']}**. Research selection: "
        f"`{result['research_selection']}`. Release controller remains "
        f"`{result['release_controller']}`.\n\n"
        f"Validation evidence: {event.get('events', 0)} events from "
        f"{event.get('recordings', 0)} recordings. Strict causal OOD "
        f"regressions: {failures.get('strict_causal_regressions', 0)}.\n\n"
        f"Failed gates: {', '.join(failed) if failed else 'none'}.\n\n"
        f"{result['reason']}.\n\n"
        f"The report is `{args.report.name}`; this validation result does not authorize a test rerun or release-config change.\n"
    )
    print(json.dumps({"decision": result["decision"], "output": str(output)}))


if __name__ == "__main__":
    main()
