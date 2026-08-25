#!/usr/bin/env python3
"""One non-development acceptance protocol for the formal world artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from hierarchical_world_model.src.protocol import (  # noqa: E402
    ACCEPTANCE_GATES,
    check_ams_readiness_gate,
    check_factual_fidelity_gate,
    check_formal_manifest_gate,
    check_intervention_gate,
    check_sampled_end_to_end_gate,
)
from world_model.src.core.utils import load_json, save_json  # noqa: E402

CONFIG = ROOT / "hierarchical_world_model/config/release.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check the formal acceptance gates for the promoted world artifact."
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    config = load_protocol_config(args.config)
    root = Path(config["paths"]["output_dir"])
    evaluation = load_json(root / "evaluation.json")
    readiness = load_json(root / "final" / "ams_readiness.json")
    manifest = load_json(root / "checkpoints" / "final_model_manifest.json")
    sampled = load_json(root / "sampled_end_to_end.json")
    factual = evaluation["factual_fidelity"]["diffusion_guided_hiqr"]
    effects = evaluation["intervention_effectiveness"]
    gates = {
        "one_checkpoint_one_config": (
            readiness["formal_checkpoint_config_match"]
            and check_formal_manifest_gate(manifest)
        ),
        "factual_conditional_fidelity": check_factual_fidelity_gate(
            factual, limits=ACCEPTANCE_GATES["factual_limits_m"]
        ),
        "intervention_responsiveness": check_intervention_gate(
            effects, thresholds=ACCEPTANCE_GATES["intervention"]
        ),
        "physical_numerical_validity": readiness["finite_state_rate"] == 1.0,
        "replayability_and_crn": bool(
            readiness["same_world_same_ads_exact"]
            and readiness["snapshot_restore_exact"]
            and readiness["world_serialization_exact"]
        ),
        "finite_evt_risk_interface": check_ams_readiness_gate(readiness),
        "stochastic_risk_non_degeneracy": (
            readiness["response_risk_variance_under_pcn_mutation"] > 0.0
        ),
        "sampled_end_to_end_complete": (
            check_sampled_end_to_end_gate(sampled)
            and sampled.get("provenance", {}).get("code_commit") == manifest.get("code_commit")
            and sampled.get("provenance", {}).get("release_tag") == manifest.get("release_tag")
        ),
    }
    report = {
        "protocol": (
            "formal acceptance: fidelity, intervention, physical/numerical validity, "
            "replayability and stochastic-risk non-degeneracy; excludes centimetre "
            "diversity development gates"
        ),
        "gates": gates,
        "all_passed": bool(all(gates.values())),
        "history_diagnostic": (
            "reported separately; no strong history-effect claim is accepted when its "
            "sensitivity is absent"
        ),
        "sources": {
            "evaluation": str(root / "evaluation.json"),
            "ams_readiness": str(root / "final" / "ams_readiness.json"),
            "final_manifest": str(root / "checkpoints" / "final_model_manifest.json"),
            "sampled_end_to_end": str(root / "sampled_end_to_end.json"),
        },
    }
    save_json(report, root / "final" / "acceptance.json")
    if not report["all_passed"]:
        raise RuntimeError("formal world-model acceptance failed")


if __name__ == "__main__":
    main()
