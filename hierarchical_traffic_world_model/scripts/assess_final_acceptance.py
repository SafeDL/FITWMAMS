#!/usr/bin/env python3
"""One non-development acceptance protocol for the formal world artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.core.utils import load_json, load_yaml, save_json  # noqa: E402

CONFIG = ROOT / "hierarchical_traffic_world_model/configs/highd_stochastic_causal_hiqr_full.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check the formal acceptance gates for the promoted world artifact."
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    config = load_yaml(args.config)
    root = Path(config["paths"]["output_dir"])
    evaluation = load_json(root / "evaluation.json")
    readiness = load_json(root / "final" / "ams_readiness.json")
    manifest = load_json(root / "checkpoints" / "final_model_manifest.json")
    factual = evaluation["factual_fidelity"]["diffusion_guided_hiqr"]
    effects = evaluation["intervention_effectiveness"]
    gates = {
        "one_checkpoint_one_config": (
            readiness["formal_checkpoint_config_match"]
            and bool(manifest["checkpoint_sha256"])
            and bool(manifest.get("code_commit"))
        ),
        "factual_conditional_fidelity": all(
            factual[key] <= limit
            for key, limit in (("ADE_m", 0.06), ("FDE_m", 0.06), ("P95_displacement_error_m", 0.12))
        ),
        "intervention_responsiveness": (
            effects["brake"]["direction_success_rate"] >= 0.95
            and effects["accelerate"]["direction_success_rate"] >= 0.95
            and effects["brake"]["dose_monotonicity_rate"] >= 0.95
            and effects["accelerate"]["dose_monotonicity_rate"] >= 0.95
            and effects["left"]["separation_non_decrease_rate"] >= 0.90
            and all(value["locality_ratio_far_to_near"] < 0.15 for value in effects.values())
            and all(value["response_latency_s"] >= 0.04 for value in effects.values())
        ),
        "physical_numerical_validity": readiness["finite_state_rate"] == 1.0,
        "replayability_and_crn": (
            readiness["same_world_same_ads_exact"]
            and readiness["snapshot_restore_exact"]
            and readiness["world_serialization_exact"]
        ),
        "finite_evt_risk_interface": (
            readiness["finite_evt_score_rate"] == 1.0
            and readiness["evt_score_monotone_on_calibration_probe"]
        ),
        "stochastic_risk_non_degeneracy": (
            readiness["response_risk_variance_under_pcn_mutation"] > 0.0
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
        },
    }
    save_json(report, root / "final" / "acceptance.json")
    if not report["all_passed"]:
        raise RuntimeError("formal world-model acceptance failed")


if __name__ == "__main__":
    main()
