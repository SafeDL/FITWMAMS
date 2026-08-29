"""Protocol acceptance checks for a completed TrafficBotsV1.5-HighD run."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from world_model.trafficbots.config import load_config
from world_model.trafficbots.data import TrafficBotsHighDDataset
from world_model.src.core.utils import load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a full TrafficBotsV1.5-HighD evaluation")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config_path = args.config or Path("world_model/trafficbots/config/highd.yaml")
    config = load_config(config_path)
    output = Path(config["paths"]["output_dir"])
    report = load_json(output / "evaluation.json")
    audit = load_json(output / "audit.json")
    full_test = TrafficBotsHighDDataset(config["paths"]["sequence_cache_dir"], "test", seed=int(config["evaluation"]["seed"]))
    crn = np.load(output / "intervention_crn.npz", allow_pickle=False)
    checks = {
        "full_test_cohort": int(report["test_sequences"]) == len(full_test),
        "schema": int(report["evaluation_schema_version"]) >= 2,
        "ego_replay_gate": bool(report["ego_replay_gate_passed"]),
        "prior_mode": "deterministic_prior_mode" in report["factual_fidelity"],
        "oracle_diagnostic": "TrafficBots_Oracle" in report["factual_fidelity"],
        "stochastic_16": int(report["reproducibility"]["stochastic_samples"]) == 16,
        "paired_crn": bool(report["evaluation_protocol"]["intervention_common_random_numbers"]),
        "crn_sequence_alignment": len(crn["sequence_id"]) == int(report["evaluation_protocol"]["intervention_subset_sequences"]),
        "checkpoint_provenance": bool(report["reproducibility"].get("checkpoint_sha256")),
        "method_audit": bool(audit.get("method_identity_passed")),
        "audit_checkpoint_alignment": (
            audit["provenance"]["checkpoint_sha256"]
            == report["reproducibility"]["checkpoint_sha256"]
        ),
        "audit_test_cohort_alignment": (
            int(audit["data_contract"]["test_sequences"])
            == int(report["test_sequences"])
        ),
    }
    payload = {"all_passed": all(checks.values()), "checks": checks, "evaluation": str((output / "evaluation.json").resolve())}
    save_json(payload, output / "full_acceptance.json")
    if not payload["all_passed"]:
        raise RuntimeError(f"TrafficBotsV1.5-HighD acceptance failed: {checks}")


if __name__ == "__main__":
    main()
