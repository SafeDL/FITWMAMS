#!/usr/bin/env python3
"""Record what existing A2 artifacts establish before calibration training.

This preflight deliberately does not call a full-background, stochastic
policy rollout a factual reconstruction. That experiment is retained only as
a separately named stress diagnostic after its scope is approved.
"""

from __future__ import annotations

import json
import subprocess

import torch

from common import ROOT, load_config, result_root
from world_model.src.core.utils import file_sha256


def command(*args: str) -> str:
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()


def main() -> None:
    config, world = load_config()
    root = result_root(config)
    root.mkdir(parents=True, exist_ok=True)
    output = root / "baseline_protocol_audit.json"
    if output.exists():
        raise RuntimeError("baseline protocol audit is immutable and will not be overwritten")

    bridge_path = ROOT / "results/hierarchical_world_model/ppo_idm_response/evaluation/factual_reconstruction.json"
    effects_path = ROOT / "results/hierarchical_world_model/ppo_idm_response/evaluation/intervention_effects/all_test_intervention_effects.json"
    bridge = json.loads(bridge_path.read_text())
    effects = json.loads(effects_path.read_text())
    report = {
        "schema": "a2_human_calibration_baseline_protocol_audit",
        "status": "blocked_pending_aligned_policy_on_factual_protocol",
        "git_head": command("git", "rev-parse", "HEAD"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "artifact_sha256": {
            "world": file_sha256(ROOT / world["paths"]["evaluation_checkpoint"]),
            "legacy_a2": file_sha256(ROOT / config["paths"]["baseline_checkpoint"]),
            "idm": file_sha256(ROOT / config["paths"]["rule_model"]),
        },
        "historical_bridge": {
            "controller": bridge["controller"],
            "scope": "NoReactionController bridge consistency; not legacy-A2 policy factual fidelity",
            "all_background_slots": bridge["all_background_slots"],
        },
        "legacy_a2_response_test": {
            "scope": "held-out paired braking intervention; not factual reconstruction",
            "events": effects["events"],
            "recordings": effects["recordings"],
            "statistics": effects["statistics"],
        },
        "required_before_training": [
            "Specify the activated-agent mask and activation event for policy-on factual replay.",
            "Fix deterministic and stochastic arms, seeds, diffusion plans, and logged ego controls.",
            "Implement the aligned evaluator and reproduce the historical bridge independently.",
        ],
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "status": report["status"]}, indent=2))


if __name__ == "__main__":
    main()
