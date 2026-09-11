#!/usr/bin/env python3
"""Run the single post-training validation pass for human-response A2."""

from __future__ import annotations

import json
import sys
from dataclasses import replace

import torch

from common import ROOT, load_config, result_root
from train import arrays_for, event_rows, factual_quick, plans_for, training_config

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.reaction_controller import HumanResponseA2Controller, IDMResidualReactionController  # noqa: E402
from hierarchical_world_model.src.reaction_evidence import ReactionEventReference  # noqa: E402
from hierarchical_world_model.src.reaction_training import validation_energy_score  # noqa: E402
from hierarchical_world_model.src.rule_models import RuleModelBundle  # noqa: E402
from hierarchical_world_model.src.train import load_checkpoint  # noqa: E402
from world_model.src.core.utils import save_json, select_device  # noqa: E402


def load_controller(path, rule, device, kind: str):
    payload = torch.load(path, map_location=device, weights_only=False)
    controller = HumanResponseA2Controller(rule).to(device) if kind == "human" else IDMResidualReactionController(rule).to(device)
    controller.load_state_dict(payload["state_dict"], strict=True)
    controller.eval()
    return controller


def main() -> None:
    config, base = load_config()
    root, training = result_root(config), config["training"]
    if not (root / "checkpoint.pt").is_file():
        raise FileNotFoundError("post-early-stop checkpoint is missing")
    if (root / "full_validation.json").exists():
        raise RuntimeError("full validation is single-use; existing report will not be overwritten")
    device = select_device(training["device"])
    model, _ = load_checkpoint(ROOT / base["paths"]["evaluation_checkpoint"], device=device)
    model.eval()
    experiment = prepare_experiment_data(base, ROOT)
    reference = ReactionEventReference.load(ROOT / config["paths"]["event_reference"] / "validation")
    rows = event_rows(reference)
    arrays, plans = plans_for(experiment, rows, base, root / "cache" / "validation", device)
    rule = RuleModelBundle.load(ROOT / config["paths"]["rule_model"])
    candidate = load_controller(root / "checkpoint.pt", rule, device, "human")
    supervised = load_controller(root / "supervised.pt", rule, device, "human")
    baseline = load_controller(ROOT / config["paths"]["baseline_checkpoint"], rule, device, "a2")
    environment = training_config(training)
    full = replace(environment, validation_events=265, validation_futures=32)
    factual = factual_quick(model, candidate, arrays, plans, device, full, int(training["seed"]), count=len(arrays["agent_states"]))
    scores = {
        "candidate": validation_energy_score(model, candidate, arrays=arrays, soft_plans=plans, reference=reference, device=device, config=full),
        "supervised": validation_energy_score(model, supervised, arrays=arrays, soft_plans=plans, reference=reference, device=device, config=full),
        "a2_baseline": validation_energy_score(model, baseline, arrays=arrays, soft_plans=plans, reference=reference, device=device, config=full),
    }
    replayable = [
        index for index in reference.events.indices(reference.supported_cells)
        if reference.events.leader_slot[index] == 0
    ]
    beats_baselines = scores["candidate"] < scores["supervised"] and scores["candidate"] < scores["a2_baseline"]
    status = "pending_remaining_bootstrap_and_safety_diagnostics" if factual["passed"] and beats_baselines else "rejected_human_energy_gate"
    report = {
        "schema_name": "human_response_a2_full_validation", "events": len(replayable),
        "futures_per_event": 32, "factual": factual, "raw_energy_score": scores,
        "candidate_selected_by": "quick factual gate then lowest paired quick Energy Score",
        "acceptance_complete": False,
        "status": status,
    }
    save_json(report, root / "full_validation.json")
    save_json({"status": report["status"], "test_not_run": True, "reason": "full acceptance is incomplete or failed; no test tuning permitted"}, root / "decision.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
