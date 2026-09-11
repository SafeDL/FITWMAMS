#!/usr/bin/env python3
"""Mandatory full-validation gate between teacher forcing and PPO."""

from __future__ import annotations

import json
import sys

import torch

from common import ROOT, load_config, require_aligned_factual_protocol, result_root
from evaluate import event_scores, load_adapter, policy_on_stress
from train import plans_for, policy_config

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.data import prepare_experiment_data
from hierarchical_world_model.src.human_response_prior import HumanResponseQuerySet
from hierarchical_world_model.src.reaction_controller import IDMResidualReactionController, NoReactionController
from hierarchical_world_model.src.reaction_evidence import ReactionEventReference
from hierarchical_world_model.src.rule_models import RuleModelBundle
from hierarchical_world_model.src.train import load_checkpoint
from world_model.src.core.utils import select_device


def main() -> None:
    config, world = load_config(); require_aligned_factual_protocol(config)
    root = result_root(config); output = root / "supervised_gate.json"
    if output.exists(): raise RuntimeError("supervised gate is immutable and will not be overwritten")
    device = select_device("auto"); model, _ = load_checkpoint(ROOT / world["paths"]["evaluation_checkpoint"], device=device)
    experiment = prepare_experiment_data(world, ROOT); arrays, plans = plans_for(experiment.bundle, experiment.validation_rows, world, root / "cache" / "validation", device)
    rule = RuleModelBundle.load(ROOT / config["paths"]["rule_model"]); runtime = policy_config(config["ppo"])
    frozen = NoReactionController().to(device)
    a2 = IDMResidualReactionController(rule).to(device); a2.load_state_dict(torch.load(ROOT / config["paths"]["baseline_checkpoint"], map_location=device, weights_only=False)["state_dict"], strict=True)
    supervised = load_adapter(root / "supervised.pt", rule, config, device)
    stress = {"frozen_hiqr": policy_on_stress(model, arrays, plans, frozen, runtime, device, chunk=int(config["evaluation"]["policy_on_stress_chunk_rows"])), "supervised": policy_on_stress(model, arrays, plans, supervised, runtime, device, chunk=int(config["evaluation"]["policy_on_stress_chunk_rows"]))}
    reference = ReactionEventReference.load(ROOT / config["paths"]["event_reference"] / "validation"); query = HumanResponseQuerySet.load(root / "human_support" / "validation")
    scores = {"a2": event_scores(model, arrays, plans, reference, query, a2.eval(), runtime, device, int(config["evaluation"]["validation_futures"])), "supervised": event_scores(model, arrays, plans, reference, query, supervised, runtime, device, int(config["evaluation"]["validation_futures"]))}
    tolerance = config["evaluation"]["factual_absolute_tolerance_m"]
    stress_pass = all(stress["supervised"][f"{name}_m"] - stress["frozen_hiqr"][f"{name}_m"] <= float(tolerance[name]) and (stress["supervised"][f"{name}_m"] - stress["frozen_hiqr"][f"{name}_m"]) / max(stress["frozen_hiqr"][f"{name}_m"], 1.e-6) <= float(config["evaluation"]["factual_relative_tolerance"]) for name in ("ade", "fde", "p95"))
    response_pass = scores["supervised"]["raw_mean"] <= float(config["supervised"]["maximum_a2_es_ratio"]) * scores["a2"]["raw_mean"]
    result = {"schema": "a2_human_calibration_supervised_gate", "policy_on_stress": stress, "event_scores": scores, "policy_on_stress_pass": stress_pass, "response_pass": response_pass, "passed": bool(stress_pass and response_pass)}
    output.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))
    if not result["passed"]: raise SystemExit(2)


if __name__ == "__main__": main()
