#!/usr/bin/env python3
"""Post-acceptance paired Monte-Carlo risk sensitivity diagnostic; never runs AMS."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

from common import ROOT, load_config, result_root
from evaluate import load_adapter
from train import plans_for, policy_config

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.data import prepare_experiment_data
from hierarchical_world_model.src.reaction_training import reaction_controller_rollout
from hierarchical_world_model.src.rule_models import RuleModelBundle
from hierarchical_world_model.src.train import load_checkpoint
from world_model.src.core.utils import select_device


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--pairs", type=int, default=2000); args = parser.parse_args()
    config, world = load_config(); root = result_root(config)
    acceptance = json.loads((root / "acceptance.json").read_text()) if (root / "acceptance.json").is_file() else {}
    if not acceptance.get("accepted", False): raise RuntimeError("sensitivity diagnostic requires an accepted validation candidate")
    output = root / "sensitivity.json"
    if output.exists(): raise RuntimeError("sensitivity diagnostic is single-use")
    device = select_device("auto"); model, _ = load_checkpoint(ROOT / world["paths"]["evaluation_checkpoint"], device=device)
    experiment = prepare_experiment_data(world, ROOT); arrays, plans = plans_for(experiment.bundle, experiment.test_rows, world, root / "cache" / "test", device)
    rule = RuleModelBundle.load(ROOT / config["paths"]["rule_model"]); controller = load_adapter(root / "checkpoint.pt", rule, config, device); runtime = policy_config(config["ppo"])
    rng = np.random.default_rng(int(config["ppo"]["seed"]) + 700); changes, collisions = [], []
    for start in range(0, int(args.pairs), 64):
        rows = rng.choice(len(plans), size=min(64, int(args.pairs) - start), replace=True)
        values = {name: value[rows] for name, value in arrays.items() if name != "row_index"}; seed = 71000 + start
        base = reaction_controller_rollout(model, states=values["agent_states"], valid=values["agent_valid"], soft_plans=plans[rows], maps=values["map_polylines"], map_valid=values["map_polyline_valid"], controller=controller, device=device, motion_seed=seed, config=runtime, deterministic_response=False)
        brake = reaction_controller_rollout(model, states=values["agent_states"], valid=values["agent_valid"], soft_plans=plans[rows], maps=values["map_polylines"], map_valid=values["map_polyline_valid"], controller=controller, device=device, motion_seed=seed, intervention="brake", dose=6.0, intervention_start=25, intervention_duration_frames=25, config=runtime, deterministic_response=False)
        changes.extend((brake.background_actions[..., 0] - base.background_actions[..., 0]).mean(axis=(1, 2)).tolist())
        collisions.extend((brake.collision.any(axis=1).astype(int) - base.collision.any(axis=1).astype(int)).tolist())
    output.write_text(json.dumps({"schema": "a2_human_calibration_paired_sensitivity", "pairs": int(args.pairs), "mean_action_delta": float(np.mean(changes)), "p05_action_delta": float(np.quantile(changes, .05)), "p95_action_delta": float(np.quantile(changes, .95)), "collision_delta_mean": float(np.mean(collisions)), "ams_started": False}, indent=2) + "\n")
    print(output)


if __name__ == "__main__": main()
