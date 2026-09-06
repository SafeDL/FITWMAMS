#!/usr/bin/env python3
"""Train the candidate calibrated residual policy without changing A2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.influence_graph import dynamic_candidate_scene_mask  # noqa: E402
from hierarchical_world_model.src.planner import complete_missing_background_plans, frozen_diffusion_plans  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from hierarchical_world_model.src.reaction_training import (  # noqa: E402
    PolicyTrainingConfig, train_reaction_policy,
)
from hierarchical_world_model.src.reaction_evidence import ReactionEventReference  # noqa: E402
from hierarchical_world_model.src.rule_models import RuleModelBundle  # noqa: E402
from hierarchical_world_model.src.train import load_checkpoint  # noqa: E402
from world_model.src.core.utils import ensure_dir, save_json, select_device  # noqa: E402


DEFAULT = ROOT / "hierarchical_world_model/config/reaction_policy.yaml"


def _arrays(bundle, rows: np.ndarray) -> dict[str, np.ndarray]:
    arrays = {name: bundle.arrays[name][rows] for name in (
        "agent_states", "agent_valid", "map_polylines", "map_polyline_valid",
    )}
    arrays["row_index"] = np.asarray(rows, np.int64)
    return arrays


def _eligible_rows(bundle, rows: np.ndarray, training: dict) -> np.ndarray:
    keep = dynamic_candidate_scene_mask(
        np.asarray(bundle.arrays["agent_states"]), np.asarray(bundle.arrays["agent_valid"]), rows=rows,
        radius_m=float(training["influence_radius_m"]),
        prediction_horizon_s=float(training["influence_prediction_horizon_s"]),
    )
    return np.asarray(rows, np.int64)[keep]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--limit", type=int, default=None, help="bounded smoke-run row limit")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--a2-transfer-checkpoint", type=Path, required=True,
                        help="frozen A2-transfer checkpoint used only to initialise the residual actor")
    parser.add_argument("--events-dir", type=Path, default=None)
    parser.add_argument("--updates", type=int, default=None)
    args = parser.parse_args()

    config = load_protocol_config(args.config.resolve())
    base = load_protocol_config(ROOT / config.get("base_config", "hierarchical_world_model/config/release.yaml"))
    device = select_device(config["training"].get("device", "auto"))
    output_root = Path(config["paths"]["output_dir"]) if args.output_dir is None else args.output_dir
    model, _ = load_checkpoint(base["paths"]["evaluation_checkpoint"], device=device)
    experiment = prepare_experiment_data(base, ROOT)
    requested = experiment.train_rows if args.limit is None else experiment.train_rows[:int(args.limit)]
    rows = _eligible_rows(experiment.bundle, requested, config["training"])
    if not len(rows):
        raise RuntimeError("no autonomous influence-graph candidates in requested training rows")
    validation_rows = _eligible_rows(experiment.bundle, experiment.validation_rows, config["training"])
    if args.limit is not None:
        validation_rows = validation_rows[:max(1, min(len(validation_rows), int(args.limit)))]

    def plans_for(rows: np.ndarray, split: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
        arrays = _arrays(experiment.bundle, rows)
        plans = frozen_diffusion_plans(
            experiment.bundle, rows, checkpoint=base["paths"]["diffusion_checkpoint"],
            output_dir=output_root / "cache" / split, device=device,
            batch_size=int(base["training"]["validation_batch_size"]),
            ddim_steps=int(config["training"].get("diffusion_ddim_steps", 20)),
            experiment_scope=base["training"].get("experiment_scope", "full"),
        )
        return arrays, complete_missing_background_plans(plans, arrays["agent_states"], arrays["agent_valid"])

    arrays, plans = plans_for(rows, "train")
    validation_arrays, validation_plans = plans_for(validation_rows, "validation")
    payload = torch.load(args.a2_transfer_checkpoint, map_location=device, weights_only=False)
    valid_legacy = (
        payload.get("schema") == "reaction_residual_ppo_dynamic_v2"
        and payload.get("controller_mode") == "rl_residual_idm"
    )
    valid_current = (
        payload.get("schema_name") == "reaction_policy"
        and payload.get("schema_version") == 1
        and payload.get("controller_mode") == "rl_residual_idm"
    )
    if not (valid_legacy or valid_current):
        raise ValueError(
            "A2-transfer checkpoint has an unsupported schema; archived naturalness checkpoints are rejected"
        )
    fields = {name: value for name, value in config["training"].items() if name in PolicyTrainingConfig.__dataclass_fields__}
    training_config = PolicyTrainingConfig(**fields)
    if args.updates is not None:
        from dataclasses import replace
        training_config = replace(training_config, updates=int(args.updates))
    events_root = args.events_dir or Path(config["paths"]["event_reference"])
    train_events = ReactionEventReference.load(events_root / "train")
    validation_events = ReactionEventReference.load(events_root / "validation")
    metadata = {
        "world_checkpoint": str(base["paths"]["evaluation_checkpoint"]),
        "diffusion_checkpoint": str(base["paths"]["diffusion_checkpoint"]),
        "a2_transfer_checkpoint": str(args.a2_transfer_checkpoint),
        "event_reference": str(events_root),
        "reference_protocol": "fixed_k_gt_conditional_resimulation_no_longitudinal_rebase",
    }
    output = ensure_dir(output_root / "controllers" / "calibrated_residual")
    summary = train_reaction_policy(
        model, train_arrays=arrays, train_plans=plans, output_dir=output, config=training_config,
        device=device, controller_mode="calibrated_residual",
        rule_model=RuleModelBundle.load(ROOT / config["paths"]["rule_model"]),
        initial_state_dict=payload["state_dict"], artifact_metadata=metadata,
        train_events=train_events, validation_arrays=validation_arrays,
        validation_plans=validation_plans, validation_events=validation_events,
    )
    summary.update({"candidate_training_rows": int(len(rows)), "full_training_split": args.limit is None})
    save_json(summary, output / "training_summary.json")
    print(summary["checkpoint"])


if __name__ == "__main__":
    main()
