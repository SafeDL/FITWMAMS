#!/usr/bin/env python3
"""Train the candidate calibrated residual policy without changing A2."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
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
A2_TRANSFER_SCHEMAS = frozenset({
    "reaction_residual_ppo_dynamic_v2",
    "reaction_residual_ppo",
})


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


def _run_manifest(
    config_path: Path, base_path: Path, config: dict, base: dict,
    *, stage: str, a2_checkpoint: Path | None,
    supervised_checkpoint: Path | None, events_root: Path,
) -> dict:
    artifacts = {
        "world_model": str(base["paths"]["evaluation_checkpoint"]),
        "diffusion": str(base["paths"]["diffusion_checkpoint"]),
        "idm": str(ROOT / config["paths"]["rule_model"]),
    }
    if a2_checkpoint is not None:
        artifacts["a2_transfer"] = str(a2_checkpoint)
    if supervised_checkpoint is not None:
        artifacts["supervised_start"] = str(supervised_checkpoint)
    return {
        "schema_name": "reaction_policy_run", "schema_version": 2,
        "stage": stage,
        "config": {"reaction": str(config_path), "base": str(base_path)},
        "artifacts": artifacts,
        "event_reference": str(events_root),
        "reference_protocol": "fixed_k_gt_conditional_resimulation_no_longitudinal_rebase",
        "reference_slot_sources": "diffusion_from_K_GT with logged_future_completion for missing valid slots",
        "same_rear_runtime_valid": True,
        "randomness": {"seed": int(config["training"]["seed"]), "shared_exogenous_arrays": True},
        "training": config["training"], "acceptance_thresholds": {
            "factual_absolute_m": {"ade": 0.02, "fde": 0.06, "p95": 0.10},
            "factual_relative": 0.05, "human_lcb95": 0.0,
        },
    }


def _is_a2_transfer_checkpoint(payload: dict) -> bool:
    """Accept only the historical and current IDM-residual A2 formats."""
    if payload.get("controller_mode") != "rl_residual_idm":
        return False
    if payload.get("schema_name") == "reaction_policy":
        return payload.get("schema_version") == 1
    return payload.get("schema") in A2_TRANSFER_SCHEMAS


def _training_config(training: dict, updates: int | None) -> PolicyTrainingConfig:
    fields = {
        name: value
        for name, value in training.items()
        if name in PolicyTrainingConfig.__dataclass_fields__
    }
    config = PolicyTrainingConfig(**fields)
    return replace(config, updates=int(updates)) if updates is not None else config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--limit", type=int, default=None, help="bounded smoke-run row limit")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--plans-cache-root", type=Path)
    parser.add_argument("--stage", choices=("supervised", "ppo"), required=True)
    parser.add_argument("--a2-transfer-checkpoint", type=Path,
                        help="frozen A2-transfer baseline; only compatible actor hidden layers may be copied")
    parser.add_argument("--supervised-checkpoint", type=Path,
                        help="calibrated supervised checkpoint used to start PPO")
    parser.add_argument("--events-dir", type=Path, default=None)
    parser.add_argument("--updates", type=int, default=None)
    args = parser.parse_args()

    config = load_protocol_config(args.config.resolve())
    base_path = ROOT / config.get("base_config", "hierarchical_world_model/config/release.yaml")
    base = load_protocol_config(base_path)
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
        cache_root = args.plans_cache_root or output_root / "cache"
        plans = frozen_diffusion_plans(
            experiment.bundle, rows, checkpoint=base["paths"]["diffusion_checkpoint"],
            output_dir=cache_root / split, device=device,
            batch_size=int(base["training"]["validation_batch_size"]),
            ddim_steps=int(config["training"].get("diffusion_ddim_steps", 20)),
            experiment_scope=base["training"].get("experiment_scope", "full"),
        )
        return arrays, complete_missing_background_plans(plans, arrays["agent_states"], arrays["agent_valid"])

    arrays, plans = plans_for(rows, "train")
    validation_arrays, validation_plans = plans_for(validation_rows, "validation")
    a2_state = None
    supervised_state = None
    if args.stage == "supervised":
        if args.a2_transfer_checkpoint is None:
            raise ValueError("supervised stage requires --a2-transfer-checkpoint")
        payload = torch.load(args.a2_transfer_checkpoint, map_location=device, weights_only=False)
        if not _is_a2_transfer_checkpoint(payload):
            raise ValueError("unsupported A2-transfer checkpoint")
        a2_state = payload["state_dict"]
    else:
        if args.supervised_checkpoint is None:
            raise ValueError("PPO stage requires --supervised-checkpoint")
        payload = torch.load(args.supervised_checkpoint, map_location=device, weights_only=False)
        if payload.get("controller_mode") != "calibrated_residual":
            raise ValueError("PPO start is not a calibrated supervised checkpoint")
        supervised_state = payload["state_dict"]
    training_config = _training_config(config["training"], args.updates)
    events_root = args.events_dir or Path(config["paths"]["event_reference"])
    train_events = ReactionEventReference.load(events_root / "train")
    validation_events = ReactionEventReference.load(events_root / "validation")
    output = ensure_dir(output_root / "controllers" / "calibrated_residual")
    manifest = _run_manifest(
        args.config.resolve(), base_path.resolve(), config, base,
        stage=args.stage,
        a2_checkpoint=(
            args.a2_transfer_checkpoint.resolve()
            if args.a2_transfer_checkpoint is not None else None
        ),
        supervised_checkpoint=(
            args.supervised_checkpoint.resolve()
            if args.supervised_checkpoint is not None else None
        ),
        events_root=events_root.resolve(),
    )
    save_json(manifest, output_root / "run_manifest.json")
    preflight = {
        "passed": True,
        "device": str(device),
        "same_rear_valid_rows": int(np.asarray(experiment.bundle.arrays["agent_valid"])[rows, 24, 2].sum()),
        "candidate_training_rows": int(len(rows)),
        "validation_rows": int(len(validation_rows)),
        "train_supported_events": int(len(train_events.events.indices(train_events.supported_cells))),
        "validation_supported_events": int(len(validation_events.events.indices(validation_events.supported_cells))),
        "training_stage": args.stage,
        "initialization": (
            "compatible A2 actor hidden layers; fresh action head, log std and critic"
            if args.stage == "supervised" else "complete supervised calibrated policy"
        ),
    }
    save_json(preflight, output_root / "preflight.json")
    audit_path = events_root / "data_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"reaction evidence audit is missing: {audit_path}")
    save_json(json.loads(audit_path.read_text()), output_root / "data_audit.json")
    metadata = {
        "world_checkpoint": str(base["paths"]["evaluation_checkpoint"]),
        "diffusion_checkpoint": str(base["paths"]["diffusion_checkpoint"]),
        "a2_transfer_checkpoint": (
            str(args.a2_transfer_checkpoint) if args.a2_transfer_checkpoint else None
        ),
        "supervised_start_checkpoint": (
            str(args.supervised_checkpoint) if args.supervised_checkpoint else None
        ),
        "event_reference": str(events_root),
        "reference_protocol": "fixed_k_gt_conditional_resimulation_no_longitudinal_rebase",
    }
    summary = train_reaction_policy(
        model, train_arrays=arrays, train_plans=plans, output_dir=output, config=training_config,
        device=device, controller_mode="calibrated_residual",
        rule_model=RuleModelBundle.load(ROOT / config["paths"]["rule_model"]),
        initial_actor_hidden_state_dict=a2_state,
        supervised_state_dict=supervised_state,
        stage=args.stage,
        artifact_metadata=metadata,
        train_events=train_events, validation_arrays=validation_arrays,
        validation_plans=validation_plans, validation_events=validation_events,
    )
    summary.update({"candidate_training_rows": int(len(rows)), "full_training_split": args.limit is None})
    save_json(summary, output / "training_summary.json")
    print(summary["checkpoint"])


if __name__ == "__main__":
    main()
