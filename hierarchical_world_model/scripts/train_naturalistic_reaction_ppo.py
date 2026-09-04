#!/usr/bin/env python3
"""Train an A1/A2/A3 frozen-HiQR reaction controller without overwriting A1."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from diffusion.src.data import ANCHOR_INDEX  # noqa: E402
from hierarchical_world_model.src.human_prior import HumanActionPrior  # noqa: E402
from hierarchical_world_model.src.influence_graph import dynamic_candidate_scene_mask  # noqa: E402
from hierarchical_world_model.src.planner import complete_missing_background_plans, frozen_diffusion_plans  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from hierarchical_world_model.src.reaction_ppo import PPOConfig, train_reaction_ppo  # noqa: E402
from hierarchical_world_model.src.rule_models import RuleModelBundle  # noqa: E402
from hierarchical_world_model.src.train import load_checkpoint  # noqa: E402
from world_model.src.core.utils import ensure_dir, file_sha256, save_json, select_device  # noqa: E402


DEFAULT = ROOT / "hierarchical_world_model/config/reaction_naturalistic.yaml"


def _prior(path: Path, device: torch.device) -> HumanActionPrior:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "longitudinal_gail_human_prior_v4":
        raise ValueError("A3 requires a retrained HumanActionPriorV4 checkpoint")
    prior = HumanActionPrior().to(device); prior.load_state_dict(payload["state_dict"]); prior.eval()
    return prior


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pure, IDM-referenced, or GAIL-constrained residual PPO.")
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--arm", choices=("rl_residual", "rl_residual_idm", "rl_residual_gail"), required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="candidate artifact root; formal release output is untouched")
    parser.add_argument("--human-prior", type=Path, default=None,
                        help="GAIL V4 checkpoint override for A3")
    parser.add_argument("--a2-checkpoint", type=Path, default=None,
                        help="validated frozen A2 checkpoint used to initialize candidate A3")
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--naturalness-weight", type=float, default=None,
                        help="optional A3 KL reward override for a focused fine-tune")
    args = parser.parse_args()
    config = load_protocol_config(args.config.resolve())
    base = load_protocol_config(ROOT / config.get("base_config", "hierarchical_world_model/config/release.yaml"))
    device = select_device(config["training"].get("device", "auto"))
    model, _ = load_checkpoint(base["paths"]["evaluation_checkpoint"], device=device)
    experiment = prepare_experiment_data(base, ROOT)
    candidate_rows = experiment.train_rows if args.limit is None else experiment.train_rows[:int(args.limit)]
    # Retain every reset with at least one potential dynamic influence-graph
    # candidate.  Role assignment itself remains causal and is recomputed
    # from realized HighwayEnv states on every tick.
    eligible = dynamic_candidate_scene_mask(
        np.asarray(experiment.bundle.arrays["agent_states"]),
        np.asarray(experiment.bundle.arrays["agent_valid"]),
        rows=candidate_rows,
        radius_m=float(config["training"].get("influence_radius_m", 50.0)),
        prediction_horizon_s=float(config["training"].get("influence_prediction_horizon_s", 1.5)),
    )
    rows = candidate_rows[eligible]
    if not len(rows):
        raise RuntimeError("no dynamic causal-influence candidates in requested highD rows")
    plans = frozen_diffusion_plans(experiment.bundle, rows, checkpoint=base["paths"]["diffusion_checkpoint"],
        output_dir=Path(config["paths"]["output_dir"]) / "cache" / "train", device=device,
        batch_size=int(base["training"]["validation_batch_size"]), ddim_steps=int(config["training"]["diffusion_ddim_steps"]),
        experiment_scope=base["training"].get("experiment_scope", "full"))
    arrays = {name: experiment.bundle.arrays[name][rows] for name in ("agent_states", "agent_valid", "map_polylines", "map_polyline_valid")}
    plans = complete_missing_background_plans(plans, arrays["agent_states"], arrays["agent_valid"])
    validation_candidates = experiment.validation_rows
    validation_eligible = dynamic_candidate_scene_mask(
        np.asarray(experiment.bundle.arrays["agent_states"]),
        np.asarray(experiment.bundle.arrays["agent_valid"]),
        rows=validation_candidates,
        radius_m=float(config["training"].get("influence_radius_m", 50.0)),
        prediction_horizon_s=float(config["training"].get("influence_prediction_horizon_s", 1.5)),
    )
    validation_rows = validation_candidates[validation_eligible]
    if args.limit is not None:
        validation_rows = validation_rows[:max(1, min(int(args.limit), len(validation_rows)))]
    validation_plans = frozen_diffusion_plans(
        experiment.bundle, validation_rows,
        checkpoint=base["paths"]["diffusion_checkpoint"],
        output_dir=Path(config["paths"]["output_dir"]) / "cache" / "validation",
        device=device, batch_size=int(base["training"]["validation_batch_size"]),
        ddim_steps=int(config["training"]["diffusion_ddim_steps"]),
        experiment_scope=base["training"].get("experiment_scope", "full"),
    )
    validation_arrays = {
        name: experiment.bundle.arrays[name][validation_rows]
        for name in ("agent_states", "agent_valid", "map_polylines", "map_polyline_valid")
    }
    validation_plans = complete_missing_background_plans(
        validation_plans, validation_arrays["agent_states"], validation_arrays["agent_valid"]
    )
    fields = {key: value for key, value in config["training"].items() if key in PPOConfig.__dataclass_fields__}
    rule = None if args.arm == "rl_residual" else RuleModelBundle.load(config["paths"]["rule_model"])
    prior_path = Path(config["paths"]["human_prior"]) if args.human_prior is None else args.human_prior
    prior = _prior(prior_path, device) if args.arm == "rl_residual_gail" else None
    output_root = Path(config["paths"]["output_dir"]) if args.output_dir is None else args.output_dir
    output = ensure_dir(output_root / "controllers" / args.arm)
    initial = None
    if args.arm == "rl_residual_gail":
        # Preserve A2's learned emergency response, then optimize only the
        # safe-regime naturalness trade-off in A3.
        a2_path = (
            output_root / "controllers" / "rl_residual_idm" / "reaction_ppo.pt"
            if args.a2_checkpoint is None else args.a2_checkpoint
        )
        a2_payload = torch.load(a2_path,
            map_location=device, weights_only=False)
        if a2_payload.get("schema") != "reaction_residual_ppo_dynamic_v2":
            raise ValueError("A3 initialization requires the dynamic-scope A2 checkpoint")
        initial = a2_payload["state_dict"]
    world_checkpoint = (ROOT / base["paths"]["evaluation_checkpoint"]).resolve()
    diffusion_checkpoint = (ROOT / base["paths"]["diffusion_checkpoint"]).resolve()
    artifact_metadata = {
        "training_rows_sha256": hashlib.sha256(np.asarray(rows, np.int64).tobytes()).hexdigest(),
        "validation_rows_sha256": hashlib.sha256(np.asarray(validation_rows, np.int64).tobytes()).hexdigest(),
        "world_checkpoint_sha256": file_sha256(world_checkpoint),
        "diffusion_checkpoint_sha256": file_sha256(diffusion_checkpoint),
        "rule_checkpoint_sha256": None if args.arm == "rl_residual" else file_sha256(ROOT / config["paths"]["rule_model"]),
        "human_prior_checkpoint_sha256": None if args.arm != "rl_residual_gail" else file_sha256((ROOT / prior_path).resolve()),
        "a2_initial_checkpoint_sha256": None if args.arm != "rl_residual_gail" else file_sha256(Path(a2_path).resolve()),
        "influence_graph": {key: config["training"][key] for key in (
            "influence_radius_m", "influence_secondary_radius_m",
            "influence_prediction_horizon_s", "influence_stable_release_frames",
        )},
    }
    ppo_config = PPOConfig(**fields)
    if args.updates is not None:
        from dataclasses import replace
        ppo_config = replace(ppo_config, updates=int(args.updates))
    if args.naturalness_weight is not None:
        from dataclasses import replace
        ppo_config = replace(ppo_config, naturalness_weight=float(args.naturalness_weight))
    summary = train_reaction_ppo(model, train_arrays=arrays, soft_plans=plans, output_dir=output,
        config=ppo_config, device=device, controller_mode=args.arm, rule_model=rule, human_prior=prior,
        initial_state_dict=initial, artifact_metadata=artifact_metadata,
        validation_arrays=validation_arrays, validation_plans=validation_plans)
    summary.update({
        "candidate_training_rows": int(len(candidate_rows)), "causally_eligible_training_rows": int(len(rows)),
        "causally_eligible_validation_rows": int(len(validation_rows)),
        "full_training_split": args.limit is None,
        "eligibility": "valid 50 m rear-sector or adjacent swept-corridor candidate at any recorded response frame; online roles still use realized state only",
    })
    save_json(summary, output / "training_summary.json")
    print(summary["checkpoint"])


if __name__ == "__main__":
    main()
