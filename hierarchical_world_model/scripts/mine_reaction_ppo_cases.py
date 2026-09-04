#!/usr/bin/env python3
"""Mine diverse dynamic-influence HighwayEnv cases for auditable GIFs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import ANCHOR_INDEX  # noqa: E402
from hierarchical_world_model.scripts.render_reaction_ppo_comparative_playbacks import _authority, _controller  # noqa: E402
from hierarchical_world_model.scripts.evaluate_naturalistic_reaction_controllers import _batched_rollout  # noqa: E402
from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.influence_graph import dynamic_candidate_scene_mask  # noqa: E402
from hierarchical_world_model.src.planner import complete_missing_background_plans, frozen_diffusion_plans  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from hierarchical_world_model.src.reaction_ppo import highway_controller_rollout  # noqa: E402
from hierarchical_world_model.src.train import load_checkpoint  # noqa: E402
from world_model.src.core.utils import ensure_dir, save_json, select_device  # noqa: E402


DEFAULT = ROOT / "hierarchical_world_model/config/reaction_naturalistic.yaml"


def _first(values: np.ndarray) -> int | None:
    found = np.flatnonzero(values)
    return None if not len(found) else int(found[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine diverse full-test NPC causal-reaction GIF cases.")
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--dose", type=float, default=8.0)
    parser.add_argument("--duration-frames", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--artifact-dir", type=Path, default=None,
                        help="candidate/formal root containing controllers/")
    parser.add_argument("--a1-checkpoint", type=Path, default=None)
    parser.add_argument("--a2-checkpoint", type=Path, default=None)
    parser.add_argument("--a3-checkpoint", type=Path, default=None)
    parser.add_argument("--human-prior", type=Path, default=None)
    args = parser.parse_args()
    config = load_protocol_config(args.config.resolve())
    base = load_protocol_config(ROOT / config.get("base_config", "hierarchical_world_model/config/release.yaml"))
    device = select_device(config["training"].get("device", "auto"))
    model, _ = load_checkpoint(base["paths"]["evaluation_checkpoint"], device=device)
    experiment = prepare_experiment_data(base, ROOT)
    arrays, candidates = experiment.bundle.arrays, experiment.test_rows
    eligible = dynamic_candidate_scene_mask(
        arrays["agent_states"], arrays["agent_valid"], rows=candidates,
        radius_m=float(config["training"].get("influence_radius_m", 50.)),
        prediction_horizon_s=float(config["training"].get("influence_prediction_horizon_s", 1.5)),
    )
    rows = candidates[eligible]
    root = (args.artifact_dir if args.artifact_dir is not None else Path(config["paths"]["output_dir"])).resolve()
    plans = frozen_diffusion_plans(experiment.bundle, rows, checkpoint=base["paths"]["diffusion_checkpoint"],
        output_dir=root / "cache" / "case_mining", device=device, batch_size=int(base["training"]["validation_batch_size"]),
        ddim_steps=int(config["training"]["diffusion_ddim_steps"]), experiment_scope=base["training"].get("experiment_scope", "full"))
    states, present = arrays["agent_states"][rows], arrays["agent_valid"][rows]
    plans = complete_missing_background_plans(plans, states, present)
    maps, map_valid = arrays["map_polylines"][rows], arrays["map_polyline_valid"][rows]
    kwargs = dict(batch_size=int(args.batch_size), states=states, valid=present, soft_plans=plans, maps=maps, map_valid=map_valid, device=device,
        motion_seed=20260902, intervention="brake", dose=float(args.dose),
        intervention_duration_frames=int(args.duration_frames), **_authority(config))
    a0 = _batched_rollout(model=model, controller=_controller("A0", config, device), **kwargs)
    # Case mining compares the corrected candidate A3 against A0.  The
    # explicit checkpoint/prior arguments make it impossible to accidentally
    # mine cases with a stale formal controller after a retraining pass.
    a3 = _batched_rollout(model=model, controller=_controller(
        "A3", config, device, checkpoint=args.a3_checkpoint,
        human_prior=args.human_prior,
    ), **kwargs)
    authority = a3.controller_diagnostics["influence_authority"] > 0.
    role = a3.controller_diagnostics["influence_role"]
    controlled_a3 = (a3.crashed[:, :, 1:] & authority).any(axis=(1, 2))
    controlled_a0 = (a0.crashed[:, :, 1:] & (a0.controller_diagnostics["influence_authority"] > 0.)).any(axis=(1, 2))
    ego_a3 = a3.crashed[:, :, 0].any(1)
    min_ttc = np.min(np.where(authority, a3.controller_diagnostics["influence_predicted_ttc_s"], np.inf), axis=(1, 2))
    # Detect a dynamically influenced child that crosses from behind to ahead
    # of ego; authority must still obey the safe/opening dwell before release.
    relative_x = a3.states[:, :, 1:, 0] - a3.states[:, :, :1, 0]
    overtaking = ((relative_x < 0.) & authority).any(axis=(1, 2)) & ((relative_x > 5.) & authority).any(axis=(1, 2))
    category_masks = {
        "same_rear": (authority & (role == 1)).any(axis=(1, 2)),
        "cut_in_conflict": (authority & (role == 2)).any(axis=(1, 2)),
        "secondary_follower": (authority & (role == 4)).any(axis=(1, 2)),
        "high_closing": np.isfinite(min_ttc),
        "overtake_handoff": overtaking,
        "residual_collision": controlled_a3 | ego_a3,
        "rescued_vs_a0": controlled_a0 & ~controlled_a3 & ~ego_a3,
    }
    records = {}
    for category, mask in category_masks.items():
        indices = np.flatnonzero(mask)
        if category == "high_closing":
            indices = indices[np.argsort(min_ttc[indices])]
        elif category in {"residual_collision", "rescued_vs_a0"}:
            indices = indices[np.argsort(min_ttc[indices])]
        else:
            indices = indices[np.argsort(min_ttc[indices])]
        records[category] = [{
            "row": int(rows[index]), "minimum_predicted_ttc_s": float(min_ttc[index]) if np.isfinite(min_ttc[index]) else None,
            "a0_influenced_collision": bool(controlled_a0[index]),
            "a3_influenced_collision": bool(controlled_a3[index]),
            "a3_ego_collision": bool(ego_a3[index]),
            "active_role_frames": {str(value): int((authority[index] & (role[index] == value)).sum()) for value in (1, 2, 3, 4)},
        } for index in indices[:max(int(args.per_category), 1)]]
    output = ensure_dir(root / "visualization" / "case_catalog")
    report = {
        "schema": "full_test_dynamic_influence_case_mining_v2", "backend": "HighwayEnvClosedLoopWorld",
        "test_sequences": int(len(candidates)), "eligible_dynamic_sequences": int(len(rows)),
        "intervention": {"dose_mps2": float(args.dose), "duration_frames": int(args.duration_frames)},
        "counts": {name: int(mask.sum()) for name, mask in category_masks.items()},
        "cases": records,
    }
    save_json(report, output / "case_catalog.json")
    print(output / "case_catalog.json")


if __name__ == "__main__":
    main()
