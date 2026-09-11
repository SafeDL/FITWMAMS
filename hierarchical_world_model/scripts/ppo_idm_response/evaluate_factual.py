#!/usr/bin/env python3
"""Chunked P0 factual reconstruction for the nominal-preserving arm.

P0 is deliberately separate from the prefix-only P2 evaluator.  It uses the
declared factual compatibility input (frozen diffusion plans and logged ego
controls) and reports the old ADE/FDE/P95 displacement metrics.  Each chunk is
independently replayable and writes one JSON shard.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from diffusion.src.data import ANCHOR_INDEX
from hierarchical_world_model.src.composition import HierarchicalWorldSampler
from hierarchical_world_model.src.data import ego_controls, prepare_experiment_data
from hierarchical_world_model.src.highway import HighwayEnvClosedLoopWorld
from hierarchical_world_model.src.nominal_reference import build_nominal_reference
from hierarchical_world_model.src.planner import complete_missing_background_plans, frozen_diffusion_plans
from hierarchical_world_model.src.prefix_reference import prefix_only_sample, row_prefix_inputs
from hierarchical_world_model.src.randomness import WorldExogenousState
from hierarchical_world_model.src.reaction_controller import CommonPhysicsReactionController, NominalPreservingReactionController, NoReactionController
from common import ROOT, load_response_config

IDM = {"a_max": 1.0, "b_comfort": 2.0, "desired_speed": 30.0, "time_headway": 1.5, "minimum_gap": 2.0}


def metrics(errors: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    values = errors[valid]
    final = errors[:, -1][valid[:, -1]]
    return {
        "ADE_m": float(values.mean()) if values.size else float("nan"),
        "FDE_m": float(final.mean()) if final.size else float("nan"),
        "P95_displacement_error_m": float(np.quantile(values, 0.95)) if values.size else float("nan"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("decision", "frozen_hiqr_common", "frozen_hiqr_legacy_raw"), required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=64)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--error-output", type=Path,
        help="optional .npy sidecar for exact cohort-wide displacement quantiles",
    )
    p.add_argument(
        "--diagnostic-output", type=Path,
        help="optional .npz sidecar with per-step controller activity and trigger",
    )
    a = p.parse_args()
    if a.arm == "decision" and a.checkpoint is None:
        p.error("--checkpoint is required for --arm decision")
    response, base = load_response_config()
    data = prepare_experiment_data(base, ROOT)
    rows_all = np.asarray(getattr(data, f"{a.split}_rows"), np.int64)
    if not 0 <= a.start < len(rows_all):
        p.error(f"--start must be in [0,{len(rows_all)-1}]")
    rows = rows_all[a.start:a.start + a.count]
    device = "cuda"
    sampler = HierarchicalWorldSampler(
        flow_checkpoint=base["paths"]["flow_checkpoint"],
        flow_output_dir=base["paths"]["flow_output_dir"],
        diffusion_checkpoint=base["paths"]["diffusion_checkpoint"],
        diffusion_contract=base["paths"]["diffusion_contract"],
        response_checkpoint=base["paths"]["evaluation_checkpoint"], repo_root=ROOT,
        device=device, ddim_steps=20, excluded_slots=(),
    )
    if a.arm == "decision":
        controller = NominalPreservingReactionController().to(device)
        controller.load_state_dict(torch.load(a.checkpoint, map_location=device, weights_only=False)["state_dict"])
    elif a.arm == "frozen_hiqr_common":
        controller = CommonPhysicsReactionController().to(device)
    else:
        controller = NoReactionController().to(device)
    controller.eval()
    arrays = data.bundle.arrays
    states = np.asarray(arrays["agent_states"])[rows]
    valid = np.asarray(arrays["agent_valid"])[rows]
    maps = np.asarray(arrays["map_polylines"])[rows]
    map_valid = np.asarray(arrays["map_polyline_valid"])[rows]
    plans = frozen_diffusion_plans(
        data.bundle, rows, checkpoint=base["paths"]["diffusion_checkpoint"],
        output_dir=a.output.parent / "p0_plan_cache", device=device,
        batch_size=32, ddim_steps=20, experiment_scope=base["training"].get("experiment_scope", "full"),
    )
    plans = complete_missing_background_plans(plans, states, valid)
    exogenous = WorldExogenousState.sample(
        len(rows), seed=int(response["training"]["seed"]) + a.start, response_steps=149,
        scene_dim=sampler.response.cfg.scene_latent_dim, agent_dim=sampler.response.cfg.agent_latent_dim,
    )
    nominal = None
    if a.arm == "decision":
        c0, mask = row_prefix_inputs(data.bundle, rows)
        nominal = build_nominal_reference(sampler, prefix_only_sample(sampler, c0=c0, slot_mask=mask, exogenous=exogenous), idm_config=IDM)
    world = HighwayEnvClosedLoopWorld(sampler.response, device=device, controller=controller, idm_config=IDM)
    history = torch.as_tensor(states[:, ANCHOR_INDEX - 24:ANCHOR_INDEX + 1])
    history_valid = torch.as_tensor(valid[:, ANCHOR_INDEX - 24:ANCHOR_INDEX + 1])
    prior = np.stack([ego_controls(s[:ANCHOR_INDEX, 0], s[1:ANCHOR_INDEX + 1, 0], .04) for s in states])
    ego = np.stack([ego_controls(s[ANCHOR_INDEX:ANCHOR_INDEX + 149, 0], s[ANCHOR_INDEX + 1:ANCHOR_INDEX + 150, 0], .04) for s in states])
    world.reset(
        torch.as_tensor(states[:, ANCHOR_INDEX]), torch.as_tensor(valid[:, ANCHOR_INDEX]), torch.as_tensor(plans),
        torch.as_tensor(maps), torch.as_tensor(map_valid), exogenous_state=exogenous,
        initial_history=history, initial_history_valid=history_valid,
        committed_ego_controls=torch.as_tensor(prior),
        nominal_reference_states=None if nominal is None else nominal.states,
        nominal_reference_actions=None if nominal is None else nominal.background_actions,
        nominal_initial_states=None if nominal is None else nominal.initial_states,
    )
    realized, collisions = [], []
    diagnostic_active, diagnostic_trigger = [], []
    diagnostic_gap_delta, diagnostic_speed_delta, diagnostic_base_delta = [], [], []
    for t in range(149):
        if a.diagnostic_output is not None and a.arm == "decision":
            actual_now = world.states
            nominal_now = world.nominal_initial_states if t == 0 else world.nominal_reference_states[:, t - 1]
            actual_gap = actual_now[:, 0, 0, None] - actual_now[:, 1:, 0] - 4.8
            nominal_gap = nominal_now[:, 0, 0, None] - nominal_now[:, 1:, 0] - 4.8
            diagnostic_gap_delta.append((actual_gap - nominal_gap).detach().cpu().numpy())
            diagnostic_speed_delta.append((actual_now[:, 1:, 2] - nominal_now[:, 1:, 2]).detach().cpu().numpy())
        transition = world.advance_response(torch.as_tensor(ego[:, t], device=device))
        realized.append(transition["agent_state_frames"].cpu().numpy())
        collisions.append(transition["collision"].cpu().numpy())
        if a.diagnostic_output is not None and a.arm == "decision":
            diagnostic_active.append(transition["controller_active"].cpu().numpy())
            diagnostic_trigger.append(transition["intervention_trigger"].cpu().numpy())
            base_now = transition["base_background_actions"][:, 0, :, 0]
            nominal_now_action = world.nominal_reference_actions[:, t, :, 0]
            diagnostic_base_delta.append((base_now - nominal_now_action).detach().cpu().numpy())
    realized = np.concatenate(realized, axis=1)
    errors = np.linalg.norm(realized[:, :, :, :2] - states[:, ANCHOR_INDEX + 1:ANCHOR_INDEX + 150, :, :2], axis=-1)
    target_valid = valid[:, ANCHOR_INDEX + 1:ANCHOR_INDEX + 150]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    target_errors = errors[:, :, 1:][target_valid[:, :, 1:]]
    target_final = errors[:, -1, 1:][target_valid[:, -1, 1:]]
    if a.error_output is not None:
        a.error_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(a.error_output, target_errors.astype(np.float32, copy=False))
    if a.diagnostic_output is not None and a.arm == "decision":
        a.diagnostic_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            a.diagnostic_output,
            active=np.asarray(diagnostic_active, dtype=np.bool_),
            intervention_trigger=np.asarray(diagnostic_trigger, dtype=np.float32),
            gap_delta=np.asarray(diagnostic_gap_delta, dtype=np.float32),
            speed_delta=np.asarray(diagnostic_speed_delta, dtype=np.float32),
            base_delta=np.asarray(diagnostic_base_delta, dtype=np.float32),
        )
    json.dump({
        "status": "p0_factual_shard", "arm": a.arm, "split": a.split,
        "start": a.start, "rows": len(rows), "metrics": metrics(errors[:, :, 1:], target_valid[:, :, 1:]),
        "metric_sufficient_statistics": {
            "error_count": int(target_errors.size), "error_sum": float(target_errors.sum()),
            "fde_count": int(target_final.size), "fde_sum": float(target_final.sum()),
        },
        "error_output": None if a.error_output is None else str(a.error_output),
        "diagnostic_output": None if a.diagnostic_output is None else str(a.diagnostic_output),
        "collision_count": int(np.asarray(collisions).any(axis=0).sum()),
        "per_row": [{"row_index": int(row), "ADE_m": float(errors[i][target_valid[i]].mean()),
                     "FDE_m": float(errors[i, -1][target_valid[i, -1]].mean())} for i, row in enumerate(rows)],
    }, open(a.output, "w"), indent=2)


if __name__ == "__main__":
    main()
