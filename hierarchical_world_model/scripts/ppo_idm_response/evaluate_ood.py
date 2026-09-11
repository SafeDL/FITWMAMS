#!/usr/bin/env python3
"""Causal E4 OOD response audit for the nominal-preserving controller.

The scene is sampled from the validation event prefix only.  The six fixed
ego profiles are offsets to the logged ego command, while NPC futures come
only from the frozen prefix sampler and shared exogenous seeds.
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
from hierarchical_world_model.src.prefix_reference import prefix_only_sample, row_prefix_inputs
from hierarchical_world_model.src.randomness import WorldExogenousState
from hierarchical_world_model.src.reaction_controller import (
    CommonPhysicsReactionController,
    NominalPreservingReactionController,
)
from hierarchical_world_model.src.reaction_evidence import ReactionEventReference
from common import ROOT, event_directory, load_response_config

IDM = {"a_max": 1.0, "b_comfort": 2.0, "desired_speed": 30.0, "time_headway": 1.5, "minimum_gap": 2.0}
PROFILES: dict[str, np.ndarray] = {}
for magnitude in (2.0, 4.0, 6.0, 8.0):
    profile = np.zeros(149, np.float32)
    profile[25:50] = -magnitude
    PROFILES[f"constant_brake_{magnitude:g}"] = profile
ramp = np.zeros(149, np.float32); ramp[25:35] = np.linspace(0.0, -6.0, 10); ramp[35:40] = -6.0; ramp[40:50] = np.linspace(-6.0, 0.0, 10)
PROFILES["unseen_ramp"] = ramp
pulse = np.zeros(149, np.float32); pulse[25:30] = -6.0
PROFILES["unseen_pulse"] = pulse


def recording_balanced_events(reference: ReactionEventReference, limit: int) -> np.ndarray:
    events = reference.events
    supported = reference.events.indices(reference.supported_cells)
    supported = supported[events.leader_slot[supported] == 0]
    pools: dict[int, list[int]] = {}
    for record in np.unique(events.recording_id[supported]):
        pool = supported[events.recording_id[supported] == record]
        order = np.argsort(events.initial_conditions[pool, 0], kind="stable")
        zigzag = np.ravel(np.column_stack((pool[order], pool[order][::-1])))
        pools[int(record)] = list(dict.fromkeys(int(i) for i in zigzag))
    selected: list[int] = []
    while len(selected) < min(limit, len(supported)):
        for record in sorted(pools):
            if pools[record]:
                selected.append(pools[record].pop(0))
                if len(selected) == limit:
                    break
    return np.asarray(selected, np.int64)


def run_scenario(sampler, controller, states, valid, maps, map_valid, event_index: int,
                 reference: ReactionEventReference, futures: int, profile: np.ndarray) -> dict:
    row = int(reference.events.row_index[event_index])
    onset = int(reference.events.local_onset_frame[event_index])
    c0, mask = row_prefix_inputs(sampler._experiment_bundle, np.repeat(row, futures)) if hasattr(sampler, "_experiment_bundle") else (None, None)
    # The sampler does not own the data bundle; callers attach it explicitly.
    if c0 is None:
        raise RuntimeError("sampler data bundle is not attached")
    exogenous = WorldExogenousState.sample(
        futures, seed=20260907 + event_index,
        response_steps=149, scene_dim=sampler.response.cfg.scene_latent_dim,
        agent_dim=sampler.response.cfg.agent_latent_dim,
    )
    sample = prefix_only_sample(sampler, c0=c0, slot_mask=mask, exogenous=exogenous)
    nominal = build_nominal_reference(sampler, sample, idm_config=IDM)
    row_states = states[row]; row_valid = valid[row]
    plan = sample.soft_plan[:, max(0, onset - ANCHOR_INDEX):]
    plan = np.asarray(plan)
    if plan.shape[1] < 149:
        plan = np.concatenate((plan, np.repeat(plan[:, -1:], 149 - plan.shape[1], axis=1)), axis=1)
    plan = plan[:, :149]
    history = np.repeat(row_states[onset - 24:onset + 1][None], futures, axis=0)
    history_valid = np.repeat(row_valid[onset - 24:onset + 1][None], futures, axis=0)
    prior = ego_controls(row_states[onset - 24:onset, 0], row_states[onset - 23:onset + 1, 0], .04)
    available = max(0, min(149, len(row_states) - onset - 1))
    if available:
        ego = ego_controls(row_states[onset:onset + available, 0], row_states[onset + 1:onset + available + 1, 0], .04)
        if available < 149:
            ego = np.concatenate((ego, np.repeat(ego[-1:], 149 - available, axis=0)), axis=0)
    else:
        ego = np.zeros((149, 2), np.float32)
    ego[:, 0] += profile
    world = HighwayEnvClosedLoopWorld(sampler.response, device="cuda", controller=controller, idm_config=IDM)
    world.reset(
        torch.as_tensor(np.repeat(row_states[onset][None], futures, axis=0)),
        torch.as_tensor(np.repeat(row_valid[onset][None], futures, axis=0)),
        torch.as_tensor(plan), torch.as_tensor(np.repeat(maps[row][None], futures, axis=0)),
        torch.as_tensor(np.repeat(map_valid[row][None], futures, axis=0)), exogenous_state=exogenous,
        initial_history=torch.as_tensor(history), initial_history_valid=torch.as_tensor(history_valid),
        committed_ego_controls=torch.as_tensor(np.repeat(prior[None], futures, axis=0)),
        nominal_reference_states=nominal.states, nominal_reference_actions=nominal.background_actions,
        nominal_initial_states=nominal.initial_states,
    )
    realized, actions, active, collisions = [], [], [], []
    for t in range(149):
        transition = world.advance_response(torch.as_tensor(np.repeat(ego[t][None], futures, axis=0), device="cuda"))
        realized.append(transition["agent_state_frames"].cpu().numpy()[:, 0])
        actions.append(transition["background_actions"].cpu().numpy()[:, 0])
        active.append(transition["controller_active"].cpu().numpy())
        collisions.append(transition["collision"].cpu().numpy())
    realized = np.stack(realized, axis=1); actions = np.stack(actions, axis=1)
    active = np.stack(active, axis=1); collisions = np.stack(collisions, axis=1)
    jerk = np.abs(np.diff(actions[..., 0], axis=1)) / .04
    gap = realized[:, :, :1, 0] - realized[:, :, 1:, 0] - 4.8
    valid_bg = np.repeat(row_valid[onset, 1:][None, None], futures, axis=0).repeat(149, axis=1)
    finite_gap = gap[valid_bg]
    recovery = slice(50, 125)
    nominal_speed = nominal.states[:, :149, 1:, 2].cpu().numpy()
    speed_error = np.abs(realized[:, recovery, 1:, 2] - nominal_speed[:, recovery]).mean()
    return {
        "event_index": event_index, "row_index": row, "recording": int(reference.events.recording_id[event_index]),
        "futures": futures, "profile": None, "action_bounds_valid": bool(np.isfinite(actions).all() and (actions[..., 0] >= -8.0001).all() and (actions[..., 0] <= 4.0001).all()),
        "jerk_max_mps3": float(jerk.max()) if jerk.size else 0.0,
        "jerk_limiter_failed": bool((jerk > 60.01).any()), "collision_rate": float(collisions.any(axis=1).mean()),
        "active_rate": float(active.mean()), "response_onset_frame": (int(np.flatnonzero(active.any(axis=(0, 2)))[0]) if active.any() else None),
        "peak_brake_mps2": float(actions[..., 0].min()), "dose_mps": float(np.abs(actions[..., 0][:, 25:50]).mean()),
        "minimum_real_gap_m": float(gap.min()), "minimum_predicted_gap_m": float(finite_gap.min()) if finite_gap.size else None,
        "recovery_speed_error_mps": float(speed_error),
    }


def run_profile_batch(sampler, controller, bundle, arrays: dict[str, np.ndarray],
                      events: np.ndarray, reference: ReactionEventReference,
                      futures: int, profile: np.ndarray, seed: int) -> list[dict]:
    """Run all fixed scenes for one profile in one GPU batch."""
    event_rows = np.asarray(reference.events.row_index[events], np.int64)
    onsets = np.asarray(reference.events.local_onset_frame[events], np.int64)
    rows = np.repeat(event_rows, futures)
    onset = np.repeat(onsets, futures)
    c0, mask = row_prefix_inputs(bundle, rows)
    exogenous = WorldExogenousState.sample(
        len(rows), seed=seed, response_steps=149,
        scene_dim=sampler.response.cfg.scene_latent_dim,
        agent_dim=sampler.response.cfg.agent_latent_dim,
    )
    sample = prefix_only_sample(sampler, c0=c0, slot_mask=mask, exogenous=exogenous)
    nominal = build_nominal_reference(sampler, sample, idm_config=IDM)
    states = np.asarray(arrays["agent_states"])[rows]
    valid = np.asarray(arrays["agent_valid"])[rows]
    maps = np.asarray(arrays["map_polylines"])[rows]
    map_valid = np.asarray(arrays["map_polyline_valid"])[rows]
    plans = []
    histories, history_valid, priors, egos = [], [], [], []
    for i, (row, start) in enumerate(zip(rows, onset)):
        off = max(0, int(start) - ANCHOR_INDEX)
        plan = np.asarray(sample.soft_plan[i, off:])
        if len(plan) < 149:
            plan = np.concatenate((plan, np.repeat(plan[-1:], 149 - len(plan), axis=0)), axis=0)
        plans.append(plan[:149])
        histories.append(states[i, int(start) - 24:int(start) + 1])
        history_valid.append(valid[i, int(start) - 24:int(start) + 1])
        prior = ego_controls(states[i, int(start) - 24:int(start), 0], states[i, int(start) - 23:int(start) + 1, 0], .04)
        priors.append(prior)
        available = max(0, min(149, len(states[i]) - int(start) - 1))
        if available:
            ego = ego_controls(states[i, int(start):int(start) + available, 0], states[i, int(start) + 1:int(start) + available + 1, 0], .04)
            if available < 149:
                ego = np.concatenate((ego, np.repeat(ego[-1:], 149 - available, axis=0)), axis=0)
        else:
            ego = np.zeros((149, 2), np.float32)
        ego[:, 0] += profile
        egos.append(ego)
    plans = np.asarray(plans); histories = np.asarray(histories); history_valid = np.asarray(history_valid)
    priors = np.asarray(priors); egos = np.asarray(egos)
    world = HighwayEnvClosedLoopWorld(sampler.response, device="cuda", controller=controller, idm_config=IDM)
    world.reset(
        torch.as_tensor(np.asarray(arrays["agent_states"])[rows, onset]),
        torch.as_tensor(np.asarray(arrays["agent_valid"])[rows, onset]),
        torch.as_tensor(plans), torch.as_tensor(maps), torch.as_tensor(map_valid), exogenous_state=exogenous,
        initial_history=torch.as_tensor(histories), initial_history_valid=torch.as_tensor(history_valid),
        committed_ego_controls=torch.as_tensor(priors), nominal_reference_states=nominal.states,
        nominal_reference_actions=nominal.background_actions, nominal_initial_states=nominal.initial_states,
    )
    realized, actions, active, collisions = [], [], [], []
    for t in range(149):
        transition = world.advance_response(torch.as_tensor(egos[:, t], device="cuda"))
        realized.append(transition["agent_state_frames"].cpu().numpy()[:, 0])
        actions.append(transition["background_actions"].cpu().numpy()[:, 0])
        active.append(transition["controller_active"].cpu().numpy())
        collisions.append(transition["collision"].cpu().numpy())
    realized = np.stack(realized, axis=1); actions = np.stack(actions, axis=1)
    active = np.stack(active, axis=1); collisions = np.stack(collisions, axis=1)
    jerk = np.abs(np.diff(actions[..., 0], axis=1)) / .04
    gap = realized[:, :, :1, 0] - realized[:, :, 1:, 0] - 4.8
    valid_bg = np.repeat(np.asarray(arrays["agent_valid"])[rows, onset, 1][:, None], 149, axis=1)
    nominal_speed = nominal.states[:, :149, 1:, 2].cpu().numpy()
    output = []
    for e, event_index in enumerate(events):
        block = slice(e * futures, (e + 1) * futures)
        finite_gap = gap[block][valid_bg[block]]
        recovery = slice(50, 125)
        speed_error = np.abs(realized[block, recovery, 1:, 2] - nominal_speed[block, recovery]).mean()
        output.append({
            "event_index": int(event_index), "row_index": int(event_rows[e]),
            "recording": int(reference.events.recording_id[event_index]), "futures": futures,
            "action_bounds_valid": bool(np.isfinite(actions[block]).all() and (actions[block, ..., 0] >= -8.0001).all() and (actions[block, ..., 0] <= 4.0001).all()),
            "jerk_max_mps3": float(jerk[block].max()) if jerk[block].size else 0.0,
            "jerk_limiter_failed": bool((jerk[block] > 60.01).any()),
            "collision_rate": float(collisions[block].any(axis=1).mean()),
            "active_rate": float(active[block].mean()),
            "response_onset_frame": int(np.flatnonzero(active[block].any(axis=(0, 2)))[0]) if active[block].any() else None,
            "peak_brake_mps2": float(actions[block, ..., 0].min()),
            "dose_mps": float(np.abs(actions[block, ..., 0][:, 25:50]).mean()),
            "minimum_real_gap_m": float(gap[block].min()),
            "minimum_predicted_gap_m": float(finite_gap.min()) if finite_gap.size else None,
            "recovery_speed_error_mps": float(speed_error),
        })
    return output


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--scenes", type=int, default=64); p.add_argument("--futures", type=int, default=8)
    args = p.parse_args()
    response, base = load_response_config()
    data = prepare_experiment_data(base, ROOT)
    ref = ReactionEventReference.load(event_directory(response) / args.split)
    events = recording_balanced_events(ref, args.scenes)
    sampler = HierarchicalWorldSampler(flow_checkpoint=base["paths"]["flow_checkpoint"], flow_output_dir=base["paths"]["flow_output_dir"], diffusion_checkpoint=base["paths"]["diffusion_checkpoint"], diffusion_contract=base["paths"]["diffusion_contract"], response_checkpoint=base["paths"]["evaluation_checkpoint"], repo_root=ROOT, device="cuda", ddim_steps=20, excluded_slots=())
    arrays = data.bundle.arrays
    candidate = NominalPreservingReactionController().to("cuda"); candidate.load_state_dict(torch.load(args.checkpoint, map_location="cuda", weights_only=False)["state_dict"]); candidate.eval()
    common = CommonPhysicsReactionController().to("cuda").eval()
    arms = {"decision": candidate, "frozen_hiqr_common": common}; results = {name: [] for name in arms}
    for profile_index, (profile_name, profile) in enumerate(PROFILES.items()):
        for name, controller in arms.items():
            seed = int(response["training"]["seed"]) + profile_index * 1009
            items = run_profile_batch(sampler, controller, data.bundle, arrays, events, ref, args.futures, profile, seed)
            for item in items:
                item["profile"] = profile_name
            results[name].extend(items)
    summary = {}
    for name, rows in results.items():
        summary[name] = {"scenes": args.scenes, "futures_per_profile": args.futures, "profiles": list(PROFILES), "action_bounds_valid": all(r["action_bounds_valid"] for r in rows), "jerk_limiter_failures": int(sum(r["jerk_limiter_failed"] for r in rows)), "collision_rate": float(np.mean([r["collision_rate"] for r in rows])), "max_jerk_mps3": float(max(r["jerk_max_mps3"] for r in rows)), "min_real_gap_m": float(min(r["minimum_real_gap_m"] for r in rows)), "min_predicted_gap_m": float(min(r["minimum_predicted_gap_m"] for r in rows if r["minimum_predicted_gap_m"] is not None)), "mean_recovery_speed_error_mps": float(np.mean([r["recovery_speed_error_mps"] for r in rows])), "mean_dose_mps": float(np.mean([r["dose_mps"] for r in rows]))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"status": "e4_nominal_ood", "events": [int(i) for i in events], "profiles": list(PROFILES), "summary": summary, "per_scenario": results}, open(args.output, "w"), indent=2)


if __name__ == "__main__":
    main()
