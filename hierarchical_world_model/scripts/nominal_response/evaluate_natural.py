#!/usr/bin/env python3
"""Evaluate natural highD responses from prefix-only nominal references."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

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
from hierarchical_world_model.src.reaction_evidence import ReactionEventReference, energy_score
from common import ROOT, event_directory, load_response_config

IDM = {"a_max": 1.0, "b_comfort": 2.0, "desired_speed": 30.0, "time_headway": 1.5, "minimum_gap": 2.0}


def load_controller(arm: str, checkpoint: Path | None, device: str):
    if arm == "frozen_hiqr_common":
        return CommonPhysicsReactionController().to(device).eval()
    if checkpoint is None:
        raise ValueError("--checkpoint is required for --arm decision")
    controller = NominalPreservingReactionController().to(device)
    controller.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["state_dict"])
    return controller.eval()


def evaluate_event(
    event_index: int,
    reference: ReactionEventReference,
    bundle,
    sampler: HierarchicalWorldSampler,
    controller,
    futures: int,
    device: str,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    events = reference.events
    row = int(events.row_index[event_index])
    onset = int(events.local_onset_frame[event_index])
    follower = int(events.follower_slot[event_index])
    c0, mask = row_prefix_inputs(bundle, np.repeat(row, futures))
    exogenous = WorldExogenousState.sample(
        futures, seed=20260907 + event_index, response_steps=149,
        scene_dim=sampler.response.cfg.scene_latent_dim,
        agent_dim=sampler.response.cfg.agent_latent_dim,
    )
    sample = prefix_only_sample(sampler, c0=c0, slot_mask=mask, exogenous=exogenous)
    nominal = build_nominal_reference(sampler, sample, idm_config=IDM)
    states = bundle.arrays["agent_states"][row]
    valid = bundle.arrays["agent_valid"][row]
    maps = bundle.arrays["map_polylines"][row]
    map_valid = bundle.arrays["map_polyline_valid"][row]
    offset = onset - 24
    plan = sample.soft_plan[:, offset:]
    if plan.shape[1] < 149:
        plan = np.concatenate((plan, np.repeat(plan[:, -1:], 149 - plan.shape[1], axis=1)), axis=1)
    history = np.repeat(states[onset - 24:onset + 1][None], futures, axis=0)
    history_valid = np.repeat(valid[onset - 24:onset + 1][None], futures, axis=0)
    prior_ego = ego_controls(states[onset - 24:onset, 0], states[onset - 23:onset + 1, 0], 0.04)
    world = HighwayEnvClosedLoopWorld(sampler.response, device=device, controller=controller, idm_config=IDM)
    world.reset(
        torch.as_tensor(np.repeat(states[onset][None], futures, axis=0)),
        torch.as_tensor(np.repeat(valid[onset][None], futures, axis=0)),
        torch.as_tensor(plan[:, :149]),
        torch.as_tensor(np.repeat(maps[None], futures, axis=0)),
        torch.as_tensor(np.repeat(map_valid[None], futures, axis=0)),
        exogenous_state=exogenous,
        initial_history=torch.as_tensor(history),
        initial_history_valid=torch.as_tensor(history_valid),
        committed_ego_controls=torch.as_tensor(np.repeat(prior_ego[None], futures, axis=0)),
        nominal_reference_states=nominal.states,
        nominal_reference_actions=nominal.background_actions,
        nominal_initial_states=nominal.initial_states,
    )
    ego_actions = ego_controls(states[onset:onset + 25, 0], states[onset + 1:onset + 26, 0], 0.04)
    predicted = []
    for ego_action in ego_actions:
        transition = world.advance_response(torch.as_tensor(np.repeat(ego_action[None], futures, axis=0), device=device))
        predicted.append(transition["background_actions"][:, 0, follower - 1, 0].cpu().numpy())
    prediction = np.stack(predicted, axis=1)
    observed = events.trajectory[event_index, 25:50, 0]
    predicted_jerk = np.abs(np.diff(prediction, axis=1, prepend=prediction[:, :1])) / 0.04
    observed_jerk = np.abs(np.diff(observed, prepend=observed[:1])) / 0.04
    result = {
        "event_index": event_index,
        "recording": int(events.recording_id[event_index]),
        "futures": futures,
        "es_acceleration": energy_score(prediction, observed),
        "es_acceleration_abs_jerk": energy_score(
            np.concatenate((prediction, predicted_jerk), axis=1),
            np.concatenate((observed, observed_jerk), axis=0),
        ),
    }
    return result, prediction, observed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--arm", choices=("decision", "frozen_hiqr_common"), default="decision")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--events", type=int, default=32)
    parser.add_argument("--futures", type=int, default=8)
    parser.add_argument("--start", type=int, default=0, help="offset in the deterministic supported-event panel")
    parser.add_argument("--count", type=int, help="number of events to evaluate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-event", type=int)
    parser.add_argument("--trace-output", type=Path)
    args = parser.parse_args()

    response, base = load_response_config()
    data = prepare_experiment_data(base, ROOT)
    reference = ReactionEventReference.load(event_directory(response) / args.split)
    selected = reference.events.indices(reference.supported_cells)
    selected = selected[reference.events.leader_slot[selected] == 0][:args.events]
    if args.start < 0 or args.start >= len(selected):
        raise ValueError(f"--start {args.start} outside 0..{len(selected) - 1}")
    selected = selected[args.start:args.start + args.count if args.count is not None else None]
    if args.trace_event is not None:
        selected = selected[selected == args.trace_event]

    device = "cuda"
    sampler = HierarchicalWorldSampler(
        flow_checkpoint=base["paths"]["flow_checkpoint"],
        flow_output_dir=base["paths"]["flow_output_dir"],
        diffusion_checkpoint=base["paths"]["diffusion_checkpoint"],
        diffusion_contract=base["paths"]["diffusion_contract"],
        response_checkpoint=base["paths"]["evaluation_checkpoint"],
        repo_root=ROOT, device=device, ddim_steps=20, excluded_slots=(),
    )
    controller = load_controller(args.arm, args.checkpoint, device)
    rows = []
    trace_prediction = trace_observation = None
    for event_index in selected:
        result, prediction, observation = evaluate_event(
            int(event_index), reference, data.bundle, sampler, controller, args.futures, device,
        )
        rows.append(result)
        if args.trace_event == event_index:
            trace_prediction, trace_observation = prediction, observation

    report = {
        "status": "completed", "arm": args.arm, "panel_events": args.events,
        "start": args.start, "events": len(rows), "futures": args.futures,
        "per_event": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.trace_output is not None and trace_prediction is not None:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.trace_output, predicted=trace_prediction, observed=trace_observation)


if __name__ == "__main__":
    main()
