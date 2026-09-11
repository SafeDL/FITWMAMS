#!/usr/bin/env python3
"""Measure PPO--IDM response effects on every eligible test intervention.

Each test event applies the fixed front-ego braking probe used by the causal
playbacks. Candidate and frozen HiQR worlds use the same prefix
sample and exogenous seed.  The report describes behavioural differences; it
does not label stronger braking as a safety improvement.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from diffusion.src.data import ANCHOR_INDEX
from hierarchical_world_model.src.data import ego_controls, prepare_experiment_data
from hierarchical_world_model.src.highway import HighwayEnvClosedLoopWorld
from hierarchical_world_model.src.nominal_reference import build_nominal_reference
from hierarchical_world_model.src.prefix_reference import prefix_only_sample, row_prefix_inputs
from hierarchical_world_model.src.randomness import WorldExogenousState
from hierarchical_world_model.src.reaction_controller import IDMResidualReactionController, NoReactionController
from hierarchical_world_model.src.reaction_evidence import ReactionEventReference
from hierarchical_world_model.src.rule_models import RuleModelBundle

from common import ROOT, event_directory, load_response_config, result_directory
from render_playbacks import IDM, _load_sampler


DT_S = 0.04
BRAKE_WINDOW = slice(25, 50)
def _ego_probe(states: np.ndarray, onsets: np.ndarray) -> np.ndarray:
    commands = []
    for state, onset in zip(states, onsets):
        available = max(0, min(149, len(state) - int(onset) - 1))
        if available:
            action = ego_controls(
                state[int(onset) : int(onset) + available, 0],
                state[int(onset) + 1 : int(onset) + available + 1, 0], DT_S,
            )
            if available < 149:
                action = np.concatenate((action, np.repeat(action[-1:], 149 - available, axis=0)))
        else:
            action = np.zeros((149, 2), np.float32)
        action[BRAKE_WINDOW, 0] -= 8.0
        commands.append(action)
    return np.asarray(commands, np.float32)


@torch.no_grad()
def _simulate(
    *, controller, sampler, bundle, reference: ReactionEventReference, events: np.ndarray,
    seed: int, steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.asarray(reference.events.row_index[events], np.int64)
    onsets = np.asarray(reference.events.local_onset_frame[events], np.int64)
    c0, mask = row_prefix_inputs(bundle, rows)
    exogenous = WorldExogenousState.sample(
        len(rows), seed=seed, response_steps=149,
        scene_dim=sampler.response.cfg.scene_latent_dim,
        agent_dim=sampler.response.cfg.agent_latent_dim,
    )
    sample = prefix_only_sample(sampler, c0=c0, slot_mask=mask, exogenous=exogenous)
    nominal = build_nominal_reference(sampler, sample, idm_config=IDM)
    arrays = bundle.arrays
    states = np.asarray(arrays["agent_states"])[rows]
    valid = np.asarray(arrays["agent_valid"])[rows]
    maps = np.asarray(arrays["map_polylines"])[rows]
    map_valid = np.asarray(arrays["map_polyline_valid"])[rows]
    plans, histories, history_valid, committed = [], [], [], []
    for state, present, onset, plan in zip(states, valid, onsets, sample.soft_plan):
        offset = max(0, int(onset) - ANCHOR_INDEX)
        plan = np.asarray(plan[offset:])
        if len(plan) < steps:
            plan = np.concatenate((plan, np.repeat(plan[-1:], steps - len(plan), axis=0)))
        plans.append(plan[:steps])
        histories.append(state[int(onset) - 24 : int(onset) + 1])
        history_valid.append(present[int(onset) - 24 : int(onset) + 1])
        committed.append(ego_controls(
            state[int(onset) - 24 : int(onset), 0], state[int(onset) - 23 : int(onset) + 1, 0], DT_S,
        ))
    commands = _ego_probe(states, onsets)[:, :steps]
    world = HighwayEnvClosedLoopWorld(sampler.response, device="cuda", controller=controller, idm_config=IDM)
    world.reset(
        torch.as_tensor(states[np.arange(len(states)), onsets]), torch.as_tensor(valid[np.arange(len(valid)), onsets]),
        torch.as_tensor(np.asarray(plans)), torch.as_tensor(maps), torch.as_tensor(map_valid),
        exogenous_state=exogenous, initial_history=torch.as_tensor(np.asarray(histories)),
        initial_history_valid=torch.as_tensor(np.asarray(history_valid)), committed_ego_controls=torch.as_tensor(np.asarray(committed)),
        nominal_reference_states=nominal.states, nominal_reference_actions=nominal.background_actions,
        nominal_initial_states=nominal.initial_states,
    )
    states_out, actions_out, collisions = [], [], []
    for command in commands.transpose(1, 0, 2):
        transition = world.advance_response(torch.as_tensor(command, device="cuda"))
        states_out.append(transition["agent_state_frames"][:, 0].cpu().numpy())
        actions_out.append(transition["background_actions"][:, 0].cpu().numpy())
        collisions.append(transition["collision"].cpu().numpy())
    return np.stack(states_out, axis=1), np.stack(actions_out, axis=1), np.stack(collisions, axis=1)


def _simulate_all(*, batch_size: int, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    events = kwargs.pop("events")
    outputs = []
    for start in range(0, len(events), batch_size):
        outputs.append(_simulate(events=events[start : start + batch_size], **kwargs))
    return tuple(np.concatenate([item[index] for item in outputs], axis=0) for index in range(3))


def _plot(records: list[dict[str, object]], output: Path, baseline_label: str) -> None:
    braking = np.asarray([row["rear_braking_increment_mps2"] for row in records], np.float64)
    gap = np.asarray([row["minimum_gap_change_m"] for row in records], np.float64)
    collision = np.asarray([row["collision_change"] for row in records], np.int8)
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.0), constrained_layout=True)
    axes[0].hist(braking, bins=30, color="#2ca25f", edgecolor="white")
    axes[0].axvline(0.0, color="#333333", linewidth=1.0)
    axes[0].set(title="Rear braking increment", xlabel=f"candidate − {baseline_label} mean rear ax [m/s²]\nnegative = extra candidate braking", ylabel="test events")
    axes[1].hist(gap, bins=30, color="#1f78b4", edgecolor="white")
    axes[1].axvline(0.0, color="#333333", linewidth=1.0)
    axes[1].set(title="Minimum rear-gap change", xlabel=f"candidate − {baseline_label} minimum gap [m]\npositive = more clearance", ylabel="test events")
    labels = ("candidate\nreduces", "unchanged", "candidate\nadds")
    counts = [int((collision == value).sum()) for value in (-1, 0, 1)]
    axes[2].bar(labels, counts, color=("#2ca25f", "#9e9e9e", "#d62728"))
    axes[2].set(title="Any-collision change", ylabel="test events")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    response, base = load_response_config()
    output = args.output or result_directory(response) / "evaluation/intervention_effects"
    checkpoint = args.checkpoint or ROOT / response["paths"]["checkpoint"]
    baseline_label = "frozen HiQR"
    candidate_label = "PPO + IDM response"
    output.mkdir(parents=True, exist_ok=True)
    data = prepare_experiment_data(base, ROOT)
    reference = ReactionEventReference.load(event_directory(response) / "test")
    event = reference.events
    events = event.indices(reference.supported_cells)
    events = events[(event.leader_slot[events] == 0) & (event.follower_slot[events] > 0)]
    sampler = _load_sampler(base)
    rule = RuleModelBundle.load(ROOT / response["paths"]["rule_model"])
    payload = torch.load(checkpoint, map_location="cuda", weights_only=False)
    if payload.get("controller_mode") != response["model"]["controller_mode"]:
        raise ValueError("--checkpoint is not a PPO + IDM response controller")
    candidate = IDMResidualReactionController(rule).to("cuda")
    candidate.load_state_dict(payload["state_dict"], strict=True)
    baseline = NoReactionController().to("cuda").eval()
    candidate.eval()
    seed = int(response["training"]["seed"])
    candidate_states, candidate_actions, candidate_collisions = _simulate_all(batch_size=args.batch_size, controller=candidate, sampler=sampler, bundle=data.bundle, reference=reference, events=events, seed=seed, steps=149)
    baseline_states, baseline_actions, baseline_collisions = _simulate_all(batch_size=args.batch_size, controller=baseline, sampler=sampler, bundle=data.bundle, reference=reference, events=events, seed=seed, steps=149)
    records: list[dict[str, object]] = []
    for index, event_index in enumerate(events):
        follower = int(event.follower_slot[event_index])
        rear = follower - 1
        candidate_ax = candidate_actions[index, BRAKE_WINDOW, rear, 0]
        baseline_ax = baseline_actions[index, BRAKE_WINDOW, rear, 0]
        candidate_gap = candidate_states[index, :, 0, 0] - candidate_states[index, :, follower, 0] - 4.8
        baseline_gap = baseline_states[index, :, 0, 0] - baseline_states[index, :, follower, 0] - 4.8
        records.append({
            "event_index": int(event_index), "row_index": int(event.row_index[event_index]),
            "recording": int(event.recording_id[event_index]), "follower_slot": follower,
            "rear_braking_increment_mps2": float(candidate_ax.mean() - baseline_ax.mean()),
            "minimum_gap_change_m": float(candidate_gap.min() - baseline_gap.min()),
            "collision_change": int(candidate_collisions[index].any()) - int(baseline_collisions[index].any()),
            "candidate_collision": bool(candidate_collisions[index].any()),
            "baseline_collision": bool(baseline_collisions[index].any()),
        })
    columns = list(records[0])
    with (output / "all_test_intervention_effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(records)
    _plot(records, output / "all_test_intervention_effects.png", baseline_label)
    difference = np.abs(np.asarray([row["rear_braking_increment_mps2"] for row in records]))
    selected = records[int(np.argmax(difference))]
    summary = {
        "role": "all eligible highD test intervention effects",
        "candidate": candidate_label, "baseline": baseline_label,
        "checkpoint": str(checkpoint.relative_to(ROOT)), "events": len(records),
        "recordings": len({row["recording"] for row in records}),
        "intervention": "logged ego controls plus -8 m/s² from 1.00 s to 2.00 s",
        "comparison": f"{candidate_label} minus {baseline_label}",
        "randomness": "one shared prefix sample and exogenous seed per event and controller",
        "statistics": {
            "rear_braking_increment_mps2": {"mean": float(np.mean([row["rear_braking_increment_mps2"] for row in records])), "median": float(np.median([row["rear_braking_increment_mps2"] for row in records]))},
            "minimum_gap_change_m": {"mean": float(np.mean([row["minimum_gap_change_m"] for row in records])), "median": float(np.median([row["minimum_gap_change_m"] for row in records]))},
            "collision_change_counts": {str(value): int(sum(row["collision_change"] == value for row in records)) for value in (-1, 0, 1)},
        },
        "max_absolute_rear_response_event": selected,
        "warning": "The selected event maximizes response magnitude and is diagnostic, not representative or evidence of improvement.",
    }
    (output / "all_test_intervention_effects.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
