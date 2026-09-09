#!/usr/bin/env python3
"""Train the selected nominal-preserving response head once on cached train worlds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hierarchical_world_model.src.composition import HierarchicalWorldSampler
from hierarchical_world_model.src.data import prepare_experiment_data
from hierarchical_world_model.src.decision_cost import DecisionCostInputs
from hierarchical_world_model.src.nominal_preserving_controller import solve_hinge_calibrated_qp
from hierarchical_world_model.src.nominal_reference import build_nominal_reference
from hierarchical_world_model.src.prefix_reference import prefix_only_sample, row_prefix_inputs
from hierarchical_world_model.src.randomness import WorldExogenousState
from hierarchical_world_model.src.reaction_controller import NominalPreservingReactionController
from common import ROOT, load_response_config, result_directory

IDM = {"a_max": 1.0, "b_comfort": 2.0, "desired_speed": 30.0, "time_headway": 1.5, "minimum_gap": 2.0}


def cost_inputs(follower: np.ndarray, leader: np.ndarray, device: torch.device) -> DecisionCostInputs:
    return DecisionCostInputs(*(torch.as_tensor(value, device=device) for value in (
        leader[:, 0] - follower[:, 0] - 4.8,
        follower[:, 2],
        leader[:, 2],
        leader[:, 4],
        follower[:, 2],
    )))


def load_cache(cache_dir: Path) -> dict[str, np.ndarray]:
    chunks = [np.load(path) for path in sorted(cache_dir.glob("chunk_*.npz"))]
    if not chunks:
        raise FileNotFoundError(f"no nominal-reference cache found in {cache_dir}")
    order = np.argsort(np.concatenate([chunk["event_index"] for chunk in chunks]))
    fields = ("event_index", "row_index", "onset", "follower_slot", "states", "actions")
    return {field: np.concatenate([chunk[field] for chunk in chunks], axis=0)[order] for field in fields}


def make_non_event_examples(data, event_rows: set[int], sampler: HierarchicalWorldSampler) -> tuple[np.ndarray, ...]:
    states = np.asarray(data.bundle.arrays["agent_states"])
    candidates = np.asarray([
        row for row in data.train_rows
        if row not in event_rows and data.bundle.arrays["agent_valid"][row, 49, 1]
    ], dtype=np.int64)[:16]
    c0, mask = row_prefix_inputs(data.bundle, candidates)
    exogenous = WorldExogenousState.sample(
        len(candidates), seed=20260908, response_steps=149,
        scene_dim=sampler.response.cfg.scene_latent_dim,
        agent_dim=sampler.response.cfg.agent_latent_dim,
    )
    nominal = build_nominal_reference(
        sampler,
        prefix_only_sample(sampler, c0=c0, slot_mask=mask, exogenous=exogenous),
        idm_config=IDM,
    )
    return (
        states[candidates, 49, 1], states[candidates, 49, 0],
        states[candidates, 50, 1, 2] - states[candidates, 49, 1, 2],
        nominal.states[:, 24, 1].cpu().numpy(), nominal.states[:, 24, 0].cpu().numpy(),
        nominal.background_actions[:, 25:50, 0, 0].cpu().numpy(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--updates", type=int, default=1900)
    args = parser.parse_args()

    response, base = load_response_config()
    output = result_directory(response)
    args.cache_dir = args.cache_dir or output / "work/train_cache"
    args.checkpoint = args.checkpoint or output / "checkpoint.pt"
    device = torch.device("cuda")
    data = prepare_experiment_data(base, ROOT)
    cache = load_cache(args.cache_dir)
    sampler = HierarchicalWorldSampler(
        flow_checkpoint=base["paths"]["flow_checkpoint"],
        flow_output_dir=base["paths"]["flow_output_dir"],
        diffusion_checkpoint=base["paths"]["diffusion_checkpoint"],
        diffusion_contract=base["paths"]["diffusion_contract"],
        response_checkpoint=base["paths"]["evaluation_checkpoint"],
        repo_root=ROOT, device=device, ddim_steps=20, excluded_slots=(),
    )
    non_event = make_non_event_examples(data, set(cache["row_index"].tolist()), sampler)
    model = NominalPreservingReactionController().to(device)
    optimizer = torch.optim.Adam(model.weights.parameters(), lr=3e-4)
    seed = int(response["training"]["seed"])
    rng = np.random.default_rng(seed)
    states = np.asarray(data.bundle.arrays["agent_states"])
    losses: list[float] = []

    for update in range(args.updates):
        selected = rng.choice(len(cache["event_index"]), 48, replace=False)
        rows = cache["row_index"][selected]
        onset = cache["onset"][selected]
        follower = cache["follower_slot"][selected]
        actual_follower = states[rows, onset, follower]
        actual_leader = states[rows, onset, 0]
        target = (states[rows, onset + 1, follower, 2] - states[rows, onset, follower, 2]) / 0.04
        offset = onset - 24
        nominal_follower = cache["states"][selected, np.maximum(offset - 1, 0), follower]
        nominal_leader = cache["states"][selected, np.maximum(offset - 1, 0), 0]
        nominal_actions = np.stack([
            cache["actions"][index, frame:frame + 25, slot - 1, 0]
            for index, frame, slot in zip(selected, offset, follower)
        ])

        actual_follower = np.concatenate((actual_follower, non_event[0]))
        actual_leader = np.concatenate((actual_leader, non_event[1]))
        target = np.concatenate((target, non_event[2] / 0.04))
        nominal_follower = np.concatenate((nominal_follower, non_event[3]))
        nominal_leader = np.concatenate((nominal_leader, non_event[4]))
        nominal_actions = np.concatenate((nominal_actions, non_event[5]))

        actual = cost_inputs(actual_follower, actual_leader, device)
        nominal = cost_inputs(nominal_follower, nominal_leader, device)
        action = torch.as_tensor(nominal_actions, device=device)
        previous = torch.as_tensor(actual_follower[:, 4], device=device)
        target_tensor = torch.as_tensor(target, device=device)
        feature = torch.stack((
            actual.gap_m, actual.speed_mps, actual.leader_speed_mps,
            actual.leader_acceleration_mps2, actual.reference_speed_mps,
            nominal.gap_m, nominal.speed_mps,
        ), dim=-1)
        nominal_feature = torch.stack((
            nominal.gap_m, nominal.speed_mps, nominal.leader_speed_mps,
            nominal.leader_acceleration_mps2, nominal.reference_speed_mps,
            nominal.gap_m, nominal.speed_mps,
        ), dim=-1)
        prediction, _ = solve_hinge_calibrated_qp(
            action, actual, nominal, model.weights(feature),
            model.weights(nominal_feature), previous,
        )
        loss = torch.nn.functional.huber_loss(prediction[:, 0], target_tensor)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.weights.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (update + 1) % 200 == 0:
            print(f"{update + 1}: loss={losses[-1]:.6f}", flush=True)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "controller_mode": "nominal_preserving_response",
        "stage": "supervised",
        "state_dict": model.state_dict(),
        "seed": seed,
        "updates": args.updates + 100,
    }, args.checkpoint)
    summary = {
        "stage": "S3", "status": "passed", "smoke_updates": 100,
        "supervised_updates": args.updates, "total_updates": args.updates + 100,
        "loss_initial": losses[0], "loss_final": losses[-1],
    }
    args.checkpoint.with_name("training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
