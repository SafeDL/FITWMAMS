#!/usr/bin/env python3
"""Build the temporary prefix-only nominal-reference cache for S3 training."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from hierarchical_world_model.src.composition import HierarchicalWorldSampler
from hierarchical_world_model.src.data import prepare_experiment_data
from hierarchical_world_model.src.nominal_reference import build_nominal_reference
from hierarchical_world_model.src.prefix_reference import prefix_only_sample, row_prefix_inputs
from hierarchical_world_model.src.randomness import WorldExogenousState
from hierarchical_world_model.src.reaction_evidence import ReactionEventReference
from common import ROOT, event_directory, load_response_config

IDM = {"a_max": 1.0, "b_comfort": 2.0, "desired_speed": 30.0, "time_headway": 1.5, "minimum_gap": 2.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    response, base = load_response_config()
    data = prepare_experiment_data(base, ROOT)
    reference = ReactionEventReference.load(event_directory(response) / "train")
    selected = reference.events.indices((2,))
    selected = selected[reference.events.leader_slot[selected] == 0]
    sampler = HierarchicalWorldSampler(
        flow_checkpoint=base["paths"]["flow_checkpoint"],
        flow_output_dir=base["paths"]["flow_output_dir"],
        diffusion_checkpoint=base["paths"]["diffusion_checkpoint"],
        diffusion_contract=base["paths"]["diffusion_contract"],
        response_checkpoint=base["paths"]["evaluation_checkpoint"],
        repo_root=ROOT,
        device="cuda" if torch.cuda.is_available() else "cpu",
        ddim_steps=20,
        excluded_slots=(),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(selected), args.batch_size):
        path = args.output / f"chunk_{start:05d}.npz"
        if path.exists():
            continue
        event_index = selected[start:start + args.batch_size]
        rows = reference.events.row_index[event_index]
        c0, mask = row_prefix_inputs(data.bundle, rows)
        exogenous = WorldExogenousState.sample(
            len(event_index), seed=int(response["training"]["seed"]) + start, response_steps=149,
            scene_dim=sampler.response.cfg.scene_latent_dim,
            agent_dim=sampler.response.cfg.agent_latent_dim,
        )
        nominal = build_nominal_reference(
            sampler,
            prefix_only_sample(sampler, c0=c0, slot_mask=mask, exogenous=exogenous),
            idm_config=IDM,
        )
        np.savez_compressed(
            path,
            event_index=event_index,
            row_index=rows,
            onset=reference.events.local_onset_frame[event_index],
            follower_slot=reference.events.follower_slot[event_index],
            recording_id=reference.events.recording_id[event_index],
            states=nominal.states.cpu().numpy(),
            actions=nominal.background_actions.cpu().numpy(),
            seed=np.asarray(int(response["training"]["seed"]) + start),
        )
        print(f"{start + len(event_index)}/{len(selected)}", flush=True)


if __name__ == "__main__":
    main()
