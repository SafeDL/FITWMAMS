#!/usr/bin/env python3
"""Evaluate scenario-constraint and motion randomness through public APIs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_traffic_world_model.src.composition import (  # noqa: E402
    HierarchicalWorldSampler,
)
from normalizing_flow.src.constraints import derived_modes  # noqa: E402
from world_model.src.core.utils import load_yaml, save_json  # noqa: E402

CONFIG = (
    ROOT
    / "hierarchical_traffic_world_model/configs/highd_hierarchical_world_model.yaml"
)


def pairwise_distance(values: np.ndarray) -> float:
    distances = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            distances.append(
                np.linalg.norm(values[left] - values[right], axis=-1).mean()
            )
    return float(np.mean(distances)) if distances else 0.0


def entropy(values: np.ndarray) -> float:
    _, count = np.unique(values, return_counts=True)
    probability = count / count.sum()
    return float(-(probability * np.log(probability)).sum())


def main() -> None:
    config = load_yaml(CONFIG)
    paths = config["paths"]
    if config["training"].get("experiment_scope") != "full":
        raise ValueError("sampling evaluation requires the maintained full protocol")
    output = Path(paths["output_dir"])
    arrays = np.load(Path(paths["flow_output_dir"]) / "dataset.npz")
    candidates = np.flatnonzero(np.asarray(arrays["split_index"]) == 2)
    slot_count = np.asarray(arrays["slot_mask"])[candidates].sum(axis=1)
    eligible = candidates[slot_count > 0]
    generator = np.random.default_rng(int(config["training"]["seed"]))
    rows = generator.choice(eligible, size=64, replace=False)
    c0 = np.asarray(arrays["features"][rows], np.float32)
    mask = np.asarray(arrays["slot_mask"][rows], bool)
    sampler = HierarchicalWorldSampler(
        flow_checkpoint=paths["flow_checkpoint"],
        flow_output_dir=paths["flow_output_dir"],
        diffusion_checkpoint=paths["diffusion_checkpoint"],
        diffusion_contract=paths["diffusion_contract"],
        response_checkpoint=output / "checkpoints/best_hierarchical_world_model.pt",
        repo_root=ROOT,
        device="cuda" if torch.cuda.is_available() else "cpu",
        ddim_steps=20,
    )
    execute_frames = sampler.response.cfg.execute_frames
    response_steps = int(np.ceil(149 / execute_frames))
    draws = 16
    scenario = sampler.sample_constraints(
        np.repeat(c0, draws, axis=0),
        np.repeat(mask, draws, axis=0),
        len(rows) * draws,
        scenario_seed=20260814,
        motion_seed=20260815,
    )
    probe = sampler.sample_constraints(
        c0[0], mask[0], 4, scenario_seed=20260814, motion_seed=20260815
    )
    repeated = sampler.sample_constraints(
        c0[0],
        mask[0],
        4,
        scenario_seed=20260814,
        motion_seed=20260815,
    )
    reproducible = np.array_equal(
        probe.scenario.trajectory_constraint,
        repeated.scenario.trajectory_constraint,
    ) and np.allclose(probe.soft_plan, repeated.soft_plan)
    motion = [
        sampler.sample_constraints(
            c0[0],
            mask[0],
            1,
            scenario_seed=20260814,
            motion_seed=20260900 + index,
        )
        for index in range(8)
    ]
    motion_plan = np.concatenate([item.soft_plan for item in motion])
    motion_trajectories = []
    for item in motion:
        motion_world = sampler.create_world(item)
        motion_actions = torch.zeros((1, execute_frames, 2), device=sampler.device)
        motion_frames = []
        for _ in range(response_steps):
            motion_step = motion_world.advance_response(motion_actions)
            motion_frames.append(
                motion_step["agent_state_frames"][:, :, 1:, :2].cpu().numpy()
            )
        motion_trajectories.append(np.concatenate(motion_frames, axis=1)[0])
    motion_trajectories = np.stack(motion_trajectories)
    modes = derived_modes(
        scenario.scenario.trajectory_constraint,
        scenario.scenario.slot_mask,
    ).reshape(len(rows), draws, 6, 2)
    mode_counts = []
    mode_entropies = []
    for index in range(len(rows)):
        valid_modes = modes[index][:, mask[index]].reshape(draws, -1)
        _, mode_codes = np.unique(valid_modes, axis=0, return_inverse=True)
        mode_counts.append(len(np.unique(mode_codes)))
        mode_entropies.append(entropy(mode_codes))
    world = sampler.create_world(scenario)
    actions = torch.zeros(
        (len(rows) * draws, execute_frames, 2), device=sampler.device
    )
    background_actions = []
    for _ in range(response_steps):
        step = world.advance_response(actions)
        background_actions.append(step["background_actions"].cpu().numpy())
    controls = np.concatenate(background_actions, axis=1)
    final = np.asarray(step["agent_states"].cpu()).reshape(
        len(rows), draws, 7, 6
    )
    knot_terminal = scenario.state_knot_reference[:, -1].reshape(
        len(rows), draws, 6, 2
    )
    soft_at_closed_horizon = scenario.soft_plan[:, 144].reshape(
        len(rows), draws, 6, 2
    )
    knot_pairwise = np.asarray(
        [
            pairwise_distance(knot_terminal[index][:, mask[index]])
            for index in range(len(rows))
        ]
    )
    closed_pairwise = np.asarray(
        [
            pairwise_distance(final[index, :, 1:][:, mask[index], :2])
            for index in range(len(rows))
        ]
    )
    soft_pairwise = np.asarray(
        [
            pairwise_distance(soft_at_closed_horizon[index][:, mask[index]])
            for index in range(len(rows))
        ]
    )
    terminal_error = np.linalg.norm(
        final[:, :, 1:, :2] - soft_at_closed_horizon, axis=-1
    )
    terminal_error = terminal_error[
        np.broadcast_to(mask[:, None], terminal_error.shape)
    ]
    report = {
        "experiment_scope": "full",
        "test_c0_conditions": int(len(rows)),
        "valid_background_slots_mean": float(mask.sum(axis=1).mean()),
        "seed_reproducible": bool(reproducible),
        "scenario_constraint_randomness": {
            "draws_per_condition": draws,
            "unique_joint_modes_mean": float(np.mean(mode_counts)),
            "joint_mode_entropy_mean_nats": float(np.mean(mode_entropies)),
            "state_knot_terminal_pairwise_distance_m": {
                "mean": float(knot_pairwise.mean()),
                "p10": float(np.quantile(knot_pairwise, 0.1)),
                "p90": float(np.quantile(knot_pairwise, 0.9)),
            },
            "closed_loop_terminal_pairwise_distance_m": {
                "mean": float(closed_pairwise.mean()),
                "p10": float(np.quantile(closed_pairwise, 0.1)),
                "p90": float(np.quantile(closed_pairwise, 0.9)),
            },
            "soft_plan_terminal_pairwise_distance_m": {
                "mean": float(soft_pairwise.mean()),
                "p10": float(np.quantile(soft_pairwise, 0.1)),
                "p90": float(np.quantile(soft_pairwise, 0.9)),
            },
            "closed_loop_to_soft_terminal_error_m": {
                "mean": float(terminal_error.mean()),
                "p50": float(np.quantile(terminal_error, 0.5)),
                "p95": float(np.quantile(terminal_error, 0.95)),
            },
        },
        "fixed_constraint_motion_randomness": {
            "draws": 8,
            "trajectory_constraint_K_held_fixed": True,
            "soft_plan_pairwise_distance_m": pairwise_distance(
                motion_plan[:, :, mask[0]]
            ),
            "soft_terminal_pairwise_distance_m": pairwise_distance(
                motion_plan[:, -1, mask[0]]
            ),
            "closed_loop_pairwise_distance_m": pairwise_distance(
                motion_trajectories[:, :, mask[0]]
            ),
            "closed_loop_terminal_pairwise_distance_m": pairwise_distance(
                motion_trajectories[:, -1, mask[0]]
            ),
        },
        "physical_validity": {
            "finite": bool(np.isfinite(final).all() and np.isfinite(controls).all()),
            "longitudinal_action_bounds": bool(
                (controls[..., 0] >= -8.0).all() and (controls[..., 0] <= 4.0).all()
            ),
            "yaw_rate_bounds": bool((np.abs(controls[..., 1]) <= 0.6).all()),
        },
    }
    save_json(report, output / "sampling_hierarchy_evaluation.json")
    print(report)


if __name__ == "__main__":
    main()
