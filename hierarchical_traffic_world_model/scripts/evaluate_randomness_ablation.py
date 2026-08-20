#!/usr/bin/env python3
"""Evaluate the two sources of motion randomness on a matched test cohort."""

from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import ANCHOR_INDEX  # noqa: E402
from hierarchical_traffic_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_traffic_world_model.src.evaluation import (  # noqa: E402
    _distribution_metrics,
    rollout,
)
from hierarchical_traffic_world_model.src.planner import (  # noqa: E402
    frozen_diffusion_plans,
    stochastic_diffusion_plan_samples,
)
from hierarchical_traffic_world_model.src.train import load_checkpoint  # noqa: E402
from world_model.src.core.dynamics import KinematicTrafficDynamics  # noqa: E402
from world_model.src.core.utils import load_yaml, save_json, select_device  # noqa: E402

CONFIG = ROOT / "hierarchical_traffic_world_model/configs/highd_hierarchical_world_model.yaml"
OUTPUT = ROOT / "results/hierarchical_traffic_world_model"
CHECKPOINT = OUTPUT / "checkpoints/best_hierarchical_world_model.pt"
SAMPLES = 16


def _relative_degradation(current: float, baseline: float) -> float:
    return float(current / max(abs(baseline), 1.0e-8) - 1.0)


def main() -> None:
    config = load_yaml(CONFIG)
    if config["training"].get("experiment_scope") != "full":
        raise ValueError("randomness evaluation requires the maintained full protocol")
    device = select_device(config["training"].get("device", "auto"))
    experiment = prepare_experiment_data(config, CONFIG.parent)
    rows = experiment.test_rows[:1024]
    arrays = experiment.bundle.arrays
    states = np.asarray(arrays["agent_states"])[rows].astype(np.float32)
    valid = np.asarray(arrays["agent_valid"])[rows].astype(bool)
    maps = np.asarray(arrays["map_polylines"])[rows].astype(np.float32)
    map_valid = np.asarray(arrays["map_polyline_valid"])[rows].astype(bool)
    active = valid[:, ANCHOR_INDEX, 1:]
    target = states[:, ANCHOR_INDEX + 1 : 174]
    target_highd = np.asarray(arrays["actions_highd"])[rows].astype(np.float32)
    source = states[:, ANCHOR_INDEX:173, 1:]
    target_actions = KinematicTrafficDynamics.controls_from_highd_actions(
        torch.from_numpy(target_highd.copy()), torch.from_numpy(source.copy())
    ).numpy()
    seed = int(config["training"]["seed"])
    motion_seeds = tuple(seed + sample for sample in range(SAMPLES))
    with tempfile.TemporaryDirectory(prefix="hierarchical_randomness_") as cache:
        fixed_plan = frozen_diffusion_plans(
            experiment.bundle,
            rows,
            checkpoint=config["paths"]["diffusion_checkpoint"],
            output_dir=cache,
            device=device,
            batch_size=32,
            ddim_steps=20,
            experiment_scope="full",
        )
    sampled_plans = stochastic_diffusion_plan_samples(
        experiment.bundle,
        rows,
        checkpoint=config["paths"]["diffusion_checkpoint"],
        device=device,
        batch_size=32,
        ddim_steps=20,
        motion_seeds=motion_seeds,
    )
    stochastic, _ = load_checkpoint(CHECKPOINT, device=device)
    stochastic.cfg = replace(
        stochastic.cfg,
        stochastic_longitudinal_jerk_mps3=0.60,
        stochastic_yaw_acceleration_rps2=0.006,
        agent_noise_correlation=0.999,
    )
    stochastic.decoder.cfg = stochastic.cfg
    deterministic, _ = load_checkpoint(CHECKPOINT, device=device)
    deterministic.cfg = replace(deterministic.cfg, stochastic_latents=False)
    deterministic.decoder.cfg = deterministic.cfg
    stochastic.eval()
    deterministic.eval()

    def evaluate(model, plans: list[np.ndarray], response_random: bool) -> dict:
        samples = [
            rollout(
                model,
                states,
                valid,
                plan,
                maps,
                map_valid,
                device=device,
                history_frames=25,
                motion_seed=motion_seed + 100_000 if response_random else None,
            )
            for plan, motion_seed in zip(plans, motion_seeds)
        ]
        return _distribution_metrics(
            samples,
            states[:, ANCHOR_INDEX],
            target,
            target_actions,
            target_highd,
            active,
        )

    fixed_plans = [fixed_plan] * SAMPLES
    quadrants = {
        "fully_deterministic": evaluate(deterministic, fixed_plans, False),
        "diffusion_only_random": evaluate(deterministic, sampled_plans, False),
        "response_only_random": evaluate(stochastic, fixed_plans, True),
        "full_hierarchical_random": evaluate(stochastic, sampled_plans, True),
    }
    baseline = quadrants["fully_deterministic"]
    motion_baseline = quadrants["diffusion_only_random"]
    full = quadrants["full_hierarchical_random"]
    energy_improvement = (
        baseline["energy_score_m"] - full["energy_score_m"]
    ) / baseline["energy_score_m"]
    response_energy_improvement = (
        motion_baseline["energy_score_m"] - full["energy_score_m"]
    ) / motion_baseline["energy_score_m"]
    distribution_degradation = {
        name: _relative_degradation(
            full["motion_distribution"][name]["KS"],
            motion_baseline["motion_distribution"][name]["KS"],
        )
        for name in ("speed", "ax")
    }
    windowed_jerk_degradation = {
        name: _relative_degradation(
            full["jerk_resolution_diagnostic"]["windowed_0p2s"][name]["KS"],
            motion_baseline["jerk_resolution_diagnostic"]["windowed_0p2s"][name][
                "KS"
            ],
        )
        for name in ("jx", "jy")
    }
    gates = {
        "energy_improvement_at_least_5pct": energy_improvement >= 0.05,
        "response_layer_improves_diffusion_only_energy": (
            response_energy_improvement > 0.0
        ),
        "trajectory_pairwise_at_least_2cm": full[
            "mean_pairwise_trajectory_distance_m"
        ]
        >= 0.02,
        "terminal_pairwise_at_least_5cm": full[
            "terminal_pairwise_distance_m"
        ]
        >= 0.05,
        "speed_ax_degradation_within_10pct": all(
            value <= 0.10 for value in distribution_degradation.values()
        ),
        "windowed_jerk_degradation_within_10pct": all(
            value <= 0.10 for value in windowed_jerk_degradation.values()
        ),
        "response_latent_is_used": quadrants["response_only_random"][
            "mean_pairwise_trajectory_distance_m"
        ]
        > 0.001,
    }
    gates["all_passed"] = all(gates.values())
    report = {
        "experiment_scope": "full",
        "cohort_sequences": int(len(rows)),
        "samples_per_condition": SAMPLES,
        "fixed_condition": "C0(40)+M(6)+K(72)",
        "selected_on": "validation",
        "stochastic_checkpoint": str(CHECKPOINT),
        "deterministic_checkpoint": str(CHECKPOINT),
        "same_weights_deterministic": True,
        "stochastic_response_config": {
            "longitudinal_jerk_mps3": 0.60,
            "yaw_acceleration_rps2": 0.006,
            "agent_noise_correlation": 0.999,
        },
        "quadrants": quadrants,
        "energy_improvement_fraction": float(energy_improvement),
        "response_energy_improvement_fraction": float(response_energy_improvement),
        "distribution_guard_baseline": "diffusion_only_random",
        "distribution_degradation_fraction": distribution_degradation,
        "windowed_jerk_degradation_fraction": windowed_jerk_degradation,
        "gates": gates,
    }
    save_json(report, OUTPUT / "randomness_ablation.json")
    print(
        {
            "energy": {
                name: values["energy_score_m"] for name, values in quadrants.items()
            },
            "pairwise": {
                name: values["mean_pairwise_trajectory_distance_m"]
                for name, values in quadrants.items()
            },
            "gates": gates,
        }
    )
    if not gates["all_passed"]:
        raise RuntimeError("matched hierarchical randomness evaluation failed")


if __name__ == "__main__":
    main()
