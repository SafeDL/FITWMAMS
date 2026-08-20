#!/usr/bin/env python3
"""Diagnose response-noise scales against a diffusion-only matched baseline."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import ANCHOR_INDEX, pilot_rows  # noqa: E402
from hierarchical_traffic_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_traffic_world_model.src.evaluation import (  # noqa: E402
    _distribution_metrics,
    rollout,
)
from hierarchical_traffic_world_model.src.planner import (  # noqa: E402
    stochastic_diffusion_plan_samples,
)
from hierarchical_traffic_world_model.src.train import load_checkpoint  # noqa: E402
from world_model.src.core.dynamics import KinematicTrafficDynamics  # noqa: E402
from world_model.src.core.utils import load_yaml, save_json, select_device  # noqa: E402

CONFIG = ROOT / "hierarchical_traffic_world_model/configs/highd_hierarchical_world_model.yaml"
CANDIDATES = (
    ("yaw_0p003", 0.60, 0.003, 1.0, 1.0, 0.999),
    ("yaw_0p006", 0.60, 0.006, 1.0, 1.0, 0.999),
    ("yaw_0p009", 0.60, 0.009, 1.0, 1.0, 0.999),
    ("yaw_0p012", 0.60, 0.012, 1.0, 1.0, 0.999),
)


def _set_motion(
    model,
    longitudinal_scale: float,
    yaw_scale: float,
    scene_multiplier: float,
    agent_multiplier: float,
    correlation: float,
    *,
    stochastic: bool,
) -> None:
    model.cfg = replace(
        model.cfg,
        stochastic_latents=bool(stochastic),
        stochastic_longitudinal_jerk_mps3=float(longitudinal_scale),
        stochastic_yaw_acceleration_rps2=float(yaw_scale),
        scene_noise_scale=0.17 * float(scene_multiplier),
        agent_noise_scale=0.06 * float(agent_multiplier),
        agent_noise_correlation=float(correlation),
    )
    model.decoder.cfg = model.cfg


def main() -> None:
    config = load_yaml(CONFIG)
    if config["training"].get("experiment_scope") != "full":
        raise ValueError("motion calibration requires the maintained full protocol")
    root_output = Path(config["paths"]["output_dir"])
    output = root_output / "motion_calibration.json"
    device = select_device(config["training"].get("device", "auto"))
    experiment = prepare_experiment_data(config, CONFIG.parent)
    seed = int(config["training"]["seed"])
    rows = pilot_rows(experiment.bundle, "val", maximum=512, seed=seed)
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
    motion_seeds = tuple(seed + sample for sample in range(16))
    plans = stochastic_diffusion_plan_samples(
        experiment.bundle,
        rows,
        checkpoint=config["paths"]["diffusion_checkpoint"],
        device=device,
        batch_size=32,
        ddim_steps=20,
        motion_seeds=motion_seeds,
    )
    checkpoint = root_output / "checkpoints/best_hierarchical_world_model.pt"
    model, _ = load_checkpoint(checkpoint, device=device)
    model.eval()

    def evaluate(
        target_model,
        longitudinal_scale: float,
        yaw_scale: float,
        scene_multiplier: float,
        agent_multiplier: float,
        correlation: float,
        stochastic: bool,
    ) -> dict:
        _set_motion(
            target_model,
            longitudinal_scale,
            yaw_scale,
            scene_multiplier,
            agent_multiplier,
            correlation,
            stochastic=stochastic,
        )
        samples = [
            rollout(
                target_model,
                states,
                valid,
                plan,
                maps,
                map_valid,
                device=device,
                history_frames=25,
                motion_seed=motion_seed + 100_000 if stochastic else None,
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

    baseline = evaluate(model, 0.0, 0.0, 0.0, 0.0, 0.0, False)
    candidates = []
    for name, longitudinal, yaw, scene, agent, correlation in CANDIDATES:
        metrics = evaluate(
            model, longitudinal, yaw, scene, agent, correlation, True
        )
        improvement = (
            baseline["energy_score_m"] - metrics["energy_score_m"]
        ) / max(abs(baseline["energy_score_m"]), 1.0e-8)
        degradation = {
            name: metrics["motion_distribution"][name]["KS"]
            / max(baseline["motion_distribution"][name]["KS"], 1.0e-8)
            - 1.0
            for name in ("speed", "ax")
        }
        passed = (
            improvement >= 0.05
            and metrics["mean_pairwise_trajectory_distance_m"] >= 0.02
            and all(value <= 0.10 for value in degradation.values())
        )
        candidates.append(
            {
                "name": name,
                "stochastic_longitudinal_jerk_mps3": longitudinal,
                "stochastic_yaw_acceleration_rps2": yaw,
                "scene_multiplier": scene,
                "agent_multiplier": agent,
                "agent_noise_correlation": correlation,
                "energy_improvement_fraction": improvement,
                "distribution_degradation_fraction": degradation,
                "passed": passed,
                "metrics": metrics,
            }
        )
    report = {
        "status": "diagnostic",
        "selection_split": "validation",
        "validation_sequences": len(rows),
        "samples_per_condition": len(motion_seeds),
        "diffusion_only_baseline": baseline,
        "candidates": candidates,
        "any_candidate_passed": any(item["passed"] for item in candidates),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, output)
    print(
        {
            "baseline_energy": baseline["energy_score_m"],
            "candidates": [
                (item["name"], item["energy_improvement_fraction"], item["passed"])
                for item in candidates
            ],
        }
    )


if __name__ == "__main__":
    main()
