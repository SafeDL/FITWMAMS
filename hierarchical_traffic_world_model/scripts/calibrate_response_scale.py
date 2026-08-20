#!/usr/bin/env python3
"""Select the causal-response scale on validation data, then test it once."""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import ANCHOR_INDEX, pilot_rows, split_rows  # noqa: E402
from hierarchical_traffic_world_model.src.calibration import (  # noqa: E402
    matched_response_calibration,
)
from hierarchical_traffic_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_traffic_world_model.src.evaluation import (  # noqa: E402
    _factual_metrics,
    _intervention_metrics,
    evaluate_world_model,
    rollout,
)
from hierarchical_traffic_world_model.src.planner import (  # noqa: E402
    frozen_diffusion_plans,
)
from hierarchical_traffic_world_model.src.train import load_checkpoint  # noqa: E402
from world_model.src.core.utils import (  # noqa: E402
    load_json,
    load_yaml,
    save_json,
    select_device,
)

CONFIG = (
    ROOT
    / "hierarchical_traffic_world_model/configs/highd_hierarchical_world_model.yaml"
)
CANDIDATES = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.25, 3.5, 3.75, 4.0)


def _set_scale(model, scale: float) -> None:
    model.cfg = replace(model.cfg, causal_response_scale=float(scale))
    model.decoder.cfg = model.cfg


def _validation_metrics(config: dict, model, experiment, device) -> list[dict]:
    seed = int(config["training"]["seed"])
    rows = pilot_rows(
        experiment.bundle, "val", maximum=512, seed=seed
    )
    arrays = experiment.bundle.arrays
    states = np.asarray(arrays["agent_states"])[rows].astype(np.float32)
    valid = np.asarray(arrays["agent_valid"])[rows].astype(bool)
    maps = np.asarray(arrays["map_polylines"])[rows].astype(np.float32)
    map_valid = np.asarray(arrays["map_polyline_valid"])[rows].astype(bool)
    active = valid[:, ANCHOR_INDEX, 1:]
    target = states[:, ANCHOR_INDEX + 1 : 174]
    with tempfile.TemporaryDirectory(prefix="response_scale_plans_") as cache:
        plans = frozen_diffusion_plans(
            experiment.bundle,
            rows,
            checkpoint=config["paths"]["diffusion_checkpoint"],
            output_dir=cache,
            device=device,
            batch_size=32,
            ddim_steps=20,
            experiment_scope="full",
        )
    distance = np.linalg.norm(plans - target[..., 1:, :2], axis=-1)
    mask = np.broadcast_to(active[:, None], distance.shape)
    diffusion = {
        "ADE_m": float(distance[mask].mean()),
        "FDE_m": float(distance[:, -1][active].mean()),
        "P95_displacement_error_m": float(np.quantile(distance[mask], 0.95)),
    }
    _, natural = matched_response_calibration(
        arrays,
        split_rows(arrays, "val", seed=seed),
        minimum_events=30,
    )
    common_seed = seed + 9_000
    rows_out = []
    for scale in CANDIDATES:
        _set_scale(model, scale)
        factual_rollout = rollout(
            model,
            states,
            valid,
            plans,
            maps,
            map_valid,
            device=device,
            history_frames=25,
            motion_seed=None,
        )
        factual = _factual_metrics(factual_rollout.states, target, active)
        baseline = rollout(
            model,
            states,
            valid,
            plans,
            maps,
            map_valid,
            device=device,
            history_frames=25,
            motion_seed=common_seed,
        )
        interventions = {}
        for kind, doses in {"brake": (1.5, 3.0), "accelerate": (1.0, 2.0)}.items():
            sampled = []
            for dose in doses:
                sampled.append(
                    rollout(
                        model,
                        states,
                        valid,
                        plans,
                        maps,
                        map_valid,
                        device=device,
                        history_frames=25,
                        motion_seed=common_seed,
                        intervention=kind,
                        dose=dose,
                    )
                )
            interventions[kind] = _intervention_metrics(
                baseline,
                sampled[0],
                sampled[1],
                states,
                active,
                kind,
                np.asarray(natural[kind]["effect_samples_mps2"], np.float32),
            )
        training = config["training"]
        factual_pass = (
            factual["ADE_m"]
            <= diffusion["ADE_m"] + float(training["factual_ade_tolerance_m"])
            and factual["FDE_m"]
            <= diffusion["FDE_m"] + float(training["factual_fde_tolerance_m"])
            and factual["P95_displacement_error_m"]
            <= diffusion["P95_displacement_error_m"]
            + float(training["factual_p95_tolerance_m"])
        )
        intervention_pass = all(
            value["direction_success_rate"] >= 0.90
            and value["dose_monotonicity_rate"] >= 0.95
            and value["locality_ratio_far_to_near"] <= 0.20
            and value["response_within_natural_p10_p90_rate"] >= 0.50
            for value in interventions.values()
        )
        rows_out.append(
            {
                "scale": scale,
                "factual": factual,
                "diffusion": diffusion,
                "interventions": interventions,
                "factual_gate_passed": factual_pass,
                "intervention_gate_passed": intervention_pass,
                "selection_wasserstein": sum(
                    value["response_distribution_wasserstein_mps2"]
                    for value in interventions.values()
                ),
            }
        )
    return rows_out


def _test_gates(report: dict, config: dict) -> dict:
    factual = report["factual_fidelity"]
    baseline = factual["open_loop_diffusion"]
    guided = factual["diffusion_guided_hiqr"]
    training = config["training"]
    factual_pass = (
        guided["ADE_m"] <= baseline["ADE_m"] + training["factual_ade_tolerance_m"]
        and guided["FDE_m"] <= baseline["FDE_m"] + training["factual_fde_tolerance_m"]
        and guided["P95_displacement_error_m"]
        <= baseline["P95_displacement_error_m"]
        + training["factual_p95_tolerance_m"]
    )
    distribution = report["distribution_stochasticity"]
    distribution_pass = (
        distribution["mean_pairwise_trajectory_distance_m"] >= 0.02
        and distribution["energy_score_m"] < distribution["sample_mean_ADE_m"]
    )
    intervention_pass = all(
        report["intervention_effectiveness"][name]["direction_success_rate"] >= 0.90
        and report["intervention_effectiveness"][name]["dose_monotonicity_rate"] >= 0.95
        and report["intervention_effectiveness"][name]["locality_ratio_far_to_near"]
        <= 0.20
        and report["intervention_effectiveness"][name][
            "response_within_natural_p10_p90_rate"
        ]
        >= 0.50
        for name in ("brake", "accelerate")
    )
    return {
        "factual_fidelity": factual_pass,
        "distribution_stochasticity": distribution_pass,
        "intervention_effectiveness": intervention_pass,
        "all_passed": factual_pass and distribution_pass and intervention_pass,
    }


def main() -> None:
    config = load_yaml(CONFIG)
    if config["training"].get("experiment_scope") != "full":
        raise ValueError("response calibration requires the maintained full protocol")
    root_output = Path(config["paths"]["output_dir"])
    device = select_device(config["training"].get("device", "auto"))
    experiment = prepare_experiment_data(config, CONFIG.parent)
    source_checkpoint = root_output / "checkpoints/best_hierarchical_world_model.pt"
    model, payload = load_checkpoint(source_checkpoint, device=device)
    candidates = _validation_metrics(config, model, experiment, device)
    eligible = [
        row
        for row in candidates
        if row["factual_gate_passed"] and row["intervention_gate_passed"]
    ]
    if not eligible:
        save_json(
            {"status": "failed", "split": "validation", "candidates": candidates},
            root_output / "response_scale_calibration.json",
        )
        raise RuntimeError("no validation-calibrated response scale passed all gates")
    selected = min(eligible, key=lambda row: row["selection_wasserstein"])
    scale = float(selected["scale"])
    payload["model_config"]["causal_response_scale"] = scale
    staging = root_output / "response_scale_staging"
    checkpoint = staging / "checkpoints/best_hierarchical_world_model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint)
    test_config = copy.deepcopy(config)
    test_config["paths"]["output_dir"] = str(staging)
    report = evaluate_world_model(test_config, config_dir=CONFIG.parent)
    gates = _test_gates(report, config)
    calibration = {
        "status": "complete" if gates["all_passed"] else "test_gate_failed",
        "selection_split": "validation",
        "validation_sequences": 512,
        "candidates": candidates,
        "selected_scale": scale,
        "selected_validation": selected,
        "held_out_test_gates": gates,
    }
    save_json(calibration, staging / "response_scale_calibration.json")
    if not gates["all_passed"]:
        raise RuntimeError(f"selected scale {scale} failed held-out test gates")
    shutil.copy2(checkpoint, source_checkpoint)
    shutil.copy2(staging / "evaluation.json", root_output / "evaluation.json")
    shutil.copy2(
        staging / "response_scale_calibration.json",
        root_output / "response_scale_calibration.json",
    )
    full_manifest = load_json(root_output / "full_training_manifest.json")
    full_manifest["response_scale"] = scale
    full_manifest["response_scale_selection_split"] = "validation"
    full_manifest["held_out_test_gates"] = gates
    save_json(full_manifest, root_output / "full_training_manifest.json")
    shutil.rmtree(staging)
    print(json.dumps({"selected_scale": scale, "test_gates": gates}, indent=2))


if __name__ == "__main__":
    main()
