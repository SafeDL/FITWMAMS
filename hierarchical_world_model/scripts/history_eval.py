#!/usr/bin/env python3
"""Diagnose whether legal realized history changes a current response."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import ANCHOR_INDEX  # noqa: E402
from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.planner import frozen_diffusion_plans  # noqa: E402
from hierarchical_world_model.src.train import load_checkpoint  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from world_model.src.core.utils import ensure_dir, save_json, select_device  # noqa: E402
from world_model.src.core.evaluation_scope import (  # noqa: E402
    evaluation_scope_contract,
    scoped_canonical_trajectory,
)

CONFIG = ROOT / "hierarchical_world_model/config/release.yaml"


def _response(
    model,
    history: torch.Tensor,
    valid: torch.Tensor,
    reference: torch.Tensor,
    maps: torch.Tensor,
    map_valid: torch.Tensor,
    scene: torch.Tensor,
    agent: torch.Tensor,
) -> torch.Tensor:
    return model(
        history,
        valid,
        history[:, -1],
        valid[:, -1],
        reference,
        history[:, -1, 1:, :2],
        maps,
        map_valid,
        scene_standard_normal=scene,
        agent_standard_normal=agent,
    ).actions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose sensitivity to legal realized history under frozen weights."
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--sequences", type=int, default=128)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_protocol_config(config_path)
    device = select_device(config["training"].get("device", "auto"))
    output = ensure_dir(Path(config["paths"]["output_dir"]) / "final")
    experiment = prepare_experiment_data(config, config_path.parent)
    rows = experiment.validation_rows[: int(args.sequences)]
    with torch.no_grad():
        plans = frozen_diffusion_plans(
            experiment.bundle, rows, checkpoint=config["paths"]["diffusion_checkpoint"],
            output_dir=output / "history_plans", device=device, batch_size=64,
            ddim_steps=20, experiment_scope="history_response_diagnostic",
        )
        model, _ = load_checkpoint(config["paths"]["evaluation_checkpoint"], device=device)
        model.eval()
        states = np.asarray(experiment.bundle.arrays["agent_states"][rows], np.float32)
        valid = np.asarray(experiment.bundle.arrays["agent_valid"][rows], bool)
        states, valid = scoped_canonical_trajectory(states, valid)
        histories = torch.from_numpy(
            states[:, ANCHOR_INDEX - 24 : ANCHOR_INDEX + 1]
        ).to(device)
        history_valid = torch.from_numpy(
            valid[:, ANCHOR_INDEX - 24 : ANCHOR_INDEX + 1]
        ).to(device)
        maps = torch.from_numpy(
            np.asarray(experiment.bundle.arrays["map_polylines"][rows], np.float32)
        ).to(device)
        map_valid = torch.from_numpy(
            np.asarray(experiment.bundle.arrays["map_polyline_valid"][rows], bool)
        ).to(device)
        reference = torch.from_numpy(plans).to(device)
        generator = torch.Generator(device=device).manual_seed(917)
        scene = torch.randn(
            (len(rows), model.cfg.scene_latent_dim),
            generator=generator,
            device=device,
        )
        agent = torch.randn(
            (len(rows), 7, model.cfg.agent_latent_dim),
            generator=generator,
            device=device,
        )
        baseline = _response(
            model,
            histories,
            history_valid,
            reference,
            maps,
            map_valid,
            scene,
            agent,
        )
        permuted = histories.clone()
        permuted[:, :-1] = torch.flip(permuted[:, :-1], dims=(1,))
        permutation = _response(
            model,
            permuted,
            history_valid,
            reference,
            maps,
            map_valid,
            scene,
            agent,
        )
        intervention_history = histories.clone()
        # This is an observed, past ego braking history; the current state and
        # any future ego command remain unchanged.
        intervention_history[:, -6:-1, 0, 4] -= 2.0
        counterfactual = _response(
            model,
            intervention_history,
            history_valid,
            reference,
            maps,
            map_valid,
            scene,
            agent,
        )
    active = history_valid[:, -1, None, 1:, None]
    def delta(value: torch.Tensor) -> float:
        return float((value.abs() * active).sum().cpu() / active.sum().clamp_min(1))

    report = {
        "evaluation_scope": evaluation_scope_contract(),
        "sequences": int(len(rows)),
        "contract": (
            "same current state, soft plan and response innovations; only legal prior "
            "realized history changes"
        ),
        "history_permutation_mean_action_delta": delta(permutation - baseline),
        "intervention_history_mean_action_delta": delta(counterfactual - baseline),
        "history_module_observable": bool(delta(permutation - baseline) > 1.0e-8),
        "intervention_history_observable": bool(delta(counterfactual - baseline) > 1.0e-8),
    }
    report["all_passed"] = bool(
        report["history_module_observable"]
        and report["intervention_history_observable"]
    )
    report["interpretation"] = (
        "diagnostic only: a false result requires restrained history claims, "
        "not an automatic architecture change or retraining"
    )
    save_json(report, output / "history_response_sensitivity.json")


if __name__ == "__main__":
    main()
