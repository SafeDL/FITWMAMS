#!/usr/bin/env python3
"""Freeze Val-selected weights for Test and the separate generated ADS branch."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.composition import DEFAULT_WORLD_SELECTION
from hierarchical_world_model.src.core.utils import load_json, save_json
from hierarchical_world_model.src.data import artifact_identity, prepare_experiment_data
from hierarchical_world_model.src.config import load_protocol_config
from hierarchical_world_model.src.shared_evaluation import (
    MOBIL_PATH,
    load_shared_components,
)
from hierarchical_world_model.src.stochastic_drivers.idm_population import MODEL_PATH
from hierarchical_world_model.src.stochastic_drivers.lane_trajectory import (
    MODEL_PATH as LANE_TRAJECTORY_PATH,
)


def main():
    config = load_protocol_config(ROOT / "hierarchical_world_model/config.yaml")
    experiment = prepare_experiment_data(config, ROOT)
    _, training, population, fit = load_shared_components(config, experiment, "cpu")
    output = Path(config["paths"]["output_dir"])
    summary = load_json(output / "training_stochastic_heads_summary.json")
    if summary["status"] != "complete" or training["test_used_for_fit_or_selection"]:
        raise ValueError(
            "publication requires completed Train/Val training without Test selection"
        )
    checkpoint = Path(config["paths"]["evaluation_checkpoint"])
    lane = load_json(LANE_TRAJECTORY_PATH)
    if lane["test_used_for_parameter_selection"]:
        raise ValueError("lane parameters must not be selected on Test")
    selection = {
        "status": "selected_after_shared_val_review",
        "selected_model": "mobil",
        "selected_arm": "rules_recovery",
        "response_checkpoint": str(checkpoint),
        "response_checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
        "diffusion": artifact_identity(config["paths"]["diffusion_checkpoint"]),
        "driver_parameter_source": str(MODEL_PATH),
        "driver_population": {
            name: population[name].tolist() for name in ("log_mean", "log_cov")
        },
        "mobil_source": str(MOBIL_PATH),
        "mobil": fit["mobil"],
        "npc_lane_model": fit["mobil"].get("lane_model"),
        "lane_trajectory_model": str(LANE_TRAJECTORY_PATH),
        "lane_trajectory_model_mtime_ns": LANE_TRAJECTORY_PATH.stat().st_mtime_ns,
        "reference_recovery_enabled": True,
        "response_selection": "state-triggered rules; no learned selector",
        "source": fit["source"],
        "fit_split": fit["fit_split"],
        "selection_split": fit["selection_split"],
        "test_used_for_selection": fit["test_used_for_fit_or_selection"],
        "selection_reason": "Train calibration and bounded search, frozen Val selection",
        "evidence": {
            "training": str(output / "training_stochastic_heads_summary.json"),
            "npc_calibration": fit["selection_report"],
            "raw_data": fit["raw_data_identity"],
            "lane_labels": fit["label_protocol"],
        },
        "limitations": "Test with observed K_GT measures conditional reconstruction. Generated ADS scenes have no unique trajectory truth; neither branch is safety certification.",
    }
    if (
        not DEFAULT_WORLD_SELECTION.exists()
        or load_json(DEFAULT_WORLD_SELECTION) != selection
    ):
        save_json(selection, DEFAULT_WORLD_SELECTION)
    print(f"Frozen Val-selected world: {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
