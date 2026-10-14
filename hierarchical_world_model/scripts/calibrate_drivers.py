#!/usr/bin/env python3
"""Learn Train behavior and freeze Val-selected population refinement."""

from pathlib import Path
import sys
import shutil
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hierarchical_world_model.src.config import load_protocol_config
from hierarchical_world_model.src.core.utils import load_json, save_json
from hierarchical_world_model.src.data import artifact_identity
from hierarchical_world_model.src.npc_experiments.calibration import (
    fit_stage,
    freeze_stage_model,
)
from hierarchical_world_model.src.npc_experiments.search import run_search
from hierarchical_world_model.src.stochastic_drivers.raw_data import (
    DRIVER_DATA_DIR,
    prepare_raw_data,
)


def main():
    torch.set_num_threads(1)
    config = load_protocol_config(ROOT / "hierarchical_world_model/config.yaml")
    settings = config["npc_response"]
    output = ROOT / settings["output_dir"]
    (output / "policy_calibration").mkdir(parents=True, exist_ok=True)
    if not (DRIVER_DATA_DIR / "manifest.json").exists():
        prepare_raw_data(DRIVER_DATA_DIR, workers=4)
    for source, name in (
        ("idm_population.npz", "reference.npz"),
        ("mobil.json", "reference_mobil.json"),
        ("lane_trajectory.json", "lane_model.json"),
    ):
        target = output / "policy_calibration" / name
        if not target.exists():
            shutil.copy2(
                ROOT / "results/hierarchical_world_model/model_assets" / source, target
            )
    sources = output / "policy_calibration/upstream_sources.json"
    if not sources.exists():
        save_json(
            {
                name: artifact_identity(config["paths"][name])
                for name in (
                    "flow_checkpoint",
                    "diffusion_checkpoint",
                    "evaluation_checkpoint",
                )
            },
            sources,
        )
    selected_path = output / "policy_calibration/selected_model.json"
    if selected_path.exists():
        selected = load_json(selected_path)
        population = dict(np.load(selected["driver_parameter_source"]))
        mobil = selected["mobil"]
    else:
        population, mobil = fit_stage(output, settings)
    population, mobil, selection = run_search(output, population, mobil, settings)
    freeze_stage_model(output, population, mobil, selection)
    print("Frozen validation model", selection["selected"], flush=True)


if __name__ == "__main__":
    main()
