"""Publish a Train/Val-selected NPC policy after its independent evaluation."""

from pathlib import Path

import numpy as np

from ..core.utils import load_json, save_json
from ..data import artifact_identity
from ..stochastic_drivers.idm_population import MODEL_PATH
from ..stochastic_drivers.lane_trajectory import MODEL_PATH as LANE_PATH
from .calibration import MOBIL_PATH, compact_population
from .search import load_candidate

ROOT = Path(__file__).resolve().parents[3]


def publish_npc(config, output):
    output = Path(output)
    if not load_json(output / "verification.json")["passed"]:
        raise ValueError("NPC evaluation has not passed verification")
    expected = load_json(output / "policy_calibration/upstream_sources.json")
    if any(
        value != artifact_identity(config["paths"][name])
        for name, value in expected.items()
    ):
        raise ValueError("Flow or Diffusion changed during NPC calibration")
    fit = load_json(output / "policy_calibration/selected_model.json")
    population, _ = load_candidate(
        output / "policy_calibration", fit["selected_candidate"]
    )
    np.savez_compressed(MODEL_PATH, **compact_population(population))
    fit["status"] = "npc_response_verified"
    fit["driver_parameter_source"] = str(MODEL_PATH)
    save_json(fit, MOBIL_PATH)
    save_json(fit["mobil"]["lane_model"], LANE_PATH)
    selection = {
        "status": "selected_after_shared_val_review",
        "selected_model": "mobil",
        "selected_arm": "tracking_npc_takeover",
        "fit_split": "train",
        "selection_split": "val",
        "test_used_for_selection": False,
        "source": fit["source"],
        "diffusion": artifact_identity(config["paths"]["diffusion_checkpoint"]),
        "driver_parameter_source": str(MODEL_PATH),
        "driver_population": {
            k: population[k].tolist() for k in ("log_mean", "log_cov")
        },
        "lane_trajectory_model": str(LANE_PATH),
        "lane_trajectory_model_mtime_ns": LANE_PATH.stat().st_mtime_ns,
        "mobil": fit["mobil"],
        "mobil_source": str(MOBIL_PATH),
        "evidence": {
            "npc_calibration": str(output / "policy_calibration/selected_model.json"),
            "verification": str(output / "verification.json"),
        },
    }
    save_json(selection, MODEL_PATH.parent / "selection.json")
    return selection
