"""Refit the existing longitudinal correction with causal entry observations."""

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.core.utils import file_sha256, load_json, save_json
from hierarchical_world_model.src.data import artifact_identity
from hierarchical_world_model.src.npc_experiments.calibration import (
    conditional_nll,
    fit_longitudinal_correction,
    longitudinal_design,
)
from hierarchical_world_model.src.stochastic_drivers.idm_population import (
    idm_acceleration,
)
from hierarchical_world_model.src.stochastic_drivers.response_data import (
    PROTOCOL,
    RESPONSE_DATA_DIR,
    load_response,
    observation_signature,
    prepare_response_data,
)

OUTPUT = ROOT / "results/hierarchical_world_model/bridge"
ASSETS = ROOT / "results/hierarchical_world_model/model_assets"


def main():
    threadpool_limits(4)
    prepare_response_data(workers=8)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    population = dict(np.load(ASSETS / "idm_population.npz"))
    original = load_json(ASSETS / "mobil.json")
    data = load_response(splits=("train", "val"))
    candidates = []
    for ridge in (0.001, 0.01, 0.1, 1.0):
        path = OUTPUT / ".cache" / "longitudinal" / f"ridge_{ridge}.json"
        if path.exists():
            parameters = load_json(path)
        else:
            parameters = fit_longitudinal_correction(data["train"], population, ridge)
            save_json(parameters, path)
        mobil = deepcopy(original["mobil"])
        mobil["correction"]["longitudinal"] = parameters
        nll = conditional_nll(data["val"], population, mobil, samples=16)
        candidates.append({"ridge": ridge, "Val_NLL": nll, "parameters": parameters})
        print("Entry correction", ridge, nll, flush=True)
    chosen = min(candidates, key=lambda c: c["Val_NLL"])
    fit = deepcopy(original)
    fit["mobil"]["correction"]["longitudinal"] = chosen["parameters"]
    fit["response_protocol"] = PROTOCOL
    fit["behavior_protocol"] = PROTOCOL
    fit["correction_protocol"] = (
        "current_leader_and_causal_entry_single_residual_process"
    )
    fit["response_data_identity"] = artifact_identity(
        RESPONSE_DATA_DIR / "manifest.json"
    )
    report = {
        "complete": True,
        "fit_split": "train",
        "selection_split": "val",
        "test_used": False,
        "frozen_assets": {
            name: file_sha256(ASSETS / name)
            for name in ("idm_population.npz", "lane_trajectory.json")
        },
        "frozen_MOBIL_and_lane_correction": True,
        "source": fit["response_data_identity"],
        "candidates": [
            {k: v for k, v in c.items() if k != "parameters"} for c in candidates
        ],
        "selected_ridge": chosen["ridge"],
        "baseline_equation": "Published population-mean IDM; within-physical-driver centering for residual dispersion.",
        "validation_score": "Gaussian-mixture NLL proxy before action clipping and the unchanged internal IDM braking guard.",
        "observation_definition_sha256": observation_signature(),
        "counts": {},
        "conditional_groups": {},
    }
    for split, values in data.items():
        report["counts"][split] = {
            "states": len(values["gap"]),
            "short_gaps": int(((values["gap"] > 0) & (values["gap"] <= 2)).sum()),
            "anticipated_entries": int(values["entry_features"][:, 0].sum()),
        }
        x, _, _ = longitudinal_design(
            values,
            chosen["parameters"]["feature_mean"],
            chosen["parameters"]["feature_std"],
        )
        baseline = idm_acceleration(
            values["gap"],
            values["speed"],
            values["leader_speed"],
            np.exp(population["log_mean"]),
        ).clip(-8, 4)
        prediction = (baseline + x @ np.array(chosen["parameters"]["mean_coef"])).clip(
            -8, 4
        )
        report["conditional_groups"][split] = {}
        for name, mask in {
            "all": np.ones(len(x), bool),
            "entry": values["entry_features"][:, 0] > 0,
            "short_gap": (values["gap"] > 0) & (values["gap"] <= 2),
            "lane_change": values["changing"] > 0,
        }.items():
            residual = prediction[mask] - values["acceleration"][mask]
            report["conditional_groups"][split][name] = {
                "states": int(mask.sum()),
                "RMSE_mps2": float(np.sqrt(np.mean(residual**2)))
                if mask.any()
                else None,
                "bias_mps2": float(np.mean(residual)) if mask.any() else None,
            }
    save_json(report, OUTPUT / "npc_calibration.json")
    save_json(fit, ASSETS / "mobil.json")
    selection = load_json(ASSETS / "selection.json")
    selection["mobil"] = fit["mobil"]
    selection["response_protocol"] = PROTOCOL
    selection["response_selection"] = "observed interaction plan applicability"
    selection["evidence"]["longitudinal_calibration"] = str(
        OUTPUT / "npc_calibration.json"
    )
    selection["bridge_validation"] = str(OUTPUT / "test_report.json")
    save_json(selection, ASSETS / "selection.json")
    print(
        "Published longitudinal correction; frozen IDM/MOBIL/lane parameters retained.",
        flush=True,
    )


if __name__ == "__main__":
    main()
