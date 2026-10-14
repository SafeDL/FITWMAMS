#!/usr/bin/env python3
"""Verify current weights and complete Test reconstruction, independently of ADS."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.composition import load_selected_components
from hierarchical_world_model.src.core.utils import load_json, save_json
from hierarchical_world_model.src.data import artifact_identity, prepare_experiment_data
from hierarchical_world_model.src.config import load_protocol_config
from hierarchical_world_model.src.shared_evaluation import load_shared_components


def _recording_ids(sequence_ids):
    recordings = set()
    for identifier in sequence_ids:
        parts = str(identifier).split("_")
        if len(parts) < 2 or parts[0] != "nat":
            raise ValueError(f"unexpected highD sequence ID: {identifier!r}")
        recordings.add(int(parts[1]))
    return recordings


def test_scene_identity_matches(records, sequence_ids):
    identifiers = np.asarray(sequence_ids).astype(str)
    return (
        not records.sequence_id.duplicated().any()
        and set(records.sequence_id) == set(identifiers)
        and all(
            int(identifier.split("_")[1]) == rec
            for identifier, rec in zip(records.sequence_id, records.recording_id)
        )
    )


def report_matches_selection(report, selection):
    return (
        Path(report["response_checkpoint"]).resolve()
        == Path(selection["response_checkpoint"]).resolve()
        and report["response_checkpoint_mtime_ns"]
        == selection["response_checkpoint_mtime_ns"]
        and report.get("diffusion") == selection["diffusion"]
        and report.get("npc_lane_model") == selection.get("npc_lane_model")
        and report["mobil_parameters"] == selection["mobil"]
        and not report["test_used_for_selection"]
    )


def main():
    config = load_protocol_config(ROOT / "hierarchical_world_model/config.yaml")
    experiment = prepare_experiment_data(config, ROOT)
    _, training, _, _ = load_shared_components(config, experiment, "cpu")
    selection, _ = load_selected_components(config["paths"]["evaluation_checkpoint"])
    identifiers = np.asarray(experiment.bundle.arrays["sequence_id"]).astype(str)
    recordings = [
        _recording_ids(identifiers[rows])
        for rows in (
            experiment.train_rows,
            experiment.validation_rows,
            experiment.test_rows,
        )
    ]
    output = ROOT / "results/hierarchical_world_model/factual_test"
    report = load_json(output / "report.json")
    records = pd.read_csv(output / "scene_metrics.csv")
    contract = load_json(config["paths"]["diffusion_contract"])
    metrics = report["factual"]["rules_recovery"]
    checks = {
        "natural_train_and_val_only": not training["test_used_for_fit_or_selection"],
        "recording_disjoint": all(
            not recordings[a] & recordings[b] for a, b in ((0, 1), (0, 2), (1, 2))
        ),
        "condition_dim_68_without_dimensions": contract["condition_dim"] == 68,
        "current_diffusion": selection["diffusion"]
        == artifact_identity(config["paths"]["diffusion_checkpoint"]),
        "report_matches_frozen_components": report_matches_selection(report, selection),
        "complete_test": report["status"]
        == "frozen_test_factual_complete_not_world_acceptance"
        and report["sequences_completed"]
        == report["sequences_requested"]
        == len(experiment.test_rows),
        "test_scene_identity": test_scene_identity_matches(
            records, identifiers[experiment.test_rows]
        ),
        "test_row_identity": np.array_equal(
            identifiers[records.row.to_numpy()], records.sequence_id
        ),
        "ADE_matches_records": np.isclose(
            metrics["ADE_m"],
            records.distance_sum_m.sum() / records.actor_frames.sum(),
            atol=1e-9,
            rtol=0,
        ),
        "FDE_matches_records": np.isclose(
            metrics["FDE_m"],
            records.final_distance_sum_m.sum() / records.final_actors.sum(),
            atol=1e-9,
            rtol=0,
        ),
        "contacts_match_records": metrics["collision"] == int(records.collision.sum()),
    }
    with np.load(output / "error_samples.npz") as samples:
        checks["P95_matches_samples"] = np.isclose(
            metrics["P95_m"], np.quantile(samples["errors"], 0.95)
        )
        checks["jerk_P99_matches_samples"] = np.isclose(
            metrics["preimpact_longitudinal_jerk_P99_mps3"],
            np.quantile(samples["jerks"], 0.99),
        )
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "split_sizes": {
            "train": len(experiment.train_rows),
            "val": len(experiment.validation_rows),
            "test": len(experiment.test_rows),
        },
        "scope": "current Test K_GT reconstruction and component provenance; ADS generation is a separate application, not required reconstruction evidence",
    }
    save_json(result, output / "verification.json")
    print(result)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
