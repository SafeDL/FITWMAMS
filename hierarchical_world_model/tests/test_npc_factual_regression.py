import numpy as np

from hierarchical_world_model.scripts.verify import (
    test_scene_identity_matches as scene_matches,
)
from hierarchical_world_model.src.shared_evaluation import scene_randomness


def test_complete_test_identity_cannot_be_replaced_by_same_count_subset():
    import pandas as pd

    ids = np.array(["seq_01_20_300", "seq_02_21_301", "seq_03_22_302"])
    frame = pd.DataFrame({"sequence_id": ids, "recording_id": [1, 2, 3]})
    assert scene_matches(frame, ids)
    frame.loc[2, "sequence_id"] = ids[1]
    assert not scene_matches(frame, ids)


def test_scene_randomness_is_stable_when_batch_size_and_order_change():
    ids = np.array(["seq_01_20_300", "seq_02_21_301", "seq_03_22_302"])
    first = scene_randomness(ids, seed=20261002, frames=149)
    reordered = scene_randomness(ids[[2, 0, 1]], seed=20261002, frames=149)
    np.testing.assert_array_equal(
        first.driver_latent[[2, 0, 1]], reordered.driver_latent
    )
    np.testing.assert_array_equal(
        first.agent_response_innovations[[2, 0, 1]],
        reordered.agent_response_innovations,
    )


def test_factual_consolidation_keeps_samples_identity_and_execution(tmp_path):
    import pandas as pd
    from hierarchical_world_model.src.core.utils import save_json, load_json
    from hierarchical_world_model.src.npc_experiments.factual_regression import (
        consolidate_factual_batches,
    )

    batches = tmp_path / "batches"
    batches.mkdir()
    rows, identifiers = [8, 3], ["nat_49_1_200", "nat_50_2_400"]
    save_json(
        {"rows": rows, "sequence_ids": identifiers, "batch_size": 1},
        tmp_path / "protocol.json",
    )
    pd.DataFrame({"row": rows, "sequence_id": identifiers}).to_csv(
        tmp_path / "scene_metrics.csv", index=False
    )
    for index in range(2):
        save_json(
            {
                "status": "complete",
                "rows": [rows[index]],
                "sequence_ids": [identifiers[index]],
                "road_steps": 149,
                "elapsed_seconds": 1.5,
                "records": [],
            },
            batches / f"{index:06d}.json",
        )
        np.savez_compressed(
            batches / f"{index:06d}.npz",
            errors=np.array([index], np.float32),
            jerks=np.array([index, 2], np.float32),
        )
    consolidate_factual_batches(tmp_path)
    assert not batches.exists()
    with np.load(tmp_path / "error_samples.npz") as data:
        np.testing.assert_array_equal(data["errors"], [0, 1])
        np.testing.assert_array_equal(data["jerks"], [0, 2, 1, 2])
        np.testing.assert_array_equal(data["error_offsets"], [0, 1, 2])
        np.testing.assert_array_equal(data["jerk_offsets"], [0, 2, 4])
    assert (
        sum(row["road_steps"] for row in load_json(tmp_path / "execution.json")) == 298
    )


def test_incomplete_factual_batch_is_preserved(tmp_path):
    import pytest
    from hierarchical_world_model.src.core.utils import save_json
    from hierarchical_world_model.src.npc_experiments.factual_regression import (
        consolidate_factual_batches,
    )

    save_json(
        {"rows": [1], "sequence_ids": ["nat_49_1_200"], "batch_size": 1},
        tmp_path / "protocol.json",
    )
    path = tmp_path / "batches/000000.json"
    save_json({"status": "failed", "reason": "nonfinite state"}, path)
    with pytest.raises(ValueError, match="Incomplete factual batch"):
        consolidate_factual_batches(tmp_path)
    assert path.exists()
    assert not (tmp_path / "error_samples.npz").exists()


def test_published_regression_is_reused_without_creating_a_pending_run(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from hierarchical_world_model.src.npc_experiments import (
        factual_regression as regression,
    )
    from hierarchical_world_model.src.npc_experiments import search as npc_search
    from hierarchical_world_model.src.core.utils import save_json

    monkeypatch.setattr(regression, "ROOT", tmp_path)
    monkeypatch.setattr(
        regression, "artifact_identity", lambda path: {"path": str(path)}
    )
    monkeypatch.setattr(npc_search, "load_candidate", lambda *args: ({}, {}))
    ids = np.array(["seq_01_20_300", "seq_02_21_301"])
    experiment = SimpleNamespace(
        test_rows=np.array([0, 1]), bundle=SimpleNamespace(arrays={"sequence_id": ids})
    )
    monkeypatch.setattr(regression, "prepare_experiment_data", lambda *args: experiment)
    output = tmp_path / "npc"
    fit = {
        "fit_split": "train",
        "selected_candidate": "reference",
        "mobil": {"bias": -3},
        "driver_population": {"log_mean": [1] * 5},
    }
    save_json(fit, output / "policy_calibration/selected_model.json")
    config = {
        "npc_response": {"world_batch_size": 64},
        "paths": {"evaluation_checkpoint": "hiq.pt", "diffusion_checkpoint": "diff.pt"},
    }
    protocol = {
        "rows": [0, 1],
        "sequence_ids": ids.tolist(),
        "batch_size": 64,
        "mobil": fit["mobil"],
        "driver_population": fit["driver_population"],
        "selected_candidate": "reference",
        "seed": 20261002,
        "response": {"path": "hiq.pt"},
        "diffusion": {"path": "diff.pt"},
        "lane": {"path": str(regression.LANE_PATH)},
        "npc_lane_model": None,
        "npc_fit_split": "train",
        "horizon_frames": 149,
        "response_activation": "longitudinal_override_requires_longitudinal_component",
    }
    destination = tmp_path / "results/hierarchical_world_model/factual_test"
    save_json(protocol, destination / "protocol.json")
    report = {
        "status": "frozen_test_factual_complete_not_world_acceptance",
        "sequences_completed": 2,
    }
    save_json(report, destination / "report.json")
    save_json({"passed": True}, output / "verification.json")
    assert regression.evaluate_factual_regression(config, output) == report
    assert not (destination.parent / "test_candidate").exists()


def test_publication_points_to_runtime_parameters_not_the_candidate_snapshot(
    tmp_path, monkeypatch
):
    from hierarchical_world_model.scripts import publish, verify
    from hierarchical_world_model.src.npc_experiments import calibration
    from hierarchical_world_model.src.npc_experiments import (
        factual_regression as regression,
    )
    from hierarchical_world_model.src.stochastic_drivers import idm_population
    from hierarchical_world_model.src.core.utils import load_json, save_json

    output = tmp_path / "npc"
    folder = output / "policy_calibration"
    folder.mkdir(parents=True)
    artifacts = tmp_path / "model_assets"
    artifacts.mkdir()
    parameters = {
        "log_mean": np.zeros(5),
        "log_cov": np.eye(5),
        "residual_std": np.array(0.1),
        "fit_driver_count": np.array(10),
        "regularization": np.array(0.1),
    }
    model_path = artifacts / "idm_population.npz"
    np.savez_compressed(model_path, **parameters)
    np.savez_compressed(folder / "selected.npz", **parameters)
    lane_path = artifacts / "lane_trajectory.json"
    save_json({"tracking": {}}, lane_path)
    mobil = {"lane_model": load_json(lane_path)}
    save_json({"mobil": mobil}, folder / "selected.json")
    fit = {
        "selected_candidate": "selected",
        "driver_parameter_source": str(folder / "selected.npz"),
        "mobil": mobil,
    }
    save_json(fit, folder / "selected_model.json")
    save_json({}, folder / "upstream_sources.json")
    save_json({"passed": True}, output / "verification.json")
    pending = tmp_path / "results/hierarchical_world_model/factual_test_candidate"
    report = {"status": "frozen_test_factual_complete_not_world_acceptance"}
    save_json(report, pending / "report.json")
    monkeypatch.setattr(regression, "ROOT", tmp_path)
    monkeypatch.setattr(regression, "LANE_PATH", lane_path)
    monkeypatch.setattr(idm_population, "MODEL_PATH", model_path)
    monkeypatch.setattr(calibration, "MOBIL_PATH", artifacts / "mobil.json")
    monkeypatch.setattr(publish, "main", lambda: None)
    monkeypatch.setattr(verify, "main", lambda: None)

    assert regression.publish_response({"paths": {}}, output) == report
    published = load_json(artifacts / "mobil.json")
    assert published["driver_parameter_source"] == str(model_path)
    assert load_json(folder / "selected_model.json") == fit
    assert not pending.exists()
    assert load_json(output / "publication.json")["status"] == "published"
