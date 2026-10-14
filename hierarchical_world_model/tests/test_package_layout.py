"""Public entrypoints and component boundaries after world-model consolidation."""

from pathlib import Path

import numpy as np
import pandas as pd

import hierarchical_world_model as world
from hierarchical_world_model import src
from hierarchical_world_model.src.core.utils import save_json
from hierarchical_world_model.src.data import event_metadata
from hierarchical_world_model.src.npc_experiments import (
    calibration,
    factual_regression,
    road,
    search,
)
from hierarchical_world_model.src.stochastic_drivers import driver_policy, online
from hierarchical_world_model.src.stochastic_drivers.idm_population import (
    MODEL_PATH as IDM_PATH,
)
from hierarchical_world_model.src.stochastic_drivers.lane_trajectory import (
    MODEL_PATH as LANE_PATH,
)
from hierarchical_world_model.src.stochastic_drivers.raw_data import DRIVER_DATA_DIR


def test_public_exports_are_unique_and_resolve_to_the_maintained_components():
    for package in (world, src):
        assert len(package.__all__) == len(set(package.__all__))
        for name in package.__all__:
            assert getattr(package, name) is not None
    assert world.FactualDynamicsModel is src.FactualDynamicsModel
    assert world.WorldModelConfig is src.WorldModelConfig


def test_npc_package_keeps_repository_and_model_paths():
    root = Path(__file__).resolve().parents[2]
    assert calibration.ROOT == factual_regression.ROOT == search.ROOT == root
    assert (
        calibration.MOBIL_PATH
        == root / "results/hierarchical_world_model/model_assets/mobil.json"
    )


def test_runtime_artifacts_have_descriptive_names_without_single_file_folders():
    artifacts = (
        Path(__file__).resolve().parents[2]
        / "results/hierarchical_world_model/model_assets"
    )
    assert IDM_PATH == artifacts / "idm_population.npz"
    assert LANE_PATH == artifacts / "lane_trajectory.json"
    assert calibration.MOBIL_PATH == artifacts / "mobil.json"
    assert not any(path.is_dir() for path in artifacts.iterdir())
    assert all(
        (artifacts / name).is_file()
        for name in (
            "idm_population.npz",
            "mobil.json",
            "lane_trajectory.json",
            "lane_fit.json",
            "selection.json",
            "response_gate.pt",
        )
    )


def test_code_directory_contains_no_model_files_or_training_data():
    code = Path(__file__).resolve().parents[1]
    assert not (code / "model_assets").exists()
    assert not any(
        path.suffix in {".npz", ".npy", ".pt", ".csv"}
        for path in code.rglob("*")
        if path.is_file()
    )


def test_driver_and_gate_training_inputs_are_owned_by_the_shared_data_directory():
    shared = Path(__file__).resolve().parents[2] / "results/highd_shared_training_data"
    assert DRIVER_DATA_DIR == shared / "driver_behavior"
    assert (DRIVER_DATA_DIR / "manifest.json").is_file()
    assert (DRIVER_DATA_DIR / "road_cohorts.npz").is_file()
    assert all(
        (shared / "response_gate_training" / name).is_file()
        for name in ("simulation_traces.npz", "training_samples.npz", "events.csv")
    )


def test_autonomous_experiments_and_world_response_share_the_driving_kernel():
    for name in (
        "corrected_acceleration",
        "causal_maneuver_features",
        "exclusive_lane_probabilities",
        "response_history_features",
        "physical_leader_observation",
        "sample_lane_choice",
    ):
        assert (
            getattr(road, name) is getattr(online, name) is getattr(driver_policy, name)
        )


def test_event_metadata_uses_physical_identity_after_row_reordering(tmp_path):
    source = tmp_path / "events.csv"
    pd.DataFrame({"segment_id": ["seq_b", "seq_a"], "recording_id": [2, 1]}).to_csv(
        source, index=False
    )
    save_json({"source_dataset": str(source)}, tmp_path / "manifest.json")
    metadata = event_metadata(tmp_path, {"sequence_id": np.array(["seq_a", "seq_b"])})
    assert metadata.segment_id.tolist() == ["seq_a", "seq_b"]
    assert metadata.recording_id.tolist() == [1, 2]
