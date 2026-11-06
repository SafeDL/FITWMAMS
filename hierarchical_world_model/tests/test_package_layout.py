"""Public entrypoints and component boundaries after world-model consolidation."""

from pathlib import Path

import hierarchical_world_model as world
from hierarchical_world_model import src
from hierarchical_world_model.src.npc_experiments import (
    calibration,
    road,
    road_evaluation,
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
    assert world.WorldModelConfig is src.WorldModelConfig


def test_npc_package_keeps_repository_and_model_paths():
    root = Path(__file__).resolve().parents[2]
    assert calibration.ROOT == road_evaluation.ROOT == root
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
