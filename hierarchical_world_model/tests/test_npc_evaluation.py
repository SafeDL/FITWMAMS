import numpy as np

from hierarchical_world_model.src.core.utils import save_json
from hierarchical_world_model.src.npc_experiments import evaluation


def test_current_table_keeps_fitting_and_evaluation_scope_distinct(tmp_path):
    import pandas as pd

    result = {
        "acceleration": [],
        "lane": {
            "opportunities": 10,
            "Brier": 0.1,
            "Brier_CI95": [0.08, 0.12],
            "observed_choice_frequency": [8, 1, 1],
            "acceptance_Hellinger": [0.1, 0.2],
            "acceptance_SE": [0.01, 0.02],
            "acceptance_CI95": [[0.08, 0.12], [0.16, 0.24]],
        },
    }
    output = tmp_path / "npc_response"
    output.mkdir()
    evaluation._save_conditional_table(
        output, {"splits": {"all": {"reference": result}}}
    )
    table = pd.read_csv(output / "tables/conditional_metrics.csv")
    assert table.phase.eq("evaluation").all()
    assert table.fit_split.eq("train").all()
    assert table.split.eq("all").all()
    assert table.model.eq("reference").all()


def test_closed_table_reports_road_collision_once_with_half_sample_value():
    result = {
        "exposure": {"road_collision_run_fraction": 0.1},
        "exposure_CI95": {"road_collision_run_fraction": [0.0, 0.2]},
        "first_8_repeats_exposure": {"road_collision_run_fraction": 0.15},
        "road_collision": {
            "road_collision_run_fraction": {
                "value": 0.1,
                "first_half": 0.15,
                "CI95": [0.0, 0.2],
            }
        },
    }
    rows = evaluation._closed_rows("long", "joint", result, 120)
    assert len(rows) == 1
    assert rows[0]["metric"] == "road_collision_run_fraction"
    assert rows[0]["first_8_repeats_value"] == 0.15


def test_evaluation_loads_only_the_selected_frozen_policy(tmp_path):
    folder = tmp_path / "policy_calibration"
    folder.mkdir()
    parameters = folder / "selected.npz"
    np.savez(parameters, log_mean=np.full(5, 3.0), log_cov=np.eye(5))
    np.savez(folder / "unused.npz", log_mean=np.full(5, -999.0))
    save_json(
        {
            "selected_candidate": "selected",
            "driver_parameter_source": str(parameters),
            "mobil": {"bias": 2},
        },
        folder / "selected_model.json",
    )
    selected, models = evaluation.stage_models(tmp_path)
    assert selected == "reference"
    assert set(models) == {"reference"}
    assert models["reference"][0] == "selected"
    assert models["reference"][2]["bias"] == 2
    np.testing.assert_array_equal(models["reference"][1]["log_mean"], np.full(5, 3.0))


def test_convergence_subset_is_identity_stable_after_state_reordering():
    data = {
        "recording": np.repeat([4, 5], 100),
        "vehicle": np.tile(np.arange(100), 2),
        "frame": np.full(200, 25),
        "speed": np.full(200, 20.0),
        "leader_speed": np.full(200, 18.0),
        "free": np.tile([True, False], 100),
        "label": np.tile([0, 1], 100),
    }
    reorder = np.random.default_rng(4).permutation(200)
    for lane in (False, True):
        selected = evaluation.convergence_subset(data, 0.1, lane=lane)
        repeated = evaluation.convergence_subset(
            {key: value[reorder] for key, value in data.items()}, 0.1, lane=lane
        )
        assert len(selected["recording"]) == 20
        for field in ("recording", "vehicle", "frame"):
            np.testing.assert_array_equal(selected[field], repeated[field])


def test_demographic_groups_partition_states_with_frozen_density_boundaries():
    data = {
        "class": np.array([0, 1, 0, 1]),
        "direction": np.array([1, 1, 2, 2]),
        "density": np.array([10.0, 20.0, 30.0, 40.0]),
    }
    groups = evaluation.subgroup_masks(data, [20, 30])
    for names in (
        evaluation.SUBGROUPS[:2],
        evaluation.SUBGROUPS[2:4],
        evaluation.SUBGROUPS[4:],
    ):
        np.testing.assert_array_equal(sum(groups[name] for name in names), np.ones(4))
    assert groups["density_low"].tolist() == [True, False, False, False]
    assert groups["density_middle"].tolist() == [False, True, False, False]


def test_demographic_metrics_use_saved_recording_counts_and_keep_empty_groups():
    numeric = {
        "real": np.array([[2.0, 3.0], [1.0, 4.0]]),
        "model": np.array([[2.0, 3.0], [1.0, 4.0]]),
        "squared_error": np.array([0.2, 0.2]),
        "lane_count": np.array([100, 50]),
        "lane_error": np.array([6.0, 3.0]),
        "lane_observed": np.array([[80, 10, 10], [40, 5, 5]]),
        "lane_predicted": np.array([[80, 10, 10], [40, 5, 5]]),
    }
    report = evaluation.subgroup_report(numeric, 100)
    assert report["following_states"] == 10
    assert report["lane_opportunities"] == 150
    assert report["acceleration_Hellinger"] == 0
    assert np.isclose(report["mean_acceleration_RMSE_mps2"], 0.2)
    assert np.isclose(report["Brier"], 0.06)
    np.testing.assert_allclose(report["observed_choice_probability"], [0.8, 0.1, 0.1])
    empty = evaluation.subgroup_report(
        {key: np.zeros_like(value) for key, value in numeric.items()}, 100
    )
    assert empty["following_states"] == empty["lane_opportunities"] == 0
    assert empty["acceleration_Hellinger"] is empty["Brier"] is None
    assert empty["observed_choice_probability"] is None


def test_completed_road_shards_are_consolidated_without_losing_failures(tmp_path):
    import pandas as pd
    from hierarchical_world_model.src.core.utils import load_json
    from hierarchical_world_model.src.npc_experiments.search import (
        consolidate_road_results,
    )

    keys = ["01_1", "02_2"]
    for index, key in enumerate(keys):
        np.savez_compressed(tmp_path / f"{key}.npz", model_speed=np.array([index, 3.5]))
        save_json(
            {
                "runs": [
                    {"repeat": 0, "status": "failed", "failure": "recorded failure"}
                ]
            },
            tmp_path / f"{key}.json",
        )
        pd.DataFrame({"source": ["model"], "duration": [3.456789012345678]}).to_csv(
            tmp_path / f"{key}_events.csv", index=False
        )
    consolidate_road_results(tmp_path, keys)
    assert {p.name for p in tmp_path.iterdir()} == {
        "runs.json",
        "statistics.npz",
        "events.csv",
    }
    with np.load(tmp_path / "statistics.npz") as saved:
        for index, key in enumerate(keys):
            np.testing.assert_array_equal(saved[key + "_model_speed"], [index, 3.5])
    assert (
        load_json(tmp_path / "runs.json")[keys[0]]["runs"][0]["failure"]
        == "recorded failure"
    )
    events = pd.read_csv(tmp_path / "events.csv", float_precision="round_trip")
    assert events.cohort.tolist() == keys
    assert events.duration.tolist() == [3.456789012345678] * 2


def test_missing_road_shard_does_not_delete_available_results(tmp_path):
    import pytest
    from hierarchical_world_model.src.npc_experiments.search import (
        consolidate_road_results,
    )

    save_json({"runs": []}, tmp_path / "01_1.json")
    with pytest.raises(FileNotFoundError):
        consolidate_road_results(tmp_path, ["01_1"])
    assert (tmp_path / "01_1.json").exists()
    assert not (tmp_path / "runs.json").exists()
