import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from hierarchical_world_model.scripts import evaluate as validation
from hierarchical_world_model.src import shared_evaluation as candidates
from hierarchical_world_model.src.core.utils import save_json
from hierarchical_world_model.src.stochastic_drivers.idm_population import sample_theta


def test_all_response_arms_use_the_same_shared_driver_draws():
    population = {
        "log_mean": np.log([33.0, 2.0, 1.2, 1.0, 2.0]),
        "log_cov": np.eye(5) * 0.02,
    }
    rng = np.random.default_rng(1)
    randomness = SimpleNamespace(
        driver_latent=rng.standard_normal((2, 6, 5)),
        maneuver_latent=rng.standard_normal((2, 6, 4)),
    )
    fit = {"mobil": {"politeness": 0.0, "bias": -4.0, "slope": 0.02}}
    for arm in ("rules", "rules_recovery"):
        controller = validation.make_controller(randomness, fit, population)
        np.testing.assert_array_equal(
            controller.theta.numpy(), sample_theta(population, randomness.driver_latent)
        )
        assert controller.response_gate is None


@pytest.mark.parametrize("split", ["val", "test"])
def test_full_split_figure_uses_recording_weighted_errors_and_retains_failed_rows(
    tmp_path, monkeypatch, split
):
    monkeypatch.setattr(validation, "OUTPUT", tmp_path)
    rows = [
        {
            "recording_id": 1,
            "ADE_m": 0.1,
            "distance_sum_m": 1.0,
            "actor_frames": 10,
            "final_distance_sum_m": 0.2,
            "final_actors": 2,
            "collision": False,
            "introduced_offroad": False,
            "npc_introduced_offroad": False,
            "ego_introduced_offroad": False,
        },
        {
            "recording_id": 1,
            "ADE_m": 0.3,
            "distance_sum_m": 9.0,
            "actor_frames": 30,
            "final_distance_sum_m": 0.8,
            "final_actors": 2,
            "collision": True,
            "introduced_offroad": True,
            "npc_introduced_offroad": False,
            "ego_introduced_offroad": True,
        },
    ]
    result = validation.plot_validation(
        {"hiqr": rows, "learned": rows}, split, tmp_path
    )
    for arm in result.values():
        assert arm["ADE_m"] == {"value": 0.25, "recording_bootstrap_95CI": [0.25, 0.25]}
        assert arm["FDE_m"]["value"] == 0.25
        assert arm["collision"]["rate"] == 0.5
        assert arm["introduced_offroad"]["rate"] == 0.5
        assert arm["npc_introduced_offroad"]["rate"] == 0
        assert arm["ego_introduced_offroad"]["rate"] == 0.5
    assert (tmp_path / "figures" / "trajectory_accuracy.png").stat().st_size > 1000
    assert not (tmp_path / "figures" / "trajectory_accuracy.pdf").exists()


@pytest.mark.parametrize(
    "split,selected_rows,directory",
    [
        ("val", [0, 1], "factual_validation"),
        ("test", [2, 3, 4], "factual_test"),
    ],
)
def test_full_world_entry_uses_all_requested_split_rows_and_separate_output(
    tmp_path, monkeypatch, split, selected_rows, directory
):
    states = np.zeros((5, 174, 7, 6), np.float32)
    states[..., 0] = np.arange(5)[:, None, None]
    valid = np.ones((5, 174, 7), bool)
    experiment = SimpleNamespace(
        validation_rows=np.array([0, 1]),
        test_rows=np.array([2, 3, 4]),
        bundle=SimpleNamespace(
            arrays={
                "agent_states": states,
                "agent_valid": valid,
                "sequence_id": np.asarray([f"scene_{i}" for i in range(5)]),
            }
        ),
    )
    config = {
        "paths": {
            "output_dir": str(tmp_path),
            "sequence_cache_dir": str(tmp_path),
            "diffusion_checkpoint": str(tmp_path / "diffusion.pt"),
        }
    }
    checkpoint = tmp_path / "checkpoints/stochastic_heads_best.pt"
    checkpoint.parent.mkdir()
    torch.save({}, checkpoint)
    (tmp_path / "diffusion.pt").write_bytes(b"diffusion")
    lane_path = tmp_path / "lane.json"
    save_json({}, lane_path)
    monkeypatch.setattr(validation, "LANE_TRAJECTORY_PATH", lane_path)
    monkeypatch.setattr(validation, "OUTPUT", tmp_path / "factual_validation")
    monkeypatch.setattr(validation, "BATCH_SIZE", 2)
    monkeypatch.setattr(validation, "load_protocol_config", lambda *a: config)
    monkeypatch.setattr(validation, "prepare_experiment_data", lambda *a: experiment)
    model = torch.nn.Linear(1, 1)
    population = {"log_mean": np.zeros(5), "log_cov": np.eye(5)}
    monkeypatch.setattr(
        validation,
        "load_shared_components",
        lambda *a: (model, {}, population, {"mobil": {}}),
    )
    monkeypatch.setattr(
        validation,
        "load_selected_components",
        lambda *a: ({"selected_arm": "rules_recovery", "mobil": {}}, population),
    )
    monkeypatch.setattr(validation, "scene_randomness", lambda *a: None)
    monkeypatch.setattr(validation, "make_controller", lambda *a: object())
    monkeypatch.setattr(
        validation,
        "event_metadata",
        lambda *a: pd.DataFrame({"recording_id": [1, 1, 2, 2, 3]}),
    )

    def plans(bundle, rows, **kwargs):
        assert rows.tolist() == selected_rows
        assert kwargs["output_dir"].name == split
        return np.zeros((len(rows), 149, 6, 2))

    executed = {arm: [] for arm in validation.ARMS}

    def rollout(model, arrays, rows, plans, randomness, *, controller, recovery):
        arm = "rules_recovery" if recovery else "rules"
        executed[arm].extend(rows.tolist())
        return None

    def records(rows, logged, observed, trace):
        np.testing.assert_array_equal(logged[:, 0, 0, 0], rows)
        return (
            [{"row": int(row)} for row in rows],
            np.zeros(len(rows)),
            np.zeros(len(rows)),
        )

    def summary(records, *a):
        return {
            "sequences": len(records),
            "finite": len(records),
            **dict.fromkeys(
                (
                    "ADE_m",
                    "FDE_m",
                    "P95_m",
                    "collision",
                    "new_offroad",
                    "introduced_offroad",
                    "committed_target_changes",
                ),
                0,
            ),
        }

    def plot(records, received_split, output):
        assert received_split == split and output.name == directory
        assert all(
            [row["row"] for row in rows] == selected_rows for rows in records.values()
        )
        return {"synthetic": True}

    monkeypatch.setattr(validation, "frozen_diffusion_plans", plans)
    monkeypatch.setattr(validation, "highway_rollout", rollout)
    monkeypatch.setattr(validation, "factual_records", records)
    monkeypatch.setattr(validation, "summarize_records", summary)
    monkeypatch.setattr(validation, "plot_validation", plot)
    validation.main(split)
    report = json.loads((tmp_path / directory / "report.json").read_text())
    assert report["split"] == split
    assert report["fit_split"] == "train"
    assert report["selection_split"] == "val"
    assert report["sequences_completed"] == len(selected_rows)
    assert not report["test_used_for_selection"]
    expected_arms = validation.ARMS if split == "val" else ("rules_recovery",)
    assert tuple(report["arms"]) == expected_arms
    assert all(executed[arm] == selected_rows for arm in expected_arms)
    if split == "test":
        assert not executed["rules"]
        assert "factual_gates" not in report
        with np.load(tmp_path / directory / "error_samples.npz") as samples:
            assert len(samples["errors"]) == len(selected_rows)
            assert samples["error_offsets"].tolist() == [0, 2, 3]
    for arm in expected_arms:
        filename = "scene_metrics.csv" if split == "test" else f"{arm}_records.csv"
        saved = pd.read_csv(tmp_path / directory / filename)
        assert saved.row.tolist() == selected_rows


@pytest.fixture
def shared_chain(tmp_path, monkeypatch):
    cache = tmp_path / "cache/sequence_cache"
    diffusion = tmp_path / "diffusion.pt"
    diffusion.write_bytes(b"diffusion")
    config = {
        "paths": {
            "output_dir": str(tmp_path),
            "sequence_cache_dir": str(cache.parent),
            "diffusion_checkpoint": str(diffusion),
        },
        "npc_response": {"output_dir": str(tmp_path / "npc_response")},
    }
    experiment = SimpleNamespace(
        train_rows=np.array([1, 3]),
        validation_rows=np.array([4]),
        bundle=SimpleNamespace(
            arrays={"sequence_id": np.asarray([f"scene_{i}" for i in range(5)])}
        ),
    )
    training = {
        "train_sequence_ids": ["scene_1", "scene_3"],
        "validation_sequence_ids": ["scene_4"],
        "diffusion": candidates.artifact_identity(diffusion),
        "test_used_for_fit_or_selection": False,
        "history_selection": "last_observed_frame_index",
        "history_controls": "valid_observed_transitions",
    }
    population = {"log_mean": np.log([33, 2, 1.2, 1, 2]), "log_cov": np.eye(5) * 0.02}
    model_path = tmp_path / "following.npz"
    np.savez(model_path, **population)
    raw_manifest = tmp_path / "shared_data/driver_behavior/manifest.json"
    save_json({"label_protocol": candidates.LABEL_PROTOCOL}, raw_manifest)
    fit = {
        "driver_parameter_source": str(model_path),
        "driver_population": {key: value.tolist() for key, value in population.items()},
        "fit_split": "train",
        "selection_split": "val",
        "source": str(raw_manifest.parent),
        "data_protocol": "highd_raw_drivers",
        "label_protocol": candidates.LABEL_PROTOCOL,
        "raw_data_identity": candidates.artifact_identity(raw_manifest),
        "test_used_for_fit_or_selection": False,
        "mobil": {"politeness": 0.0, "bias": -4.0, "slope": 0.02},
    }
    mobil_path = tmp_path / "mobil.json"
    save_json(fit, mobil_path)
    model = object()
    monkeypatch.setattr(candidates, "MODEL_PATH", model_path)
    monkeypatch.setattr(candidates, "DRIVER_DATA_DIR", raw_manifest.parent)
    monkeypatch.setattr(candidates, "MOBIL_PATH", mobil_path)
    monkeypatch.setattr(
        candidates,
        "load_checkpoint",
        lambda *a, **kw: (model, {"training_data": training}),
    )
    return config, experiment, fit, mobil_path, model


def test_rule_components_do_not_load_a_selector(shared_chain, monkeypatch):
    config, experiment, fit, _, model = shared_chain

    def forbidden_selector_load(*args, **kwargs):
        pytest.fail("rule response must not load a learned selector")

    monkeypatch.setattr(candidates.torch, "load", forbidden_selector_load)
    loaded, training, population, decision = candidates.load_shared_components(
        config, experiment, "cpu"
    )
    assert loaded is model
    assert training["train_sequence_ids"] == ["scene_1", "scene_3"]
    assert decision == fit
    assert population["log_mean"].shape == (5,)


@pytest.mark.parametrize(
    "change", ["driver", "source", "test", "protocol", "labels", "raw_data"]
)
def test_rule_components_reject_mixed_calibration(shared_chain, change):
    config, experiment, fit, mobil_path, _ = shared_chain
    if change == "driver":
        fit["driver_population"]["log_mean"] = [0.0] * 5
    elif change == "source":
        fit["source"] = str(mobil_path.parent / "different_cache")
    elif change == "test":
        fit["test_used_for_fit_or_selection"] = True
    elif change == "protocol":
        fit["data_protocol"] = "old_short_windows"
    elif change == "labels":
        fit["label_protocol"] = {"target": "lateral_motion"}
    else:
        fit["raw_data_identity"]["mtime_ns"] -= 1
    save_json(fit, mobil_path)
    with pytest.raises(ValueError, match="raw-highD Train/Val calibration"):
        candidates.load_shared_components(config, experiment, "cpu")
