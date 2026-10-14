"""Publish Train/Val weights before Test; ADS evidence is not a release gate."""

import json
from pathlib import Path

import numpy as np
import pytest

from hierarchical_world_model.scripts import publish
from hierarchical_world_model.src import composition
from hierarchical_world_model.src.core.utils import save_json


@pytest.fixture
def publication(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoints/stochastic_heads_best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"HiQR")
    diffusion = tmp_path / "diffusion.pt"
    diffusion.write_bytes(b"current 68D diffusion")
    lane = tmp_path / "lane.json"
    save_json({"test_used_for_parameter_selection": False}, lane)
    population = {"log_mean": np.zeros(5), "log_cov": np.eye(5)}
    following = tmp_path / "following.npz"
    np.savez(following, **population)
    training = {"test_used_for_fit_or_selection": False}
    fit = {
        "fit_split": "train",
        "selection_split": "val",
        "test_used_for_fit_or_selection": False,
        "mobil": {"bias": -4.0, "politeness": 0.0, "slope": 0.02},
        "source": str(tmp_path / "shared_data/driver_behavior"),
        "data_protocol": "highd_raw_drivers",
        "raw_data_identity": {
            "path": "raw/manifest.json",
            "mtime_ns": 1,
            "size_bytes": 1,
        },
        "label_protocol": {
            "confirmation": "actual adjacent crossing in the same motion episode"
        },
        "selection_report": str(tmp_path / "npc_response/search/selection.json"),
    }
    summary_path = tmp_path / "training_stochastic_heads_summary.json"
    save_json({"status": "complete"}, summary_path)
    selection = tmp_path / "selection.json"
    monkeypatch.setattr(publish, "MODEL_PATH", following)
    monkeypatch.setattr(publish, "MOBIL_PATH", tmp_path / "mobil.json")
    monkeypatch.setattr(publish, "LANE_TRAJECTORY_PATH", lane)
    monkeypatch.setattr(publish, "DEFAULT_WORLD_SELECTION", selection)
    monkeypatch.setattr(
        publish,
        "load_protocol_config",
        lambda *a: {
            "paths": {
                "output_dir": str(tmp_path),
                "evaluation_checkpoint": str(checkpoint),
                "diffusion_checkpoint": str(diffusion),
            }
        },
    )
    monkeypatch.setattr(publish, "prepare_experiment_data", lambda *a: object())
    monkeypatch.setattr(
        publish, "load_shared_components", lambda *a: (None, training, population, fit)
    )
    return selection, summary_path, lane, training


def test_publication_does_not_require_test_or_generated_results(publication):
    selection, _, _, _ = publication
    publish.main()
    result = json.loads(selection.read_text())
    assert result["selection_split"] == "val"
    assert not result["test_used_for_selection"]
    assert result["selected_arm"] == "rules_recovery"
    assert result["evidence"]["npc_calibration"] == str(
        selection.parent / "npc_response/search/selection.json"
    )
    assert "bounded" in result["selection_reason"]
    assert "sha256" not in str(result)
    assert not (selection.parent / "test").exists()


@pytest.mark.parametrize("change", ["incomplete", "test", "lane"])
def test_publication_refuses_incomplete_or_test_selected_weights(publication, change):
    selection, summary, lane, training = publication
    if change == "incomplete":
        save_json({"status": "running"}, summary)
    elif change == "test":
        training["test_used_for_fit_or_selection"] = True
    else:
        save_json({"test_used_for_parameter_selection": True}, lane)
    with pytest.raises(ValueError):
        publish.main()
    assert not selection.exists()


@pytest.mark.parametrize("change", [None, "checkpoint", "diffusion", "lane", "driver"])
def test_runtime_rejects_changed_published_components(publication, monkeypatch, change):
    selection, _, lane, _ = publication
    publish.main()
    manifest = json.loads(selection.read_text())
    checkpoint = Path(manifest["response_checkpoint"])
    monkeypatch.setattr(composition, "DEFAULT_WORLD_SELECTION", selection)
    monkeypatch.setattr(
        composition, "DRIVER_MODEL_PATH", selection.parent / "following.npz"
    )
    monkeypatch.setattr(composition, "LANE_MODEL_PATH", lane)
    if change == "checkpoint":
        manifest["response_checkpoint_mtime_ns"] -= 1
    elif change == "diffusion":
        manifest["diffusion"]["mtime_ns"] -= 1
    elif change == "lane":
        manifest["lane_trajectory_model_mtime_ns"] -= 1
    elif change == "driver":
        manifest["driver_population"]["log_mean"][0] += 1
    save_json(manifest, selection)
    if change:
        with pytest.raises(ValueError, match="published shared chain"):
            composition.load_selected_components(checkpoint)
    else:
        current, population = composition.load_selected_components(checkpoint)
        assert current["reference_recovery_enabled"]
        np.testing.assert_array_equal(population["log_mean"], np.zeros(5))
