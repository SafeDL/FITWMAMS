"""Test evidence follows physical scene identities and the current upstream."""

import numpy as np
import pandas as pd
import pytest
from hierarchical_world_model.src.core.utils import load_json, save_json

from hierarchical_world_model.scripts.verify import (
    _recording_ids,
    test_scene_identity_matches as scene_matches,
    report_matches_selection,
)


def test_recordings_are_read_from_physical_scene_identity():
    assert _recording_ids(["nat_04_a", "nat_49_b", "nat_04_c"]) == {4, 49}
    with pytest.raises(ValueError):
        _recording_ids(["unknown_04_a"])


def test_all_test_scene_identities_survive_record_reordering():
    rows = pd.DataFrame(
        {"sequence_id": ["nat_04_a", "nat_49_b"], "recording_id": [4, 49]}
    )
    assert scene_matches(rows.iloc[::-1], np.array(["nat_04_a", "nat_49_b"]))
    assert not scene_matches(rows.assign(recording_id=[49, 4]), rows.sequence_id)
    assert not scene_matches(pd.concat([rows, rows.iloc[[0]]]), rows.sequence_id)


def test_old_upstream_report_cannot_pass_component_matching():
    selection = {
        "response_checkpoint": "/response.pt",
        "response_checkpoint_mtime_ns": 1,
        "diffusion": {"path": "/diffusion.pt", "mtime_ns": 2},
        "lane_trajectory_model_mtime_ns": 3,
        "mobil": {"bias": -4},
    }
    report = {
        **selection,
        "mobil_parameters": selection["mobil"],
        "test_used_for_selection": False,
    }
    assert report_matches_selection(report, selection)
    assert not report_matches_selection(
        {**report, "diffusion": {"path": "/diffusion.pt", "mtime_ns": 1}}, selection
    )


def test_failed_report_serialization_preserves_the_last_complete_report(tmp_path):
    path = tmp_path / "report.json"
    save_json({"status": "complete"}, path)
    with pytest.raises(TypeError):
        save_json({"passed": np.bool_(True)}, path)
    assert load_json(path) == {"status": "complete"}
