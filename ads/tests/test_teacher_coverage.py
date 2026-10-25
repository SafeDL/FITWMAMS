import json

import pytest

from ads.common.instances import file_hash
from ads.common.teacher_coverage import coverage_for_training, teacher_coverage


def fixture_directory(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    curriculum = {
        "categories": {
            "normal": [{"instance_hash": "prior"}],
            "new_failure": [{"instance_hash": "defect"}],
        },
        "success_neighbours": {"defect": {"instance_hash": "neighbour"}},
        "teacher_history": [{"instance_hash": "old"}],
    }
    (inputs / "curriculum.json").write_text(json.dumps(curriculum))
    reports = [
        {"instance_hash": "prior", "qualified_demonstration": False},
        {"instance_hash": "defect", "qualified_demonstration": True, "samples": 20},
        {"instance_hash": "neighbour", "qualified_demonstration": True, "samples": 20},
        {"instance_hash": "old", "qualified_demonstration": True, "samples": 0},
    ]
    metadata = {
        "baseline": False,
        "curriculum_sha256": file_hash(inputs / "curriculum.json"),
        "teacher": {"reports": reports},
    }
    (tmp_path / "training.json").write_text(json.dumps(metadata))
    (tmp_path / "teacher_selection.json").write_text(
        json.dumps([{"instance_hash": "defect"}, {"instance_hash": "neighbour"}])
    )
    outcomes = [
        {"instance_hash": "defect", "safety_failure": True},
        {"instance_hash": "neighbour", "safety_failure": False},
    ]
    (tmp_path / "search.jsonl").write_text(
        "\n".join(json.dumps(row) for row in outcomes) + "\n"
    )
    return {
        "repair": {"plant_normal_teacher_instances": 1},
    }


def coverage(directory, config):
    rows = coverage_for_training(
        directory, config, seed=7, method="guided", round_index=2
    )
    return {row["role"]: row for row in rows}


def test_true_defect_coverage_separates_normal_neighbour_and_history(tmp_path):
    config = fixture_directory(tmp_path)
    result = coverage(tmp_path, config)
    assert result["normal_prior"]["qualified_fraction"] == 0
    assert result["new_failure"]["selected_instances"] == 1
    assert result["new_failure"]["qualified_fraction"] == 1
    assert result["success_neighbour"]["qualified_instances"] == 1
    assert result["new_failure"]["available_label_frames"] == 20
    assert result["history"]["qualified_instances"] == 1
    assert result["history"]["instances_with_labels"] == 0
    assert result["history"]["available_label_frames"] == 0


@pytest.mark.parametrize(
    "outcomes", [[], [{"instance_hash": "neighbour", "safety_failure": True}]]
)
def test_neighbour_requires_executed_success_under_search_policy(tmp_path, outcomes):
    config = fixture_directory(tmp_path)
    rows = [{"instance_hash": "defect", "safety_failure": True}] + outcomes
    (tmp_path / "search.jsonl").write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match="executed search outcome"):
        coverage(tmp_path, config)


def test_teacher_report_order_cannot_change_role_denominators(tmp_path):
    config = fixture_directory(tmp_path)
    path = tmp_path / "training.json"
    metadata = json.loads(path.read_text())
    metadata["teacher"]["reports"].reverse()
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="ordered training selections"):
        coverage(tmp_path, config)


def test_unsuccessful_teacher_cannot_contribute_positive_labels(tmp_path):
    config = fixture_directory(tmp_path)
    path = tmp_path / "training.json"
    metadata = json.loads(path.read_text())
    metadata["teacher"]["reports"][0]["samples"] = 20
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="positive supervision"):
        coverage(tmp_path, config)


def test_baseline_has_no_search_or_defect_denominator(tmp_path):
    (tmp_path / "training.json").write_text(json.dumps({"baseline": True}))
    assert (
        coverage_for_training(
            tmp_path, {}, seed=20260926, method="initial", round_index=0
        )
        == []
    )


def test_other_ads_do_not_require_expert_records(tmp_path):
    assert teacher_coverage(tmp_path, {"algorithms": ["dqn"]}) == []
