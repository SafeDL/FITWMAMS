"""An expert success must also survive the ADS's native trajectory tracker."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ads.common import expert
from ads.common.instances import UsageLedger, file_hash
from ads.diffusion_planner import expert_materials as module


def tracking_report(states, path, success):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, states=states, controls=np.zeros((len(states) - 1, 2)))
    return {
        "success": success,
        "physics_steps": len(states) - 1,
        "trajectory_path": str(path),
        "trajectory_sha256": file_hash(path),
        "event_risk": 0.1,
        "helper_sha256": file_hash(
            Path(module.__file__).with_name("label_tracking.py")
        ),
    }


def test_untrackable_positive_expert_is_not_a_learning_label(
    instance, tmp_path, monkeypatch
):
    path = tmp_path / "instance.npz"
    instance.save(path)
    monkeypatch.setattr(
        module,
        "_settings",
        {
            "historical_run": str(tmp_path / "no_history"),
            "repair": {"expert_candidates": 128},
        },
        raising=False,
    )
    monkeypatch.setattr(module, "_output", tmp_path, raising=False)
    monkeypatch.setattr(module, "_factory", SimpleNamespace(steps=0), raising=False)
    monkeypatch.setattr(
        module, "_ledger", UsageLedger(tmp_path / "usage.sqlite"), raising=False
    )
    states = np.tile(instance.event.initial_states, (150, 1, 1))
    states[:, :, :2] += np.arange(150)[:, None, None] * 0.04 * states[:, :, 2:4]

    def successful(value, factory, budget):
        return expert.TeacherExecution(
            expert.Candidate(0, 20),
            True,
            1,
            100,
            states,
            np.zeros((149, 2), np.float32),
        ), {
            "instance_hash": value.instance_hash,
            "candidates": 1,
            "qualified_demonstration": True,
        }

    monkeypatch.setattr(expert, "successful_expert", successful)
    monkeypatch.setattr(
        module,
        "check_labels",
        lambda *args, **kwargs: tracking_report(states, kwargs["trace_path"], False),
    )
    report = module.execute((instance.instance_hash, str(path), instance.assets))
    assert report["expert_success"]
    assert not report["qualified_demonstration"]
    assert report["samples"] == 0
    assert report["new_candidates"] == 1
    assert report["label_tracking_new_executions"] == 1


@pytest.mark.parametrize(
    "expert_success,tracking_success", [(True, True), (True, False), (False, False)]
)
def test_initial_expert_budget_includes_tracking_and_resume(
    instance, tmp_path, monkeypatch, expert_success, tracking_success
):
    path = tmp_path / "instance.npz"
    instance.save(path)
    monkeypatch.setattr(
        module,
        "_settings",
        {
            "historical_run": str(tmp_path / "no_history"),
            "repair": {"expert_candidates": 128},
        },
        raising=False,
    )
    monkeypatch.setattr(module, "_output", tmp_path, raising=False)
    monkeypatch.setattr(module, "_factory", SimpleNamespace(steps=0), raising=False)
    monkeypatch.setattr(
        module, "_ledger", UsageLedger(tmp_path / "usage.sqlite"), raising=False
    )
    lane = instance.event.geometry.nearest_lane(instance.event.initial_states[0, :2])
    monkeypatch.setattr(
        expert,
        "coarse_candidates",
        lambda event: [expert.Candidate(lane, 10 + i / 10) for i in range(64)],
    )
    monkeypatch.setattr(
        expert,
        "refine_candidates",
        lambda values: [expert.Candidate(lane, 20 + i / 10) for i in range(64)],
    )
    calls = []
    tracking_calls = []
    states = np.tile(instance.event.initial_states, (150, 1, 1))
    states[:, :, :2] += np.arange(150)[:, None, None] * 0.04 * states[:, :, 2:4]

    def execute(value, factory, candidate):
        calls.append(candidate)
        return expert.TeacherExecution(
            candidate, expert_success, 1, 100, states, np.zeros((149, 2), np.float32)
        )

    def check(*args, **kwargs):
        tracking_calls.append(args[0].instance_hash)
        return tracking_report(states, kwargs["trace_path"], tracking_success)

    monkeypatch.setattr(expert, "execute_candidate", execute)
    monkeypatch.setattr(module, "check_labels", check)
    job = (instance.instance_hash, str(path), instance.assets)
    report = module.execute(job)
    assert report["candidate_budget"] == len(calls) == 127
    assert len(tracking_calls) == int(expert_success)
    assert report["new_world_executions"] == 127 + int(expert_success)
    assert report["candidate_and_tracking_executions"] <= 128
    assert report["expert_execution_budget"] == 128
    assert report["qualified_demonstration"] == (expert_success and tracking_success)
    if not report["qualified_demonstration"]:
        assert report["samples"] == 0
    resumed = module.execute(job)
    assert resumed["resumed"]
    assert (
        resumed["candidate_and_tracking_executions"]
        == report["candidate_and_tracking_executions"]
    )
    assert len(calls) == 127
    assert len(tracking_calls) == int(expert_success)
    cached = tmp_path / "experts/inputs" / (instance.instance_hash + ".json")
    previous = json.loads(cached.read_text())
    previous.pop("expert_protocol")
    cached.write_text(json.dumps(previous))
    with pytest.raises(ValueError, match="different execution protocol"):
        module.execute(job)
    assert len(calls) == 127
    assert len(tracking_calls) == int(expert_success)
