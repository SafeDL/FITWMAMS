"""Reference successes retain their executed behavior instead of faster experts."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from ads.common.instances import UsageLedger, file_hash
from ads.common.randomness import PolicyRandomContext
from ads.diffusion_planner import supervision


def setup_campaign(tmp_path, instance, monkeypatch, *, failure=False, stalled=False):
    base = tmp_path / "policy.pth"
    base.write_bytes(b"fixed deployment")
    directory = tmp_path / "diffusion_planner/20260926/guided/round_1"
    directory.mkdir(parents=True)
    (directory / "search_summary.json").write_text(
        json.dumps({"policy_sha256": file_hash(base)})
    )
    (directory / "search.jsonl").write_text("")
    campaign = SimpleNamespace(
        output=tmp_path,
        factory=None,
        evt=None,
        ledger=UsageLedger(tmp_path / "usage.sqlite"),
        config={
            "master_seed": 20260926,
            "repair": {"plant_normal_teacher_instances": 32},
            "teacher": {"candidate_budget": 128},
        },
    )
    calls = {"reference": 0, "expert": 0}
    states = np.tile(instance.event.initial_states, (150, 1, 1))
    states[:, 0, 0] += np.arange(150) * 0.04 * 10

    class Reference:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, value):
            calls["reference"] += 1
            return SimpleNamespace(
                failure=failure,
                record={
                    "instance_hash": value.instance_hash,
                    "policy_stream_id": PolicyRandomContext(
                        "diffusion_planner", 20260926, "search", value.instance_hash
                    ).identity,
                    "safety_failure": failure,
                    "stalled": stalled,
                },
                states=states,
                controls=np.zeros((149, 2), np.float32),
            )

    class Expert:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, value):
            calls["expert"] += 1
            trace = tmp_path / "expert.npz"
            np.savez_compressed(trace, states=states * 2)
            return [{"origin": "expert"}], {
                "instance_hash": value.instance_hash,
                "cache_path": str(trace.with_suffix(".json")),
                "trajectory_sha256": file_hash(trace),
                "qualified_demonstration": True,
                "new_candidates": 64,
                "new_elapsed_s": 1.0,
            }

    monkeypatch.setattr(supervision, "WorldEvaluator", Reference)
    monkeypatch.setattr(supervision, "DemonstrationCache", Expert)
    monkeypatch.setattr(
        supervision,
        "demonstration",
        lambda value, executed, algorithm: [
            {"origin": "reference", "states": executed}
        ],
    )
    return campaign, directory, base, calls, states


def test_successful_prior_keeps_its_executed_behavior(tmp_path, instance, monkeypatch):
    c, directory, base, calls, states = setup_campaign(tmp_path, instance, monkeypatch)
    material, reports = supervision.prepare_materials(
        c, object(), base, directory, {"normal": [(instance, "prior")]}, [], [], 0
    )
    assert calls == {"reference": 1, "expert": 0}
    np.testing.assert_array_equal(material["normal"][0]["states"], states)
    assert reports[0]["label_source"] == "policy_success"
    assert material["normal"][0]["label_source"] == "policy_success"
    assert material["normal"][0]["scene_id"] == instance.root_sample_id
    supervision.prepare_materials(
        c, object(), base, directory, {"normal": [(instance, "prior")]}, [], [], 0
    )
    assert calls == {"reference": 1, "expert": 0}


def test_shared_reference_cache_executes_once_under_concurrent_requests(
    tmp_path, instance, monkeypatch
):
    c, _, base, calls, _ = setup_campaign(tmp_path, instance, monkeypatch)
    original = supervision.WorldEvaluator.__call__

    def delayed(reference, value):
        time.sleep(0.05)
        return original(reference, value)

    monkeypatch.setattr(supervision.WorldEvaluator, "__call__", delayed)
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(
            workers.map(
                lambda _: supervision.reference_execution(
                    c, object(), instance, file_hash(base)
                ),
                range(2),
            )
        )
    assert sum(result[1] for result in results) == 1
    assert calls == {"reference": 1, "expert": 0}


@pytest.mark.parametrize("failure,stalled", [(True, False), (False, True)])
def test_failed_or_stalled_prior_requests_corrective_expert(
    tmp_path, instance, monkeypatch, failure, stalled
):
    c, directory, base, calls, _ = setup_campaign(
        tmp_path, instance, monkeypatch, failure=failure, stalled=stalled
    )
    material, reports = supervision.prepare_materials(
        c, object(), base, directory, {"normal": [(instance, "prior")]}, [], [], 0
    )
    assert calls == {"reference": 1, "expert": 1}
    assert material["normal"][0]["origin"] == "expert"
    assert reports[0]["label_source"] == "expert"
    assert material["normal"][0]["label_source"] == "expert"


def test_executed_search_failure_reuses_its_label_without_another_replay(
    tmp_path, instance, monkeypatch
):
    c, directory, base, calls, _ = setup_campaign(tmp_path, instance, monkeypatch)
    (directory / "search.jsonl").write_text(
        json.dumps(
            {
                "instance_hash": instance.instance_hash,
                "execution_kind": "conditional_proposal",
                "safety_failure": True,
                "stalled": False,
            }
        )
        + "\n"
    )
    material, reports = supervision.prepare_materials(
        c, object(), base, directory, {"normal": []}, [(instance, "branch")], [], 0
    )
    assert calls == {"reference": 0, "expert": 1}
    assert material["new_failure"][0]["origin"] == "expert"
    assert reports[0]["reference_new_executions"] == 0


def test_history_reuses_previously_qualified_policy_trajectory(
    tmp_path, instance, monkeypatch
):
    c, first, base, calls, states = setup_campaign(tmp_path, instance, monkeypatch)
    _, reports = supervision.prepare_materials(
        c, object(), base, first, {"normal": [(instance, "prior")]}, [], [], 0
    )
    (first / "training.json").write_text(json.dumps({"teacher": {"reports": reports}}))
    second = first.parent / "round_2"
    second.mkdir()
    for name in ("search_summary.json", "search.jsonl"):
        (second / name).write_bytes((first / name).read_bytes())
    material, reports = supervision.prepare_materials(
        c, object(), base, second, {"normal": []}, [], [(instance, "history")], 1
    )
    assert calls == {"reference": 1, "expert": 0}
    np.testing.assert_array_equal(material["history"][0]["states"], states)
    assert reports[0]["historical_reuse"]


@pytest.mark.parametrize("purpose", ["development", "final_test"])
def test_evaluation_instances_cannot_supply_selective_supervision(
    tmp_path, instance, monkeypatch, purpose
):
    c, directory, base, _, _ = setup_campaign(tmp_path, instance, monkeypatch)
    with pytest.raises(ValueError, match="Evaluation instances"):
        supervision.prepare_materials(
            c,
            object(),
            base,
            directory,
            {"normal": [(replace(instance, purpose=purpose), "prior")]},
            [],
            [],
            0,
        )
