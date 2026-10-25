import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ads.common.curriculum import build_repair_banks, persist_curriculum
from ads.common.diagnostics import capture_failure
from ads.common.evaluation import EpisodeResult
from ads.common.instances import GeneratedInstance, UsageLedger


def result(instance, index, *, failure=True, kind="prior"):
    return EpisodeResult(
        instance,
        {
            "safety_failure": failure,
            "event_risk": float(1000 - index),
            "branch_id": "root",
            "descriptor": [index / 40, 0, 0, 0, 0],
            "execution_kind": kind,
        },
        np.empty((0, 7, 6)),
        np.empty((0, 2)),
        [],
    )


def test_single_search_branch_supplies_full_failure_selection(source):
    normal, *cases = source.sample(41)
    records = [result(instance, index) for index, instance in enumerate(cases)]
    known = {r.instance.instance_hash: r for r in records}
    banks = build_repair_banks("diffusion_planner", [normal], records, [], known)
    assert banks["normal"] == [(normal, "prior")]
    selected = [instance for instance, _ in banks["new_failure"]]
    assert len(selected) == 32
    assert len({item.root_sample_id for item in selected}) == 32
    assert {item.instance_hash for item in selected} <= set(known)


def test_learning_root_limit_precedes_candidate_capacity(source):
    cases = source.sample(40)
    cases[:31] = [
        replace(item, root_sample_id=cases[0].root_sample_id) for item in cases[:31]
    ]
    records = [result(instance, index) for index, instance in enumerate(cases)]
    known = {r.instance.instance_hash: r for r in records}
    banks = build_repair_banks("diffusion_planner", [], records, [], known)
    assert len(banks["new_failure"]) == 10
    assert len({item.root_sample_id for item, _ in banks["new_failure"]}) == 10


def test_repair_uses_executed_failures_and_success_edges_excluding_checks(source):
    failure, check, replay = source.sample(3)
    success = source.mutation(
        failure,
        failure.event.exogenous_state.mutate("driver_latent", beta=0.2, seed=10),
        "driver_latent",
    )
    records = [
        result(failure, 0),
        result(success, 1, failure=False),
        result(check, 2, kind="branch_check"),
        result(replay, 3, kind="archive_replay"),
    ]
    known = {r.instance.instance_hash: r for r in records}
    banks = build_repair_banks("diffusion_planner", [], records, [], known)
    assert [item.instance_hash for item, _ in banks["new_failure"]] == [
        failure.instance_hash
    ]
    assert (
        banks["success_neighbours"][failure.instance_hash][0].instance_hash
        == success.instance_hash
    )


def test_curriculum_preserves_exact_order_and_rejects_changes(source, tmp_path):
    first, second = source.sample(2)
    ledger = UsageLedger(tmp_path / "usage.sqlite")
    banks = {
        "normal": [(second, "prior"), (first, "prior")],
        "new_failure": [(first, "front")],
    }
    digest = persist_curriculum(tmp_path / "run", banks, ledger)
    assert persist_curriculum(tmp_path / "run", banks, ledger) == digest
    payload = json.loads((tmp_path / "run/inputs/curriculum.json").read_text())
    assert [r["instance_hash"] for r in payload["categories"]["normal"]] == [
        second.instance_hash,
        first.instance_hash,
    ]
    for instance in (first, second):
        restored = GeneratedInstance.load(
            tmp_path / "run/inputs/instances" / (instance.instance_hash + ".npz")
        )
        np.testing.assert_array_equal(
            restored.event.soft_reference, instance.event.soft_reference
        )
    with pytest.raises(ValueError, match="curriculum differs"):
        persist_curriculum(
            tmp_path / "run", {"normal": list(reversed(banks["normal"]))}, ledger
        )


def test_failure_capture_preserves_error_and_stack(tmp_path):
    try:
        raise ValueError("invalid rollout")
    except ValueError as error:
        directory = capture_failure(error, tmp_path)
    payload = json.loads((Path(directory) / "failure.json").read_text())
    assert payload["error"] == "invalid rollout"
    assert (
        payload["frames"][-1]["function"]
        == "test_failure_capture_preserves_error_and_stack"
    )
