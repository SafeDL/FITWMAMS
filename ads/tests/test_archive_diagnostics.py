import json
from types import SimpleNamespace

import pytest

from ads.common.archive_diagnostics import (
    archive_role_outcomes,
    collect_archive_records,
)
from ads.common.instances import json_hash


def record(instance, failure=False):
    return {
        "event_id": instance.instance_hash,
        "instance_hash": instance.instance_hash,
        "root_sample_id": instance.root_sample_id,
        "assets_hash": json_hash(instance.assets),
        "purpose": "search",
        "safety_failure": failure,
        "elapsed_s": 1.0,
    }


def test_diagnostic_reuses_only_the_exact_pre_or_post_policy(source, tmp_path):
    instances = source.sample(2)
    journal = tmp_path / "search.jsonl"
    journal.write_text(json.dumps(record(instances[0], True)) + "\n")
    calls = []

    def evaluate(missing):
        calls.extend(instance.instance_hash for instance in missing)
        return [record(instance) for instance in missing]

    options = {
        "protocol_hash": "protocol",
        "journals": [(journal, "pre")],
        "caches": [],
        "evaluate": evaluate,
    }
    before = collect_archive_records(
        instances, tmp_path / "before.json", policy_sha256="pre", **options
    )
    assert calls == [instances[1].instance_hash]
    assert before["cost"]["reused_search_records"] == 1
    assert before["cost"]["new_executions"] == 1
    calls.clear()
    after = collect_archive_records(
        instances, tmp_path / "after.json", policy_sha256="post", **options
    )
    assert set(calls) == {instance.instance_hash for instance in instances}
    assert after["cost"]["reused_search_records"] == 0
    calls.clear()
    collect_archive_records(
        instances, tmp_path / "after.json", policy_sha256="post", **options
    )
    assert not calls
    with pytest.raises(ValueError, match="cached archive evaluation"):
        collect_archive_records(
            instances, tmp_path / "after.json", policy_sha256="changed", **options
        )


def test_diagnostic_rejects_changed_external_random_ancestry(instance, tmp_path):
    row = record(instance)
    row["root_sample_id"] = "different"
    journal = tmp_path / "search.jsonl"
    journal.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="full generated instance"):
        collect_archive_records(
            [instance],
            tmp_path / "result.json",
            policy_sha256="pre",
            protocol_hash="p",
            journals=[(journal, "pre")],
            caches=[],
            evaluate=lambda missing: [],
        )


def test_round_diagnostic_does_not_treat_old_labels_as_current_failures(source):
    failure, old_failure, neighbour = source.sample(3)
    archive = SimpleNamespace(
        domains={
            "front_tracking": [
                SimpleNamespace(instance=failure),
                SimpleNamespace(instance=old_failure),
            ]
        },
        neighbours={failure.instance_hash: SimpleNamespace(instance=neighbour)},
    )
    before = [
        record(failure, True),
        record(old_failure, False),
        record(neighbour, False),
    ]
    after = [
        record(failure, False),
        record(old_failure, False),
        record(neighbour, True),
    ]
    result = archive_role_outcomes(archive, before, after)["front_tracking"]
    assert result["pre_repair_failures"] == result["repaired_failures"] == 1
    assert result["already_safe_representatives"] == 1
    assert (
        result["pre_repair_success_neighbours"]
        == result["regressed_success_neighbours"]
        == 1
    )
