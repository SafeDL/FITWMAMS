"""Paired admission preserves previous skills for both repair learning routes."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ads.common import admission as module
from ads.common.instances import UsageLedger, file_hash
from ads.common.reporting import write_json
from ads.search.archive import FailureArchive
from ads.tests.test_search import evaluate


def record(name, collision=False, progress=100.0):
    return {
        "event_id": name,
        "instance_hash": name,
        "root_sample_id": name,
        "assets_hash": "fixed-world",
        "policy_stream_id": name + "-stream",
        "collision": collision,
        "safety_failure": collision,
        "offroad": False,
        "stalled": False,
        "progress_m": progress,
    }


@pytest.mark.parametrize("algorithm", ["carl", "diffusion_planner"])
def test_admission_rejects_natural_regression_and_retains_original_cache(
    tmp_path, monkeypatch, algorithm
):
    policies = {}
    for name in ("official", "regression", "improved"):
        policies[name] = tmp_path / (name + ".pth")
        policies[name].write_text(name)
    root = tmp_path / "materials"
    development = {
        name: [Path(name + "-a.npz"), Path(name + "-b.npz")]
        for name in ("tail_generated", "natural_generated")
    }
    for name, paths in development.items():
        for index, path in enumerate(paths):
            write_json(
                root / "initial" / name / (path.stem + ".json"),
                {
                    "policy_sha256": file_hash(policies["official"]),
                    "record": record(
                        path.stem, collision=name == "tail_generated" and index == 0
                    ),
                },
            )
    calls = []

    def execute(paths, policy, directory, settings, **options):
        assert options["algorithm"] == algorithm
        assert policy != policies["official"]
        calls.append(policy)
        natural = Path(directory).name == "natural_generated"
        return [
            {
                "policy_sha256": file_hash(policy),
                "record": record(
                    path.stem,
                    collision=natural and policy == policies["regression"] and i == 0,
                    progress=99.0,
                ),
            }
            for i, path in enumerate(paths)
        ]

    monkeypatch.setattr(module, "evaluate_paths", execute)
    campaign = SimpleNamespace(
        config={"capability_preserving_repair": {"minimum_progress_ratio": 0.98}},
        ledger=UsageLedger(tmp_path / "usage.sqlite"),
    )
    directory = tmp_path / "round_1"
    FailureArchive().save(directory / "archive")
    selected, report = module.admission(
        [(500, policies["regression"]), (1000, policies["improved"])],
        policies["official"],
        campaign,
        directory,
        algorithm=algorithm,
        official=policies["official"],
        material_root=root,
        development=development,
        training_instances=[],
    )
    assert selected == policies["improved"] and report["accepted_update"]
    assert not report["candidates"][0]["highway_passed"]
    assert report["candidates"][1]["highway_passed"]
    field = "decision_steps" if algorithm == "carl" else "updates"
    assert [row[field] for row in report["candidates"]] == [500, 1000]
    assert len(calls) == 4
    assert json.loads((directory / "admission.json").read_text()) == report
    with pytest.raises(ValueError, match="another deployment policy"):
        cache = root / "initial/tail_generated/tail_generated-a.json"
        data = json.loads(cache.read_text())
        data["policy_sha256"] = "another-policy"
        write_json(cache, data)
        module.development_results(
            policies["official"],
            campaign,
            algorithm=algorithm,
            official=policies["official"],
            material_root=root,
            development=development,
        )


@pytest.mark.parametrize("regress_neighbour", [False, True])
def test_unchanged_development_needs_archive_repair_without_neighbour_regression(
    source, tmp_path, monkeypatch, regress_neighbour
):
    old, success = [evaluate(instance) for instance in source.sample(2)]
    old.record.update(safety_failure=True, collision=True)
    success.record.update(safety_failure=False, collision=False)
    archive = FailureArchive()
    archive.domains["root"] = [old]
    archive.neighbours[old.instance.instance_hash] = success
    directory = tmp_path / "round_2"
    archive.save(directory / "archive")
    official, previous, candidate = [
        tmp_path / (name + ".pth") for name in ("official", "previous", "candidate")
    ]
    for path in (official, previous, candidate):
        path.write_text(path.stem)
    monkeypatch.setattr(
        module,
        "development_results",
        lambda *a, **k: {
            name: [record(name)] for name in ("tail_generated", "natural_generated")
        },
    )

    def execute(paths, policy, *args, **options):
        assert options["purpose"] == "search" and options["algorithm"] == "carl"
        return [
            {
                "record": record(
                    path.stem,
                    collision=(
                        path.stem == old.instance.instance_hash
                        if policy in (official, previous)
                        else regress_neighbour
                        and path.stem == success.instance.instance_hash
                    ),
                )
            }
            for path in paths
        ]

    monkeypatch.setattr(module, "evaluate_paths", execute)
    campaign = SimpleNamespace(
        config={"capability_preserving_repair": {"minimum_progress_ratio": 0.98}},
        ledger=UsageLedger(tmp_path / "usage.sqlite"),
    )
    selected, report = module.admission(
        [(8192, candidate)],
        previous,
        campaign,
        directory,
        algorithm="carl",
        official=official,
        material_root=tmp_path,
        development={},
        training_instances=[old.instance.instance_hash],
    )
    assert report["accepted_update"] == (not regress_neighbour)
    assert selected == (previous if regress_neighbour else candidate)
    endpoints = json.loads((directory / "admission_endpoints.json").read_text())
    assert endpoints[0]["instance_hash"] == success.instance.instance_hash
    assert not endpoints[0]["used_for_training"]
    assert endpoints[1]["used_for_training"]
