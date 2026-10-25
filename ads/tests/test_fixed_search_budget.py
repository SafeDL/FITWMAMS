import json
from collections import Counter

import pytest
import yaml

from ads.common.instances import file_hash, json_hash
from ads.run_campaign import Scheduler
from ads.search.analytic import performance
from ads.search.archive import FailureArchive
from ads.search.engine import SearchConfig, SearchEngine
from ads.tasks import Task


def config(**overrides):
    return SearchConfig(
        **{
            "budget": 37,
            "fixed_budget": True,
            "initial_samples": 8,
            "fresh_samples": 4,
            "batch_size": 4,
            "generation_batch_size": 3,
            "graph_size": 4,
            "branch_check_budget": 8,
            "subset_max_levels": 1,
            "terminal_proposals": 0,
            "archive_capacity": 4,
            **overrides,
        }
    )


def failed(instance):
    result = performance(instance, "unbalanced")
    result.record.update(safety_failure=True, collision=True)
    return result


@pytest.mark.parametrize("method", ("monte_carlo", "subset", "guided"))
@pytest.mark.parametrize("budget", (1, 11, 37))
def test_each_method_spends_exact_budget_including_partial_batches(
    source, method, budget
):
    calls = []

    def evaluate(instance):
        calls.append(instance.instance_hash)
        return failed(instance)

    result = SearchEngine(source, evaluate, config=config(budget=budget)).run(method)
    summary = result.summary()
    assert len(calls) == len(result.records) == sum(result.costs.values()) == budget
    assert summary["evaluations"] == budget
    assert summary["unused_budget"] == 0
    assert summary["budget_mode"] == "fixed_budget"
    assert result.stop_reason == "execution_budget"
    assert result.costs.get("branch_check", 0) <= 8
    assert [r.record["evaluation_index"] for r in result.records] == list(
        range(1, budget + 1)
    )


@pytest.mark.parametrize("method", ("subset", "guided"))
def test_completed_failure_paths_restart_with_new_independent_samples(source, method):
    result = SearchEngine(source, failed, config=config(branch_check_budget=0)).run(
        method
    )
    assert result.summary()["search_restarts"] == 4
    assert result.costs["restart_prior"] == 25
    restarts = {
        key: value
        for key, value in result.branches.items()
        if key.startswith("restart_")
    }
    assert [len(node["seed_states"]) for node in restarts.values()] == [8, 8, 8, 1]
    assert all(
        node["status"] == "representative_target"
        for key, node in restarts.items()
        if key != "restart_3"
    )
    assert len({r.instance.instance_hash for r in result.records}) == 37
    assert len({r.instance.root_sample_id for r in result.records}) == 37
    assert all(r.record["independent_sampling_root"] for r in result.records)


@pytest.mark.parametrize("method", ("subset", "guided"))
def test_early_stop_mode_keeps_original_completion_behavior(source, method):
    result = SearchEngine(
        source, failed, config=config(fixed_budget=False, branch_check_budget=0)
    ).run(method)
    assert not SearchConfig().fixed_budget
    assert len(result.records) == 12 < result.budget
    assert result.stop_reason == "scheduled_tasks_complete"
    assert result.summary()["budget_mode"] == "early_stop"
    assert result.summary()["search_restarts"] == 0
    assert result.costs.get("restart_prior", 0) == 0


@pytest.mark.parametrize("method", ("subset", "guided"))
def test_depth_limits_restart_and_rejected_proposals_still_spend_budget(source, method):
    def safe(instance):
        result = performance(instance, "unbalanced")
        result.record.update(
            safety_failure=False,
            collision=False,
            event_risk=0.0 if instance.parent_id else 1.0,
        )
        return result

    result = SearchEngine(source, safe, config=config(branch_check_budget=0)).run(
        method
    )
    proposals = [
        r
        for r in result.records
        if r.record["execution_kind"] == "conditional_proposal"
    ]
    assert len(result.records) == sum(result.costs.values()) == 37
    assert proposals and all(not r.record["accepted"] for r in proposals)
    assert any(node["status"] == "depth_limit" for node in result.branches.values())
    assert result.summary()["search_restarts"] > 0
    assert not result.archive.failures


@pytest.mark.parametrize("method", ("monte_carlo", "subset"))
def test_baseline_diagnostics_do_not_generate_after_fixed_budget(
    source, method, monkeypatch
):
    def unexpected_mutation(*args):
        pytest.fail("exhausted baseline diagnostics must only inspect cached evidence")

    monkeypatch.setattr(source, "mutation", unexpected_mutation)
    result = SearchEngine(source, failed, config=config()).run(method)
    assert len(result.records) == 37
    assert result.costs.get("branch_check", 0) == 0
    assert result.branches["posthoc"]["evidence"]["status"] == "incomplete"
    assert {r.domain for r in result.archive.failures} == {"posthoc.0"}


@pytest.mark.parametrize("method", ("monte_carlo", "subset", "guided"))
def test_archive_replays_are_included_in_fixed_budget(source, method):
    archive = FailureArchive(capacity=4)
    old = [failed(instance) for instance in source.sample(4)]
    archive.add(old, {})
    result = SearchEngine(
        source, failed, config=config(branch_check_budget=0, replay_limit=3)
    ).run(method, archive=archive)
    assert len(result.records) == sum(result.costs.values()) == 37
    assert result.costs["archive_replay"] == 3
    assert Counter(r.record["execution_kind"] for r in result.records) == result.costs
    assert all(
        not r.record["independent_sampling_root"]
        for r in result.records
        if r.record["execution_kind"] == "archive_replay"
    )


@pytest.mark.parametrize("fixed_budget", (False, True))
@pytest.mark.parametrize("evaluations", (0, 11, 37, 38))
def test_scheduler_requires_exact_cost_only_for_fixed_budget(
    tmp_path, fixed_budget, evaluations
):
    settings = {
        "output_dir": str(tmp_path / "output"),
        "algorithms": ["dqn"],
        "seeds": [20260926],
        "methods": ["guided"],
        "rounds": 2,
        "repair": {"rl_steps": 8},
        "search": {"budget": 37, "fixed_budget": fixed_budget},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(settings))
    scheduler = Scheduler(path)
    protocol = {"settings": settings}
    (scheduler.output / "protocol.json").write_text(json.dumps(protocol))
    task = Task("repair", "dqn", 20260926, "guided", 1)
    directory = task.directory(scheduler.output)
    inputs = directory / "inputs/curriculum.json"
    inputs.parent.mkdir(parents=True)
    inputs.write_text("{}")
    checkpoint = directory / "policy.zip"
    checkpoint.write_bytes(b"test checkpoint")
    (directory / "training.json").write_text(
        json.dumps(
            {
                "protocol_hash": json_hash(protocol),
                "policy_sha256": file_hash(checkpoint),
                "curriculum_sha256": file_hash(inputs),
                "learning": {"actual_steps": 8},
            }
        )
    )
    (directory / "search_summary.json").write_text(
        json.dumps({"evaluations": evaluations})
    )
    archive = directory / "archive/archive.json"
    archive.parent.mkdir()
    archive.write_text("{}")
    if evaluations == 37 or (not fixed_budget and 0 < evaluations < 37):
        assert scheduler.training_done(task)
    else:
        with pytest.raises(ValueError, match="incomplete search budget"):
            scheduler.training_done(task)


def test_report_rejects_a_short_run_as_equal_budget(tmp_path):
    from ads.report import export_campaign

    settings = {
        "algorithms": ["carl"],
        "seeds": [20260926],
        "methods": ["monte_carlo"],
        "rounds": 1,
        "search": {"budget": 5000, "fixed_budget": True},
    }
    directory = tmp_path / "carl/20260926/monte_carlo/round_1"
    directory.mkdir(parents=True)
    row = {
        "instance_hash": "one-case",
        "root_sample_id": "one-root",
        "safety_failure": True,
        "collision": True,
    }
    (directory / "search.jsonl").write_text(json.dumps(row) + "\n")
    (directory / "search_summary.json").write_text('{"evaluations": 1}')
    with pytest.raises(ValueError, match="unequal actual search budget"):
        export_campaign(tmp_path, settings)
    assert not (tmp_path / "reports/discovery.csv").exists()
