"""Common post-hoc analysis uses the real search caps, without traffic models."""

import json
from types import SimpleNamespace

import pytest

from ads.common.instances import GeneratedSource, InstanceStore, UsageLedger
from ads.common.reporting import write_json
from ads.search.branches import HillValleyPartitioner
from ads.search.comparison import common_discovery_partition
from ads.search.engine import SearchEngine
from ads.tests.test_search import evaluate, small_config


@pytest.mark.parametrize("fixed_budget", (False, True))
def test_common_partition_charges_checks_and_does_not_create_repair_samples(
    source, tmp_path, monkeypatch, fixed_budget
):
    config = small_config(budget=100, branch_check_budget=20, fixed_budget=fixed_budget)
    methods = ("monte_carlo", "subset", "guided")
    ledger = UsageLedger(tmp_path / "usage.sqlite")

    def new_source(*parts):
        return GeneratedSource(
            source.sampler, source.assets, master_seed=123, purpose="search"
        )

    def evaluator(instance):
        result = evaluate(instance)
        result.record.update(
            instance_hash=instance.instance_hash,
            event_id=instance.instance_hash,
            root_sample_id=instance.root_sample_id,
            parent_id=instance.parent_id,
        )
        return result

    for method in methods:
        run = SearchEngine(new_source(), evaluator, config=config).run(method)
        directory = tmp_path / "dqn/20260926" / method / "round_1"
        store = InstanceStore(
            directory / "search_instances", purpose="search", ledger=ledger
        )
        for result in run.records:
            store.add(result.instance)
        (directory / "search.jsonl").write_text(
            "".join(json.dumps(r.record) + "\n" for r in run.records)
        )
        write_json(directory / "search_summary.json", run.summary())
    monkeypatch.setattr(
        "ads.search.comparison.load_policy", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        "ads.search.comparison.WorldEvaluator", lambda *args, **kwargs: evaluator
    )
    campaign = SimpleNamespace(
        output=tmp_path,
        config={
            "methods": methods,
            "search": {
                "budget": 100,
                "branch_check_budget": 20,
                "graph_size": 6,
                "fixed_budget": fixed_budget,
            },
        },
        source=new_source,
        checkpoint=lambda directory, algorithm: directory / "policy.zip",
        factory=None,
        evt=None,
        device="cpu",
        ledger=ledger,
    )
    original_split = HillValleyPartitioner.split
    partition_populations = []

    def check_failure_population(partitioner, population, check):
        partition_populations.append(population)
        assert population and all(result.failure for result in population)
        return original_split(partitioner, population, check)

    monkeypatch.setattr(
        "ads.search.comparison.HillValleyPartitioner.split", check_failure_population
    )
    if fixed_budget:

        def forbidden(*args, **kwargs):
            pytest.fail("post-hoc analysis cannot generate or execute beyond budget")

        monkeypatch.setattr(GeneratedSource, "mutation", forbidden)
        monkeypatch.setattr("ads.search.comparison.WorldEvaluator", forbidden)
    report = common_discovery_partition(campaign, "dqn", 20260926)
    assert len(partition_populations) == 1
    assert report["partition_population"] == "executed_ego_failures_only"
    assert report["unique_partition_instances"] == len(partition_populations[0])
    assert "resolved" in report
    for method in methods:
        directory = tmp_path / "dqn/20260926" / method / "round_1"
        summary = json.loads((directory / "search_summary.json").read_text())
        rows = [
            json.loads(line)
            for line in (directory / "search.jsonl").read_text().splitlines()
        ]
        assert len(rows) == summary["evaluations"] <= 100
        if fixed_budget:
            assert len(rows) == 100
        assert sum(summary["execution_costs"].values()) == len(rows)
        assert summary["execution_costs"].get("branch_check", 0) <= 20
        assert all(
            not row["independent_sampling_root"]
            for row in rows
            if row["execution_kind"] == "branch_check"
        )
        assert all("common_partition_resolved" in row for row in rows)
