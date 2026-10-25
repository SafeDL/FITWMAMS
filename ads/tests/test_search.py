from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from ads.common.evaluation import EpisodeResult
from ads.search.archive import FailureArchive, diverse_representatives
from ads.search.branches import HillValleyPartitioner, midpoint
from ads.search.engine import SearchConfig, SearchEngine


def evaluate(instance):
    omega = instance.event.exogenous_state
    x = float(omega.c0_base_latent[0, 0])
    risk = abs(x)
    failure = risk > 1.5
    return EpisodeResult(
        instance,
        {
            "safety_failure": failure,
            "collision": failure,
            "event_risk": risk,
            "evt_score": min(risk, 0.5),
            "interaction_type": "front_tracking",
            "branch_id": "root",
            "stage": "tracking",
            "descriptor": [x, 0, 0, 0, 0.4],
            "critical_slot": 1,
            "critical_frame": 40,
            "physics_steps": 149,
            "elapsed_s": 0,
        },
        np.empty((0, 7, 6)),
        np.empty((0, 2)),
        [],
    )


def small_config(**kwargs):
    return SearchConfig(
        **{
            "budget": 100,
            "initial_samples": 12,
            "fresh_samples": 8,
            "batch_size": 8,
            "generation_batch_size": 4,
            "graph_size": 6,
            "branch_check_budget": 16,
            "terminal_proposals": 8,
            "subset_max_levels": 2,
            "archive_capacity": 4,
            **kwargs,
        }
    )


@pytest.mark.parametrize("method", ["monte_carlo", "subset", "guided"])
def test_execution_cap_includes_rejections_checks_and_replay(source, method):
    journal = []
    result = SearchEngine(
        source,
        evaluate,
        config=small_config(budget=63),
        on_evaluation=lambda item: journal.append(dict(item.record)),
    ).run(method)
    summary = result.summary()
    assert len(result.records) <= 63
    assert sum(summary["execution_costs"].values()) == len(result.records)
    assert len(journal) == len(result.records)
    assert [row["evaluation_index"] for row in journal] == list(
        range(1, len(result.records) + 1)
    )
    assert summary["execution_costs"].get("branch_check", 0) <= 16
    assert summary["hit_fraction_is_prior_probability"] is False
    assert "probability" not in summary
    assert all(
        r.record.get("execution_kind") != "branch_check"
        for r in result.archive.failures
    )


def test_original_risk_orders_beyond_evt_saturation_and_failure_is_terminal(source):
    low, high = [evaluate(i) for i in source.sample(2)]
    low.record.update(event_risk=2, evt_score=27.63, safety_failure=False)
    high.record.update(event_risk=3, evt_score=27.63, safety_failure=False)
    assert high.score > low.score
    assert high.level_score > low.level_score
    low.record["safety_failure"] = True
    assert low.level_score == 1 > high.level_score


def test_safe_search_stops_at_depth_after_all_prior_quotas(source):
    def safe(instance):
        result = evaluate(instance)
        result.record.update(safety_failure=False, collision=False)
        return result

    result = SearchEngine(
        source, safe, config=small_config(branch_check_budget=0, subset_max_levels=1)
    ).run()
    assert len(result.records) < result.budget
    assert result.costs["initial_prior"] == 12
    assert result.costs["fresh_prior"] == 8
    assert result.stop_reason == "scheduled_tasks_complete"
    assert any(node["status"] == "depth_limit" for node in result.branches.values())


def test_hill_valley_detects_separated_failures_without_semantic_labels(source):
    instances = source.sample(12)
    results = []
    for index, instance in enumerate(instances):
        arrays = instance.event.exogenous_state.as_dict()
        arrays["c0_base_latent"][0, 0] = -3 if index < 6 else 3
        results.append(
            evaluate(
                source.mutation(
                    instance,
                    instance.event.exogenous_state.replace_arrays(arrays),
                    "c0_base_latent",
                )
            )
        )

    def check(left, right):
        return evaluate(
            source.mutation(
                left.instance, midpoint(left.instance, right.instance), "branch_check"
            )
        )

    partition, evidence = HillValleyPartitioner(graph_size=12).split(results, check)
    assert evidence["status"] == "split"
    labels = partition.predict([r.instance for r in results])
    assert len(set(labels)) == 2
    assert len(set(labels[:6])) == len(set(labels[6:])) == 1
    assert labels[0] != labels[-1]
    assert {r.record["interaction_type"] for r in results} == {"front_tracking"}


def test_unknown_pairs_do_not_create_false_disconnection(source):
    results = [evaluate(i) for i in source.sample(12)]
    partition, evidence = HillValleyPartitioner().split(
        results, lambda left, right: None
    )
    assert partition is None
    assert evidence["status"] == "incomplete"


def test_archive_diversity_and_root_limits(source):
    results = [evaluate(i) for i in source.sample(20)]
    for index, result in enumerate(results):
        result.instance = replace(result.instance, root_sample_id=str(index // 8))
    selected = diverse_representatives(results, capacity=8, root_limit=2)
    assert len(selected) == 6
    assert max(Counter(r.instance.root_sample_id for r in selected).values()) == 2


def test_partial_success_retests_use_actual_denominator(source):
    instances = source.sample(4)
    archive = FailureArchive()
    failures, neighbours = [], []
    for instance in instances[:2]:
        parent = evaluate(instance)
        parent.record["safety_failure"] = False
        child = evaluate(
            source.mutation(
                instance,
                instance.event.exogenous_state.mutate(
                    "driver_latent", beta=0.2, seed=2
                ),
                "driver_latent",
            )
        )
        child.record["safety_failure"] = True
        failures.append(child)
        neighbours.append(parent)
        archive.add([child], {parent.instance.instance_hash: parent})
    repaired = replace(
        failures[0], record={**failures[0].record, "safety_failure": False}
    )
    regressed = replace(
        neighbours[0], record={**neighbours[0].record, "safety_failure": True}
    )
    report = archive.feedback([repaired, regressed])["root"]
    assert report["success_neighbours"] == 1
    assert report["regression_fraction"] == 1
    assert report["weight"] == 3


def test_two_round_replay_is_separate_from_new_conditional_particles(source):
    config = small_config(budget=128, branch_check_budget=0)
    first = SearchEngine(source, evaluate, config=config).run()
    second = SearchEngine(source, evaluate, config=config, seed=1).run(
        archive=first.archive
    )
    assert len(second.records) <= 128
    assert second.costs.get("archive_replay", 0) <= config.replay_limit
    assert all(
        "old_branch_id" in r.record
        for r in second.records
        if r.record["execution_kind"] == "archive_replay"
    )


def test_unretested_and_repaired_endpoints_survive_the_second_archive(source, tmp_path):
    previous = FailureArchive()
    old = [evaluate(i) for i in source.sample(6)]
    for index, result in enumerate(old):
        result.record.update(safety_failure=True, branch_id=f"old.{index % 2}")
    previous.add(old, {})

    def safe(instance):
        result = evaluate(instance)
        result.record.update(safety_failure=False, collision=False)
        return result

    run = SearchEngine(
        source, safe, config=small_config(replay_limit=2, branch_check_budget=0)
    ).run(archive=previous)
    old_ids = {r.instance.instance_hash for r in old}
    retested = {
        r.instance.instance_hash
        for r in run.records
        if r.record["execution_kind"] == "archive_replay"
    }
    assert len(retested) == 2
    assert {r.domain for r in old if r.instance.instance_hash in retested} == {
        "old.0",
        "old.1",
    }
    assert set(run.archive.retained) == old_ids
    assert not run.archive.failures  # Old outcomes never become current particles.
    for identity, result in run.archive.retained.items():
        assert result.failure == (identity not in retested)
    run.archive.save(tmp_path / "archive")
    restored = FailureArchive.load(tmp_path / "archive")
    assert {r.instance.instance_hash for r in restored.endpoints()} == old_ids
    assert set(restored.retained) == old_ids
    assert len(restored.endpoints(limit=3)) == 3


def test_teacher_neighbours_share_scene_budget(source):
    from ads.common.expert import select_teacher_instances

    failures = source.sample(4)
    neighbours = {
        i.instance_hash: (
            source.mutation(
                i,
                i.event.exogenous_state.mutate("driver_latent", beta=0.2, seed=5),
                "driver_latent",
            ),
            "root",
        )
        for i in failures
    }
    selected = select_teacher_instances(
        [(i, "root") for i in failures], neighbours, limit=3
    )
    assert len(selected) == 3
    assert selected[0][0].root_sample_id == selected[1][0].root_sample_id
    assert selected[2][0].root_sample_id != selected[0][0].root_sample_id


@pytest.mark.parametrize("problem", ["unbalanced", "weak_early_branch"])
def test_high_dimensional_analytic_search_comparison(problem):
    from ads.search.analytic import compare

    results = compare(
        problem, config=small_config(budget=160, initial_samples=24, fresh_samples=16)
    )
    for result in results.values():
        assert result["ambient_dimension"] > 100
        assert result["active_dimensions"] == 2
        assert result["evaluations"] <= 160
        assert result["execution_costs"].get("branch_check", 0) <= 16
        assert not result["hit_fraction_is_prior_probability"]


def test_partition_protects_a_branch_with_weaker_early_risk(source):
    results = []
    for index, instance in enumerate(source.sample(12)):
        arrays = instance.event.exogenous_state.as_dict()
        arrays["c0_base_latent"][0, 0] = -3 if index < 6 else 3
        result = evaluate(
            source.mutation(
                instance,
                instance.event.exogenous_state.replace_arrays(arrays),
                "c0_base_latent",
            )
        )
        result.record.update(
            safety_failure=False, collision=False, event_risk=0.3 if index < 6 else 3.0
        )
        results.append(result)

    def check(left, right):
        instance = source.mutation(
            left.instance, midpoint(left.instance, right.instance), "branch_check"
        )
        result = evaluate(instance)
        x = instance.event.exogenous_state.c0_base_latent[0, 0]
        result.record.update(
            safety_failure=False,
            collision=False,
            event_risk=0.3 if x < -1 else 3.0 if x > 1 else 0.0,
        )
        return result

    partition, _ = HillValleyPartitioner().split(results, check)
    assert partition is not None
    labels = partition.predict([r.instance for r in results])
    threshold = np.quantile([r.level_score for r in results], 0.9)
    assert {labels[i] for i, r in enumerate(results) if r.level_score >= threshold} == {
        labels[-1]
    }
    for label in set(labels):
        population = [r for r, actual in zip(results, labels) if actual == label]
        branch_threshold = np.quantile([r.level_score for r in population], 0.9)
        assert any(r.level_score >= branch_threshold for r in population)


def test_partition_roundtrip_preserves_membership(source):
    # The actual full coordinate layout is retained by the persisted classifier.
    from ads.search.branches import Partition, coordinates

    instances = source.sample(3)
    n = len(coordinates(instances[0]))
    basis = np.zeros((1, n))
    basis[0, 0] = 1
    partition = Partition(
        np.zeros(n), np.ones(n), basis, np.ones((1, 1)), np.zeros(1), np.array([0, 1])
    )
    restored = Partition.from_dict(partition.as_dict())
    np.testing.assert_array_equal(
        partition.predict(instances), restored.predict(instances)
    )


def test_protocol_rejects_caps_above_the_agreed_limits():
    with pytest.raises(ValueError, match="5,000"):
        SearchConfig(budget=5001)
    with pytest.raises(ValueError, match="500 branch"):
        SearchConfig(branch_check_budget=501)


def test_default_full_execution_cap_truncates_remaining_analytic_work(source):
    def safe(instance):
        result = evaluate(instance)
        result.record.update(safety_failure=False, collision=False, event_risk=0.0)
        return result

    run = SearchEngine(source, safe).run()
    assert len(run.records) == sum(run.costs.values()) == 5000
    assert run.costs["branch_check"] <= 500
    assert run.costs["initial_prior"] == 500
    assert run.costs["fresh_prior"] == 500
    assert run.stop_reason == "execution_budget"
    assert run.pending["active_branches"]
    assert any(node.get("unfinished") for node in run.branches.values())
    assert all(node["depth"] <= 8 for node in run.branches.values())
