import csv
import math

import numpy as np
import pytest
from scipy.stats import multinomial

from ads.common.reporting import (
    paired_binary_difference_ci,
    paired_summary,
    repair_method_comparisons,
    shared_world_seed_summary,
)


def rows(failures):
    return [
        {
            "event_id": str(i),
            "instance_hash": str(i),
            "assets_hash": "world",
            "policy_stream_id": str(i),
            "source_kind": "tail_generated",
            "root_sample_id": str(i),
            "safety_failure": failure,
            "progress_m": 100,
            "collision": failure,
            "offroad": False,
            "stalled": False,
        }
        for i, failure in enumerate(failures)
    ]


def test_safety_pair_and_content_identity():
    before, after = rows([True, False, True]), rows([False, True, False])
    result = paired_summary(before, after, bootstrap_replicates=50)
    assert result["repaired"] == 2
    assert result["regressed"] == 1
    after[0]["assets_hash"] = "changed"
    with pytest.raises(ValueError, match="assets_hash"):
        paired_summary(before, after, bootstrap_replicates=50)


def test_training_seeds_do_not_multiply_independent_world_count():
    before, after = rows([True, False, True]), rows([False, False, False])
    summary = shared_world_seed_summary([(before, after)] * 3, bootstrap_replicates=50)
    assert summary["independent_units"] == 3
    assert summary["training_seeds"] == 3
    assert summary["mean_collision_difference"] == pytest.approx(-2 / 3)


def test_repair_comparison_uses_shared_instances_and_correct_difference_sign():
    records = {}
    for seed in (1, 2, 3):
        records[f"dqn/{seed}/monte_carlo"] = rows([True, False, True])
        records[f"dqn/{seed}/subset"] = rows([True, False, False])
        records[f"dqn/{seed}/guided"] = rows([False, False, False])
    result = repair_method_comparisons(
        records,
        ["dqn"],
        [1, 2, 3],
        ["monte_carlo", "subset", "guided"],
        bootstrap_replicates=50,
    )
    summary = result["shared_world_aggregate"]["dqn/guided_vs_monte_carlo"]
    assert summary["independent_units"] == 3
    assert summary["mean_collision_difference"] == pytest.approx(-2 / 3)
    records["dqn/2/subset"][0]["root_sample_id"] = "wrong"
    with pytest.raises(ValueError, match="root_sample_id"):
        repair_method_comparisons(
            records,
            ["dqn"],
            [1, 2, 3],
            ["subset", "guided"],
            bootstrap_replicates=50,
        )


def test_policy_stream_mismatch_and_recording_cluster():
    before, after = rows([True, False, True, False]), rows([False, True, False, False])
    for group in (before, after):
        for index, row in enumerate(group):
            row.update(source_kind="empirical_natural", recording_id=index // 2)
    result = paired_summary(before, after, bootstrap_replicates=50)
    assert (
        result["bootstrap_unit"] == "recording_id" and result["independent_units"] == 2
    )
    after[0]["policy_stream_id"] = "changed"
    with pytest.raises(ValueError, match="policy_stream_id"):
        paired_summary(before, after, bootstrap_replicates=50)


def test_zero_discordance_has_nonzero_iid_uncertainty_and_shared_seed_agrees():
    before = rows([False] * 2000)
    after = rows([False] * 2000)
    result = paired_summary(before, after, bootstrap_replicates=50)
    expected = -math.expm1(math.log(0.0125) / 2000)
    assert result["collision_rate_difference_ci95"] == pytest.approx(
        [-expected, expected]
    )
    assert result["collision_rate_difference_bootstrap_ci95"] == [0, 0]
    assert result["collision_discordant_instances"] == 0
    assert not result["collision_ci_is_degenerate"]
    shared = shared_world_seed_summary([(before, after)], bootstrap_replicates=50)
    assert (
        shared["paired_collision_difference_ci95"]
        == result["collision_rate_difference_ci95"]
    )
    assert shared["independent_units"] == 2000


def test_paired_interval_reverses_sign_and_ignores_concordant_outcome_value():
    delta = np.array([-1] * 7 + [1] * 2 + [0] * 31)
    low, high = paired_binary_difference_ci(delta)
    assert paired_binary_difference_ci(-delta) == pytest.approx([-high, -low])
    both_safe = paired_summary(
        rows([False] * 40), rows([False] * 40), bootstrap_replicates=50
    )
    both_collide = paired_summary(
        rows([True] * 40), rows([True] * 40), bootstrap_replicates=50
    )
    assert (
        both_safe["collision_rate_difference_ci95"]
        == both_collide["collision_rate_difference_ci95"]
    )


@pytest.mark.parametrize(
    "positive,negative",
    [
        (0, 0),
        (0.001, 0.001),
        (0.2, 0.01),
        (0.01, 0.2),
        (0.4, 0.4),
        (0.95, 0.01),
        (1, 0),
    ],
)
def test_paired_interval_exact_multinomial_coverage(positive, negative):
    n, coverage = 12, 0.0
    for added in range(n + 1):
        for removed in range(n - added + 1):
            delta = np.array([1] * added + [-1] * removed + [0] * (n - added - removed))
            low, high = paired_binary_difference_ci(delta)
            if low <= positive - negative <= high:
                coverage += multinomial.pmf(
                    [added, removed, n - added - removed],
                    n,
                    [positive, negative, 1 - positive - negative],
                )
    assert coverage >= 0.95 - 1e-12


def test_cluster_bootstrap_retains_recording_unit_and_marks_no_variation():
    before, after = rows([False] * 12), rows([False] * 12)
    for group in (before, after):
        for index, row in enumerate(group):
            row.update(source_kind="empirical_natural", recording_id=index // 4)
    result = paired_summary(before, after, bootstrap_replicates=50)
    assert result["bootstrap_unit"] == "recording_id"
    assert result["independent_units"] == 3
    assert result["collision_ci_method"] == "cluster_percentile_bootstrap"
    assert result["collision_ci_is_degenerate"]
    assert result["collision_rate_difference_ci95"] == [0, 0]
    archive = paired_summary(
        rows([False] * 12), rows([False] * 12), diagnostic=True, bootstrap_replicates=50
    )
    assert archive["bootstrap_unit"] == "root_sample_id"
    assert archive["collision_ci_is_degenerate"]


def test_export_keeps_cluster_interval_method_and_degeneracy(tmp_path):
    from ads.common.reporting import write_json
    from ads.report import export_evaluation

    before, after = rows([False] * 12), rows([False] * 12)
    for group in (before, after):
        for index, row in enumerate(group):
            row.update(source_kind="empirical_natural", recording_id=index // 4)
    single = paired_summary(before, after, bootstrap_replicates=50)
    shared = shared_world_seed_summary([(before, after)], bootstrap_replicates=50)
    summary = {
        "cohorts": {
            "empirical_natural": {
                "per_seed": {"dqn/20260926/guided": single},
                "shared_world_aggregate": {"dqn/guided": shared},
                "between_repair_methods": {
                    "per_seed": {},
                    "shared_world_aggregate": {},
                },
            }
        },
        "archive_diagnostics": {},
    }
    write_json(tmp_path / "evaluation/summary.json", summary)
    reports = tmp_path / "reports"
    reports.mkdir()
    export_evaluation(tmp_path, reports, {})
    with (reports / "repair_comparisons.csv").open() as stream:
        result = next(csv.DictReader(stream))
    assert result["ci_method"] == "cluster_percentile_bootstrap"
    assert result["ci_is_degenerate"] == "True"
    assert result["independent_units"] == "3"
    assert (reports / "repair.pdf").exists()


def test_discovery_coverage_includes_offroad_only_failure_branches():
    from ads.report import discovery_curve

    records = rows([True] * 4)
    for index, row in enumerate(records):
        row.update(common_branch_id=str(index), common_partition_resolved=True)
    records[1].update(collision=False, offroad=True)
    records[2]["execution_kind"] = "branch_check"
    records[3]["execution_kind"] = "archive_replay"
    curve = discovery_curve(records)
    assert curve[-1]["common_failure_branches"] == 2
    assert curve[-1]["distinct_collisions"] == 1
    assert curve[-1]["distinct_instance_failures"] == 2
    assert curve[-1]["failure_roots"] == 1
