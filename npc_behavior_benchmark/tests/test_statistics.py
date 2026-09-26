import numpy as np
import pandas as pd

from npc_behavior_benchmark.evaluation.statistics import (
    clustered_bootstrap,
    clustered_event_macro_interval,
    clustered_mean_interval,
    clustered_paired_interval,
    event_macro_mean,
    paired_metric_frame,
)


def test_clustered_bootstrap_is_reproducible_and_contains_estimate():
    frame = pd.DataFrame({"recording_id": [1, 1, 2, 2], "value": [0.0, 2.0, 4.0, 6.0]})
    first = clustered_bootstrap(
        frame, lambda data: data.value.mean(), replicates=100, seed=7
    )
    second = clustered_bootstrap(
        frame, lambda data: data.value.mean(), replicates=100, seed=7
    )
    assert first == second
    assert first["estimate"] == 3.0
    assert first["ci_low"] <= first["estimate"] <= first["ci_high"]


def test_event_macro_gives_families_equal_weight():
    frame = pd.DataFrame(
        {"event_type": ["a", "a", "a", "b"], "score": [0.0, 0.0, 0.0, 2.0]}
    )
    assert np.isclose(event_macro_mean(frame, "score"), 1.0)


def test_fast_cluster_intervals_preserve_estimands():
    frame = pd.DataFrame(
        {
            "recording_id": [1, 1, 2, 2],
            "event_type": ["a", "b", "a", "b"],
            "score": [0.0, 2.0, 2.0, 4.0],
        }
    )
    assert clustered_mean_interval(frame, "score", replicates=20)["estimate"] == 2.0
    assert (
        clustered_event_macro_interval(frame, "score", replicates=20)["estimate"] == 2.0
    )


def test_paired_cluster_interval_uses_candidate_minus_reference():
    reference = pd.DataFrame(
        {
            "scenario_id": ["a", "b", "c"],
            "recording_id": [1, 1, 2],
            "score": [1.0, 2.0, 4.0],
        }
    )
    candidate = pd.DataFrame(
        {
            "scenario_id": ["a", "b", "c"],
            "recording_id": [1, 1, 2],
            "score": [0.5, 1.0, 3.0],
        }
    )
    paired = paired_metric_frame(reference, candidate, "score")
    assert np.allclose(paired.paired_delta, [-0.5, -1.0, -1.0])
    effect = clustered_paired_interval(
        reference, candidate, "score", replicates=50, seed=3
    )
    assert np.isclose(effect["estimate"], -5 / 6)
    assert effect["paired_rows"] == 3


def test_paired_comparison_rejects_incomplete_key_sets():
    reference = pd.DataFrame(
        {"scenario_id": ["a", "b"], "recording_id": [1, 1], "score": [1.0, 2.0]}
    )
    candidate = reference.iloc[:1].copy()
    with np.testing.assert_raises(ValueError):
        paired_metric_frame(reference, candidate, "score")
