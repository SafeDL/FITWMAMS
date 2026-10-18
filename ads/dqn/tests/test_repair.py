from __future__ import annotations

import numpy as np
import pandas as pd

from ads.dqn.env import WorldEvent
from ads.dqn.event_sampler import build_event_mixture
from ads.dqn.paired import paired_repair_summary
from hierarchical_world_model.src.randomness import WorldExogenousState


def _event(name: str, *, split: str = "train") -> WorldEvent:
    maps = np.zeros((1, 2, 6), np.float32)
    maps[..., 0] = ((0.0, 1.0))
    maps[..., 2] = 1.0
    maps[..., 4] = 3.6
    states = np.zeros((7, 6), np.float32)
    valid = np.zeros(7, bool)
    valid[0] = True
    return WorldEvent(
        name, name, states, valid, np.zeros((2, 6, 2), np.float32),
        maps, np.ones((1, 2), bool),
        WorldExogenousState.sample(1, seed=len(name), response_steps=2),
        split=split,
    )


def test_repair_event_mixture_uses_exact_requested_ratio() -> None:
    normal = [_event("n0"), _event("n1")]
    new = [_event("f0")]
    history = [_event("h0")]
    mixture = build_event_mixture(
        normal, new, history, ratios=(0.5, 0.3, 0.2), size=100, seed=3
    )
    assert len(mixture.events) == 100
    assert mixture.categories.count("normal") == 50
    assert mixture.categories.count("new_failure") == 30
    assert mixture.categories.count("history") == 20


def test_repair_event_mixture_rejects_test_leakage() -> None:
    try:
        build_event_mixture(
            [_event("n")], [_event("held", split="test")], [_event("h")], size=10
        )
    except ValueError as error:
        assert "held-out" in str(error)
    else:
        raise AssertionError("held-out event leaked into repair")


def test_paired_repair_summary_reports_four_cells_and_progress() -> None:
    before = pd.DataFrame(
        {"event_id": list("abcd"), "failure": [False, False, True, True], "progress_m": [10, 10, 8, 7]}
    )
    after = pd.DataFrame(
        {"event_id": list("abcd"), "failure": [False, True, False, True], "progress_m": [11, 9, 10, 8]}
    )
    result = paired_repair_summary(before, after)
    assert result["n00_success_to_success"] == 1
    assert result["n01_success_to_failure"] == 1
    assert result["n10_failure_to_success"] == 1
    assert result["n11_failure_to_failure"] == 1
    assert result["repair_rate"] == 0.5
    assert result["regression_rate"] == 0.5
    assert result["net_gain"] == 0.0
    assert result["mean_progress_delta_m"] == 0.75
    assert result["bootstrap_unit"] == "event_id"
    assert result["bootstrap_cluster_count"] == 4


def test_paired_repair_summary_clusters_related_branches() -> None:
    before = pd.DataFrame(
        {
            "event_id": ["a0", "a1", "b0"],
            "source_event_id": ["a", "a", "b"],
            "failure": [True, True, False],
            "progress_m": [5.0, 5.0, 8.0],
        }
    )
    after = before.copy()
    after["failure"] = [False, False, False]
    result = paired_repair_summary(before, after, bootstrap_replicates=100)
    assert result["bootstrap_unit"] == "source_event_id"
    assert result["bootstrap_cluster_count"] == 2
