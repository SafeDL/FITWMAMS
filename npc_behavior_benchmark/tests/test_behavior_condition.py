import numpy as np
import pandas as pd
import pytest
import torch

from npc_behavior_benchmark.data.dataset import event_behavior_condition
from npc_behavior_benchmark.evaluation.action_rollout import _append_t4a_condition
from npc_behavior_benchmark.scripts.evaluate_controlled_action import _mode_metrics
from npc_behavior_benchmark.policies.rolling_action_policy import (
    BEHAVIOR_CONDITION_MODES,
    ActionDiffusionConfig,
    RollingActionDiffusion,
    RollingActionPolicy,
    append_behavior_condition,
)
from npc_behavior_benchmark.policies.interface import RandomKey


def test_behavior_condition_is_targeted_constant_and_future_free():
    feature = np.zeros((25, 7, 12), np.float32)
    encoded = append_behavior_condition(
        feature, {"target_agent_index": 3, "mode": "lane_left"}
    )
    assert encoded.shape == (25, 7, 16)
    assert np.array_equal(encoded[..., :12], feature)
    expected = np.zeros((7, 4), np.float32)
    expected[3, BEHAVIOR_CONDITION_MODES.index("lane_left")] = 1.0
    assert np.array_equal(encoded[0, :, 12:], expected)
    assert np.array_equal(encoded[-1, :, 12:], expected)


@pytest.mark.parametrize(
    "condition",
    (
        {"target_agent_index": 0, "mode": "brake"},
        {"target_agent_index": 7, "mode": "brake"},
        {"target_agent_index": 1, "mode": "unknown"},
        {"target_agent_index": 1},
    ),
)
def test_behavior_condition_rejects_non_npc_or_unknown_mode(condition):
    with pytest.raises(ValueError):
        append_behavior_condition(np.zeros((25, 7, 12), np.float32), condition)


@pytest.mark.parametrize(
    ("event", "expected"),
    (
        (
            {
                "event_type": "lane_change",
                "response_agent_index": 4,
                "stimulus_agent_index": 4,
                "direction": "left",
            },
            {"target_agent_index": 4, "mode": "lane_left"},
        ),
        (
            {
                "event_type": "cut_out",
                "response_agent_index": 2,
                "stimulus_agent_index": 5,
                "direction": None,
            },
            {"target_agent_index": 2, "mode": "recover"},
        ),
        (
            {
                "event_type": "following_brake",
                "response_agent_index": 2,
                "stimulus_agent_index": 5,
                "direction": None,
            },
            {"target_agent_index": 5, "mode": "brake"},
        ),
        (
            {
                "event_type": "cut_in",
                "response_agent_index": 0,
                "stimulus_agent_index": 0,
                "direction": None,
            },
            None,
        ),
    ),
)
def test_d1_semantic_condition_mapping_never_targets_ego(event, expected):
    assert event_behavior_condition(pd.Series(event)) == expected


def test_batched_t4a_condition_expands_per_scene_not_future():
    raw = torch.zeros((4, 25, 7, 12))
    encoded = _append_t4a_condition(
        raw,
        16,
        [
            {"target_agent_index": 2, "mode": "brake"},
            {"target_agent_index": 5, "mode": "recover"},
        ],
        futures=2,
    )
    assert tuple(encoded.shape) == (4, 25, 7, 16)
    assert encoded[:2, :, 2, 12].all() and not encoded[:2, :, 5, 13].any()
    assert encoded[2:, :, 5, 13].all() and not encoded[2:, :, 2, 12].any()


def test_t4a_fixed_goal_metrics_right_censor_unmet_behavior():
    initial = np.zeros((7, 6), np.float32)
    initial[2, 2] = 10.0
    states = np.broadcast_to(initial, (2, 75, 7, 6)).copy()
    states[0, 9:, 2, 2] = 7.5  # one of two futures brakes by 2.5 m/s
    metrics = _mode_metrics(states, initial, 2, "brake")
    assert metrics["completion_probability"] == 0.5
    assert metrics["right_censored_completion_time_s"] == pytest.approx((0.4 + 3.0) / 2)


def test_unconditional_policy_refuses_explicit_t4a_goal():
    model = RollingActionDiffusion(
        ActionDiffusionConfig(hidden_dim=8, num_heads=4, num_layers=1),
        np.zeros(2),
        np.ones(2),
    )
    policy = RollingActionPolicy(model, np.zeros(12), np.ones(12))
    with pytest.raises(ValueError, match="unconditional"):
        policy.reset(
            np.zeros((25, 7, 6), np.float32),
            np.ones((25, 7), bool),
            np.arange(7),
            {},
            RandomKey("benchmark", "scenario", 1, 0),
            {"target_agent_index": 1, "mode": "brake"},
        )
