import numpy as np
import torch

from hierarchical_world_model.src.npc_experiments.lateral import (
    distance,
    event_histograms,
    lane_events,
    lateral_protocol,
)
from hierarchical_world_model.src.stochastic_drivers.driver_policy import (
    causal_maneuver_features,
)


def lane_trace():
    t = np.arange(301) * 0.04
    phase = ((t - 1.5) / 7).clip(0, 1)
    y = 4 * (10 * phase**3 - 15 * phase**4 + 6 * phase**5)
    state = np.zeros((len(t), 1, 6), np.float32)
    state[:, 0, 0] = 20 * t
    state[:, 0, 1] = y
    state[:, 0, 2] = 20
    state[:, 0, 3] = np.gradient(y, 0.04)
    state[:, 0, 5] = np.gradient(state[:, 0, 3], 0.04)
    return state


def test_common_event_definition_counts_real_and_simulated_motion_identically():
    import pandas as pd

    states = lane_trace()
    valid = np.ones(states.shape[:2], bool)
    events, completed, curves = lane_events(
        states, valid, [0, 4, 8], np.array([[4.5, 1.8]])
    )
    complete = [row for row in events if row["observed_complete"]]
    assert len(complete) == completed.sum() == 1
    assert complete[0]["direction"] == 1
    assert curves.shape == (1, 2, 101)
    # No-neighbor categories survive; finite gaps supplied only for bin fitting.
    train = pd.DataFrame(complete * 20)
    train["front_gap"] = np.linspace(10, 60, 20)
    train["rear_gap"] = np.linspace(15, 70, 20)
    train["front_relative_speed"] = np.linspace(-3, 3, 20)
    train["rear_relative_speed"] = np.linspace(-2, 2, 20)
    protocol = lateral_protocol(train)
    hist = event_histograms(events, protocol)
    assert hist["start_gap_joint"][-1, -1] == 1
    assert distance(hist["direction"], hist["direction"]) == 0


def test_truncated_observation_is_kept_as_censored_not_a_completed_event():
    states = lane_trace()[:120]
    events, completed, curves = lane_events(
        states, np.ones(states.shape[:2], bool), [0, 4, 8], np.array([[4.5, 1.8]])
    )
    assert any(row["status"] == "end_censored" for row in events)
    assert not completed.any()
    assert not len(curves)


def test_maneuver_anchor_uses_past_motion_and_changes_only_after_settling():
    source = torch.tensor(0.0)
    outputs = []
    for y, vy, centre in [(0, 0, 0), (1, 0.5, 0), (2.5, 0.5, 4), (4, 0, 4)]:
        current = torch.tensor([0.0, float(y), 20.0, float(vy), 0.0, 0.0])
        features, source = causal_maneuver_features(
            current, source, torch.tensor(float(centre)), torch.tensor(4.0)
        )
        outputs.append(features)
    torch.testing.assert_close(outputs[2], torch.tensor([1.0, 0.625, 0.5]))
    torch.testing.assert_close(outputs[3], torch.zeros(3))
    assert source == 4


def test_empty_event_distribution_is_not_silently_a_good_match():
    assert distance([0, 0], [0, 0]) == 0
    assert distance([0, 0], [1, 0]) == 1
