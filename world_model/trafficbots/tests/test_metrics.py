from __future__ import annotations

import numpy as np
from types import SimpleNamespace

from world_model.src.core.highd_metrics import distribution_metrics, ego_replay_gate, ego_replay_metrics, factual_metrics, intervention_metrics


def test_factual_metric_and_ego_gate():
    generated = np.zeros((2, 3, 7, 6), np.float32)
    target = generated.copy(); target[..., 1:, 0] = 1.0
    active = np.ones((2, 6), bool)
    assert factual_metrics(generated, target, active)["ADE_m"] == 1.0
    ego = ego_replay_metrics(generated, generated)
    assert ego_replay_gate(ego, {key: 0.0 for key in ego})


def test_shared_distribution_and_paired_intervention_metrics():
    states = np.zeros((2, 149, 7, 6), np.float32)
    states[..., 2] = 20.0
    actions = np.zeros((2, 149, 6, 2), np.float32)
    active = np.ones((2, 6), bool)
    samples = [SimpleNamespace(states=states, background_actions=actions) for _ in range(2)]
    distribution = distribution_metrics(samples, states[:, 0], states, actions, actions, active)
    assert distribution["samples_per_condition"] == 2
    assert distribution["min_ADE_m"] == 0.0
    full = np.zeros((2, 174, 7, 6), np.float32); full[..., 2] = 20.0
    response = intervention_metrics(samples[0], samples[0], samples[1], full, active, "brake")
    assert response["committed_response_invariant"]
