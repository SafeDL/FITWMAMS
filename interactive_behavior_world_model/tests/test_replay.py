import numpy as np
import torch

from interactive_behavior_world_model.evaluation.rollout import rollout_shared_bc
from interactive_behavior_world_model.evaluation.semantic_rollout import rollout_semantic_policy
from interactive_behavior_world_model.policies.interface import PolicyObservation, RandomKey
from interactive_behavior_world_model.policies.rolling_action_policy import (
    ActionDiffusionConfig,
    RollingActionDiffusion,
)
from interactive_behavior_world_model.policies.shared_bc import SharedBCModel, SharedBCPolicy


def _synthetic_scene():
    history = np.zeros((25, 7, 6), np.float32)
    history[..., 2] = 15.0
    history[:, :, 0] = np.arange(7)[None] * 12.0
    valid = np.ones((25, 7), bool)
    lines = np.zeros((8, 8, 5), np.float32)
    line_valid = np.ones((8, 8), bool)
    lines[..., 0] = np.arange(8)[None]
    lines[..., 1] = np.arange(8)[:, None] * 3.5
    lines[..., 4] = 3.5
    return history, valid, lines, line_valid


def test_shared_policy_snapshot_restores_random_stream():
    history, valid, lines, line_valid = _synthetic_scene()
    model = SharedBCModel(hidden_dim=16, heads=4, layers=1)
    policy = SharedBCPolicy(model, np.zeros(12, np.float32), np.ones(12, np.float32))
    ids = np.arange(7)
    lengths = np.full(7, 4.5, np.float32)
    widths = np.full(7, 1.8, np.float32)
    policy.reset(history, valid, ids, {}, RandomKey("b", "s", 1, 0))
    observation = PolicyObservation(
        history[-1],
        valid[-1],
        ids,
        lengths,
        widths,
        lines,
        line_valid,
        history,
        valid,
        np.zeros((7, 2)),
        0,
        0.0,
    )
    snapshot = policy.snapshot()
    first = policy.act(observation).requested_action
    policy.restore(snapshot)
    replay = policy.act(observation).requested_action
    assert np.array_equal(first, replay)


def test_fixed_pnc_rollout_does_not_read_logged_ego_future():
    history, valid, lines, line_valid = _synthetic_scene()
    model = SharedBCModel(hidden_dim=16, heads=4, layers=1)
    checkpoint = {
        "feature_mean": np.zeros(12, np.float32),
        "feature_std": np.ones(12, np.float32),
    }
    kwargs = dict(
        initial_history=history[None],
        history_valid=valid[None],
        agent_ids=np.arange(7)[None],
        lengths_m=np.full((1, 7), 4.5, np.float32),
        widths_m=np.full((1, 7), 1.8, np.float32),
        map_polylines=lines[None],
        map_valid=line_valid[None],
        scenario_ids=np.asarray(["scene"]),
        benchmark_id="bench",
        fit_seed=3,
        futures=2,
        device=torch.device("cpu"),
        ego_policy_id="cruise",
    )
    zero_future = np.zeros((1, 5, 6), np.float32)
    changed_future = np.full((1, 5, 6), 1000.0, np.float32)
    first = rollout_shared_bc(model, checkpoint, ego_future=zero_future, **kwargs)
    second = rollout_shared_bc(model, checkpoint, ego_future=changed_future, **kwargs)
    assert np.array_equal(first.states, second.states)
    assert np.array_equal(first.requested_actions, second.requested_actions)


def test_semantic_fixed_pnc_rollout_does_not_read_logged_ego_future():
    history, valid, lines, line_valid = _synthetic_scene()
    config = ActionDiffusionConfig(hidden_dim=16, num_layers=1, num_heads=4)
    model = RollingActionDiffusion(
        config, np.zeros(2, np.float32), np.ones(2, np.float32)
    )
    checkpoint = {
        "feature_mean": np.zeros(12, np.float32),
        "feature_std": np.ones(12, np.float32),
    }
    kwargs = dict(
        initial_history=history[None],
        history_valid=valid[None],
        lengths_m=np.full((1, 7), 4.5, np.float32),
        widths_m=np.full((1, 7), 1.8, np.float32),
        map_polylines=lines[None],
        map_valid=line_valid[None],
        scenario_ids=np.asarray(["scene"]),
        benchmark_id="bench",
        fit_seed=3,
        futures=1,
        inference_steps=1,
        candidates=2,
        response_samples=1,
        response_aware=True,
        stochastic_selection=True,
        device=torch.device("cpu"),
        ego_policy_id="cruise",
    )
    zero_future = np.zeros((1, 5, 6), np.float32)
    changed_future = np.full((1, 5, 6), 1000.0, np.float32)
    first = rollout_semantic_policy(model, checkpoint, ego_future=zero_future, **kwargs)
    second = rollout_semantic_policy(
        model, checkpoint, ego_future=changed_future, **kwargs
    )
    assert np.array_equal(first.states, second.states)
    assert np.array_equal(first.requested_actions, second.requested_actions)


def test_semantic_paired_stimulus_branches_share_pre_intervention_innovation():
    history, valid, lines, line_valid = _synthetic_scene()
    config = ActionDiffusionConfig(hidden_dim=16, num_layers=1, num_heads=4)
    model = RollingActionDiffusion(
        config, np.zeros(2, np.float32), np.ones(2, np.float32)
    )
    checkpoint = {
        "feature_mean": np.zeros(12, np.float32),
        "feature_std": np.ones(12, np.float32),
    }
    kwargs = dict(
        initial_history=history[None],
        history_valid=valid[None],
        ego_future=np.zeros((1, 10, 6), np.float32),
        lengths_m=np.full((1, 7), 4.5, np.float32),
        widths_m=np.full((1, 7), 1.8, np.float32),
        map_polylines=lines[None],
        map_valid=line_valid[None],
        scenario_ids=np.asarray(["paired-scene"]),
        benchmark_id="bench",
        fit_seed=3,
        futures=1,
        inference_steps=1,
        candidates=2,
        response_samples=1,
        response_aware=True,
        stochastic_selection=True,
        device=torch.device("cpu"),
    )
    natural = rollout_semantic_policy(model, checkpoint, **kwargs)
    changed_external = np.repeat(history[-1][None, None], 10, axis=1)
    changed_external[:, :, 1, 1] += 3.0
    changed_mask = np.zeros((1, 7), bool)
    changed_mask[:, 1] = True
    stimulated = rollout_semantic_policy(
        model,
        checkpoint,
        **kwargs,
        exogenous_future=changed_external,
        exogenous_mask=changed_mask,
    )
    # The external stimulus is first applied after the initial decision, so
    # matching keys must yield an exactly matched first semantic action.
    assert np.array_equal(
        natural.requested_actions[:, :, 0], stimulated.requested_actions[:, :, 0]
    )
