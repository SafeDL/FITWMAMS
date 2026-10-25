"""Native CaRL retention gradients, local PPO updates and checkpoint recovery."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ads.carl.policy import CaRLPolicy
from ads.carl.prepare_data import native_inputs
from ads.carl.train import configure_local_updates, retention_loss, train
from ads.common.repair import RepairEnvironment


@pytest.fixture
def policy():
    torch.set_num_threads(1)
    return CaRLPolicy.load(Path("ads/carl/artifacts/official.pth"))


@pytest.fixture
def env(instance, world_factory):
    return RepairEnvironment(
        "carl", {"normal": [(instance, "prior")]}, world_factory, round_index=0
    )


def material(observation):
    return {
        name: [{"scene_id": name, "observation": deepcopy(observation)}]
        for name in ("native", "keep")
    }


def recorded_scene():
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import (
        StateSE2,
        StateVector2D,
        TimePoint,
    )
    from nuplan.common.actor_state.tracked_objects import TrackedObjects
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

    states = [
        EgoState.build_from_rear_axle(
            StateSE2(i, 0, 0),
            StateVector2D(10, 0),
            StateVector2D(0, 0),
            0.0,
            TimePoint(i * 100_000),
            get_pacifica_parameters(),
        )
        for i in range(21)
    ]
    tracks = [DetectionsTracks(TrackedObjects([])) for _ in states]

    def forbidden(*args, **kwargs):
        pytest.fail("Native observation attempted to read future data")

    return SimpleNamespace(
        database_interval=0.1,
        get_ego_past_trajectory=lambda *a, **k: states[:-1],
        get_past_tracked_objects=lambda *a, **k: tracks[:-1],
        get_ego_state_at_iteration=lambda i: states[-1],
        get_tracked_objects_at_iteration=lambda i: tracks[-1],
        get_time_point=lambda i: states[-1].time_point,
        get_traffic_light_status_at_iteration=lambda i: [],
        get_route_roadblock_ids=lambda: ["recorded_route"],
        get_mission_goal=lambda: StateSE2(50, 0, 0),
        map_api=object(),
        get_ego_future_trajectory=forbidden,
        get_future_tracked_objects=forbidden,
    )


def test_native_history_includes_current_state_without_future():
    scenario = recorded_scene()
    current, initialization = native_inputs(scenario)
    assert len(current.history.ego_states) == 21
    assert current.history.current_state[0] is scenario.get_ego_state_at_iteration(0)
    assert current.history.current_state[
        1
    ] is scenario.get_tracked_objects_at_iteration(0)
    assert [s.time_point.time_us for s in current.history.ego_states] == list(
        range(0, 2_000_001, 100_000)
    )
    assert initialization.route_roadblock_ids == ["recorded_route"]
    assert initialization.map_api is scenario.map_api


def test_native_history_does_not_replace_missing_recordings_with_zeros():
    scenario = recorded_scene()
    past = list(scenario.get_ego_past_trajectory())[:-1]
    scenario.get_ego_past_trajectory = lambda *a, **k: past
    with pytest.raises(ValueError, match="complete recorded history"):
        native_inputs(scenario)


@pytest.mark.parametrize("roadblocks", [[], ["missing-roadblock"]])
def test_missing_native_route_is_ineligible_before_official_rendering(
    monkeypatch, roadblocks
):
    from ads.carl import prepare_data

    initialization = SimpleNamespace(
        route_roadblock_ids=roadblocks,
        map_api=SimpleNamespace(get_map_object=lambda *args: None),
    )
    monkeypatch.setattr(
        prepare_data, "native_inputs", lambda scenario: (None, initialization)
    )
    with pytest.raises(ValueError, match="Native CaRL route"):
        prepare_data.native_observation(None)


def test_beta_retention_keeps_the_student_gradient(policy, env):
    teacher = CaRLPolicy.load(Path("ads/carl/artifacts/official.pth"))
    teacher.network.eval().requires_grad_(False)
    configure_local_updates(policy)
    observation, _ = env.reset()
    batch = {key: np.stack([value, value]) for key, value in observation.items()}
    zero = retention_loss(policy, teacher, batch)
    assert zero.requires_grad and zero.item() == pytest.approx(0, abs=1e-8)
    with torch.no_grad():
        policy.network.dist_mu[0].bias.add_(0.2)
    loss = retention_loss(policy, teacher, batch)
    assert loss.item() > 0
    loss.backward()
    assert policy.network.dist_mu[0].bias.grad.abs().sum() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in policy.network.policy_head.parameters()
    )
    assert all(p.grad is None for p in policy.network.features_extractor.parameters())
    assert all(p.grad is None for p in teacher.network.parameters())
    features = policy.network.get_features(policy.tensors(batch, batched=True))
    distribution, detached_mu, detached_sigma = (
        policy.network.get_action_dist_from_features(features)
    )
    assert not detached_mu.requires_grad and not detached_sigma.requires_grad
    assert distribution.distribution.concentration1.requires_grad


def test_missing_retention_does_not_start_interaction(policy, env, monkeypatch):
    observation, _ = env.reset()
    banks = material(observation)
    banks["native"] = []
    monkeypatch.setattr(env, "reset", lambda **kw: pytest.fail("unexpected rollout"))
    with pytest.raises(ValueError, match="retention observations"):
        train(policy, env, steps=4, seed=1, settings={}, retention=banks)


def test_retention_micro_batches_preserve_the_logical_objective(env):
    observation, _ = env.reset()
    settings = {
        "learning_rate": 2.5e-5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "rollout_steps": 4,
        "epochs": 1,
        "batch_size": 4,
        "clip_range": 0.1,
        "value_coefficient": 0.5,
        "entropy_coefficient": 0.01,
    }
    policies = []
    for micro in (10, 64):
        policy = CaRLPolicy.load(Path("ads/carl/artifacts/official.pth"))
        train(
            policy,
            env,
            steps=8,
            seed=20260926,
            settings={**settings, "micro_batch_size": micro},
            retention=material(observation),
        )
        policies.append(policy)
    for name, expected in policies[0].network.state_dict().items():
        torch.testing.assert_close(
            expected, policies[1].network.state_dict()[name], rtol=1e-5, atol=1e-6
        )


def test_local_ppo_uses_new_steps_and_round_trips_training_state(policy, env, tmp_path):
    observation, _ = env.reset()
    banks = material(observation)
    before = {
        name: value.clone() for name, value in policy.network.state_dict().items()
    }
    settings = {
        "learning_rate": 2.5e-5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "rollout_steps": 4,
        "epochs": 1,
        "batch_size": 4,
        "micro_batch_size": 16,
        "clip_range": 0.1,
        "value_coefficient": 0.5,
        "entropy_coefficient": 0.01,
    }
    seen_modes = []
    hook = policy.network.features_extractor.register_forward_pre_hook(
        lambda module, inputs: seen_modes.append(module.training)
    )
    checkpoints = []

    def checkpoint(model, steps):
        target = tmp_path / f"{steps}.pth"
        model.save(target)
        checkpoints.append(target)

    report = train(
        policy,
        env,
        steps=8,
        seed=20260926,
        settings=settings,
        retention=banks,
        checkpoint=checkpoint,
    )
    hook.remove()
    assert report["actual_steps"] == 8 and report["optimizer_updates"] == 2
    assert sum(report["event_decisions"].values()) == 8
    assert set(report["event_decisions"]) <= {i.event.event_id for i in env.instances}
    assert report["retention_observations"] == {"native": 64, "keep": 64}
    assert report["frozen_state_identical"]
    assert seen_modes and not any(seen_modes)
    names = set(report["trainable_parameters"])
    assert any(
        not torch.equal(before[name], value)
        for name, value in policy.network.state_dict().items()
        if name in names
    )
    assert all(
        torch.equal(before[name], value)
        for name, value in policy.network.state_dict().items()
        if name not in names
    )
    restored = CaRLPolicy.load(checkpoints[0])
    assert restored.repair_state == policy.repair_state
    assert restored.repair_state["decision_steps"] == 8
    assert len(restored.optimizer_state["param_groups"][0]["params"]) == len(names)
    assert max(int(s["step"]) for s in restored.optimizer_state["state"].values()) == 2
    continued = train(
        restored,
        env,
        steps=4,
        seed=20260927,
        settings=settings,
        retention=banks,
    )
    assert continued["actual_steps"] == 4
    assert max(int(s["step"]) for s in restored.optimizer_state["state"].values()) == 3
    native = tmp_path / "model_best.pth"
    restored.export_native(native)
    readback = CaRLPolicy.load(native)
    assert all(
        torch.equal(value, readback.network.state_dict()[name])
        for name, value in restored.network.state_dict().items()
    )
    np.testing.assert_array_equal(
        restored.predict(observation)[0], readback.predict(observation)[0]
    )
