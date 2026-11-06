"""Published-weight inference on synthetic states only; no scenario evaluation."""

import gc
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from ads.common.control import NuplanTrajectoryController
from ads.common.environment import PolicyEnvironment
from ads.common.randomness import PolicyRandomContext
from ads.common.setup import load_policy
from ads.nuplan.assets import official_path
from ads.nuplan.features import build_features, sample_history
from ads.nuplan.losses import OBJECTIVES
from ads.nuplan.teacher import demonstration
from ads.tasks import NATIVE_PLANNERS


@pytest.fixture(params=NATIVE_PLANNERS)
def planner(request):
    torch.set_num_threads(1)
    device = "cpu"
    policy = load_policy(request.param, official_path(request.param), device=device)
    yield policy
    del policy
    gc.collect()
    torch.cuda.empty_cache()


def test_history_is_causal_and_interpolates_25_to_10_hz():
    states = np.zeros((101, 7, 6))
    states[:, :, 0] = np.arange(101)[:, None] * 0.4
    history, valid = sample_history(states, 5)
    np.testing.assert_array_equal(np.flatnonzero(valid[:, 0]), [18, 19, 20])
    np.testing.assert_allclose(history[valid[:, 0], 0, 0], [0, 1, 2])
    changed = states.copy()
    changed[6:] = 999
    np.testing.assert_array_equal(sample_history(changed, 5)[0], history)


def test_official_weights_inference_export_and_reload(
    planner, instance, world_factory, tmp_path
):
    env = PolicyEnvironment(planner.algorithm, [instance], world_factory)
    observation, _ = env.reset()
    assert env.observation_space.contains(observation)
    context = PolicyRandomContext(planner.algorithm)
    planner.reset(context)
    first, _ = planner.predict(observation)
    assert first.shape == (80, 3) and np.isfinite(first).all()
    export = tmp_path / "native.pth"
    with pytest.raises(ValueError, match="official artifacts"):
        planner.export_native(
            official_path(planner.algorithm).with_name("repaired.pth")
        )
    planner.export_native(export)
    restored = type(planner).load(export, device=planner.device)
    assert len(planner.network.state_dict()) == 276
    for key, value in planner.network.state_dict().items():
        torch.testing.assert_close(
            value, restored.network.state_dict()[key], rtol=0, atol=0
        )
    restored.reset(context)
    second, _ = restored.predict(observation)
    np.testing.assert_allclose(first, second, rtol=1e-5, atol=1e-5)
    env.step(first)
    assert len(env.physics_controls) == 5
    del restored


def test_masks_apply_to_native_training_inputs_and_objectives(planner, instance):
    states = np.tile(instance.event.initial_states, (150, 1, 1))
    states[:, :, :2] += np.arange(150)[:, None, None] * 0.04 * states[:, :, 2:4]
    samples = demonstration(instance, states, planner.algorithm)[:1]
    assert samples[0]["future_valid"][0].sum() == 59
    assert not samples[0]["future_valid"][:, 59:].any()
    changed = deepcopy(samples)
    changed[0]["target"][:, 59:] = 100000
    planner.network.train()
    with torch.no_grad():
        torch.manual_seed(23)
        first = OBJECTIVES[planner.algorithm](planner, samples)
        torch.manual_seed(23)
        second = OBJECTIVES[planner.algorithm](planner, changed)
    assert torch.isfinite(first)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_full_valid_objective_matches_published_training(
    planner, instance, world_factory
):
    from ads.nuplan.losses import targets

    env = PolicyEnvironment(planner.algorithm, [instance], world_factory)
    observation, _ = env.reset()
    count = len(observation.get("slots", [0, 1]))
    count = max(count, 2)
    target = np.zeros((count, 80, 3), np.float32)
    target[0, :, 0] = np.arange(1, 81) * 0.8
    target[1:, :, 0] = 35 + np.arange(1, 81) * 0.7
    sample = {
        "features": observation["features"],
        "observation": observation,
        "target": target,
        "future_valid": np.ones((count, 80), bool),
    }
    planner.network.train()
    for module in planner.network.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    with torch.no_grad():
        torch.manual_seed(20260926)
        actual = OBJECTIVES[planner.algorithm](planner, [sample])
        torch.manual_seed(20260926)
        from ads.diffusion_planner.vendor.diffusion_planner.loss import (
            diffusion_loss_func,
        )

        value, valid = targets([sample], planner.device, 11)
        future = torch.cat(
            (value[..., :2], value[..., 2:3].cos(), value[..., 2:3].sin()), -1
        )
        future = torch.where(valid[..., None], future, 0)
        losses, _ = diffusion_loss_func(
            planner.network,
            planner.inputs([sample["features"]]),
            planner.network.sde.marginal_prob,
            (future[:, 0], future[:, 1:], ~valid[:, 1:]),
            planner.args.state_normalizer,
            {},
            planner.args.diffusion_model_type,
        )
        expected = losses["ego_planning_loss"] + losses["neighbor_prediction_loss"]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_native_supervision_has_finite_nonzero_gradients(planner, instance):
    states = np.tile(instance.event.initial_states, (150, 1, 1))
    states[:, :, :2] += np.arange(150)[:, None, None] * 0.04 * states[:, :, 2:4]
    sample = demonstration(instance, states, planner.algorithm)[0]
    planner.network.train()
    for module in planner.network.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    torch.manual_seed(20260926)
    # Validate differentiation without any optimizer step or checkpoint update.
    loss = OBJECTIVES[planner.algorithm](planner, [sample])
    assert torch.isfinite(loss) and loss.requires_grad
    loss.backward()
    gradients = [p.grad for p in planner.network.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)
    assert any(torch.count_nonzero(value) > 0 for value in gradients)
    planner.network.zero_grad(set_to_none=True)


@pytest.mark.parametrize("algorithm", NATIVE_PLANNERS)
def test_causal_features_ignore_soft_future(algorithm, instance, world_factory):
    env = PolicyEnvironment(algorithm, [instance], world_factory)
    observation, _ = env.reset()
    event = replace(
        instance.event, soft_reference=np.full_like(instance.event.soft_reference, 999)
    )
    env.event = event
    other, _ = build_features(
        env.causal_scene(),
        observation["history"],
        observation["history_valid"],
        algorithm=algorithm,
    )
    for key in other:
        np.testing.assert_array_equal(observation["features"][key], other[key])
    assert not observation["history_valid"][:-1].any()


def test_trajectory_tracking_immediate_speed(instance):
    controller = NuplanTrajectoryController(instance.event)
    poses = np.zeros((80, 3))
    poses[:, 0] = np.arange(1, 81) * 2.0
    controller.update_target(poses)
    np.testing.assert_allclose(
        controller.path[1], instance.event.initial_states[0, :2] + [2, 0], atol=1e-6
    )
    np.testing.assert_allclose(
        controller.control(instance.event.initial_states[0]), [0, 0], atol=1e-5
    )
    poses[20:, 0] += 500
    controller.update_target(poses)
    np.testing.assert_allclose(
        controller.control(instance.event.initial_states[0]), [0, 0], atol=1e-5
    )


@pytest.mark.parametrize("algorithm", NATIVE_PLANNERS)
def test_missing_checkpoint_is_never_replaced_with_random_weights(algorithm, tmp_path):
    with pytest.raises(FileNotFoundError):
        load_policy(algorithm, tmp_path / "missing", device="cpu")


@pytest.mark.parametrize("algorithm", NATIVE_PLANNERS)
def test_export_entrypoint_preserves_official_configuration(algorithm, tmp_path):
    from ads.nuplan.export import export_policy

    with pytest.raises(ValueError, match="official artifact directory"):
        export_policy(
            algorithm,
            tmp_path / "repair.pth",
            official_path(algorithm).with_name("repaired.pth"),
        )


@pytest.mark.parametrize("algorithm", NATIVE_PLANNERS)
def test_straight_road_continues_past_the_cached_polyline_endpoint(
    algorithm, instance, world_factory
):
    event = replace(instance.event, initial_states=instance.event.initial_states.copy())
    event.initial_states[:, 0] += 300
    moved = replace(instance, event=event)
    moved = replace(moved, event=replace(event, event_id=moved.instance_hash))
    observation, _ = PolicyEnvironment(algorithm, [moved], world_factory).reset()
    assert observation["features"]["lanes"][..., 0].max() > 50
