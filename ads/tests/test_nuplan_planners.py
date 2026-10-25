"""Published-weight inference on synthetic states only; no scenario evaluation."""

import gc
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ads.common.control import NuplanTrajectoryController
from ads.common.environment import PolicyEnvironment
from ads.common.randomness import PolicyRandomContext
from ads.common.setup import load_policy
from ads.nuplan.assets import official_path
from ads.nuplan.features import build_features, sample_history
from ads.nuplan.losses import OBJECTIVES, dtpp_candidates
from ads.nuplan.teacher import demonstration
from ads.tasks import NATIVE_PLANNERS


@pytest.fixture(params=NATIVE_PLANNERS)
def planner(request):
    torch.set_num_threads(1)
    device = "cuda" if request.param == "str2" and torch.cuda.is_available() else "cpu"
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


def test_dtpp_batch_masks_keep_scenes_independent(instance):
    from ads.nuplan.policy import collate

    torch.set_num_threads(1)
    policy = load_policy("dtpp", official_path("dtpp"), device="cpu")
    policy.network.eval()
    states = np.tile(instance.event.initial_states, (150, 1, 1))
    states[:, :, :2] += np.arange(150)[:, None, None] * 0.04 * states[:, :, 2:4]
    sample = demonstration(instance, states, "dtpp")[0]
    features = [deepcopy(sample["features"]) for _ in range(2)]
    features[1]["neighbor_agents_past"][:] = 0
    bank = np.zeros((2, 30, 80, 6), np.float32)
    bank[:, :, :, :3] = sample["target"][0]
    bank[:, :, :, 3] = 20
    bank[:, :, :, 1] += np.arange(30)[None, :, None] * 0.02
    bank[:, :, 59:] = 0
    bank[1, 10:] = 0
    inputs = collate(features, "cpu")
    with torch.no_grad():
        encoded = policy.network["encoder"](inputs)
        decoder = policy.network["decoder"]
        assert decoder.batch_mask_layout == "batch_then_head"
        single = decoder(
            {key: value[:1] for key, value in encoded.items()},
            torch.as_tensor(bank[:1]),
            inputs["neighbor_agents_past"][:1],
            59,
        )
        batched = decoder(
            encoded, torch.as_tensor(bank), inputs["neighbor_agents_past"], 59
        )
    for expected, actual in zip(single, batched):
        torch.testing.assert_close(expected[0], actual[0], rtol=1e-5, atol=1e-4)


def test_dtpp_training_rejects_an_old_loaded_decoder():
    from ads.nuplan.train import train

    policy = SimpleNamespace(algorithm="dtpp", network={"decoder": object()})
    banks = {"normal": [{"future_valid": np.ones((1, 80), bool)}]}
    with pytest.raises(RuntimeError, match="fresh training worker"):
        train(policy, banks, updates=1, seed=20260926)


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
    export = tmp_path / ("native" if planner.algorithm == "str2" else "native.pth")
    with pytest.raises(ValueError, match="official artifacts"):
        planner.export_native(
            official_path(planner.algorithm).with_name("repaired.pth")
        )
    planner.export_native(export)
    restored = type(planner).load(export, device=planner.device)
    assert (
        len(planner.network.state_dict())
        == {"diffusion_planner": 276, "dtpp": 117, "flow_planner": 338, "str2": 1222}[
            planner.algorithm
        ]
    )
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


def test_flow_native_loader_reads_exported_device(
    instance, world_factory, tmp_path, monkeypatch
):
    from nuplan.planning.simulation.trajectory.trajectory_sampling import (
        TrajectorySampling,
    )

    from ads.flow_planner.vendor.flow_planner.planner import FlowPlanner

    torch.set_num_threads(1)
    policy = load_policy("flow_planner", official_path("flow_planner"), device="cpu")
    monkeypatch.syspath_prepend(str(Path("ads/flow_planner/vendor").resolve()))
    export = tmp_path / "model.pth"
    policy.export_native(export)
    native = FlowPlanner(
        str(tmp_path / "model_config.yaml"),
        str(export),
        TrajectorySampling(num_poses=20, time_horizon=2),
        TrajectorySampling(num_poses=80, time_horizon=8),
        device="cpu",
    )
    native.initialize(SimpleNamespace(map_api=None, route_roadblock_ids=[]))
    for key, value in policy.network.state_dict().items():
        torch.testing.assert_close(
            value, native._planner.state_dict()[key], rtol=0, atol=0
        )
    env = PolicyEnvironment("flow_planner", [instance], world_factory)
    observation, _ = env.reset()
    context = PolicyRandomContext("flow_planner")
    policy.reset(context)
    expected, _ = policy.predict(observation)
    with torch.no_grad(), context.step(0, "cpu", use="flow_planner_native_inference"):
        output = native.core.inference(
            native._planner,
            policy.inputs([observation["features"]]),
            use_cfg=native.use_cfg,
            cfg_weight=native.cfg_weight,
        )
    values = output[0, 0].numpy()
    actual = np.column_stack((values[:, :2], np.arctan2(values[:, 3], values[:, 2])))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_masks_apply_to_native_training_inputs_and_objectives(planner, instance):
    states = np.tile(instance.event.initial_states, (150, 1, 1))
    states[:, :, :2] += np.arange(150)[:, None, None] * 0.04 * states[:, :, 2:4]
    samples = demonstration(instance, states, planner.algorithm)[:1]
    assert samples[0]["future_valid"][0].sum() == 59
    assert not samples[0]["future_valid"][:, 59:].any()
    if planner.algorithm == "dtpp":
        samples[0]["dtpp_candidates"] = dtpp_candidates(planner, samples[0])
    changed = deepcopy(samples)
    changed[0]["target"][:, 59:] = 100000
    planner.network.eval()
    # DP's native decoder selects its training path from module.training.
    if planner.algorithm == "diffusion_planner":
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
    from ads.nuplan.policy import collate

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
    if planner.algorithm == "dtpp":
        planner.network.eval()
        sample["dtpp_candidates"] = dtpp_candidates(planner, sample)
    planner.network.train()
    for module in planner.network.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    with torch.no_grad():
        torch.manual_seed(20260926)
        actual = OBJECTIVES[planner.algorithm](planner, [sample])
        torch.manual_seed(20260926)
        if planner.algorithm == "diffusion_planner":
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
        elif planner.algorithm == "flow_planner":
            _, losses = planner.network(
                planner.inputs(
                    [sample["features"]],
                    future=torch.as_tensor(target[None, 0], device=planner.device),
                ),
                mode="train",
            )
            expected = sum(
                float(planner.native_config.core[name]) * losses[name]
                for name in ("ego_planning_loss", "consistency_loss")
            )
        elif planner.algorithm == "dtpp":
            from ads.dtpp.vendor.train_utils import calc_loss

            inputs = collate([sample["features"]], planner.device)
            encoded = planner.network["encoder"](inputs)
            value, valid = targets([sample], planner.device, 11)
            expected = value.sum() * 0
            for stage, horizon, coefficient in ((0, 30, 1.0), (1, 80, 0.2)):
                bank = torch.as_tensor(
                    sample["dtpp_candidates"][stage][None], device=planner.device
                )
                neighbors, scores, ego, weights = planner.network["decoder"](
                    encoded, bank, inputs["neighbor_agents_past"], horizon
                )
                expected += coefficient * calc_loss(
                    neighbors[..., :horizon, :],
                    bank[:, :, :horizon],
                    ego[:, :horizon],
                    scores,
                    weights,
                    value[:, 0, :horizon],
                    value[:, 1:, :horizon],
                    valid[:, 1:, :horizon, None].expand(-1, -1, -1, 3),
                )
        else:
            value, _ = targets([sample], planner.device, 1)
            label = torch.nn.functional.pad(value[:, 0, :, :2], (0, 2))
            label[..., 3] = value[:, 0, :, 2]
            expected = planner.network(
                **planner.inputs([sample["features"]], target=label)
            ).loss
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_native_supervision_has_finite_nonzero_gradients(planner, instance):
    states = np.tile(instance.event.initial_states, (150, 1, 1))
    states[:, :, :2] += np.arange(150)[:, None, None] * 0.04 * states[:, :, 2:4]
    sample = demonstration(instance, states, planner.algorithm)[0]
    if planner.algorithm == "dtpp":
        sample["dtpp_candidates"] = dtpp_candidates(planner, sample)
    planner.network.train()
    if planner.algorithm == "str2":
        planner.network.transformer.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        planner.network.transformer.config.use_cache = False
    for module in planner.network.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    torch.manual_seed(20260926)
    # Validate differentiation without any optimizer step or checkpoint update.
    with torch.autocast(
        device_type=planner.device.type,
        dtype=torch.bfloat16,
        enabled=planner.algorithm == "str2" and planner.device.type == "cuda",
    ):
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


def test_nuplan_ego_dynamics_are_longitudinal_lateral_not_world_xy(
    instance, world_factory
):
    from ads.nuplan.scene import ego_state

    env = PolicyEnvironment("dtpp", [instance], world_factory)
    env.reset()
    scene = env.causal_scene()
    state = scene.states[0].copy()
    state[2:6] = (0, 20, -2, 1)
    native = ego_state(scene, state, 0)
    np.testing.assert_allclose(
        native.dynamic_car_state.rear_axle_velocity_2d.array, [20, 0], atol=1e-6
    )
    np.testing.assert_allclose(
        native.dynamic_car_state.rear_axle_acceleration_2d.array, [1, 2], atol=1e-6
    )


def test_dtpp_highway_speed_keeps_the_full_two_stage_horizon(instance, world_factory):
    torch.set_num_threads(1)
    states = instance.event.initial_states.copy()
    states[0, 2] = 40
    fast = replace(instance, event=replace(instance.event, initial_states=states))
    fast = replace(fast, event=replace(fast.event, event_id=fast.instance_hash))
    observation, _ = PolicyEnvironment("dtpp", [fast], world_factory).reset()
    policy = load_policy("dtpp", official_path("dtpp"), device="cpu")
    trajectory, _ = policy.predict(observation)
    assert trajectory.shape == (80, 3) and np.isfinite(trajectory).all()
    assert policy.tree.first_stage_candidates
    assert policy.tree.second_stage_candidates
    assert policy.tree.max_path_len >= 8 * states[0, 2]


def test_divided_road_geometry_does_not_fill_the_median():
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    from ads.nuplan.scene import HighwayMap
    from hierarchical_world_model.src.scene_geometry import SceneGeometry

    geometry = SceneGeometry.straight_template(
        lane_centers_y=np.array([0, 4, 20, 24]), lane_widths=4, x_min=-100, x_max=100
    )
    mapped = HighwayMap(geometry)
    assert len(mapped.blocks) == 2
    assert not mapped.is_in_layer(Point2D(0, 12), SemanticMapLayer.DRIVABLE_AREA)
    assert all(block.polygon.geom_type == "Polygon" for block in mapped.blocks)


@pytest.mark.parametrize("algorithm", ["diffusion_planner", "flow_planner", "dtpp"])
def test_straight_road_continues_past_the_cached_polyline_endpoint(
    algorithm, instance, world_factory
):
    from ads.nuplan.scene import map_for_scene

    event = replace(instance.event, initial_states=instance.event.initial_states.copy())
    event.initial_states[:, 0] += 300
    moved = replace(instance, event=event)
    moved = replace(moved, event=replace(event, event_id=moved.instance_hash))
    observation, _ = PolicyEnvironment(algorithm, [moved], world_factory).reset()
    lane_key = "map_lanes" if algorithm == "dtpp" else "lanes"
    assert observation["features"][lane_key][..., 0].max() > 50
    mapped = map_for_scene(observation["scene"])
    assert min(lane.baseline_path.discrete_path[-1].x for lane in mapped.lanes) > 660
