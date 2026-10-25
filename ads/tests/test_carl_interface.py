import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from gymnasium import spaces

from ads.carl.export import export_policy
from ads.carl.policy import CaRLPolicy, observation_space
from ads.carl.vendor.config import GlobalConfig
from ads.carl.vendor.model import PPOPolicy
from ads.common.control import (
    CARL_MAX_STEERING,
    ContinuousController,
    carl_action_commands,
)
from ads.common.environment import PolicyEnvironment
from ads.common.observations import carl_observation
from ads.tasks import training_budget


def test_carl_raster_matches_official_forward_right_and_route_roadblock(
    instance, world_factory
):
    env = PolicyEnvironment("carl", [instance], world_factory)
    observation, _ = env.reset()
    bev = observation["bev_semantics"]
    assert bev[8, 128, 100] > 0
    assert bev[8, 156, 128] == 0
    np.testing.assert_array_equal(bev[0], bev[1])
    assert bev[1, 136, 100] == 255  # Adjacent lane is in the same route roadblock.
    assert bev[5, 128, 120] == 255
    assert bev[0, 130, 120] == 255 and bev[5, 130, 120] == 0
    assert not bev[[3, 4, 7]].any()

    scene = env.causal_scene()
    states = scene.states.copy()
    states[1, :2] = (20, 4)
    changed = carl_observation(replace(scene, states=states))["bev_semantics"]
    assert changed[6, 136, 140] > 0  # 20 m forward, 4 m left in official axes.
    assert changed[6, 120, 140] == 0


@pytest.mark.parametrize(
    "action,steering,speed,acceleration,expected_acceleration",
    [
        ((1, 0.25), 0.1, 20, 0, 2.4),
        ((-1, -0.25), -0.1, 20, 0, -3.2),
        ((-1, 0), 0, 0.1, -0.4, 0.2),
    ],
)
def test_carl_default_action_scaling(
    action, steering, speed, acceleration, expected_acceleration
):
    result_acceleration, rate = carl_action_commands(
        action, steering=steering, speed=speed, acceleration=acceleration
    )
    assert result_acceleration == pytest.approx(expected_acceleration)
    assert steering + 0.1 * rate == pytest.approx(action[1] * CARL_MAX_STEERING)


def test_carl_actuator_filters_limits_and_stationary_steering(instance):
    controller = ContinuousController(instance.event)
    controller.update_target((1, 0.1))
    control = controller.control(instance.event.initial_states[0])
    assert control[0] == pytest.approx(2.4 * 0.04 / 0.24)
    assert 0 < controller.steering < CARL_MAX_STEERING * 0.1
    assert controller.steering_rate == pytest.approx(controller.steering / 0.04)
    controller.update_target((1, 1))
    for _ in range(20):
        control = controller.control(instance.event.initial_states[0])
        assert abs(control[1]) <= 0.600001

    state = instance.event.initial_states[0].copy()
    state[2:] = 0
    controller = ContinuousController(instance.event)
    controller.update_target((0, 0.1))
    assert controller.control(state)[1] == 0
    assert controller.steering > 0  # Wheels can turn without forward motion.


def test_carl_observation_uses_executed_steering_and_resets(instance, world_factory):
    env = PolicyEnvironment("carl", [instance], world_factory)
    first, _ = env.reset()
    observation, *_ = env.step(np.asarray((0, 0.02), np.float32))
    assert observation["measurements"][6] == pytest.approx(env.controller.steering)
    assert observation["measurements"][7] == pytest.approx(env.controller.steering_rate)
    again, _ = env.reset()
    for key in first:
        np.testing.assert_array_equal(first[key], again[key])


def test_native_export_preserves_parameters_and_actions(
    instance, world_factory, tmp_path
):
    torch.set_num_threads(1)
    policy = CaRLPolicy(
        PPOPolicy(
            observation_space(),
            spaces.Box(-1, 1, (2,), dtype=np.float32),
            config=GlobalConfig(),
        )
    )
    checkpoint, output = tmp_path / "training.pth", tmp_path / "native.pth"
    policy.save(checkpoint)
    export_policy(checkpoint, output)
    state = torch.load(output, weights_only=True)
    assert all(torch.is_tensor(v) for v in state.values())
    restored = CaRLPolicy.load(output)
    for key, value in policy.network.state_dict().items():
        torch.testing.assert_close(value.cpu(), state[key], rtol=0, atol=0)
    observation, _ = PolicyEnvironment("carl", [instance], world_factory).reset()
    np.testing.assert_array_equal(
        policy.predict(observation)[0], restored.predict(observation)[0]
    )


def test_new_protocol_has_one_seed_and_no_carl_adaptation():
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/risk_stratified_repair.yaml").read_text()
    )
    assert config["seeds"] == [20260926]
    assert config["search"]["budget"] == 5000
    assert config["rounds"] == 2
    assert training_budget(config, "carl", baseline=True) == ("actual_steps", 0)
    assert training_budget(config, "carl", baseline=False) == ("actual_steps", 20480)


def test_carl_initialization_cannot_call_optimizer_or_world(tmp_path, monkeypatch):
    from ads.risk_stratified_repair import Campaign

    class UnchangedPolicy:
        def save(self, path):
            Path(path).write_bytes(b"unchanged weights")

    def forbidden(*args, **kwargs):
        pytest.fail(
            "initializing the native policy must not train or execute the world"
        )

    monkeypatch.setattr(
        "ads.risk_stratified_repair.load_policy", lambda *a, **k: UnchangedPolicy()
    )
    monkeypatch.setattr(
        "ads.risk_stratified_repair.persist_curriculum", lambda *a, **k: "manifest"
    )
    monkeypatch.setattr("ads.carl.train.train", forbidden)
    campaign = object.__new__(Campaign)
    campaign.config = {"run_kind": "formal"}
    campaign.output = tmp_path / "campaign"
    campaign.device = "cpu"
    campaign.factory = forbidden
    campaign.ledger = None
    campaign.protocol_hash = "protocol"
    output = campaign.baseline("carl", 20260926)
    assert output.read_bytes() == b"unchanged weights"
    metadata = json.loads((output.parent / "training.json").read_text())
    assert metadata["learning"]["actual_steps"] == 0
    assert metadata["learning"]["gradient_updates"] == 0
    assert metadata["baseline"] is True
    assert metadata["source_path"].endswith("ads/carl/artifacts/official.pth")


def test_selected_official_carl_checkpoint_matches_source_and_loads(
    instance, world_factory
):
    from ads.common.instances import file_hash

    source = Path(
        "/media/hp/DATA/nuplan_workspace/CaRL/nuPlan/checkpoints/nuplan_51892_1B/model_best.pth"
    )
    local = Path("ads/carl/artifacts/official.pth")
    assert (
        file_hash(local)
        == file_hash(source)
        == "b3d7c9f687743dd3a9b2377b177ada37dc4ae7c9a563a853c18b3158b8d2954d"
    )
    policy = CaRLPolicy.load(local)
    observation, _ = PolicyEnvironment("carl", [instance], world_factory).reset()
    action, _ = policy.predict(observation)
    assert np.isfinite(action).all() and action.shape == (2,)
