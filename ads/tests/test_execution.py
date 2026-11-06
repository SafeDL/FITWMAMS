from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch
from gymnasium.utils.env_checker import check_env

from ads.common.environment import PolicyEnvironment
from ads.common.evaluation import WorldEvaluator
from ads.common.expert import (
    coarse_candidates,
    refine_candidates,
)
from ads.common.repair import RepairEnvironment
from ads.nuplan.teacher import demonstration
from ads.tests.conftest import SmallWorld


def test_maneuver_adoption_trace_observes_after_handover_without_changing_motion(
    instance,
):
    class HandoverWorld(SmallWorld):
        def __init__(self):
            self.controller = SimpleNamespace(
                _maneuver_phase=torch.zeros((1, 6), dtype=torch.int64),
                _maneuver_cooldown_frames=torch.zeros((1, 6), dtype=torch.int64),
                plan_started=torch.zeros((1, 6), dtype=torch.bool),
            )

        def advance_response(self, action):
            result = super().advance_response(action)
            self.controller.plan_started[0, 0] = True
            self.controller._maneuver_phase[0, 0] = 2
            result["takeover_started"] = self.controller.plan_started.clone()
            result["takeover_owned"] = self.controller.plan_started.clone()
            return result

    worlds = []
    for traced in (False, True):
        env = PolicyEnvironment("carl", [instance], lambda event: HandoverWorld())
        env.record_trace = traced
        env.reset()
        env.step(np.zeros(2, np.float32))
        worlds.append(env)
    np.testing.assert_array_equal(worlds[0]._states(), worlds[1]._states())
    assert torch.equal(
        worlds[0].world.controller.plan_started, worlds[1].world.controller.plan_started
    )
    first = worlds[1].physics_diagnostics[0]
    assert first["input_maneuver_phase"][0] == 0
    assert first["input_maneuver_adopted"][0]


class Idle:
    def predict(self, observation, deterministic=True):
        path = np.zeros((80, 3), np.float32)
        path[:, 0] = np.arange(1, 81) * float(observation["scene"].states[0, 2]) * 0.1
        return path, None


def test_shared_execution_replays_all_149_physics_steps(instance, world_factory, evt):
    evaluate = WorldEvaluator(Idle(), "diffusion_planner", world_factory, evt)
    first, second = evaluate(instance), evaluate(instance)
    np.testing.assert_array_equal(first.states, second.states)
    assert len(first.states) == 150
    assert first.record["physics_steps"] == 149
    assert first.record["safety_failure"] is False


def test_initial_collision_is_retained_and_never_labelled_success(
    instance, world_factory, evt
):
    states = instance.event.initial_states.copy()
    states[1, 0] = 1
    event = replace(instance.event, initial_states=states)
    defective = replace(instance, event=event)
    defective = replace(
        defective, event=replace(event, event_id=defective.instance_hash)
    )
    result = WorldEvaluator(Idle(), "diffusion_planner", world_factory, evt)(defective)
    assert result.failure and result.record["initial_safety_failure"]
    assert result.record["physics_steps"] == 0
    env = PolicyEnvironment("diffusion_planner", [defective], world_factory)
    env.reset()
    _, _, terminated, _, info = env.step(1)
    assert terminated and info["safety_failure"]


def test_causal_encoders_and_native_action_spaces(instance, world_factory):
    from ads.tasks import POLICY_SUFFIX

    for algorithm in POLICY_SUFFIX:
        env = PolicyEnvironment(algorithm, [instance], world_factory)
        observation, _ = env.reset()
        assert env.observation_space.contains(observation)
        if algorithm == "carl":
            check_env(env, skip_render_check=True)
        else:
            action, _ = Idle().predict(observation)
            assert env.action_space.contains(action)
            env.step(action)
            assert len(env.physics_controls) == 5
        assert not any(
            k in observation for k in ("soft_reference", "exogenous_state", "evt_score")
        )
    env = PolicyEnvironment("diffusion_planner", [instance], world_factory)
    observation, _ = env.reset()
    action, _ = Idle().predict(observation)
    env.step(action)
    assert len(env.physics_controls) == 5


def test_empty_history_and_no_new_failures_transfer_to_normal(instance, world_factory):
    env = RepairEnvironment(
        "diffusion_planner",
        {"normal": [(instance, "normal")]},
        world_factory,
        round_index=1,
    )
    np.testing.assert_array_equal(env.weights, (1, 0, 0))
    env.reset()
    assert env.category_counts["normal"] == 1


def test_expert_parameter_budget_and_real_future_labels(instance, world_factory):
    coarse = coarse_candidates(instance.event)
    refined = refine_candidates(coarse[:8])
    assert len(coarse) <= 64 and len(refined) <= 64
    assert all(0 <= c.speed <= 45 and c.duration > 0 for c in coarse + refined)
    env = PolicyEnvironment("diffusion_planner", [instance], world_factory)
    observation, _ = env.reset()
    done = False
    while not done:
        action, _ = Idle().predict(observation)
        observation, _, term, trunc, _ = env.step(action)
        done = term or trunc
    samples = demonstration(
        instance, np.asarray(env.physics_states), "diffusion_planner"
    )
    assert len(samples) == 30
    assert samples[0]["target"].shape[1:] == (80, 3)
    assert samples[0]["future_valid"][0].sum() == 59
    assert not samples[0]["future_valid"][0, 59:].any()


def test_stalling_is_not_safety_failure(instance, world_factory, evt):
    states = instance.event.initial_states.copy()
    states[0, 2:] = 0
    event = replace(instance.event, initial_states=states)
    stopped = replace(instance, event=event)
    stopped = replace(stopped, event=replace(event, event_id=stopped.instance_hash))
    result = WorldEvaluator(Idle(), "diffusion_planner", world_factory, evt)(stopped)
    assert result.record["stalled"] and not result.failure


def test_ego_npc_collision_is_not_reported_as_npc_npc_collision(instance, evt):
    import torch
    from conftest import SmallWorld

    class EgoCollisionWorld(SmallWorld):
        def advance_response(self, action):
            result = super().advance_response(action)
            result["collision"][:] = True
            result["crashed"][:, :2] = True
            result["collision_pairs"] = torch.zeros((1, 7, 7), dtype=torch.bool)
            result["collision_pairs"][:, 0, 1] = True
            result["collision_pairs"][:, 1, 0] = True
            return result

    result = WorldEvaluator(
        Idle(), "diffusion_planner", lambda e: EgoCollisionWorld(), evt
    )(instance)
    assert result.failure and not result.record["npc_collision"]
