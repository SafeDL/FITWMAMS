from dataclasses import replace

import numpy as np
from gymnasium.utils.env_checker import check_env

from ads.common.environment import PolicyEnvironment
from ads.common.evaluation import WorldEvaluator
from ads.common.expert import (
    Candidate,
    TeacherExecution,
    coarse_candidates,
    refine_candidates,
)
from ads.common.repair import RepairEnvironment
from ads.plant2.demonstrations import demonstration


class Idle:
    def predict(self, observation, deterministic=True):
        return 1, None


def test_shared_execution_replays_all_149_physics_steps(instance, world_factory, evt):
    evaluate = WorldEvaluator(Idle(), "dqn", world_factory, evt)
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
    result = WorldEvaluator(Idle(), "dqn", world_factory, evt)(defective)
    assert result.failure and result.record["initial_safety_failure"]
    assert result.record["physics_steps"] == 0
    env = PolicyEnvironment("dqn", [defective], world_factory)
    env.reset()
    _, _, terminated, _, info = env.step(1)
    assert terminated and info["safety_failure"]


def test_causal_encoders_and_native_action_spaces(instance, world_factory):
    for algorithm in ("dqn", "ppo", "carl", "plant2"):
        env = PolicyEnvironment(algorithm, [instance], world_factory)
        observation, _ = env.reset()
        assert env.observation_space.contains(observation)
        check_env(env, skip_render_check=True)
    env = PolicyEnvironment("plant2", [instance], world_factory)
    observation, _ = env.reset()
    assert not any(
        k in observation for k in ("soft_reference", "exogenous_state", "evt_score")
    )
    assert observation["objects"].shape[1] == 7
    assert observation["route_original"].shape == (20, 2)
    action = (
        np.stack((np.arange(20), np.zeros(20)), -1),
        np.stack((np.arange(1, 9) * 5, np.zeros(8)), -1),
    )
    env.step(action)
    assert len(env.physics_controls) == 5


def test_empty_history_and_no_new_failures_transfer_to_normal(instance, world_factory):
    env = RepairEnvironment(
        "dqn", {"normal": [(instance, "normal")]}, world_factory, round_index=1
    )
    np.testing.assert_array_equal(env.weights, (1, 0, 0))
    env.reset()
    assert env.category_counts["normal"] == 1


def test_expert_parameter_budget_and_real_future_labels(instance, world_factory):
    from omegaconf import OmegaConf

    coarse = coarse_candidates(instance.event)
    refined = refine_candidates(coarse[:8])
    assert len(coarse) <= 64 and len(refined) <= 64
    assert all(0 <= c.speed <= 45 and c.duration > 0 for c in coarse + refined)
    env = PolicyEnvironment("dqn", [instance], world_factory)
    env.reset()
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(1)
        done = term or trunc
    cfg = OmegaConf.create(
        {
            "model": {
                "training": {"range": 50, "range_factor_front": 2},
                "pre_training": {
                    "precision_pos": 7,
                    "precision_angle": 5,
                    "precision_speed": 5,
                },
            }
        }
    )
    execution = TeacherExecution(
        Candidate(1, 20),
        True,
        1,
        100,
        np.asarray(env.physics_states),
        np.asarray(env.physics_controls),
    )
    samples, _ = demonstration(instance, execution, cfg)
    assert len(samples) == 20
    assert samples[0]["forecast"].shape == (1, 4)
    np.testing.assert_allclose(
        samples[0]["waypoints"][:, 0], np.arange(1, 9) * 5, atol=1e-3
    )
    assert samples[0]["path"].shape == (20, 2)


def test_stalling_is_not_safety_failure(instance, world_factory, evt):
    states = instance.event.initial_states.copy()
    states[0, 2:] = 0
    event = replace(instance.event, initial_states=states)
    stopped = replace(instance, event=event)
    stopped = replace(stopped, event=replace(event, event_id=stopped.instance_hash))
    result = WorldEvaluator(Idle(), "dqn", world_factory, evt)(stopped)
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

    result = WorldEvaluator(Idle(), "dqn", lambda e: EgoCollisionWorld(), evt)(instance)
    assert result.failure and not result.record["npc_collision"]


def test_agreed_existing_initial_policies_run_on_the_shared_synthetic_interface(
    instance, world_factory
):
    from pathlib import Path

    import torch

    from ads.common.setup import load_policy

    torch.set_num_threads(1)
    for algorithm, suffix in (("dqn", ".zip"), ("ppo", ".zip"), ("plant2", ".ckpt")):
        path = Path(
            f"results/risk_stratified_repair_history/formal/{algorithm}/20260926/baseline/policy{suffix}"
        )
        model = load_policy(algorithm, path, device="cpu")
        env = PolicyEnvironment(algorithm, [instance], world_factory)
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert env.action_space.contains(action)


def test_planT_spatial_labels_do_not_extend_past_executed_future(
    instance, world_factory
):
    from omegaconf import OmegaConf

    states = np.tile(instance.event.initial_states, (150, 1, 1))
    states[:, 0, 2] = 2
    states[:, :, :2] += np.arange(150)[:, None, None] * 0.04 * states[:, :, 2:4]
    cfg = OmegaConf.create(
        {
            "model": {
                "training": {"range": 50},
                "pre_training": {
                    "precision_pos": 7,
                    "precision_angle": 5,
                    "precision_speed": 5,
                },
            }
        }
    )
    execution = TeacherExecution(
        Candidate(1, 2), True, 1, 12, states, np.zeros((149, 2))
    )
    samples, _ = demonstration(instance, execution, cfg)
    assert samples[0]["path_valid"].sum() == 12
    assert not samples[0]["path_valid"][12:].any()
