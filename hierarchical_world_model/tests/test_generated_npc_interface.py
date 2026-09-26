"""Contract checks for the generated HiQR + MA-IDM default interface."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from hierarchical_world_model.src.composition import (
    HierarchicalWorldSampler,
    SampledWorldBatch,
)
from hierarchical_world_model.src.config import WorldModelConfig
from hierarchical_world_model.src.execution import _rollout_sample, rollout_world
from hierarchical_world_model.src.stochastic_drivers.online import (
    DEFAULT_MA_IDM_POSTERIOR,
    OnlineMAIDMController,
    sample_population_theta_from_world,
)


def test_online_world_commits_and_redecides_every_25hz_tick() -> None:
    cfg = WorldModelConfig()
    assert cfg.execute_frames == 1
    assert cfg.dt_s == 0.04
    assert cfg.execute_frames * cfg.dt_s == 0.04


def _exogenous(batch: int = 2) -> SimpleNamespace:
    innovations = np.random.default_rng(19).standard_normal(
        (batch, 3, 7, 16)
    ).astype(np.float32)
    return SimpleNamespace(agent_response_innovations=innovations, response_steps=3)


def test_episode_parameters_replay_independently_of_batch_position() -> None:
    exogenous = _exogenous()
    together = sample_population_theta_from_world(DEFAULT_MA_IDM_POSTERIOR, exogenous)
    alone = sample_population_theta_from_world(
        DEFAULT_MA_IDM_POSTERIOR,
        SimpleNamespace(agent_response_innovations=exogenous.agent_response_innovations[1:]),
    )
    np.testing.assert_array_equal(together[1:], alone)
    np.testing.assert_array_equal(
        together, sample_population_theta_from_world(DEFAULT_MA_IDM_POSTERIOR, exogenous)
    )
    assert np.isfinite(together).all()


def test_create_world_defaults_to_single_pass_online_ma_idm(monkeypatch) -> None:
    import hierarchical_world_model.src.composition as composition
    import hierarchical_world_model.src.execution as execution

    exogenous = _exogenous(batch=1)
    initial = np.zeros((1, 7, 6), np.float32)
    initial[:, :, 2] = 20.0
    sample = SampledWorldBatch(
        scenario=None,
        initial_states=initial,
        initial_valid=np.ones((1, 7), bool),
        soft_plan=np.zeros((1, 3, 6, 2), np.float32),
        state_knot_reference=np.zeros((1, 3, 6, 2), np.float32),
        scenario_seed=0,
        motion_seed=0,
        response_seed=0,
        exogenous_state=exogenous,
    )
    def fail_nominal(*args, **kwargs):
        raise AssertionError("single-pass world must not pre-run a passive branch")

    class FakeWorld:
        def __init__(self, model, *, controller, **kwargs):
            self.controller = controller

        def reset(self, *args, **kwargs):
            self.reset_kwargs = kwargs

    monkeypatch.setattr(execution, "_rollout_sample", fail_nominal)
    monkeypatch.setattr(composition, "HighwayEnvClosedLoopWorld", FakeWorld)
    sampler = HierarchicalWorldSampler.__new__(HierarchicalWorldSampler)
    sampler.device = torch.device("cpu")
    sampler.response = object()
    sampler.ma_idm_posterior = DEFAULT_MA_IDM_POSTERIOR

    interactive = sampler.create_world(sample)
    assert isinstance(interactive.controller, OnlineMAIDMController)
    assert not any(key.startswith("nominal_") for key in interactive.reset_kwargs)


def test_public_rollout_passes_auto_controller_by_default(monkeypatch) -> None:
    import hierarchical_world_model.src.execution as execution

    sample = SimpleNamespace()
    exogenous = object()

    class FakeSampler:
        def compose_exogenous(self, received):
            assert received is exogenous
            return sample

    captured = []

    def fake_rollout(sampler, received, policy, **kwargs):
        assert received is sample
        captured.append(kwargs["reaction_controller"])
        return object()

    monkeypatch.setattr(execution, "_rollout_sample", fake_rollout)
    rollout_world(FakeSampler(), exogenous)
    rollout_world(FakeSampler(), exogenous, reaction_controller=None)
    assert captured == ["auto", None]


def test_composed_sample_execution_uses_its_explicit_response_horizon(monkeypatch) -> None:
    import hierarchical_world_model.src.execution as execution

    state = torch.zeros((1, 7, 6))

    class FakeWorld:
        device = torch.device("cpu")

        def observe(self):
            return {"agent_states": state}

        def advance_response(self, action):
            return {
                "agent_state_frames": state[:, None],
                "ego_actions": action,
                "background_actions": torch.zeros((1, 1, 6, 2)),
                "collision_pairs": torch.zeros((1, 7, 7), dtype=torch.bool),
                "crashed": torch.zeros((1, 7), dtype=torch.bool),
                "offroad": torch.zeros((1, 7), dtype=torch.bool),
            }

    class FakeSampler:
        def create_world(self, sample, **kwargs):
            assert kwargs["controller"] is None
            return FakeWorld()

    sample = SimpleNamespace(
        exogenous_state=SimpleNamespace(response_steps=2),
        soft_plan=np.zeros((1, 5, 6, 2), np.float32),
        initial_states=np.zeros((1, 7, 6), np.float32),
        initial_valid=np.ones((1, 7), bool),
    )
    monkeypatch.setattr(
        execution,
        "trajectory_event_risk",
        lambda states, valid, **kwargs: np.zeros(len(states), np.float32),
    )
    result = _rollout_sample(
        FakeSampler(), sample, execution.hold_current_ego_action,
        reaction_controller=None,
    )
    assert result.states.shape == (1, 3, 7, 6)
    assert result.background_actions.shape == (1, 2, 6, 2)
    assert result.collision_pairs.shape == (1, 2, 7, 7)
    assert result.offroad.shape == (1, 2, 7)
