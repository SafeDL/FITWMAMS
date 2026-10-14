"""Contract checks for the generated HiQR + MA-IDM default interface."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hierarchical_world_model.src.composition import (
    HierarchicalWorldSampler,
    SampledWorldBatch,
)
from hierarchical_world_model.src.config import WorldModelConfig
from hierarchical_world_model.src.execution import _rollout_sample, rollout_world
from hierarchical_world_model.scripts.evaluate_generated_online_ads import (
    plan_diagnostics,
)
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
    rng = np.random.default_rng(19)
    innovations = rng.standard_normal((batch, 3, 7, 16)).astype(np.float32)
    return SimpleNamespace(
        agent_response_innovations=innovations,
        driver_latent=rng.standard_normal((batch, 6, 5)).astype(np.float32),
        maneuver_latent=rng.standard_normal((batch, 6, 4)).astype(np.float32),
        response_steps=3,
    )


def test_episode_parameters_replay_independently_of_batch_position() -> None:
    exogenous = _exogenous()
    together = sample_population_theta_from_world(DEFAULT_MA_IDM_POSTERIOR, exogenous)
    alone = sample_population_theta_from_world(
        DEFAULT_MA_IDM_POSTERIOR,
        SimpleNamespace(
            agent_response_innovations=exogenous.agent_response_innovations[1:],
            driver_latent=exogenous.driver_latent[1:],
        ),
    )
    np.testing.assert_array_equal(together[1:], alone)
    np.testing.assert_array_equal(
        together,
        sample_population_theta_from_world(DEFAULT_MA_IDM_POSTERIOR, exogenous),
    )
    assert np.isfinite(together).all()


def test_generated_plan_checks_the_composed_motion_and_keeps_sampled_dimensions():
    initial = np.zeros((2, 7, 6), np.float32)
    initial[..., 0] = np.arange(7) * 20
    initial[..., 2] = 20
    time = np.arange(1, 150, dtype=np.float32) * 0.04
    plan = initial[:, None, 1:, :2] + time[None, :, None, None] * np.array([20, 0])
    # The second condition accelerates strongly; it must not be repaired by the audit.
    plan[1, :, :, 0] += 0.5 * 20 * time[:, None] ** 2
    sample = SimpleNamespace(
        initial_states=initial,
        initial_valid=np.ones((2, 7), bool),
        soft_plan=plan.astype(np.float32),
        vehicle_dimensions=np.broadcast_to([4.8, 1.8], (2, 7, 2)).copy(),
    )
    original = sample.soft_plan.copy()
    result = plan_diagnostics(sample)
    assert result["plan_finite"].all()
    assert result["ax_outside_minus8_4_fraction"][0] == 0
    assert result["ax_outside_minus8_4_fraction"][1] > 0.9
    np.testing.assert_array_equal(sample.soft_plan, original)


def test_missing_driver_latent_does_not_silently_use_legacy_sampling() -> None:
    exogenous = _exogenous(batch=1)
    del exogenous.driver_latent
    with pytest.raises(AttributeError, match="driver_latent"):
        sample_population_theta_from_world(DEFAULT_MA_IDM_POSTERIOR, exogenous)
    assert np.isfinite(
        sample_population_theta_from_world(
            DEFAULT_MA_IDM_POSTERIOR, exogenous, use_driver_latent=False
        )
    ).all()


def test_create_world_uses_only_the_maintained_response_chain(monkeypatch):
    import hierarchical_world_model.src.composition as composition

    sample = SampledWorldBatch(
        scenario=None,
        initial_states=np.zeros((1, 7, 6), np.float32),
        initial_valid=np.ones((1, 7), bool),
        soft_plan=np.zeros((1, 3, 6, 2), np.float32),
        scenario_seed=0,
        motion_seed=0,
        response_seed=0,
        exogenous_state=_exogenous(batch=1),
        vehicle_dimensions=np.broadcast_to([4.8, 1.8], (1, 7, 2)).copy(),
    )

    class FakeWorld:
        def __init__(self, model, *, controller, **kwargs):
            self.controller = controller
            self.reference_recovery_enabled = kwargs["reference_recovery_enabled"]

        def reset(self, *args, **kwargs):
            self.reset_args = args
            self.reset_kwargs = kwargs

    monkeypatch.setattr(composition, "HighwayEnvClosedLoopWorld", FakeWorld)
    sampler = HierarchicalWorldSampler.__new__(HierarchicalWorldSampler)
    sampler.device = torch.device("cpu")
    sampler.generated_road = {"lane_centers_m": [-4.0, 0.0, 4.0], "lane_width_m": 4.0}
    sampler.response = object()
    sampler.selection = {
        "mobil": {"politeness": 0.0, "bias": -3.0, "slope": 0.1},
        "reference_recovery_enabled": True,
    }
    sampler.driver_population = {
        "log_mean": np.log([33, 2, 1.3, 1.2, 2]),
        "log_cov": np.eye(5) * 0.01,
    }
    world = sampler.create_world(sample)
    assert isinstance(world.controller, OnlineMAIDMController)
    for retired in ("lane_choice_model", "response_model", "residual_path_bank"):
        assert not hasattr(world.controller, retired)
    assert world.controller.autonomous_lane_changes_enabled
    assert world.controller.lane_changes_on_response_only
    assert not world.controller.mobil_parameter_variation
    assert world.controller.mobil_onset_calibration == (-3.0, 0.1)
    assert world.reference_recovery_enabled
    assert not any(key.startswith("nominal_") for key in world.reset_kwargs)
    assert world.reset_kwargs["vehicle_dimensions"].shape == (1, 7, 2)
    assert world.reset_args[4].any(-1).sum() == 3
    np.testing.assert_array_equal(world.reset_args[3][0, :3, 0, 1], [-4, 0, 4])
    with pytest.raises(ValueError, match="only controller='auto'"):
        sampler.create_world(sample, controller="auto_lane_choice")
    assert sampler.create_world(sample, controller=None).controller is None


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


def test_composed_sample_execution_uses_its_explicit_response_horizon(
    monkeypatch,
) -> None:
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
        vehicle_dimensions=np.broadcast_to(
            np.asarray((4.8, 1.8), np.float32), (1, 7, 2)
        ),
    )
    monkeypatch.setattr(
        execution,
        "trajectory_event_risk",
        lambda states, valid, **kwargs: np.zeros(len(states), np.float32),
    )
    result = _rollout_sample(
        FakeSampler(),
        sample,
        execution.hold_current_ego_action,
        reaction_controller=None,
    )
    assert result.states.shape == (1, 3, 7, 6)
    assert result.background_actions.shape == (1, 2, 6, 2)
    assert result.collision_pairs.shape == (1, 2, 7, 7)
    assert result.offroad.shape == (1, 2, 7)
