"""The explicit post-Diffusion tail cannot rewrite its frozen prefix."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hierarchical_world_model.scripts.audit_online_late_lane_continuation import _tail_plan
from hierarchical_world_model.src.continuation import append_future_reference
from hierarchical_world_model.src.highway import HighwayEnvClosedLoopWorld
from hierarchical_world_model.src.model import DiffusionGuidedHiQR
from hierarchical_world_model.src.randomness import WorldExogenousState
from hierarchical_world_model.src.stochastic_drivers.online import OnlineMAIDMController


def test_tail_plan_continues_terminal_speed_and_lane_without_mutating_prefix() -> None:
    plan = np.zeros((1, 149, 6, 2), np.float32)
    plan[0, :, :, 0] = np.arange(149, dtype=np.float32)[:, None] * 0.8
    plan[0, :, :, 1] = np.arange(6, dtype=np.float32)[None] * 3.7
    prefix = plan.copy()
    tail = _tail_plan(plan, 75)
    np.testing.assert_array_equal(plan, prefix)
    assert tail.shape == (1, 75, 6, 2)
    terminal_step = plan[0, -1, 0, 0] - plan[0, -2, 0, 0]
    np.testing.assert_allclose(
        tail[0, :, 0, 0], plan[0, -1, 0, 0] + terminal_step * np.arange(1, 76),
        atol=2.0e-5,
    )
    np.testing.assert_array_equal(
        tail[0, :, :, 1], np.broadcast_to(plan[0, -1, :, 1], (75, 6))
    )


def test_future_reference_append_preserves_prefix_and_requires_boundary() -> None:
    prefix = torch.zeros(2, 149, 6, 2)
    world = SimpleNamespace(
        reference=prefix.clone(), states=torch.zeros(2, 7, 6),
        reference_index=149, response_agent_innovations=torch.zeros(2, 224, 7, 16),
        device=torch.device("cpu"),
    )
    tail = torch.ones(2, 75, 6, 2)
    append_future_reference(world, tail)
    torch.testing.assert_close(world.reference[:, :149], prefix)
    torch.testing.assert_close(world.reference[:, 149:], tail)
    with pytest.raises(ValueError, match="plan exhaustion"):
        append_future_reference(world, tail)


def test_future_reference_append_rejects_bad_tail_and_short_randomness() -> None:
    world = SimpleNamespace(
        reference=torch.zeros(1, 149, 6, 2), states=torch.zeros(1, 7, 6),
        reference_index=149, response_agent_innovations=torch.zeros(1, 200, 7, 16),
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="pre-sampled"):
        append_future_reference(world, torch.zeros(1, 75, 6, 2))
    with pytest.raises(ValueError, match="future_xy"):
        append_future_reference(world, torch.zeros(1, 75, 6, 3))
    bad = torch.zeros(1, 25, 6, 2)
    bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        append_future_reference(world, bad)
    assert world.reference.shape == (1, 149, 6, 2)


def test_world_snapshot_restores_plan_after_alternative_tail_branch() -> None:
    model = DiffusionGuidedHiQR()
    exogenous = WorldExogenousState.sample(
        1, seed=9, response_steps=3,
        scene_refresh_responses=model.cfg.scene_refresh_responses,
        scene_dim=model.cfg.scene_latent_dim,
        agent_dim=model.cfg.agent_latent_dim,
    )
    states = torch.zeros(1, 7, 6)
    states[0, 0, 2] = 20.0
    states[0, 1, 0] = 30.0
    states[0, 1, 2] = 20.0
    valid = torch.zeros(1, 7, dtype=torch.bool)
    valid[0, :2] = True
    reference = torch.zeros(1, 1, 6, 2)
    reference[0, 0, 0, 0] = 30.8
    maps = torch.zeros(1, 3, 8, 6)
    maps[0, :, :, 1] = torch.tensor([-3.6, 0.0, 3.6])[:, None]
    map_valid = torch.ones(1, 3, 8, dtype=torch.bool)
    theta = np.tile(np.array([33.0, 2.0, 1.2, 1.4, 2.0], np.float32), (6, 1))
    world = HighwayEnvClosedLoopWorld(
        model, device="cpu", controller=OnlineMAIDMController(theta)
    )
    world.reset(
        states, valid, reference, maps, map_valid,
        exogenous_state=exogenous,
        initial_history=states[:, None].expand(-1, 25, -1, -1).clone(),
        initial_history_valid=valid[:, None].expand(-1, 25, -1).clone(),
        deterministic_response=True,
    )
    ego = torch.zeros(1, 1, 2)
    world.advance_response(ego)
    boundary = world.snapshot()
    tail_a = torch.zeros(1, 2, 6, 2)
    tail_a[0, :, 0, 0] = torch.tensor([31.6, 32.4])
    tail_b = tail_a.clone()
    tail_b[0, :, 0, 0] += 10.0

    append_future_reference(world, tail_a)
    first = world.advance_response(ego)
    world.restore(boundary)
    torch.testing.assert_close(world.reference, reference)
    append_future_reference(world, tail_b)
    world.advance_response(ego)
    world.restore(boundary)
    torch.testing.assert_close(world.reference, reference)
    append_future_reference(world, tail_a)
    replay = world.advance_response(ego)
    torch.testing.assert_close(first["agent_states"], replay["agent_states"])
    torch.testing.assert_close(first["background_actions"], replay["background_actions"])
