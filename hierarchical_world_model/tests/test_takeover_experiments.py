"""Batching preserves native execution and inactive episodes remain on plan."""

import numpy as np
import torch
from hierarchical_world_model.src.shared_evaluation import (
    highway_rollout,
    scene_randomness,
)
from hierarchical_world_model.src.takeover import TakeoverController
from hierarchical_world_model.tests.traffic_context import THETA, context


def native_scenes():
    c = context()
    initial = c.current.numpy().repeat(2, 0)
    initial[0, 0, 0], initial[0, 0, 2] = 14, 12
    time = (np.arange(174) - 24) * 0.04
    states = np.broadcast_to(initial[:, None], (2, 174, 7, 6)).copy()
    states[..., :2] += initial[:, None, :, 2:4] * time[None, :, None, None]
    valid = np.ones((2, 174, 7), bool)
    valid[:, :, 2:] = False
    arrays = dict(
        agent_states=states,
        agent_valid=valid,
        vehicle_dimensions=np.broadcast_to([4.8, 1.8], (2, 7, 2)).copy(),
        map_polylines=c.map_polylines.numpy().repeat(2, 0),
        map_polyline_valid=c.map_polyline_valid.numpy().repeat(2, 0),
        sequence_id=np.array(["nat_01_1", "nat_01_2"]),
    )
    return arrays, np.arange(2), states[:, 25:174, 1:, :2].copy()


def test_native_batching_preserves_active_and_inactive_episodes():
    arrays, rows, plans = native_scenes()
    draws = scene_randomness(arrays["sequence_id"], seed=19)
    base = highway_rollout(arrays, rows, plans, draws)
    full = highway_rollout(
        arrays,
        rows,
        plans,
        draws,
        controller=TakeoverController(THETA[None].expand(2, -1, -1)),
    )
    subset = highway_rollout(
        arrays,
        rows[:1],
        plans[:1],
        draws.select(slice(0, 1)),
        controller=TakeoverController(THETA[None]),
    )
    np.testing.assert_array_equal(full["states"][:1], subset["states"])
    np.testing.assert_array_equal(full["states"][1:], base["states"][1:])
    assert full["takeover_started"][0].any()
    assert not full["takeover_started"][1].any()
