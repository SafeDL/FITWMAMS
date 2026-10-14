"""Physical motion statistics preserve overflow, absent slots and common derivatives."""

import numpy as np
import pytest

from hierarchical_world_model.src.npc_experiments.figures import binned_density
from hierarchical_world_model.src.npc_experiments.metrics import (
    diagnostic_states,
    hellinger,
)
from hierarchical_world_model.src.npc_experiments.road import neighbors, traffic_statistics


def traffic():
    state = np.zeros((1, 7, 6), np.float32)
    state[0, :, 0] = [0, 60, -60, 30, -30, 30, -30]
    state[0, :, 1] = [0, 0, 0, 3.6, 3.6, -3.6, -3.6]
    state[..., 2] = 25
    valid = np.ones((1, 7), bool)
    dimensions = np.tile([4.8, 1.8], (1, 7, 1)).astype(np.float32)
    maps = np.zeros((1, 3, 8, 6), np.float32)
    maps[..., 0] = np.linspace(-200, 200, 8)
    maps[..., 1] = np.array([-3.6, 0, 3.6])[None, :, None]
    maps[..., 2] = 1
    maps[..., 4] = 3.6
    map_valid = np.ones((1, 3, 8), bool)
    return state, valid, dimensions, maps, map_valid


def test_hellinger_bounds_and_mass_not_density():
    assert hellinger([2, 8], [20, 80]) == pytest.approx(0)
    assert hellinger([1, 0], [0, 1]) == pytest.approx(1)
    assert hellinger([3, 7], [7, 3]) == hellinger([7, 3], [3, 7])


def test_display_density_keeps_overflow_mass_and_physical_bin_widths():
    edges = np.array([-np.inf, 0, 1, 3, np.inf])
    density = binned_density(np.array([1, 2, 4, 3]), edges)
    np.testing.assert_allclose(density, [0.2, 0.2])
    assert np.sum(density * np.diff(edges[1:-1])) == pytest.approx(0.6)


def test_display_density_does_not_smooth_into_adjacent_bins():
    edges = np.array([-np.inf, 0, 1, 2, 3, np.inf])
    density = binned_density(np.array([0, 0, 10, 0, 0]), edges)
    np.testing.assert_array_equal(density, [0, 1, 0])


def test_empty_slots_do_not_become_gap_observations():
    state, valid, dimensions, maps, _ = traffic()
    valid[:, 1] = False
    _, leader, gap, _, _ = neighbors(state, valid, dimensions[0], maps[0, :, 0, 1])
    assert leader[0, 1] == -1
    assert leader[0, 0] == -1
    assert leader[0, 2] == 0
    assert gap[0, 2] == pytest.approx(55.2, abs=1e-4)


def test_shared_motion_operator_preserves_positions_and_quadratic_dynamics():
    state, *_ = traffic()
    states = np.repeat(state[:, None], 150, axis=1)
    time = np.arange(150) * 0.04
    states[..., 0] += (25 * time + 0.5 * 0.3 * time**2)[None, :, None]
    derived = diagnostic_states(states, 25)
    np.testing.assert_array_equal(derived[..., :2], states[..., :2])
    np.testing.assert_allclose(derived[0, :, 0, 2], 25 + 0.3 * time, atol=1e-3)
    np.testing.assert_allclose(derived[0, :, 0, 4], 0.3, atol=2e-3)


def test_overflow_and_negative_gap_are_not_silently_dropped():
    state, valid, dimensions, maps, map_valid = traffic()
    state[0, 2, 0] = -1
    states = np.repeat(state[:, None], 150, axis=1)
    time = np.arange(150) * 0.04
    states[..., 0] += (70 * time + 0.5 * 5 * time**2)[None, :, None]
    states[..., 2] = (70 + 5 * time)[None, :, None]
    states[..., 4] = 5
    cohort = {
        "bounds": np.array([-200, 800]),
        "dimensions": dimensions[0],
        "lane_centers": np.array([-3.6, 0, 3.6]),
        "lane_widths": np.full(3, 3.6),
    }
    hist, exposure = traffic_statistics(
        states, np.repeat(valid[:, None], 150, axis=1), cohort
    )
    assert hist["speed"].sum() > 0
    assert hist["speed"][..., -1].sum() == hist["speed"].sum()
    assert hist["acceleration"][..., -1].sum() == hist["acceleration"].sum()
    assert hist["gap"][..., 0].sum() > 0
    assert exposure[..., 0].sum() > exposure[..., 5].sum()
