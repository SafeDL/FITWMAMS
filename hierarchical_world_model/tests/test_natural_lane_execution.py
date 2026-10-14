"""Boundary continuity, Train-only style sampling and progress-based tracking."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from hierarchical_world_model.src.npc_experiments import lane_execution as execution
from hierarchical_world_model.src.stochastic_drivers import lane_fit as fitting
from hierarchical_world_model.src.stochastic_drivers.lane_trajectory import (
    MODEL_PATH,
    LearnedLaneTrajectory,
    path_at,
    path_coefficients,
    frenet_feedback,
    limit_realized_lateral_acceleration,
)
from hierarchical_world_model.src.core.dynamics import (
    DynamicsConfig,
    HIGHD_INTEGRATION_SPEED_MAX_MPS,
    KinematicTrafficDynamics,
)


def test_lane_path_matches_current_motion_and_settles_at_target():
    state = torch.tensor([12.0, 0.5, 25.0, 0.2, 0.2, 0.1])
    distance = torch.tensor(100.0)
    coefficients = path_coefficients(
        state, torch.tensor(3.9), distance, torch.tensor([0.08, -0.04])
    )
    start = path_at(coefficients, state[0], distance, state[0])
    finish = path_at(coefficients, state[0], distance, state[0] + distance)
    torch.testing.assert_close(start[0], state[1])
    torch.testing.assert_close(start[1], state[3] / state[2])
    torch.testing.assert_close(
        start[2], (state[5] * state[2] - state[3] * state[4]) / state[2] ** 3
    )
    torch.testing.assert_close(finish[0], torch.tensor(3.9), atol=2e-5, rtol=0)
    assert abs(float(finish[1])) < 1e-6
    assert abs(float(finish[2])) < 1e-6


def test_explicit_lane_model_does_not_read_the_published_file(tmp_path):
    report = json.loads(MODEL_PATH.read_text())
    loaded = LearnedLaneTrajectory(model=report, path=tmp_path / "absent.json")
    expected = LearnedLaneTrajectory()
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


@pytest.mark.parametrize(
    "speed, acceleration", [(10.0, 4.0), (25.0, -8.0), (100.0, 4.0)]
)
def test_lane_yaw_bounds_actual_next_normal_acceleration(speed, acceleration):
    state = torch.tensor([0, 0, speed, 0, 0, 1.0])
    raw = frenet_feedback(
        state,
        torch.tensor(0.0),
        torch.tensor(0.0),
        torch.tensor(0.02),
        frequency=1.2,
        acceleration_limit=1.1,
        jerk_limit=2.0,
    )
    bounded = limit_realized_lateral_acceleration(
        state,
        raw,
        torch.tensor(acceleration),
        acceleration_limit=1.1,
        jerk_limit=2.0,
    )
    dynamics = KinematicTrafficDynamics(
        DynamicsConfig(speed_max_mps=HIGHD_INTEGRATION_SPEED_MAX_MPS)
    )
    actual = dynamics.step(
        state,
        torch.stack((torch.tensor(acceleration), bounded)),
        torch.tensor(True),
        0.04,
    )
    heading = torch.atan2(actual[3], actual[2])
    normal = -actual[4] * heading.sin() + actual[5] * heading.cos()
    assert abs(float(normal)) <= 1.1 + 1e-6
    assert abs(float(normal - 1.0)) / 0.04 <= 2.0 + 1e-4


def test_stopped_lane_response_does_not_rotate_in_place():
    state = torch.zeros(6)
    yaw = limit_realized_lateral_acceleration(
        state,
        torch.tensor(1.0),
        torch.tensor(-8.0),
        acceleration_limit=1.1,
        jerk_limit=2.0,
    )
    assert yaw == 0


def test_style_is_persistent_stochastic_and_conditioned_on_dimensions():
    model = LearnedLaneTrajectory()
    state = torch.tensor([[0, 0, 25, 0, 0, 0]] * 3, dtype=torch.float32)
    dimensions = torch.tensor([[4.8, 1.8], [4.8, 1.8], [16, 2.5]])
    noise = torch.tensor([[-2.0, 0], [2.0, 0], [-2.0, 0]])
    first = model.parameters_for(state, dimensions, torch.full((3,), 3.6), noise)
    torch.testing.assert_close(
        first, model.parameters_for(state, dimensions, torch.full((3,), 3.6), noise)
    )
    assert not torch.allclose(first[0], first[1])
    assert not torch.allclose(first[0], first[2])


def test_spatial_path_slows_its_progress_with_longitudinal_braking():
    model = LearnedLaneTrajectory()
    state = torch.tensor([0, 0, 25, 0, 0, 0], dtype=torch.float32)
    coefficients, origin, distance = model.plan(
        state, torch.tensor([4.8, 1.8]), torch.tensor(3.6)
    )
    early = path_at(coefficients, origin, distance, distance * 0.1)[0]
    later = path_at(coefficients, origin, distance, distance * 0.6)[0]
    assert 0 <= early < later < 3.6
    stopped = state.clone()
    stopped[2] = 0
    assert abs(float(model.track(stopped, coefficients, origin, distance))) < 1e-6


def test_lane_shape_is_mirrored_for_left_and_right_motion():
    model = LearnedLaneTrajectory()
    state = torch.tensor([[0, 0, 25, 0.16, 0, 0.1], [0, 0, 25, -0.16, 0, -0.1]])
    dimensions = torch.tensor([[4.8, 1.8]] * 2)
    coefficients, _, distance = model.plan(state, dimensions, torch.tensor([3.6, -3.6]))
    torch.testing.assert_close(distance[0], distance[1])
    torch.testing.assert_close(coefficients[0], -coefficients[1])


def test_fitted_asset_uses_shared_train_events_and_val_selection():
    report = json.loads(MODEL_PATH.read_text())
    assert report["train_events"] == len(report["fit"]["residuals"])
    assert (
        "highd_shared_training_data/driver_behavior/coverage.json" in report["source"]
    )
    assert report["test_used_for_parameter_selection"] is False
    assert report["validation_selection"]["selected"]["execution"]["events"] > 0


def lane_training_example():
    count, speed = 151, 25.0
    distance = (count - 1) * speed * 0.04
    state = torch.tensor([0, 0, speed, 0, 0, 0], dtype=torch.float64)
    shape = torch.tensor([0.08, -0.06], dtype=torch.float64)
    coefficients = path_coefficients(
        state, torch.tensor(3.6), torch.tensor(distance), shape
    ).numpy()
    polynomial = np.polynomial.polynomial
    u = np.linspace(0, 1, count)
    values = np.column_stack(
        (
            polynomial.polyval(u, coefficients),
            np.full(count, speed),
            polynomial.polyval(u, polynomial.polyder(coefficients)) * speed / distance,
            polynomial.polyval(u, polynomial.polyder(coefficients, 2))
            * speed**2
            / distance**2,
        )
    )
    states = np.pad(values, ((25, 25), (0, 0)), mode="edge")
    events = pd.DataFrame(
        [
            {
                "trajectory_index": 0,
                "target_y_m": float(torch.tensor(3.6)),
                "source_y_m": 0,
                "length_m": 4.8,
                "width_m": 1.8,
                "duration_s": count * 0.04,
            }
        ]
    )
    return (
        events,
        {"states": states, "offsets": np.array([0, len(states)])},
        shape.numpy(),
    )


def test_joint_shape_fit_recovers_runtime_path_and_derivatives():
    events, observations, shape = lane_training_example()
    features, parameters, _, _ = fitting.training_parameters(events, observations)
    np.testing.assert_allclose(parameters[0, 1:], shape, atol=1e-7)
    np.testing.assert_allclose(parameters[0, 0], np.log(6.0), atol=1e-7)
    np.testing.assert_allclose(features[0, [0, 4, 5]], [25, 0, 0])


@pytest.mark.parametrize("column, delta", [(2, 0.1), (3, 0.02)])
def test_joint_shape_fit_uses_velocity_and_acceleration_not_only_position(
    column, delta
):
    events, observations, _ = lane_training_example()
    original = fitting.training_parameters(events, observations)[1]
    observations["states"][75:125, column] += delta
    changed = fitting.training_parameters(events, observations)[1]
    np.testing.assert_array_equal(original[:, 0], changed[:, 0])
    assert not np.allclose(original[:, 1:], changed[:, 1:], atol=1e-6, rtol=1e-6)
