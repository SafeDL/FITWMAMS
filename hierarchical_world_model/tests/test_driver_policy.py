import numpy as np
import torch
from dataclasses import replace

from hierarchical_world_model.src.stochastic_drivers.driver_policy import (
    corrected_acceleration,
    exclusive_lane_probabilities,
    lane_frame_probabilities,
    physical_leader_observation,
    sample_lane_choice,
)
from hierarchical_world_model.src.npc_experiments.residual import (
    autoregression_parameters,
)


def correction():
    return {
        "longitudinal": {
            "feature_mean": [0] * 9,
            "feature_std": [1] * 9,
            "mean_coef": [0] * 10,
            "variance_coef": [0] * 10,
            "autoregression": autoregression_parameters([1.8, -0.85]),
            "mean_strength": 1,
            "noise_strength": 1,
        },
        "response_blend": 0.5,
        "trajectory": {"log_duration_scale": 0, "shape_shift": [0, 0]},
    }


def test_exclusive_ties_and_one_second_hazard_are_consistent():
    probability = exclusive_lane_probabilities(
        torch.zeros(2, 2),
        torch.tensor([[True, True], [False, True]]),
        {"bias": 0, "slope": 0},
    )
    torch.testing.assert_close(
        probability, torch.tensor([[0.25, 0.5, 0.25], [0.5, 0, 0.5]])
    )
    frame = lane_frame_probabilities(probability)
    torch.testing.assert_close(frame.sum(-1), torch.ones(2))
    torch.testing.assert_close(
        frame[:, 0] ** 25, probability[:, 0], atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(frame[0, 1] / frame[0, 2], torch.tensor(2.0))
    assert sample_lane_choice(probability, torch.tensor([0.98, 0.999])).tolist() == [
        -1,
        1,
    ]


def test_driver_residual_restores_and_cannot_relax_immediate_braking():
    speed = torch.tensor([20.0])
    theta = torch.tensor([[32.0, 2, 1.2, 1, 2]])
    parameters = correction()
    parameters["longitudinal"]["mean_coef"][0] = 5
    first, state = corrected_acceleration(
        torch.tensor([100.0]),
        speed,
        speed,
        theta,
        correction=parameters,
        innovation=torch.tensor([0.2]),
    )
    _, state2 = corrected_acceleration(
        torch.tensor([100.0]),
        speed,
        speed,
        theta,
        correction=parameters,
        residual_state=state.clone(),
        innovation=torch.tensor([-0.4]),
    )
    _, replay = corrected_acceleration(
        torch.tensor([100.0]),
        speed,
        speed,
        theta,
        correction=parameters,
        residual_state=state.clone(),
        innovation=torch.tensor([-0.4]),
    )
    torch.testing.assert_close(state2, replay)
    torch.testing.assert_close(state[..., 0], torch.tensor([0.2]))
    brake, _ = corrected_acceleration(
        torch.tensor([-0.5]),
        speed,
        speed,
        theta,
        correction=parameters,
        innovation=torch.tensor([10.0]),
    )
    assert brake.item() == -8
    assert torch.isfinite(first).all()


def test_swept_actual_body_detects_vehicle_before_lane_center_switch():
    states = torch.tensor(
        [[0, 0, 20, 0, 0, 0], [12, 2.1, 10, -0.3, 0, 0], [50, 0, 20, 0, 0, 0]],
        dtype=torch.float32,
    )
    dimensions = torch.tensor([[4.8, 1.8], [12, 2.6], [4.8, 1.8]])
    gap, _, _, leader = physical_leader_observation(
        states, torch.ones(3, dtype=torch.bool), dimensions
    )
    assert leader[0].item() == 1
    assert abs(gap[0].item() - 3.6) < 1e-5


def test_residual_cannot_cancel_braking_while_approaching_a_leader():
    theta = torch.tensor([[32.0, 2.0, 1.2, 1.0, 2.0]])
    gap, speed, front = torch.tensor([40.0]), torch.tensor([25.0]), torch.tensor([20.0])
    baseline, _ = corrected_acceleration(gap, speed, front, theta)
    fit = correction()
    fit["longitudinal"]["mean_coef"][0] = 10.0
    fit["longitudinal"]["noise_strength"] = 0.0
    learned, _ = corrected_acceleration(
        gap, speed, front, theta, correction=fit, innovation=torch.zeros(1)
    )
    assert baseline.item() < 0
    torch.testing.assert_close(learned, baseline)


def test_conditional_lane_correction_preserves_unsafe_side():
    model = {
        "lane": {
            "feature_mean": [0] * 8,
            "feature_std": [1] * 8,
            "coef": [5] + [0] * 8,
            "strength": 1,
        }
    }
    probability = exclusive_lane_probabilities(
        torch.zeros(1, 2),
        torch.tensor([[False, True]]),
        {"bias": -4, "slope": 0},
        features=torch.zeros(1, 2, 8),
        correction=model,
    )
    assert probability[0, 1].item() == 0
    assert probability[0, 2].item() > 0.5
    assert np.isclose(probability.sum().item(), 1)


def test_zero_probability_correction_is_exactly_the_original_physical_prior():
    incentive = torch.tensor([[0.0, 1.0], [-2.0, 3.0]])
    safe = torch.tensor([[False, True], [True, False]])
    parameters = {"bias": -3.0, "slope": 0.2}
    baseline = exclusive_lane_probabilities(incentive, safe, parameters)
    learned = {
        "lane": {
            "feature_mean": [0] * 8,
            "feature_std": [1] * 8,
            "coef": [2] + [0.1] * 8,
            "strength": 0.0,
        }
    }
    actual = exclusive_lane_probabilities(
        incentive,
        safe,
        parameters,
        features=torch.ones(2, 2, 8),
        correction=learned,
    )
    assert torch.equal(actual, baseline)


def test_persistent_lane_style_changes_intent_without_changing_feasibility():
    incentive = torch.zeros(3, 2)
    safe = torch.tensor([[True, True], [False, True], [False, False]])
    probability = exclusive_lane_probabilities(
        incentive,
        safe,
        {"bias": -4, "slope": 0.1},
        style_offset=torch.tensor([-2.0, 2.0, 10.0]),
    )
    torch.testing.assert_close(probability.sum(-1), torch.ones(3))
    assert probability[1, 1] == 0
    assert probability[2, 0] == 1
    assert probability[1, 2] > probability[0, 2]


def test_online_driver_residual_snapshot_replays_explicit_innovations():
    from hierarchical_world_model.tests.test_online_ma_idm import _context, THETA
    from hierarchical_world_model.src.stochastic_drivers.online import (
        OnlineMAIDMController,
    )

    geometry = torch.zeros(1, 3, 2, 5)
    geometry[0, :, :, 1] = torch.tensor([-4.0, 0.0, 4.0])[:, None]
    geometry[0, :, 1, 0] = 200
    geometry[..., 4] = 4
    context = replace(
        _context(ego_x=16),
        policy_standard_normal=torch.full((1, 6, 2), 0.2),
        map_polylines=geometry,
        map_polyline_valid=torch.ones(1, 3, 2, dtype=torch.bool),
    )
    controller = OnlineMAIDMController(THETA, driver_correction=correction())
    controller(context)
    snapshot = controller.snapshot_runtime()
    assert "driver_residual_state" in snapshot
    next_context = replace(context, policy_standard_normal=torch.full((1, 6, 2), -0.4))
    first = controller(next_context)
    restored = OnlineMAIDMController(THETA, driver_correction=correction())
    restored.restore_runtime(snapshot)
    replay = restored(next_context)
    torch.testing.assert_close(first.actions, replay.actions)
    torch.testing.assert_close(
        controller._driver_residual_state, restored._driver_residual_state
    )


def test_lane_only_model_cannot_activate_longitudinal_residual_overrides():
    from hierarchical_world_model.tests.test_online_ma_idm import _context, THETA
    from hierarchical_world_model.src.stochastic_drivers.online import (
        OnlineMAIDMController,
    )

    context = _context(ego_x=40.0, ego_ax=-0.5)
    history = context.current[:, None].repeat(1, 6, 1, 1)
    history[:, 0, 0, 2] += 0.2
    context = replace(
        context,
        history=history,
        history_valid=context.current_valid[:, None].repeat(1, 6, 1),
        nominal_ego_velocity=torch.zeros(1, 2),
    )
    baseline = OnlineMAIDMController(THETA)(context)
    lane_only = {
        "response_blend": 1.0,
        "trajectory": {
            "log_duration_scale": -0.075,
            "shape_shift": [0.0, 0.0],
        },
    }
    controller = OnlineMAIDMController(THETA, driver_correction=lane_only)
    actual = controller(context)
    torch.testing.assert_close(actual.actions[..., 0], baseline.actions[..., 0])
    assert controller._driver_residual_state is None
    assert controller._driver_source_y is None
