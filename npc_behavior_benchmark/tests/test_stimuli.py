import numpy as np

from npc_behavior_benchmark.evaluation.stimuli import (
    lateral_stimulus_trajectories,
    longitudinal_stimulus_trajectories,
    sustained_response_latency,
)


def test_longitudinal_pulse_uses_common_dynamics_and_expected_direction():
    states = np.zeros((2, 7, 6), np.float32)
    valid = np.zeros((2, 7), bool)
    states[:, 1, 2] = 20.0
    valid[:, 1] = True
    baseline, changed = longitudinal_stimulus_trajectories(
        states, valid, np.array([1, 1]), np.array([-3.0, 2.0])
    )
    assert changed[0, 24, 1, 2] < baseline[0, 24, 1, 2]
    assert changed[1, 24, 1, 2] > baseline[1, 24, 1, 2]
    assert np.allclose(changed[:, 74, 1, 4], 0.0)


def test_sustained_response_latency_and_right_censoring():
    delta = np.array([[0.0, -0.1, -0.3, -0.4, 0.0], [0.0, -0.3, 0.0, -0.3, 0.0]])
    responded, latency = sustained_response_latency(delta, np.array([-1.0, -1.0]))
    assert responded.tolist() == [True, False]
    assert latency[0] == 0.4
    assert np.isnan(latency[1])


def test_lateral_stimulus_moves_toward_target_lane_without_changing_speed():
    states = np.zeros((2, 7, 6), np.float32)
    states[:, 0, 2] = 20.0
    states[0, 0, 1] = -3.5
    states[1, 0, 1] = 3.5
    valid = np.zeros((2, 7), bool)
    valid[:, 0] = True
    natural, changed = lateral_stimulus_trajectories(
        states,
        valid,
        np.array([0, 0]),
        np.array([0.0, 0.0]),
        np.array([0.9, 0.9]),
    )
    assert np.all(np.abs(changed[:, -1, 0, 1]) < np.abs(natural[:, -1, 0, 1]))
    assert np.allclose(
        np.linalg.norm(changed[:, -1, 0, 2:4], axis=-1), 20.0, atol=1.0e-4
    )
