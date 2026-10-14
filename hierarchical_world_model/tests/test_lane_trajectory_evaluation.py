"""Complete raw-event replay uses the same lane planner and physical limits."""

import numpy as np
import pandas as pd
import pytest

from hierarchical_world_model.src.npc_experiments.lane_execution import (
    lane_path_rollout,
    recording_interval,
    trajectory_replay,
)
from hierarchical_world_model.src.stochastic_drivers.lane_trajectory import (
    MODEL_PATH,
    LearnedLaneTrajectory,
)


def records():
    return [
        {
            "event_id": f"event_{index}",
            "recording_id": index // 2,
            "direction": (-1) ** index,
            "speed_mps": 25 + 10 * (index % 2),
            "length_m": 4.8 + 5 * (index % 2),
            "lateral_RMSE_m": 0.25,
            "final_target_error_m": 0.01,
            "peak_vy_error_mps": 0.1,
            "duration_error_s": 0.2,
            "peak_ay_mps2": 0.5,
            "lateral_jerk_q99_mps3": 1.0,
            "logged_jerk_q99_mps3": 0.5,
            "model_peak_vy_mps": 1.1,
            "observed_peak_vy_mps": 1.0,
            "model_duration_s": 4.2,
            "observed_duration_s": 4.0,
            "completion_within_observed_exposure": index != 3,
            "longitudinal_condition_clipped_steps": 0,
            "max_vx_condition_error_mps": 0.0,
        }
        for index in range(4)
    ]


def test_recording_bootstrap_keeps_constant_event_error():
    np.testing.assert_allclose(
        recording_interval(records(), "lateral_RMSE_m"), [0.25, 0.25]
    )


def test_supplied_longitudinal_speed_still_obeys_runtime_acceleration_limit():
    model = LearnedLaneTrajectory()
    states, clipped = lane_path_rollout(
        model,
        [0, 0, 25, 0, 0, 0],
        [4.8, 1.8],
        0,
        [25, 28.6],
    )
    assert clipped == 1
    assert states[-1, 2] == pytest.approx(25.16, abs=1e-5)
    assert states[-1, 0] == pytest.approx(1.0032, abs=1e-5)
    assert states[-1, 1] == 0


def test_complete_lane_replay_uses_runtime_integrator():
    model = LearnedLaneTrajectory()
    data = np.zeros((29, 4), np.float32)
    data[:, 1] = 25.0
    data[26:, 1] = 28.6
    events = pd.DataFrame(
        [
            {
                "trajectory_index": 0,
                "event_id": "event_1",
                "recording_id": 4,
                "direction": 1,
                "speed_mps": 25.0,
                "length_m": 4.8,
                "width_m": 1.8,
                "source_y_m": 0.0,
                "target_y_m": 3.6,
                "peak_lateral_speed_mps": 0.0,
                "duration_s": 0.12,
            }
        ]
    )
    observations = {"states": data, "offsets": np.array([0, len(data)])}
    record = trajectory_replay(events, observations, MODEL_PATH)[0]
    predicted, clipped = lane_path_rollout(
        model, [0, 0, 25, 0, 0, 0], [4.8, 1.8], 3.6, data[25:, 1]
    )
    assert clipped == 3
    assert record["longitudinal_condition_clipped_steps"] == clipped
    assert record["max_vx_condition_error_mps"] == pytest.approx(
        np.max(np.abs(predicted[:, 2] - data[25:, 1]))
    )
    assert record["model_peak_vy_mps"] == pytest.approx(np.abs(predicted[:, 3]).max())
    assert record["observed_exposure_s"] == pytest.approx(0.12)
    assert record["lateral_RMSE_m"] == pytest.approx(
        np.sqrt(np.mean((predicted[:, 1] - data[25:, 0]) ** 2))
    )
    observations["states"][26:, 0] += 1
    observations["states"][26:, 2:] += 2
    changed = trajectory_replay(events, observations, MODEL_PATH)[0]
    for metric in (
        "model_peak_vy_mps",
        "lateral_jerk_q99_mps3",
        "max_vx_condition_error_mps",
    ):
        assert changed[metric] == record[metric]
    assert changed["lateral_RMSE_m"] != record["lateral_RMSE_m"]


def test_lane_replay_bounds_realized_normal_acceleration_after_longitudinal_selection():
    model = LearnedLaneTrajectory()
    initial = [0, 0, 10, 0, 0, 1.0]
    predicted, _ = lane_path_rollout(
        model, initial, [4.8, 1.8], 3.6, [10, 10.16, 10.32, 10.48]
    )
    heading = np.arctan2(predicted[:, 3], predicted[:, 2])
    normal = -predicted[:, 4] * np.sin(heading) + predicted[:, 5] * np.cos(heading)
    assert np.abs(normal[1:]).max() <= model.acceleration_limit + 1e-6
    assert np.abs(np.diff(normal)).max() <= model.jerk_limit * 0.04 + 1e-5
