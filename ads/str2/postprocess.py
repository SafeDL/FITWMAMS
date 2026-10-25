"""The released STR/PDM proposal generation, scoring and emergency braking."""

import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import (
    StateSE2,
    StateVector2D,
    TimePoint,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from ads.nuplan.scene import planner_input
from ads.str2.vendor.nuplan_simulation.str_trajectory_generator import (
    STRTrajectoryGenerator,
)
from ads.str2.vendor.tuplan_garage.planning.simulation.planner.pdm_planner.pdm_str_closed_planner import (
    STRClosedPlanner,
)
from ads.str2.vendor.tuplan_garage.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import (
    BatchIDMPolicy,
)


def make_closed_planner(observation):
    initialization, _ = planner_input(observation, 0)
    policies = {
        "fallback_target_velocity": 15.0,
        "speed_limit_fraction": [0.2, 0.4, 0.6, 0.8, 1.0],
        "min_gap_to_lead_agent": 1.0,
        "headway_time": 1.5,
        "accel_max": 1.5,
        "decel_max": 3.0,
    }
    generator = STRTrajectoryGenerator("", build_model=False)
    planner = STRClosedPlanner(
        generator,
        TrajectorySampling(num_poses=80, interval_length=0.1),
        TrajectorySampling(num_poses=40, interval_length=0.1),
        BatchIDMPolicy(**policies),
        BatchIDMPolicy(**policies),
        None,
        [-1.0, 1.0],
        50.0,
        False,
        True,
        0.8,
        10.0,
        8,
    )
    planner.initialize(initialization)
    return planner, initialization.map_api


def postprocess(planner, map_api, observation, poses, decision):
    _, current = planner_input(observation, decision, map_api)
    ego, _ = current.history.current_state
    origin = np.array((ego.rear_axle.x, ego.rear_axle.y))
    angle = ego.rear_axle.heading
    rotation = np.array(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    )
    xy = poses[:, :2] @ rotation.T + origin
    headings = poses[:, 2] + angle
    positions = np.concatenate((origin[None], xy))
    velocities = np.diff(positions, axis=0) / 0.1
    old_velocity = np.array(
        (
            ego.dynamic_car_state.rear_axle_velocity_2d.x,
            ego.dynamic_car_state.rear_axle_velocity_2d.y,
        )
    )
    old_velocity = old_velocity @ rotation.T
    accelerations = (
        np.diff(np.concatenate((old_velocity[None], velocities)), axis=0) / 0.1
    )
    predicted = [ego]
    for index in range(80):
        heading = headings[index]
        body_rotation = np.array(
            ((np.cos(heading), -np.sin(heading)), (np.sin(heading), np.cos(heading)))
        )
        velocity, acceleration = (
            velocities[index] @ body_rotation,
            accelerations[index] @ body_rotation,
        )
        predicted.append(
            EgoState.build_from_rear_axle(
                StateSE2(
                    float(xy[index, 0]), float(xy[index, 1]), float(headings[index])
                ),
                StateVector2D(*map(float, velocity)),
                StateVector2D(*map(float, acceleration)),
                ego.tire_steering_angle,
                TimePoint(ego.time_us + (index + 1) * 100000),
                ego.car_footprint.vehicle_parameters,
            )
        )
    trajectory = planner.compute_planner_trajectory(current, str_pred_states=predicted)
    selected = trajectory.get_state_at_times(
        [TimePoint(ego.time_us + (index + 1) * 100000) for index in range(80)]
    )
    absolute = np.array(
        [
            [state.rear_axle.x, state.rear_axle.y, state.rear_axle.heading]
            for state in selected
        ]
    )
    return np.column_stack(
        (
            (absolute[:, :2] - origin) @ rotation,
            (absolute[:, 2] - angle + np.pi) % (2 * np.pi) - np.pi,
        )
    )
