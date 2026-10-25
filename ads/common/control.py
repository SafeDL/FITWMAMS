"""Track semantic actions, continuous controls and trajectories on shared dynamics."""

from dataclasses import dataclass

import numpy as np

from ads.common.observations import from_local, to_local
from hierarchical_world_model.src.highway import (
    steering_to_yaw_rate,
    yaw_rate_to_steering,
)
from hierarchical_world_model.src.scene_geometry import SceneGeometry

LANE_LEFT = 0
IDLE = 1
LANE_RIGHT = 2
FASTER = 3
SLOWER = 4
ACTION_NAMES = ("lane_left", "idle", "lane_right", "faster", "slower")

CARL_MAX_STEERING = 0.83775804096
CARL_CONTROL_DT = 0.1
PHYSICS_DT = 0.04


@dataclass
class SemanticEgoController:
    """Track target speed/lane without making an implicit safety decision."""

    geometry: SceneGeometry
    initial_xy: np.ndarray
    initial_speed_mps: float
    min_speed_mps: float = 0.0
    max_speed_mps: float = 45.0
    speed_increment_mps: float = 2.0
    acceleration_limit_mps2: float = 4.0
    braking_limit_mps2: float = 8.0
    yaw_rate_limit_rps: float = 0.6

    def __post_init__(self) -> None:
        self.target_lane_id = self.geometry.nearest_lane(np.asarray(self.initial_xy))
        self.target_speed_mps = float(
            np.clip(self.initial_speed_mps, self.min_speed_mps, self.max_speed_mps)
        )

    @property
    def target_y(self) -> float:
        return self.geometry.center_y(self.target_lane_id)

    def update_target(self, action: int) -> bool:
        """Apply one semantic action; return whether it was structurally valid."""
        value = int(action)
        if value not in range(len(ACTION_NAMES)):
            raise ValueError(f"unknown semantic action {value}")
        if value in (LANE_LEFT, LANE_RIGHT):
            neighbor = self.geometry.adjacent_lane(
                self.target_lane_id, -1 if value == LANE_RIGHT else 1
            )
            if neighbor is None:
                return False
            self.target_lane_id = neighbor
        elif value == FASTER:
            self.target_speed_mps = min(
                self.max_speed_mps, self.target_speed_mps + self.speed_increment_mps
            )
        elif value == SLOWER:
            self.target_speed_mps = max(
                self.min_speed_mps, self.target_speed_mps - self.speed_increment_mps
            )
        return True

    def control(self, ego_state: np.ndarray) -> np.ndarray:
        state = np.asarray(ego_state, dtype=np.float64)
        speed = float(np.linalg.norm(state[2:4]))
        acceleration = np.clip(
            1.5 * (self.target_speed_mps - speed),
            -self.braking_limit_mps2,
            self.acceleration_limit_mps2,
        )
        heading = float(np.arctan2(state[3], max(abs(state[2]), 1.0e-6)))
        lateral_error = self.target_y - float(state[1])
        desired_vy = 2.0 * np.tanh(0.75 * lateral_error)
        target_heading = float(
            np.arcsin(np.clip(desired_vy / max(speed, 0.5), -0.9, 0.9))
        )
        heading_error = float(
            np.arctan2(
                np.sin(target_heading - heading), np.cos(target_heading - heading)
            )
        )
        yaw_rate = np.clip(
            5.0 * heading_error, -self.yaw_rate_limit_rps, self.yaw_rate_limit_rps
        )
        return np.asarray((acceleration, yaw_rate), dtype=np.float32)


def carl_action_commands(action, *, steering, speed, acceleration):
    """Official ActionTrajectoryBuilder defaults: acceleration and steering rate."""
    target_acceleration = float(action[0]) * (2.4 if action[0] >= 0 else 3.2)
    if speed + target_acceleration * CARL_CONTROL_DT < 0:
        target_acceleration = -speed - 0.75 * acceleration
    steering_rate = (float(action[1]) * CARL_MAX_STEERING - steering) / CARL_CONTROL_DT
    return float(np.clip(target_acceleration, -3.2, 2.4)), steering_rate


class ContinuousController:
    def __init__(self, event):
        self.geometry = event.geometry
        self.target_lane_id = self.geometry.nearest_lane(event.initial_states[0, :2])
        self.target_speed_mps = float(np.linalg.norm(event.initial_states[0, 2:4]))
        self.action = np.zeros(2, np.float32)
        self.vehicle_length = float(event.vehicle_dimensions[0, 0])
        state = event.initial_states[0]
        speed = float(np.linalg.norm(state[2:4]))
        heading = float(np.arctan2(state[3], state[2]))
        yaw = (-state[4] * np.sin(heading) + state[5] * np.cos(heading)) / max(
            speed, 0.5
        )
        self.steering = yaw_rate_to_steering(
            yaw, speed, self.vehicle_length, CARL_MAX_STEERING
        )
        self.steering_rate = 0.0

    @property
    def target_y(self):
        return self.geometry.center_y(self.target_lane_id)

    def update_target(self, action):
        action = np.asarray(action, np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("CaRL control must contain two finite values")
        self.action = np.clip(action, -1, 1)
        return True

    def control(self, state):
        speed = float(np.linalg.norm(state[2:4]))
        heading = float(np.arctan2(state[3], state[2]))
        acceleration = float(state[4] * np.cos(heading) + state[5] * np.sin(heading))
        target_acceleration, target_rate = carl_action_commands(
            self.action, steering=self.steering, speed=speed, acceleration=acceleration
        )
        # Match nuPlan's command filters at the frozen world's 25 Hz execution rate.
        acceleration += (
            PHYSICS_DT / (PHYSICS_DT + 0.2) * (target_acceleration - acceleration)
        )
        next_steering = self.steering + (
            PHYSICS_DT / (PHYSICS_DT + 0.05) * PHYSICS_DT * target_rate
        )
        requested_yaw = steering_to_yaw_rate(next_steering, speed, self.vehicle_length)
        yaw = np.clip(requested_yaw, -0.6, 0.6)
        # Feed back the steering actually admitted by the public yaw-rate limit.
        realised = (
            yaw_rate_to_steering(yaw, speed, self.vehicle_length, CARL_MAX_STEERING)
            if abs(requested_yaw) > 0.6
            else next_steering
        )
        self.steering_rate = (realised - self.steering) / PHYSICS_DT
        self.steering = realised
        return np.asarray((np.clip(acceleration, -8, 4), yaw), np.float32)


class TrajectoryController(ContinuousController):
    def __init__(self, event):
        super().__init__(event)
        self.current_ego = event.initial_states[0].copy()
        self.path = np.asarray(
            (event.initial_states[0, :2], event.initial_states[0, :2] + (20, 0))
        )

    def update_target(self, action):
        path, waypoints = action
        path, waypoints = np.asarray(path), np.asarray(waypoints)
        if (
            path.shape != (20, 2)
            or waypoints.shape != (8, 2)
            or not np.isfinite(path).all()
            or not np.isfinite(waypoints).all()
        ):
            raise ValueError("PlanT must return finite 20-point path and 8 waypoints")
        self.path = from_local(path, self.current_ego, carla=True)
        temporal = np.concatenate((np.zeros((1, 2)), waypoints), axis=0)
        self.target_speed_mps = float(
            np.clip(
                np.linalg.norm(np.diff(temporal, axis=0), axis=-1).mean() / 0.25, 0, 45
            )
        )
        return True

    def control(self, state):
        self.current_ego = np.asarray(state).copy()
        speed = float(np.linalg.norm(state[2:4]))
        local = to_local(self.path, state)
        ahead = np.flatnonzero(local[:, 0] >= max(3, speed * 0.35))
        target = local[ahead[0] if len(ahead) else np.argmax(local[:, 0])]
        curvature = 2 * target[1] / max(float(np.square(target).sum()), 1)
        return np.asarray(
            (
                np.clip(1.5 * (self.target_speed_mps - speed), -8, 4),
                np.clip(speed * curvature, -0.6, 0.6),
            ),
            np.float32,
        )


class NuplanTrajectoryController(TrajectoryController):
    """Track native 10 Hz rear-axle poses at the world's existing 25 Hz."""

    def __init__(self, event):
        super().__init__(event)
        from ads.nuplan.features import rear_offset

        self.rear_offset = rear_offset(event.vehicle_dimensions)
        self.time = 0.0

    def update_target(self, action):
        poses = np.asarray(action, np.float64)
        if poses.shape != (80, 3) or not np.isfinite(poses).all():
            raise ValueError(
                "nuPlan planners must return 80 finite (x, y, heading) poses"
            )
        ego = self.current_ego.copy()
        heading = np.arctan2(ego[3], ego[2])
        ego[:2] -= self.rear_offset * np.array((np.cos(heading), np.sin(heading)))
        rear = from_local(poses[:, :2], ego)
        angles = poses[:, 2] + heading
        centers = rear + self.rear_offset * np.stack(
            (np.cos(angles), np.sin(angles)), -1
        )
        self.path = np.concatenate((self.current_ego[None, :2], centers), axis=0)
        self.speeds = np.linalg.norm(np.diff(self.path, axis=0), axis=-1) / 0.1
        self.time = 0.0
        return True

    def control(self, state):
        # Use the imminent part of the temporal plan, not its eight-second mean speed.
        index = min(int(self.time / 0.1), 79)
        self.target_speed_mps = float(np.clip(self.speeds[index], 0, 45))
        control = super().control(state)
        self.time += PHYSICS_DT
        return control
