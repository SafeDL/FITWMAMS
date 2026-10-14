"""Low-level tracking for the five semantic ADS actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hierarchical_world_model.src.scene_geometry import SceneGeometry


LANE_LEFT = 0
IDLE = 1
LANE_RIGHT = 2
FASTER = 3
SLOWER = 4
ACTION_NAMES = ("lane_left", "idle", "lane_right", "faster", "slower")


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
        target_heading = float(np.arcsin(np.clip(desired_vy / max(speed, 0.5), -0.9, 0.9)))
        heading_error = float(
            np.arctan2(np.sin(target_heading - heading), np.cos(target_heading - heading))
        )
        yaw_rate = np.clip(5.0 * heading_error, -self.yaw_rate_limit_rps, self.yaw_rate_limit_rps)
        return np.asarray((acceleration, yaw_rate), dtype=np.float32)

