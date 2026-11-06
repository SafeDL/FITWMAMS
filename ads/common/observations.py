"""Policy encoders receive only current state, mapped road and causal history."""

from dataclasses import dataclass

import cv2
import numpy as np

from hierarchical_world_model.src.scene_geometry import SceneGeometry


@dataclass(frozen=True)
class CausalScene:
    states: np.ndarray
    valid: np.ndarray
    dimensions: np.ndarray
    geometry: SceneGeometry
    elapsed_s: float
    horizon_s: float
    previous_action: np.ndarray
    yaw_rate: float
    yaw_acceleration: float
    steering: float
    steering_rate: float
    progress_fraction: float = 0.0

    @property
    def heading(self):
        return float(np.arctan2(self.states[0, 3], self.states[0, 2]))


def to_local(points, ego):
    heading = np.arctan2(ego[3], ego[2])
    rotation = np.asarray(
        ((np.cos(heading), -np.sin(heading)), (np.sin(heading), np.cos(heading)))
    )
    local = (np.asarray(points) - ego[:2]) @ rotation
    return local


def from_local(points, ego):
    local = np.asarray(points).copy()
    heading = np.arctan2(ego[3], ego[2])
    rotation = np.asarray(
        ((np.cos(heading), -np.sin(heading)), (np.sin(heading), np.cos(heading)))
    )
    return local @ rotation.T + ego[:2]


def carl_observation(scene: CausalScene):
    ego = scene.states[0]

    def pixels(points):
        # Official nuPlan raster: origin 14 m ahead, forward along columns.
        local = to_local(points, ego) - np.asarray((14.0, 0.0))
        return (local * 2.0 + 128.0).astype(np.int32)

    bev = np.zeros((9, 256, 256), np.uint8)
    xs = (ego[0] - 200, ego[0] + 200)
    lane_edges = []
    for lane_id, width in zip(scene.geometry.lane_ids, scene.geometry.lane_widths):
        center = scene.geometry.center_y(int(lane_id))
        edges = (center - width / 2, center + width / 2)
        lane_edges.extend(edges)
        for y in edges:
            line = pixels(np.asarray(((xs[0], y), (xs[1], y))))
            cv2.polylines(bev[2], [line], False, 255, 1)
        line = pixels(np.asarray(((xs[0], center), (xs[1], center))))
        # highD has no legal speed-limit label: use the official unknown value.
        cv2.polylines(bev[5], [line], False, 255, 1)
    lower, upper = min(lane_edges), max(lane_edges)
    roadblock = pixels(
        np.asarray(((xs[0], lower), (xs[1], lower), (xs[1], upper), (xs[0], upper)))
    )
    cv2.fillPoly(bev[0], [roadblock], 255)
    # The existing straight road is one route roadblock, including all its lanes.
    bev[1] = bev[0]
    # Absent lights, stop signs, pedestrians and static objects remain zero.
    for slot in np.flatnonzero(scene.valid):
        state = scene.states[slot]
        length, width = scene.dimensions[slot]
        heading = np.arctan2(state[3], state[2])
        corners = np.asarray(
            (
                (-length / 2, -width / 2),
                (length / 2, -width / 2),
                (length / 2, width / 2),
                (-length / 2, width / 2),
            )
        )
        rotation = np.asarray(
            ((np.cos(heading), -np.sin(heading)), (np.sin(heading), np.cos(heading)))
        )
        color = int(127.5 * (1 + min(np.linalg.norm(state[2:4]) / 30, 1)))
        cv2.fillConvexPoly(
            bev[8 if slot == 0 else 6], pixels(corners @ rotation.T + state[:2]), color
        )
    velocity = to_local(ego[None, 2:4] + ego[:2], ego)[0]
    acceleration = to_local(ego[None, 4:6] + ego[:2], ego)[0]
    measurements = np.asarray(
        (
            *scene.previous_action,
            *velocity,
            *acceleration,
            scene.steering,
            scene.steering_rate,
            scene.yaw_rate,
            scene.yaw_acceleration,
        ),
        np.float32,
    )
    remaining = max(0, 1 - scene.elapsed_s / scene.horizon_s)
    # Critic-only causal context. Actor feature extraction never reads this key.
    _, _, comfort, ttc = carl_constraints(scene)
    value = np.asarray(
        (remaining, max(0, 1 - scene.progress_fraction), comfort, ttc), np.float32
    )
    return {
        "bev_semantics": bev,
        "measurements": measurements,
        "value_measurements": value,
    }


def carl_constraints(scene: CausalScene):
    ego = scene.states[0]
    lane = scene.geometry.nearest_lane(ego[:2])
    lane_error = abs(ego[1] - scene.geometry.center_y(lane))
    lane_factor = max(0.0, 1 - max(lane_error - 0.5, 0) / 2)
    speed_factor = min(1.0, 45 / max(float(np.linalg.norm(ego[2:4])), 0.5))
    comfort_factor = float(abs(ego[4]) <= 4 and abs(ego[5]) <= 4.89)
    relative = scene.states[1:] - ego
    widths = (scene.dimensions[1:, 1] + scene.dimensions[0, 1]) / 2
    gap = (
        np.abs(relative[:, 0]) - (scene.dimensions[1:, 0] + scene.dimensions[0, 0]) / 2
    )
    closing = np.where(relative[:, 0] >= 0, -relative[:, 2], relative[:, 2])
    active = scene.valid[1:] & (np.abs(relative[:, 1]) < widths) & (closing > 0.05)
    ttc = (
        float(
            np.clip(np.min(np.maximum(gap[active], 0) / closing[active]) / 0.95, 0, 1)
        )
        if active.any()
        else 1.0
    )
    return lane_factor, speed_factor, comfort_factor, ttc
