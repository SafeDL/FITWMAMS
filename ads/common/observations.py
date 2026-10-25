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


def to_local(points, ego, *, carla=False):
    heading = np.arctan2(ego[3], ego[2])
    rotation = np.asarray(
        ((np.cos(heading), -np.sin(heading)), (np.sin(heading), np.cos(heading)))
    )
    local = (np.asarray(points) - ego[:2]) @ rotation
    if carla:
        local[..., 1] *= -1
    return local


def from_local(points, ego, *, carla=False):
    local = np.asarray(points).copy()
    if carla:
        local[..., 1] *= -1
    heading = np.arctan2(ego[3], ego[2])
    rotation = np.asarray(
        ((np.cos(heading), -np.sin(heading)), (np.sin(heading), np.cos(heading)))
    )
    return local @ rotation.T + ego[:2]


def road_raster(scene: CausalScene, *, size=256, ppm=2.0, ahead=78.0):
    ego = scene.states[0]

    def pixels(points):
        local = to_local(points, ego)
        return np.rint(
            np.stack(
                (size / 2 - local[..., 1] * ppm, ahead * ppm - local[..., 0] * ppm), -1
            )
        ).astype(np.int32)

    road = np.zeros((size, size), np.uint8)
    boundary = road.copy()
    route = road.copy()
    nearest = scene.geometry.nearest_lane(ego[:2])
    for lane_id, width in zip(scene.geometry.lane_ids, scene.geometry.lane_widths):
        center = scene.geometry.center_y(int(lane_id))
        # Extend the existing straight lanes for the sensor window, without changing topology.
        xs = (ego[0] - 200, ego[0] + 200)
        corners = np.asarray(
            (
                (xs[0], center - width / 2),
                (xs[1], center - width / 2),
                (xs[1], center + width / 2),
                (xs[0], center + width / 2),
            )
        )
        cv2.fillConvexPoly(road, pixels(corners), 255)
        for y in (center - width / 2, center + width / 2):
            cv2.polylines(
                boundary, [pixels(np.asarray(((xs[0], y), (xs[1], y))))], False, 255, 1
            )
        if int(lane_id) == nearest:
            cv2.fillConvexPoly(route, pixels(corners), 255)
    return road, boundary, route, pixels


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


def plant_observation(scene: CausalScene, *, range_m=50.0, front_factor=2.0):
    ego = scene.states[0]
    slots = np.flatnonzero(scene.valid[1:]) + 1
    local = to_local(scene.states[slots, :2], ego, carla=True)
    objects = np.zeros((len(slots), 7), np.float32)
    keep = (
        local[:, 0] ** 2 / np.where(local[:, 0] > 0, front_factor**2, 1)
        + local[:, 1] ** 2
        <= range_m**2
    )
    objects[:, 0] = 1
    objects[:, 1:3] = local
    relative_heading = (
        np.arctan2(scene.states[slots, 3], scene.states[slots, 2]) - scene.heading
    )
    objects[:, 3] = -np.rad2deg(
        np.arctan2(np.sin(relative_heading), np.cos(relative_heading))
    )
    objects[:, 4] = np.linalg.norm(scene.states[slots, 2:4], axis=-1) * 3.6
    objects[:, 5:7] = scene.dimensions[slots][:, ::-1]
    nearest = scene.geometry.nearest_lane(ego[:2])
    route_global = np.stack(
        (
            ego[0] + np.arange(20, dtype=np.float32),
            np.full(20, scene.geometry.center_y(nearest)),
        ),
        -1,
    )
    road, boundary, _, _ = road_raster(scene, ppm=5, ahead=25.6)
    colors = np.asarray(
        ((0.485, 0.456, 0.406), (0.25, 0.25, 0.75), (0.75, 0.25, 0.25)), np.float32
    )
    labels = np.where(boundary > 0, 2, np.where(road > 0, 1, 0))
    return {
        "objects": objects[keep],
        "object_slots": slots[keep],
        "route_original": to_local(route_global, ego, carla=True).astype(np.float32),
        "speed_limit": np.int64(
            3
        ),  # Highest official category; task cap is 45 m/s, not a highD legal limit.
        "input_ego_speed": np.float32(np.linalg.norm(ego[2:4])),
        "BEV": colors[labels].transpose(2, 0, 1),
    }
