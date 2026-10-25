"""STR2's published raster convention with real or explicitly missing history."""

from types import SimpleNamespace

import numpy as np

from ads.nuplan.features import rear_offset
from ads.nuplan.scene import detections, map_for_scene


def build_features(scene, history, history_valid):
    from nuplan.common.actor_state.state_representation import Point2D

    from ads.str2.vendor.nuplan_simulation.planner import get_road_dict
    from ads.str2.vendor.nuplan_simulation.str_trajectory_generator import (
        STRTrajectoryGenerator,
    )

    map_api = map_for_scene(scene)
    map_api.select_route(scene.states[0, :2])
    # Published checkpoint uses selected_exponential_past: -2, -1, -.5, 0 seconds.
    indices = np.array((0, 10, 15, 20))
    past = np.asarray(history)[indices]
    valid = np.asarray(history_valid)[indices]
    angles = np.arctan2(past[:, 0, 3], past[:, 0, 2])
    rear = past[:, 0, :2] - rear_offset(scene.dimensions) * np.stack(
        (np.cos(angles), np.sin(angles)), -1
    )
    poses = np.column_stack((rear, angles)).astype(np.float32)
    origin = np.array(
        (
            scene.states[0, 0] - rear_offset(scene.dimensions) * np.cos(scene.heading),
            scene.states[0, 1] - rear_offset(scene.dimensions) * np.sin(scene.heading),
            scene.heading,
        ),
        np.float32,
    )
    agents = [
        detections(scene, state, mask & scene.valid).tracked_objects.get_agents()
        for state, mask in zip(past, valid)
    ]
    road = get_road_dict(map_api, Point2D(float(origin[0]), float(origin[1])), {})
    rasterizer = SimpleNamespace(
        _map_api=map_api, _route_roadblock_ids=[map_api.block.id], road_dic=road
    )
    high, low, actions, _rectangles = STRTrajectoryGenerator.compute_raster_input(
        rasterizer,
        poses,
        agents,
        [[] for _ in poses],
        [],
        scene.dimensions[0, ::-1],
        origin_ego_pose=origin,
        map_name=map_api.map_name,
        corrected_route_ids=[map_api.block.id],
    )
    dynamic = np.column_stack(
        (
            np.linalg.norm(past[:, 0, 2:4], axis=-1),
            np.linalg.norm(past[:, 0, 4:6], axis=-1),
            np.zeros(len(poses)),
        )
    ).astype(np.float32)
    for index in range(len(poses)):
        if valid[index, 0]:
            speed = max(float(np.linalg.norm(past[index, 0, 2:4])), 0.5)
            h = angles[index]
            dynamic[index, 2] = (
                -past[index, 0, 4] * np.sin(h) + past[index, 0, 5] * np.cos(h)
            ) / speed
        else:
            high[..., 26 + index : 26 + 8 * len(poses) : len(poses)] = False
            low[..., 26 + index : 26 + 8 * len(poses) : len(poses)] = False
            actions[index] = 0
            dynamic[index] = 0
    features = {
        "high_res_raster": high,
        "low_res_raster": low,
        "context_actions": np.concatenate((actions, dynamic), -1).astype(np.float32),
    }
    return features, np.flatnonzero(scene.valid)
