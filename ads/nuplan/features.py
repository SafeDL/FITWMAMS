"""Causal history and native vector features for the published nuPlan planners."""

import numpy as np
from gymnasium import spaces

from ads.common.observations import to_local

HISTORY_STEPS = 21
SAMPLE_INTERVAL = 0.1
REAR_OFFSET_RATIO = 1.461 / 5.176


def rear_offset(dimensions):
    return float(dimensions[0, 0]) * REAR_OFFSET_RATIO


def sample_history(
    states, frame, *, dt=0.04, initial_history=None, initial_history_valid=None
):
    states = np.asarray(states)[: frame + 1]
    observed = np.ones(states.shape[:2], bool)
    if initial_history is not None:
        prior = np.asarray(initial_history)
        states = np.concatenate((prior[:-1], states))
        observed = np.concatenate(
            (np.asarray(initial_history_valid, bool)[:-1], observed)
        )
    times = (np.arange(len(states)) - (len(states) - frame - 1)) * dt
    query = frame * dt + np.arange(1 - HISTORY_STEPS, 1) * SAMPLE_INTERVAL
    within = query >= times[0] - 1e-8
    flat = states.reshape(len(states), -1)
    sampled = np.stack(
        [np.interp(query, times, flat[:, i]) for i in range(flat.shape[1])], -1
    ).reshape(HISTORY_STEPS, 7, 6)
    right = np.searchsorted(times, query, side="left").clip(0, len(states) - 1)
    left = np.searchsorted(times, query, side="right").clip(1, len(states)) - 1
    valid = within[:, None] & observed[left] & observed[right]
    sampled[~valid] = 0
    return sampled.astype(np.float32), valid


def local_history(scene, history, valid):
    ego = scene.states[0].astype(np.float64).copy()
    heading = np.arctan2(history[..., 3], history[..., 2])
    past = np.asarray(history, float).copy()
    offset = rear_offset(scene.dimensions)
    past[:, 0, :2] -= offset * np.stack(
        (np.cos(heading[:, 0]), np.sin(heading[:, 0])), -1
    )
    ego[:2] -= offset * np.array((np.cos(scene.heading), np.sin(scene.heading)))
    position = to_local(past[..., :2], ego)
    velocity = to_local(past[..., 2:4] + ego[:2], ego)
    acceleration = to_local(past[..., 4:6] + ego[:2], ego)
    angles = (heading - scene.heading + np.pi) % (2 * np.pi) - np.pi
    return ego, position, velocity, acceleration, angles, np.asarray(valid, bool)


def lane_vectors(scene, ego, *, capacity=70, points=20, radius=100):
    lanes = np.zeros((capacity, points, 12), np.float32)
    candidates = []
    for line, width in zip(scene.geometry.centerlines, scene.geometry.lane_widths):
        line = np.asarray(line)
        # Segment existing mapped lanes; no additional road geometry is introduced.
        # Existing highway lanes are continuous straight roads; the stored
        # polyline endpoints delimit an encoding window, not the road's end.
        low, high = ego[0] - radius, ego[0] + radius
        if high <= low:
            continue
        for start in np.arange(low, high, 20.0):
            xs = np.linspace(start, min(start + 20, high), points + 1)
            center = np.stack((xs, np.interp(xs, line[:, 0], line[:, 1])), -1)
            center = to_local(center, ego)
            direction = np.diff(center, axis=0)
            center = center[:-1]
            left = np.tile(
                np.array((np.sin(scene.heading), np.cos(scene.heading))) * width / 2,
                (points, 1),
            )
            light = np.tile(np.array((0, 0, 0, 1)), (points, 1))
            value = np.concatenate((center, direction, left, -left, light), -1)
            candidates.append((float(np.linalg.norm(center, axis=-1).min()), value))
    for index, (_, value) in enumerate(
        sorted(candidates, key=lambda x: x[0])[:capacity]
    ):
        lanes[index] = value
    return lanes


def vector_features(scene, history, history_valid):
    ego, position, velocity, acceleration, heading, valid = local_history(
        scene, history, history_valid
    )
    slots = np.flatnonzero(scene.valid[1:]) + 1
    slots = slots[
        np.argsort(
            np.linalg.norm(scene.states[slots, :2] - scene.states[0, :2], axis=-1)
        )
    ]
    neighbors = np.zeros((32, HISTORY_STEPS, 11), np.float32)
    for index, slot in enumerate(slots):
        values = np.concatenate(
            (
                position[:, slot],
                np.cos(heading[:, slot, None]),
                np.sin(heading[:, slot, None]),
                velocity[:, slot],
                np.tile(scene.dimensions[slot, ::-1], (HISTORY_STEPS, 1)),
                np.tile((1, 0, 0), (HISTORY_STEPS, 1)),
            ),
            -1,
        )
        values[~valid[:, slot]] = 0
        neighbors[index] = values
    current = np.array(
        (
            0,
            0,
            1,
            0,
            velocity[-1, 0, 0],
            velocity[-1, 0, 1],
            acceleration[-1, 0, 0],
            acceleration[-1, 0, 1],
            scene.steering,
            scene.yaw_rate,
        ),
        np.float32,
    )
    ego_past = np.concatenate(
        (
            position[:, 0],
            np.cos(heading[:, 0, None]),
            np.sin(heading[:, 0, None]),
            velocity[:, 0],
            acceleration[:, 0],
            np.full((HISTORY_STEPS, 1), scene.steering),
            np.full((HISTORY_STEPS, 1), scene.yaw_rate),
        ),
        -1,
    ).astype(np.float32)
    ego_past[~valid[:, 0]] = 0
    lanes = lane_vectors(scene, ego)
    features = {
        "ego_current_state": current,
        "ego_agent_past": ego_past,
        "neighbor_agents_past": neighbors,
        "static_objects": np.zeros((5, 10), np.float32),
        "lanes": lanes,
        "lanes_speed_limit": np.zeros((70, 1), np.float32),
        "lanes_has_speed_limit": np.zeros((70, 1), bool),
        "route_lanes": lanes[:25].copy(),
        "route_lanes_speed_limit": np.zeros((25, 1), np.float32),
        "route_lanes_has_speed_limit": np.zeros((25, 1), bool),
    }
    return features, np.concatenate(([0], slots)), ego


def dtpp_features(scene, history, history_valid):
    ego, position, velocity, acceleration, heading, valid = local_history(
        scene, history, history_valid
    )
    slots = np.flatnonzero(scene.valid[1:]) + 1
    slots = slots[np.argsort(np.linalg.norm(position[-1, slots], axis=-1))]
    slots = slots[position[-1, slots, 0] >= -6.0]
    ego_past = np.concatenate(
        (position[:, 0], heading[:, 0, None], velocity[:, 0], acceleration[:, 0]), -1
    ).astype(np.float32)
    ego_past[~valid[:, 0]] = 0
    neighbors = np.zeros((20, HISTORY_STEPS, 11), np.float32)
    for index, slot in enumerate(slots):
        yaw = np.zeros(HISTORY_STEPS)
        pairs = valid[1:, slot] & valid[:-1, slot]
        yaw[1:] = np.where(pairs, np.diff(np.unwrap(heading[:, slot])) / 0.1, 0)
        values = np.concatenate(
            (
                position[:, slot],
                heading[:, slot, None],
                velocity[:, slot],
                yaw[:, None],
                np.tile(scene.dimensions[slot, ::-1], (HISTORY_STEPS, 1)),
                np.tile((1, 0, 0), (HISTORY_STEPS, 1)),
            ),
            -1,
        )
        values[~valid[:, slot]] = 0
        neighbors[index] = values
    lanes = np.zeros((40, 50, 7), np.float32)
    for index, line in enumerate(scene.geometry.centerlines):
        line = np.asarray(line)
        xs = np.linspace(ego[0] - 80, ego[0] + 80, 50)
        points = to_local(
            np.stack((xs, np.interp(xs, line[:, 0], line[:, 1])), -1), ego
        )
        lanes[index, :, :2] = points
        lanes[index, :, 2] = -scene.heading
        lanes[index, :, 6] = 1  # unknown traffic light, original one-hot convention
    return (
        {
            "ego_agent_past": ego_past,
            "neighbor_agents_past": neighbors,
            "map_lanes": lanes,
            "map_crosswalks": np.zeros((5, 30, 3), np.float32),
            "route_lanes": lanes[:10].copy(),
        },
        np.concatenate(([0], slots)),
        ego,
    )


def build_features(scene, history, history_valid, *, algorithm):
    if algorithm == "dtpp":
        features, slots, _ = dtpp_features(scene, history, history_valid)
    elif algorithm in ("diffusion_planner", "flow_planner"):
        features, slots, _ = vector_features(scene, history, history_valid)
        if algorithm == "flow_planner":
            shape_type = np.asarray(
                (*scene.dimensions[0, ::-1], 1, 0, 0, 0), np.float32
            )
            past = np.concatenate(
                (
                    features["ego_agent_past"][:, :8],
                    np.tile(shape_type, (HISTORY_STEPS, 1)),
                ),
                -1,
            )
            past[~np.asarray(history_valid)[:, 0]] = 0
            features["ego_agent_past"] = past
            features["ego_current_state"] = np.concatenate(
                (features["ego_current_state"], shape_type)
            )
    elif algorithm == "str2":
        from ads.str2.features import build_features as build_str2

        return build_str2(scene, history, history_valid)
    else:
        raise ValueError(f"unknown native planner {algorithm}")
    return features, slots


class PlannerObservationSpace(spaces.Space):
    def contains(self, value):
        return (
            isinstance(value, dict) and {"features", "scene", "slots"} <= value.keys()
        )

    def sample(self, mask=None):
        raise NotImplementedError("planner observations require causal world state")
