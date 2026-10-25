"""Expose the existing highway road and causal actors through nuPlan objects."""

from types import SimpleNamespace

import numpy as np
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import (
    StateSE2,
    StateVector2D,
    TimePoint,
)
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.history.simulation_history_buffer import (
    SimulationHistoryBuffer,
)
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
    SimulationIteration,
)
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from ads.nuplan.features import rear_offset


class LanePath:
    def __init__(self, line):
        self.linestring = LineString(line)
        self.length = self.linestring.length
        distances = np.linspace(0, self.length, max(2, int(self.length / 0.25) + 1))
        points = np.array([self.linestring.interpolate(s).coords[0] for s in distances])
        angles = np.arctan2(np.gradient(points[:, 1]), np.gradient(points[:, 0]))
        self.discrete_path = [
            StateSE2(float(x), float(y), float(h)) for (x, y), h in zip(points, angles)
        ]

    def get_nearest_arc_length_from_position(self, point):
        return self.linestring.project(Point(point.x, point.y))

    def get_nearest_pose_from_position(self, point):
        s = self.get_nearest_arc_length_from_position(point)
        index = min(
            int(s / max(self.length, 1e-6) * (len(self.discrete_path) - 1)),
            len(self.discrete_path) - 1,
        )
        return self.discrete_path[index]


class HighwayLane:
    def __init__(self, identity, line, width, index):
        self.id, self.index = str(identity), index
        self.baseline_path = LanePath(line)
        self.polygon = self.baseline_path.linestring.buffer(
            float(width) / 2, cap_style=2
        )
        self.incoming_edges, self.outgoing_edges = [], []
        self.speed_limit_mps = None
        self.left_boundary = SimpleNamespace(
            linestring=self.baseline_path.linestring.parallel_offset(
                float(width) / 2, "left"
            )
        )
        self.right_boundary = SimpleNamespace(
            linestring=self.baseline_path.linestring.parallel_offset(
                float(width) / 2, "right"
            )
        )
        self.adjacent_edges = (None, None)

    def contains_point(self, point):
        return self.polygon.covers(Point(point.x, point.y))

    def get_roadblock_id(self):
        return self.roadblock_id

    def has_traffic_lights(self):
        return False


class HighwayMap:
    map_name = "existing_highway_road"

    def __init__(self, geometry, *, center_x=0.0, extent=200.0):
        lines = [
            np.array(
                (
                    (
                        min(float(line[:, 0].min()), center_x - extent),
                        float(line[:, 1].mean()),
                    ),
                    (
                        max(float(line[:, 0].max()), center_x + extent),
                        float(line[:, 1].mean()),
                    ),
                )
            )
            for line in geometry.centerlines
        ]
        self.lanes = [
            HighwayLane(str(10000 + int(identity)), line, width, index + 1)
            for index, (identity, line, width) in enumerate(
                zip(geometry.lane_ids, lines, geometry.lane_widths)
            )
        ]
        # Roadblocks follow the existing adjacency components. A median gap never
        # becomes drivable merely because nuPlan asks for a roadblock polygon.
        by_id = {
            int(identity): lane for identity, lane in zip(geometry.lane_ids, self.lanes)
        }
        remaining, self.blocks = set(by_id), []
        while remaining:
            frontier, members = [min(remaining)], []
            while frontier:
                identity = frontier.pop()
                if identity not in remaining:
                    continue
                remaining.remove(identity)
                lane = by_id[identity]
                members.append(lane)
                for side in (-1, 1):
                    adjacent = geometry.adjacent_lane(identity, side)
                    if adjacent in remaining:
                        frontier.append(adjacent)
            block_id = str(20000 + len(self.blocks))
            polygon = unary_union([lane.polygon for lane in members])
            # NuPlan's renderer requires one exterior per object. Split numeric
            # polygons at actual gaps, rather than fill them with a convex hull.
            polygons = (
                list(polygon.geoms)
                if polygon.geom_type == "MultiPolygon"
                else [polygon]
            )
            for polygon_index, polygon in enumerate(polygons):
                contained = [
                    lane
                    for lane in members
                    if polygon.intersects(lane.polygon)
                    and polygon.intersection(lane.polygon).area
                    > 0.5 * lane.polygon.area
                ]
                identity = str(int(block_id) + polygon_index * 100)
                block = SimpleNamespace(
                    id=identity,
                    interior_edges=contained,
                    incoming_edges=[],
                    outgoing_edges=[],
                    polygon=polygon,
                )
                self.blocks.append(block)
                for lane in contained:
                    lane.roadblock_id = identity
        self.block = self.blocks[0]
        self.layers = {
            SemanticMapLayer.LANE: self.lanes,
            SemanticMapLayer.ROADBLOCK: self.blocks,
            SemanticMapLayer.DRIVABLE_AREA: self.blocks,
        }
        for index, lane in enumerate(self.lanes):
            left = geometry.adjacent_lane(int(geometry.lane_ids[index]), 1)
            right = geometry.adjacent_lane(int(geometry.lane_ids[index]), -1)
            lane.adjacent_edges = (by_id.get(left), by_id.get(right))

    def select_route(self, point):
        lane = min(self.lanes, key=lambda lane: lane.polygon.distance(Point(*point)))
        self.block = self.get_map_object(
            lane.get_roadblock_id(), SemanticMapLayer.ROADBLOCK
        )
        return self.block

    def get_proximal_map_objects(self, point, radius, layers):
        position = Point(point.x, point.y)
        return {
            layer: [
                obj
                for obj in self.layers.get(layer, [])
                if obj.polygon.distance(position) <= radius
            ]
            for layer in layers
        }

    def get_map_object(self, object_id, layer):
        return next(
            (obj for obj in self.layers.get(layer, []) if obj.id == str(object_id)),
            None,
        )

    def get_all_map_objects(self, point, layer):
        return [
            obj
            for obj in self.layers.get(layer, [])
            if obj.polygon.covers(Point(point.x, point.y))
        ]

    def is_in_layer(self, point, layer):
        return bool(self.get_all_map_objects(point, layer))

    def get_distance_to_nearest_map_object(self, point, layer):
        objects = self.layers.get(layer, [])
        if not objects:
            return None, float("inf")
        distances = [obj.polygon.distance(Point(point.x, point.y)) for obj in objects]
        index = int(np.argmin(distances))
        return objects[index].id, distances[index]


def map_for_scene(scene):
    # Cover the existing world horizon plus the native eight-second planning
    # horizon. This is a view of the same continuous straight road, not a new map.
    extent = (scene.horizon_s + 8) * max(
        45.0, float(np.linalg.norm(scene.states[0, 2:4]))
    ) + 50
    return HighwayMap(scene.geometry, center_x=float(scene.states[0, 0]), extent=extent)


def vehicle_parameters(scene):
    length, width = map(float, scene.dimensions[0])
    offset = rear_offset(scene.dimensions)
    return VehicleParameters(
        width,
        length / 2 + offset,
        length / 2 - offset,
        1.67 / 5.176 * length,
        3.089 / 5.176 * length,
        "measured_highD_ego",
        "vehicle",
        1.8,
    )


def ego_state(scene, state, time_s):
    heading = float(np.arctan2(state[3], state[2]))
    rear = state[:2] - rear_offset(scene.dimensions) * np.array(
        (np.cos(heading), np.sin(heading))
    )
    rotation = np.array(
        ((np.cos(heading), -np.sin(heading)), (np.sin(heading), np.cos(heading)))
    )
    velocity, acceleration = state[2:4] @ rotation, state[4:6] @ rotation
    return EgoState.build_from_rear_axle(
        StateSE2(float(rear[0]), float(rear[1]), heading),
        StateVector2D(*map(float, velocity)),
        StateVector2D(*map(float, acceleration)),
        float(scene.steering),
        TimePoint(round((time_s + 10) * 1e6)),
        vehicle_parameters(scene),
        angular_vel=float(scene.yaw_rate),
        angular_accel=float(scene.yaw_acceleration),
    )


def detections(scene, states=None, valid=None, time_s=None):
    states = scene.states if states is None else states
    valid = scene.valid if valid is None else valid
    timestamp = round(((scene.elapsed_s if time_s is None else time_s) + 10) * 1e6)
    agents = []
    for slot in np.flatnonzero(valid[1:]) + 1:
        state = states[slot]
        length, width = map(float, scene.dimensions[slot])
        pose = StateSE2(
            float(state[0]), float(state[1]), float(np.arctan2(state[3], state[2]))
        )
        agents.append(
            Agent(
                TrackedObjectType.VEHICLE,
                OrientedBox(pose, length, width, 1.8),
                StateVector2D(float(state[2]), float(state[3])),
                SceneObjectMetadata(
                    timestamp, f"highD_{slot}", int(slot), f"highD_{slot}", "vehicle"
                ),
            )
        )
    return DetectionsTracks(TrackedObjects(agents))


def planner_input(observation, decision, map_api=None):
    scene = observation["scene"]
    map_api = map_for_scene(scene) if map_api is None else map_api
    map_api.select_route(scene.states[0, :2])
    history = observation["history"]
    valid = observation["history_valid"]
    ego_states, observations = [], []
    for index in range(len(history)):
        if not valid[index, 0]:
            continue
        time_s = scene.elapsed_s + (index - len(history) + 1) * 0.1
        ego_states.append(ego_state(scene, history[index, 0], time_s))
        observations.append(
            detections(scene, history[index], valid[index] & scene.valid, time_s)
        )
    buffer = SimulationHistoryBuffer.initialize_from_list(
        max(1, len(ego_states)), ego_states, observations, sample_interval=0.1
    )
    current = PlannerInput(
        SimulationIteration(ego_states[-1].time_point, decision), buffer, []
    )
    lane = min(
        map_api.block.interior_edges,
        key=lambda lane: lane.polygon.distance(Point(*scene.states[0, :2])),
    )
    initialization = PlannerInitialization(
        [map_api.block.id], lane.baseline_path.discrete_path[-1], map_api
    )
    return initialization, current
