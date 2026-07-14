"""rounD adapter for curved maps, conflict zones, and variable agents.

The repository does not ship rounD recordings.  Callers provide tracked states
and a vector map; this adapter preserves curved geometry and derives compact
conflict zones from merging/crossing topology when an annotation is absent.
It never approximates a roundabout with highD-style straight lanes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from ..graph_builder import DynamicTrafficGraphBuilder, GraphBuilderConfig
from ..graph_schema import DynamicTrafficSequence


class RoundGraphAdapter:
    version = "round_vector_map_v3_lanelet2_conflict_zones"

    def __init__(self, top_r_lanes: int = 3, lane_width_m: float = 3.6) -> None:
        self.builder = DynamicTrafficGraphBuilder(GraphBuilderConfig(
            top_r_lanes=int(top_r_lanes), lane_width_m=float(lane_width_m),
        ))

    @staticmethod
    def infer_conflict_zones(
        map_polylines: np.ndarray,
        map_polyline_valid: np.ndarray,
        lane_graph_edges: np.ndarray,
        *,
        lane_width_m: float = 3.6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Infer merge/cross zones from vector-map geometry and topology.

        Topology kinds ``2``, ``3``, and ``4`` mean merge, diverge, and cross.
        In addition, non-parallel polylines that geometrically meet create a
        zone.  Neighbouring detections are coalesced to one physical region.
        """
        polylines = np.asarray(map_polylines, np.float32)
        valid = np.asarray(map_polyline_valid, bool)
        edges = np.asarray(lane_graph_edges, np.int64).reshape(-1, 3)
        if polylines.ndim != 3 or polylines.shape[-1] < 4:
            raise ValueError("map_polylines must have shape [M, P, >=4]")
        if valid.shape != polylines.shape[:2]:
            raise ValueError("map_polyline_valid must align with map_polylines")
        explicit = {
            tuple(sorted((int(source), int(destination))))
            for source, destination, kind in edges
            if source >= 0 and destination >= 0 and int(kind) in {2, 3, 4}
        }
        zones: list[np.ndarray] = []
        for left in range(polylines.shape[0]):
            left_points = polylines[left, valid[left]]
            if not len(left_points):
                continue
            for right in range(left + 1, polylines.shape[0]):
                right_points = polylines[right, valid[right]]
                if not len(right_points):
                    continue
                delta = left_points[:, None, :2] - right_points[None, :, :2]
                sq_distance = np.sum(delta * delta, axis=-1)
                left_idx, right_idx = np.unravel_index(int(sq_distance.argmin()), sq_distance.shape)
                separation = float(np.sqrt(sq_distance[left_idx, right_idx]))
                tangent_left, tangent_right = left_points[left_idx, 2:4], right_points[right_idx, 2:4]
                norm = float(np.linalg.norm(tangent_left) * np.linalg.norm(tangent_right))
                sin_angle = 0.0 if norm <= 1.0e-6 else abs(float(np.cross(tangent_left, tangent_right) / norm))
                topology_conflict = (left, right) in explicit
                geometric_conflict = separation <= 0.75 * float(lane_width_m) and sin_angle >= np.sin(np.deg2rad(15.0))
                if not (topology_conflict or geometric_conflict):
                    continue
                center = 0.5 * (left_points[left_idx, :2] + right_points[right_idx, :2])
                if any(float(np.linalg.norm(center - item[:2])) < 0.75 * float(lane_width_m) for item in zones):
                    continue
                priority = 1.0 if (topology_conflict or sin_angle >= np.sin(np.deg2rad(45.0))) else 0.5
                zones.append(np.asarray((center[0], center[1], 0.75 * lane_width_m, priority), np.float32))
        if not zones:
            return np.zeros((0, 4), np.float32), np.zeros((0,), bool)
        return np.stack(zones).astype(np.float32), np.ones((len(zones),), bool)

    @staticmethod
    def _column(frame, *candidates: str, required: bool = True) -> str | None:
        lookup = {str(name).lower(): str(name) for name in frame.columns}
        for candidate in candidates:
            if candidate.lower() in lookup:
                return lookup[candidate.lower()]
        if required:
            raise KeyError(f"Missing one of required rounD columns: {candidates}")
        return None

    @staticmethod
    def _normalise_polylines(polylines: np.ndarray, *, lane_width_m: float) -> np.ndarray:
        """Convert `[M,P,2|4|6]` vector-map input to the common 6-D points."""
        raw = np.asarray(polylines, np.float32)
        if raw.ndim != 3 or raw.shape[-1] < 2:
            raise ValueError("rounD map polylines must have shape [M, P, >=2]")
        if raw.shape[-1] >= 6:
            return raw[..., :6].copy()
        out = np.zeros((*raw.shape[:2], 6), np.float32)
        out[..., : raw.shape[-1]] = raw
        if raw.shape[-1] < 4:
            tangent = np.gradient(out[..., :2], axis=1)
            norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
            out[..., 2:4] = tangent / np.maximum(norm, 1.0e-6)
        out[..., 4] = float(lane_width_m)
        out[..., 5] = 1.0
        return out

    @staticmethod
    def _resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
        """Arc-length resampling used to make Lanelet2 boundary pairs align."""
        values = np.asarray(points, np.float32)
        if len(values) < 2:
            raise ValueError("a Lanelet2 boundary needs at least two points")
        distance = np.concatenate((np.zeros(1, np.float32), np.cumsum(np.linalg.norm(np.diff(values, axis=0), axis=1))))
        total = float(distance[-1])
        if total <= 1.0e-6:
            return np.repeat(values[:1], int(count), axis=0)
        query = np.linspace(0.0, total, int(count), dtype=np.float32)
        return np.stack([np.interp(query, distance, values[:, dimension]) for dimension in range(2)], axis=-1).astype(np.float32)

    @staticmethod
    def _node_xy(node: ElementTree.Element) -> tuple[float, float] | None:
        """Read custom metric ``x/y`` tags or a conventional OSM lat/lon node."""
        tags = {str(item.attrib.get("k")): str(item.attrib.get("v")) for item in node.findall("tag")}
        try:
            if "x" in tags and "y" in tags:
                return float(tags["x"]), float(tags["y"])
            if "local_x" in tags and "local_y" in tags:
                return float(tags["local_x"]), float(tags["local_y"])
            return float(node.attrib["lon"]), float(node.attrib["lat"])
        except (KeyError, ValueError):
            return None

    @classmethod
    def _load_lanelet2_osm(cls, path: Path, *, lane_width_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert an official Lanelet2 OSM map to the common vector-map form.

        rounD distributions provide Lanelet2 maps rather than the JSON/NPZ
        convenience format used in early smoke tests.  The converter is
        intentionally dependency-light: it parses boundaries and lanelet
        relations with ``ElementTree``, projects geographic nodes to a local
        metric tangent plane when needed, then derives successor, merge,
        diverge, adjacent, and crossing topology from directed centerlines.
        """
        root = ElementTree.parse(path).getroot()
        raw_nodes: dict[str, tuple[float, float]] = {}
        geographic = False
        for node in root.findall("node"):
            point = cls._node_xy(node)
            if point is None or "id" not in node.attrib:
                continue
            tags = {str(item.attrib.get("k")): str(item.attrib.get("v")) for item in node.findall("tag")}
            geographic = geographic or not ({"x", "y"} <= set(tags) or {"local_x", "local_y"} <= set(tags))
            raw_nodes[str(node.attrib["id"])] = point
        if not raw_nodes:
            raise ValueError("Lanelet2 OSM map contains no usable nodes")
        if geographic:
            # ``_node_xy`` returned (longitude, latitude).  A local
            # equirectangular projection is sufficient because every rounD
            # recording spans only one roundabout and preserves the meter
            # geometry required by lane assignment and conflict inference.
            longitudes = np.asarray([value[0] for value in raw_nodes.values()], np.float64)
            latitudes = np.asarray([value[1] for value in raw_nodes.values()], np.float64)
            lon0, lat0 = float(longitudes.mean()), float(latitudes.mean())
            scale_x = 6_371_000.0 * math.pi / 180.0 * math.cos(math.radians(lat0))
            scale_y = 6_371_000.0 * math.pi / 180.0
            raw_nodes = {
                key: ((value[0] - lon0) * scale_x, (value[1] - lat0) * scale_y)
                for key, value in raw_nodes.items()
            }
        ways: dict[str, np.ndarray] = {}
        way_tags: dict[str, dict[str, str]] = {}
        for way in root.findall("way"):
            identifier = str(way.attrib.get("id", ""))
            points = [raw_nodes[item.attrib["ref"]] for item in way.findall("nd") if item.attrib.get("ref") in raw_nodes]
            if len(points) >= 2:
                ways[identifier] = np.asarray(points, np.float32)
                way_tags[identifier] = {str(item.attrib.get("k")): str(item.attrib.get("v")) for item in way.findall("tag")}
        lanes: list[np.ndarray] = []
        lane_widths: list[float] = []
        lane_tags: list[dict[str, str]] = []
        for relation in root.findall("relation"):
            tags = {str(item.attrib.get("k")): str(item.attrib.get("v")) for item in relation.findall("tag")}
            if tags.get("type") != "lanelet":
                continue
            members = {str(item.attrib.get("role", "")): str(item.attrib.get("ref", "")) for item in relation.findall("member") if item.attrib.get("type") == "way"}
            left, right = ways.get(members.get("left", "")), ways.get(members.get("right", ""))
            if left is None or right is None:
                continue
            # The two boundaries should advance in the same direction.  Some
            # OSM exporters reverse the right boundary, so normalize it here.
            same = float(np.linalg.norm(left[0] - right[0]) + np.linalg.norm(left[-1] - right[-1]))
            reversed_cost = float(np.linalg.norm(left[0] - right[-1]) + np.linalg.norm(left[-1] - right[0]))
            if reversed_cost < same:
                right = right[::-1]
            count = max(8, min(64, max(len(left), len(right))))
            left_sample, right_sample = cls._resample_polyline(left, count), cls._resample_polyline(right, count)
            center = 0.5 * (left_sample + right_sample)
            lanes.append(center)
            width = float(np.median(np.linalg.norm(left_sample - right_sample, axis=-1)))
            lane_widths.append(width if np.isfinite(width) and width > 0.25 else float(lane_width_m))
            lane_tags.append(tags)
        # Accept a basic OSM lane-centerline export as a graceful fallback.
        if not lanes:
            for identifier, points in ways.items():
                tags = way_tags[identifier]
                if "highway" not in tags and tags.get("type") not in {"lanelet", "line_thin", "line_thick"}:
                    continue
                lanes.append(cls._resample_polyline(points, max(8, min(64, len(points)))))
                try:
                    lane_widths.append(float(tags.get("width", lane_width_m)))
                except ValueError:
                    lane_widths.append(float(lane_width_m))
                lane_tags.append(tags)
        if not lanes:
            raise ValueError("Lanelet2 OSM map contains no lanelet relations or lane centerlines")

        point_count = max(len(lane) for lane in lanes)
        polylines = np.zeros((len(lanes), point_count, 6), np.float32)
        valid = np.zeros((len(lanes), point_count), bool)
        starts, ends, tangent_samples = [], [], []
        for index, (lane, width, tags) in enumerate(zip(lanes, lane_widths, lane_tags)):
            sampled = cls._resample_polyline(lane, point_count)
            tangent = np.gradient(sampled, axis=0)
            norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
            tangent = tangent / np.maximum(norm, 1.0e-6)
            priority_text = tags.get("priority", tags.get("right_of_way", "1"))
            try:
                priority = float(priority_text)
            except ValueError:
                priority = 0.5 if str(priority_text).lower() in {"yield", "give_way", "no"} else 1.0
            polylines[index, :, :2] = sampled
            polylines[index, :, 2:4] = tangent
            polylines[index, :, 4] = max(float(width), 0.25)
            polylines[index, :, 5] = priority
            valid[index] = True
            starts.append(sampled[0]); ends.append(sampled[-1]); tangent_samples.append(tangent)

        starts_arr, ends_arr = np.asarray(starts), np.asarray(ends)
        widths = np.asarray(lane_widths, np.float32)
        successor_pairs: list[tuple[int, int]] = []
        for source in range(len(lanes)):
            for destination in range(len(lanes)):
                if source == destination:
                    continue
                threshold = max(2.0, 0.75 * float(widths[source] + widths[destination]))
                if float(np.linalg.norm(ends_arr[source] - starts_arr[destination])) <= threshold:
                    successor_pairs.append((source, destination))
        successor_count = np.bincount([source for source, _ in successor_pairs], minlength=len(lanes))
        predecessor_count = np.bincount([destination for _, destination in successor_pairs], minlength=len(lanes))
        topology: dict[tuple[int, int], int] = {}
        for source, destination in successor_pairs:
            kind = 2 if predecessor_count[destination] > 1 else 3 if successor_count[source] > 1 else 0
            topology[(source, destination)] = int(kind)

        for left in range(len(lanes)):
            for right in range(left + 1, len(lanes)):
                if (left, right) in topology or (right, left) in topology:
                    continue
                delta = polylines[left, :, :2] - polylines[right, :, :2]
                nearest = float(np.sqrt(np.sum(delta * delta, axis=-1)).mean())
                alignment = float(np.abs((tangent_samples[left] * tangent_samples[right]).sum(axis=-1)).mean())
                adjacent_limit = max(2.0, 1.8 * float(0.5 * (widths[left] + widths[right])))
                if nearest <= adjacent_limit and alignment >= 0.75:
                    topology[(left, right)] = topology[(right, left)] = 1
                    continue
                pair_delta = polylines[left, :, None, :2] - polylines[right, None, :, :2]
                closest = np.unravel_index(int(np.sum(pair_delta * pair_delta, axis=-1).argmin()), pair_delta.shape[:2])
                separation = float(np.linalg.norm(pair_delta[closest]))
                cross = abs(float(np.cross(tangent_samples[left][closest[0]], tangent_samples[right][closest[1]])))
                if separation <= max(1.0, 0.4 * float(widths[left] + widths[right])) and cross >= math.sin(math.radians(20.0)):
                    topology[(left, right)] = topology[(right, left)] = 4
        edges = np.asarray([(source, destination, kind) for (source, destination), kind in sorted(topology.items())], np.int64)
        return polylines, valid, edges.reshape(-1, 3)

    @classmethod
    def load_vector_map(
        cls,
        path: str | Path,
        *,
        lane_width_m: float = 3.6,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Read a rounD vector-map sidecar from JSON or NPZ.

        JSON/NPZ keys are ``polylines``, optional ``polyline_valid``, optional
        ``lane_graph_edges``, and optional ``conflict_zone_features`` /
        ``conflict_zone_valid``.  Official Lanelet2 ``.osm`` maps are also
        accepted and are converted directly.  A tracks CSV alone does not
        carry lane centerlines or priority geometry.
        """
        source = Path(path)
        suffix = source.suffix.lower()
        if suffix == ".osm":
            polylines, valid, edges = cls._load_lanelet2_osm(source, lane_width_m=lane_width_m)
            zones, zone_valid = cls.infer_conflict_zones(polylines, valid, edges, lane_width_m=lane_width_m)
            return polylines, valid, edges, zones, zone_valid
        if suffix == ".npz":
            archive = np.load(source, allow_pickle=False)
            payload = {key: archive[key] for key in archive.files}
        elif suffix == ".json":
            payload = json.loads(source.read_text(encoding="utf-8"))
        else:
            raise ValueError("rounD vector map must be .json, .npz, or Lanelet2 .osm")
        raw_polylines = payload.get("polylines", payload.get("map_polylines"))
        if raw_polylines is None:
            raise KeyError("rounD vector map requires 'polylines' or 'map_polylines'")
        polylines = cls._normalise_polylines(raw_polylines, lane_width_m=lane_width_m)
        valid = np.asarray(payload.get("polyline_valid", np.isfinite(polylines[..., :2]).all(axis=-1)), bool)
        if valid.shape != polylines.shape[:2]:
            raise ValueError("rounD polyline_valid must have shape [M, P]")
        edges = np.asarray(payload.get("lane_graph_edges", np.zeros((0, 3))), np.int64).reshape(-1, 3)
        zones_value = payload.get("conflict_zone_features")
        zones = None if zones_value is None else np.asarray(zones_value, np.float32).reshape(-1, 4)
        zone_valid_value = payload.get("conflict_zone_valid")
        zone_valid = None if zone_valid_value is None else np.asarray(zone_valid_value, bool).reshape(-1)
        if zones is not None and zone_valid is not None and zone_valid.shape != (len(zones),):
            raise ValueError("rounD conflict_zone_valid must align with conflict_zone_features")
        if (zones is None) != (zone_valid is None):
            raise ValueError("rounD map must provide both conflict-zone arrays or neither")
        return polylines, valid, edges, zones, zone_valid

    def adapt_from_files(
        self,
        *,
        tracks_csv: str | Path,
        vector_map: str | Path,
        ego_id: int,
        start_frame: int,
        num_frames: int,
        recording_id: str | int,
        split: str,
        primary_agent_id: int | None = None,
        frame_rate_hz: float = 25.0,
    ) -> DynamicTrafficSequence:
        """Read a standard rounD tracks CSV and emit one variable-agent sequence.

        The public rounD naming convention uses columns such as ``trackId``,
        ``frame``, ``xCenter``, and ``xVelocity``.  Common snake-case aliases
        are accepted so the loader can also read pre-cleaned exports.
        """
        import pandas as pd

        tracks = pd.read_csv(Path(tracks_csv))
        id_col = self._column(tracks, "trackId", "track_id", "id")
        frame_col = self._column(tracks, "frame", "frame_id")
        x_col, y_col = self._column(tracks, "xCenter", "x", "x_center"), self._column(tracks, "yCenter", "y", "y_center")
        vx_col, vy_col = self._column(tracks, "xVelocity", "vx", "x_velocity"), self._column(tracks, "yVelocity", "vy", "y_velocity")
        ax_col = self._column(tracks, "xAcceleration", "ax", "x_acceleration", required=False)
        ay_col = self._column(tracks, "yAcceleration", "ay", "y_acceleration", required=False)
        frame_values = np.arange(int(start_frame), int(start_frame) + int(num_frames), dtype=np.int64)
        subset = tracks[tracks[frame_col].isin(frame_values)].copy()
        if subset.empty:
            raise ValueError("requested rounD frame range has no tracks")
        observed_ids = sorted(int(value) for value in subset[id_col].unique())
        if int(ego_id) not in observed_ids:
            raise ValueError("ego_id is not present in the requested rounD frame range")
        agent_ids = np.asarray([int(ego_id), *[value for value in observed_ids if value != int(ego_id)]], np.int64)
        index_by_id = {int(value): index for index, value in enumerate(agent_ids)}
        states = np.zeros((len(frame_values), len(agent_ids), 6), np.float32)
        valid = np.zeros((len(frame_values), len(agent_ids)), bool)
        frame_index = {int(value): index for index, value in enumerate(frame_values)}
        for row in subset.itertuples(index=False):
            item = row._asdict()
            target_frame, target_id = int(item[frame_col]), int(item[id_col])
            values = (item[x_col], item[y_col], item[vx_col], item[vy_col], 0.0 if ax_col is None else item[ax_col], 0.0 if ay_col is None else item[ay_col])
            if not np.isfinite(np.asarray(values, np.float32)).all():
                continue
            states[frame_index[target_frame], index_by_id[target_id]] = values
            valid[frame_index[target_frame], index_by_id[target_id]] = True
        if not valid[:, 0].any():
            raise ValueError("ego has no finite rounD physical state in requested range")
        polylines, polyline_valid, lane_edges, zones, zone_valid = self.load_vector_map(vector_map, lane_width_m=self.builder.cfg.lane_width_m)
        primary_index = -1 if primary_agent_id is None else index_by_id.get(int(primary_agent_id), -1)
        return self.adapt(
            sequence_id=f"round-{recording_id}-{int(ego_id)}-{int(start_frame)}", recording_id=str(recording_id),
            ego_id=str(int(ego_id)), timestamps=(frame_values - frame_values[0]).astype(np.float32) / float(frame_rate_hz),
            agent_ids=agent_ids, agent_states=states, agent_valid=valid, ego_index=0,
            primary_agent_index=primary_index, map_polylines=polylines, map_polyline_valid=polyline_valid,
            lane_graph_edges=lane_edges, split=split, conflict_zone_features=zones, conflict_zone_valid=zone_valid,
        )

    def adapt(
        self,
        *,
        sequence_id: str,
        recording_id: str,
        ego_id: str,
        timestamps: np.ndarray,
        agent_ids: np.ndarray,
        agent_states: np.ndarray,
        agent_valid: np.ndarray,
        ego_index: int,
        primary_agent_index: int,
        map_polylines: np.ndarray,
        map_polyline_valid: np.ndarray,
        lane_graph_edges: np.ndarray,
        split: str,
        is_evt_tail: bool = False,
        conflict_zone_features: np.ndarray | None = None,
        conflict_zone_valid: np.ndarray | None = None,
    ) -> DynamicTrafficSequence:
        states = np.asarray(agent_states, np.float32)
        valid = np.asarray(agent_valid, bool)
        candidates = np.stack([
            self.builder.lane_candidates_from_polylines(step, mask, map_polylines, map_polyline_valid)
            for step, mask in zip(states, valid)
        ])
        if conflict_zone_features is None and conflict_zone_valid is None:
            zones, zone_valid = self.infer_conflict_zones(
                map_polylines, map_polyline_valid, lane_graph_edges,
                lane_width_m=self.builder.cfg.lane_width_m,
            )
        elif conflict_zone_features is not None and conflict_zone_valid is not None:
            zones, zone_valid = np.asarray(conflict_zone_features, np.float32), np.asarray(conflict_zone_valid, bool)
        else:
            raise ValueError("conflict_zone_features and conflict_zone_valid must be supplied together")
        return DynamicTrafficSequence(
            sequence_id=str(sequence_id), recording_id=str(recording_id), ego_id=str(ego_id),
            timestamps=np.asarray(timestamps, np.float32), agent_ids=np.asarray(agent_ids, np.int64),
            agent_states=states, agent_valid=valid, ego_index=int(ego_index),
            primary_agent_index=int(primary_agent_index), map_polylines=np.asarray(map_polylines, np.float32),
            map_polyline_valid=np.asarray(map_polyline_valid, bool), lane_graph_edges=np.asarray(lane_graph_edges, np.int64),
            agent_lane_candidates=candidates, split=str(split), is_evt_tail=bool(is_evt_tail),
            conflict_zone_features=zones, conflict_zone_valid=zone_valid,
        )
