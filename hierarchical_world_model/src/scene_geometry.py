"""Shared road geometry for model inputs and HighwayEnv execution.

The canonical highD cache stores straight lane centre-lines as map
polylines.  This module turns that representation into one small, explicit
geometry object.  Both the neural model and the physical backend therefore
use the same lane centres, widths, adjacency and road boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# Allow non-nominal map spacing without bridging highD divided-carriageway gaps.
LANE_ADJACENCY_GAP_TOLERANCE_M = 2.1


def _lane_adjacency(
    centers_y: np.ndarray,
    widths_m: np.ndarray,
    lane_ids: np.ndarray,
) -> np.ndarray:
    adjacency = np.full((len(lane_ids), 2), -1, dtype=np.int64)
    for source in range(len(lane_ids)):
        separation = centers_y - centers_y[source]
        maximum_gap = (
            0.5 * (widths_m[source] + widths_m) + LANE_ADJACENCY_GAP_TOLERANCE_M
        )
        for side_index, side in enumerate((-1.0, 1.0)):
            candidate = (separation * side > 1.0e-4) & (
                separation * side <= maximum_gap
            )
            if candidate.any():
                target = np.flatnonzero(candidate)[
                    np.argmin(separation[candidate] * side)
                ]
                adjacency[source, side_index] = lane_ids[target]
    return adjacency


def torch_map_lane_geometry(
    map_polylines: torch.Tensor,
    map_polyline_valid: torch.Tensor,
    *,
    fallback_width_m: float = 3.6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return batched lane centres, presence and measured widths.

    The online controllers consume the same map tensor as the physical world.
    Keeping this extraction in one place prevents nominal 3.6 m assumptions
    from leaking back into lane membership or adjacency tests.
    """
    polylines = torch.as_tensor(map_polylines)
    valid = torch.as_tensor(
        map_polyline_valid, dtype=torch.bool, device=polylines.device
    )
    if polylines.ndim != 4 or polylines.shape[-1] < 2:
        raise ValueError("map_polylines must have shape [batch,lanes,points,features]")
    if valid.shape != polylines.shape[:3]:
        raise ValueError("map validity must match batched lane points")
    lane_present = valid.sum(dim=-1).ge(2)
    count = valid.sum(dim=-1).clamp_min(1)
    lane_y = (
        torch.where(valid, polylines[..., 1], torch.zeros_like(polylines[..., 1])).sum(
            dim=-1
        )
        / count
    )
    if polylines.shape[-1] > 4:
        width_values = polylines[..., 4]
        width_valid = valid & width_values.gt(0.0) & torch.isfinite(width_values)
        lane_width = torch.where(
            width_valid, width_values, torch.zeros_like(width_values)
        ).sum(dim=-1) / width_valid.sum(dim=-1).clamp_min(1)
        lane_width = torch.where(
            width_valid.any(dim=-1),
            lane_width,
            torch.full_like(lane_width, float(fallback_width_m)),
        )
    else:
        lane_width = torch.full_like(lane_y, float(fallback_width_m))
    return lane_y, lane_present, lane_width


def torch_nearest_lane_geometry(
    lateral_y: torch.Tensor,
    lane_y: torch.Tensor,
    lane_present: torch.Tensor,
    lane_width: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve nearest mapped lane index, centre, width and availability."""
    y = torch.as_tensor(lateral_y, device=lane_y.device, dtype=lane_y.dtype)
    if (
        lane_y.ndim != 2
        or lane_present.shape != lane_y.shape
        or lane_width.shape != lane_y.shape
    ):
        raise ValueError("lane geometry must have matching [batch,lanes] shapes")
    if y.shape[0] != lane_y.shape[0]:
        raise ValueError("lateral positions and lane geometry must share a batch")
    distance = torch.where(
        lane_present[(slice(None),) + (None,) * (y.ndim - 1)],
        (lane_y[(slice(None),) + (None,) * (y.ndim - 1)] - y[..., None]).abs(),
        torch.full(
            (*y.shape, lane_y.shape[-1]),
            float("inf"),
            dtype=y.dtype,
            device=y.device,
        ),
    )
    index = distance.argmin(dim=-1)
    flat_index = index.reshape(index.shape[0], -1)
    center = torch.gather(lane_y, 1, flat_index).reshape(index.shape)
    width = torch.gather(lane_width, 1, flat_index).reshape(index.shape)
    available = lane_present.any(dim=-1)
    available = available[(slice(None),) + (None,) * (index.ndim - 1)].expand_as(index)
    return index, center, width, available


def torch_adjacent_lane_index(
    source_index: torch.Tensor,
    lane_y: torch.Tensor,
    lane_present: torch.Tensor,
    lane_width: torch.Tensor,
    *,
    side: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the nearest mapped lane that is topologically adjacent."""
    if int(side) not in (-1, 1):
        raise ValueError("side must be -1 or 1")
    source = torch.gather(
        lane_y, 1, source_index.reshape(source_index.shape[0], -1)
    ).reshape(source_index.shape)
    source_width = torch.gather(
        lane_width, 1, source_index.reshape(source_index.shape[0], -1)
    ).reshape(source_index.shape)
    prefix = (slice(None),) + (None,) * (source.ndim - 1)
    directed = (lane_y[prefix] - source[..., None]) * float(side)
    maximum_gap = (
        0.5 * (source_width[..., None] + lane_width[prefix])
        + LANE_ADJACENCY_GAP_TOLERANCE_M
    )
    candidates = lane_present[prefix] & directed.gt(1.0e-4) & directed.le(maximum_gap)
    score = torch.where(candidates, directed, torch.full_like(directed, float("inf")))
    index = score.argmin(dim=-1)
    available = candidates.any(dim=-1)
    return torch.where(available, index, torch.full_like(index, -1)), available


@dataclass(frozen=True)
class VehicleDimensions:
    """Per-agent longitudinal length and lateral width in metres."""

    length_m: np.ndarray
    width_m: np.ndarray

    def __post_init__(self) -> None:
        length = np.asarray(self.length_m, dtype=np.float64)
        width = np.asarray(self.width_m, dtype=np.float64)
        if length.shape != width.shape or length.ndim != 1:
            raise ValueError("vehicle lengths and widths must be matching vectors")
        if len(length) < 1 or np.any(length <= 0.0) or np.any(width <= 0.0):
            raise ValueError("vehicle dimensions must be positive")
        if not np.isfinite(length).all() or not np.isfinite(width).all():
            raise ValueError("vehicle dimensions must be finite")
        object.__setattr__(self, "length_m", length.copy())
        object.__setattr__(self, "width_m", width.copy())

    @classmethod
    def constant(
        cls, count: int, *, length_m: float = 4.8, width_m: float = 1.8
    ) -> "VehicleDimensions":
        if int(count) < 1:
            raise ValueError("vehicle count must be positive")
        return cls(
            np.full(int(count), float(length_m), dtype=np.float64),
            np.full(int(count), float(width_m), dtype=np.float64),
        )

    @classmethod
    def from_array(cls, values: np.ndarray) -> "VehicleDimensions":
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError("vehicle_dimensions must have shape [agents,2]")
        return cls(array[:, 0], array[:, 1])


@dataclass(frozen=True)
class SceneGeometry:
    """Straight-lane scene topology in the execution coordinate frame."""

    lane_ids: np.ndarray
    centerlines: tuple[np.ndarray, ...]
    lane_widths: np.ndarray
    adjacency: np.ndarray

    def __post_init__(self) -> None:
        ids = np.asarray(self.lane_ids, dtype=np.int64)
        widths = np.asarray(self.lane_widths, dtype=np.float64)
        adjacency = np.asarray(self.adjacency, dtype=np.int64)
        centers = tuple(np.asarray(line, dtype=np.float64) for line in self.centerlines)
        count = len(ids)
        if count < 1 or widths.shape != (count,) or adjacency.shape != (count, 2):
            raise ValueError("lane ids, widths and adjacency have inconsistent shapes")
        if len(centers) != count or any(
            line.ndim != 2 or line.shape[1] != 2 or len(line) < 2 for line in centers
        ):
            raise ValueError("every centerline must contain at least two xy points")
        if len(np.unique(ids)) != count or np.any(widths <= 0.0):
            raise ValueError("lane ids must be unique and widths positive")
        if not np.isfinite(widths).all() or any(
            not np.isfinite(line).all() for line in centers
        ):
            raise ValueError("scene geometry must be finite")
        valid_neighbors = (adjacency == -1) | np.isin(adjacency, ids)
        if not bool(valid_neighbors.all()):
            raise ValueError("adjacency contains an unknown lane id")
        object.__setattr__(self, "lane_ids", ids.copy())
        object.__setattr__(self, "lane_widths", widths.copy())
        object.__setattr__(self, "adjacency", adjacency.copy())
        object.__setattr__(self, "centerlines", tuple(line.copy() for line in centers))

    @classmethod
    def straight_template(
        cls,
        *,
        lane_centers_y: np.ndarray,
        lane_widths: np.ndarray | float = 3.6,
        x_min: float = -10_000.0,
        x_max: float = 10_000.0,
    ) -> "SceneGeometry":
        centers = np.asarray(lane_centers_y, dtype=np.float64)
        if centers.ndim != 1 or len(centers) < 1:
            raise ValueError("lane_centers_y must be a non-empty vector")
        widths = np.broadcast_to(
            np.asarray(lane_widths, dtype=np.float64), centers.shape
        )
        ids = np.arange(len(centers), dtype=np.int64)
        adjacency = _lane_adjacency(centers, widths, ids)
        lines = tuple(
            np.asarray(((x_min, y), (x_max, y)), dtype=np.float64) for y in centers
        )
        return cls(ids, lines, widths, adjacency)

    @classmethod
    def from_map_polylines(
        cls, map_polylines: np.ndarray, map_polyline_valid: np.ndarray
    ) -> "SceneGeometry":
        polylines = np.asarray(map_polylines, dtype=np.float64)
        valid = np.asarray(map_polyline_valid, dtype=bool)
        if polylines.ndim != 3 or polylines.shape[-1] < 2:
            raise ValueError("map_polylines must have shape [lanes,points,features]")
        if valid.shape != polylines.shape[:2]:
            raise ValueError("map validity must match the first two map dimensions")
        keep = np.flatnonzero(valid.sum(axis=1) >= 2)
        if not len(keep):
            raise ValueError("scene map contains no valid lane with two points")
        centers: list[np.ndarray] = []
        widths: list[float] = []
        for index in keep:
            line = polylines[index, valid[index], :2]
            centers.append(line.copy())
            width_values = (
                polylines[index, valid[index], 4]
                if polylines.shape[-1] > 4
                else np.empty(0)
            )
            width_values = width_values[
                np.isfinite(width_values) & (width_values > 0.0)
            ]
            widths.append(float(np.median(width_values)) if len(width_values) else 3.6)
        ids = keep.astype(np.int64)
        lateral = np.asarray([line[:, 1].mean() for line in centers])
        adjacency = _lane_adjacency(lateral, np.asarray(widths), ids)
        return cls(ids, tuple(centers), np.asarray(widths), adjacency)

    def lane_index(self, lane_id: int) -> int:
        matches = np.flatnonzero(self.lane_ids == int(lane_id))
        if not len(matches):
            raise KeyError(f"unknown lane id {lane_id}")
        return int(matches[0])

    def center_y(self, lane_id: int) -> float:
        return float(self.centerlines[self.lane_index(lane_id)][:, 1].mean())

    def nearest_lane(self, position_xy: np.ndarray) -> int:
        point = np.asarray(position_xy, dtype=np.float64)
        if point.shape != (2,):
            raise ValueError("position must have shape [2]")
        distances = []
        for line in self.centerlines:
            segment = line[-1] - line[0]
            fraction = np.dot(point - line[0], segment) / max(
                np.dot(segment, segment), 1e-12
            )
            projection = line[0] + np.clip(fraction, 0.0, 1.0) * segment
            distances.append(float(np.linalg.norm(point - projection)))
        return int(self.lane_ids[int(np.argmin(distances))])

    def adjacent_lane(self, lane_id: int, side: int) -> int | None:
        if int(side) not in (-1, 1):
            raise ValueError("side must be -1 or 1")
        neighbor = int(self.adjacency[self.lane_index(lane_id), 0 if side < 0 else 1])
        return None if neighbor < 0 else neighbor

    def relative_lane(self, source_lane_id: int, target_lane_id: int) -> int:
        """Signed lateral rank difference, independent of raw map lane IDs."""
        centers = np.asarray([line[:, 1].mean() for line in self.centerlines])
        order = np.argsort(centers, kind="stable")
        rank = np.empty(len(order), dtype=np.int64)
        rank[order] = np.arange(len(order), dtype=np.int64)
        source = self.lane_index(source_lane_id)
        target = self.lane_index(target_lane_id)
        return int(rank[target] - rank[source])

    @property
    def lateral_bounds(self) -> tuple[float, float]:
        low = min(
            float(line[:, 1].mean() - 0.5 * width)
            for line, width in zip(self.centerlines, self.lane_widths)
        )
        high = max(
            float(line[:, 1].mean() + 0.5 * width)
            for line, width in zip(self.centerlines, self.lane_widths)
        )
        return low, high

    def boundary_clearance(
        self,
        positions_xy: np.ndarray,
        *,
        vehicle_width_m: np.ndarray | float | None = None,
    ) -> np.ndarray:
        """Signed lateral clearance: positive inside, negative beyond the road edge."""
        positions = np.asarray(positions_xy, dtype=np.float64)
        if positions.shape[-1] != 2:
            raise ValueError("positions must end in xy")
        half_width = (
            0.0
            if vehicle_width_m is None
            else 0.5 * np.asarray(vehicle_width_m, dtype=np.float64)
        )
        half_width = np.broadcast_to(half_width, positions.shape[:-1])
        if not np.isfinite(half_width).all() or np.any(half_width < 0.0):
            raise ValueError("vehicle widths must be finite and non-negative")
        low, high = self.lateral_bounds
        return np.minimum(
            positions[..., 1] - half_width - low,
            high - positions[..., 1] - half_width,
        )

    def offroad(
        self,
        positions_xy: np.ndarray,
        *,
        vehicle_width_m: np.ndarray | float | None = None,
        margin_m: float = 0.0,
    ) -> np.ndarray:
        return (
            self.boundary_clearance(positions_xy, vehicle_width_m=vehicle_width_m)
            < -margin_m
        )


def rectangle_separation(position, heading, dimensions, padding):
    """Pairwise SAT gap in metres; <=0 means intersecting Minkowski boxes.

    Padding is in world x/y and includes corner forecast error. Projecting its
    support on each separating axis avoids pretending heading is certain.
    """
    forward = torch.stack((heading.cos(), heading.sin()), -1)
    lateral = torch.stack((-heading.sin(), heading.cos()), -1)
    basis = torch.stack((forward, lateral), -2)
    half = dimensions[:, None] * 0.5
    delta = position[..., None, :, :] - position[..., :, None, :]
    gap = None
    for owner in (False, True):
        axes = basis[..., None, :, :, :] if owner else basis[..., :, None, :, :]
        projection = (delta[..., None, :] * axes).sum(-1).abs()
        own_radius = (
            (basis[..., :, None, :, None, :] * axes[..., None, :, :]).sum(-1).abs()
            * half[..., :, None, :, None]
        ).sum(-2)
        other_radius = (
            (basis[..., None, :, :, None, :] * axes[..., None, :, :]).sum(-1).abs()
            * half[..., None, :, :, None]
        ).sum(-2)
        uncertainty = (
            (padding[..., :, None, :] + padding[..., None, :, :])[..., None, :]
            * axes.abs()
        ).sum(-1)
        local = (projection - own_radius - other_radius - uncertainty).amax(-1)
        gap = local if gap is None else torch.maximum(gap, local)
    return gap


def paired_rectangle_separation(pi, hi, di, pj, hj, dj):
    """SAT separation of corresponding rectangles; leading axes broadcast."""
    dx, dy = (pj - pi).unbind(-1)
    ci, si, cj, sj = hi.cos(), hi.sin(), hj.cos(), hj.sin()
    cosine = (ci * cj + si * sj).abs()
    sine = (si * cj - ci * sj).abs()
    li, wi = (di * 0.5).unbind(-1)
    lj, wj = (dj * 0.5).unbind(-1)
    return torch.maximum(
        torch.maximum(
            (dx * ci + dy * si).abs() - li - lj * cosine - wj * sine,
            (-dx * si + dy * ci).abs() - wi - lj * sine - wj * cosine,
        ),
        torch.maximum(
            (dx * cj + dy * sj).abs() - lj - li * cosine - wi * sine,
            (-dx * sj + dy * cj).abs() - wj - li * sine - wi * cosine,
        ),
    )
