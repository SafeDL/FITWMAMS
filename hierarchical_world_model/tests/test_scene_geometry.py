from __future__ import annotations

import numpy as np
import pytest
import torch

from hierarchical_world_model.src.core.sequential_dataset import recording_lane_map
from hierarchical_world_model.src.highway import HighwayEnvTraffic
from hierarchical_world_model.src.scene_geometry import SceneGeometry, VehicleDimensions
from hierarchical_world_model.src.stochastic_drivers.online import (
    nearest_leader_observation,
)


@pytest.mark.parametrize("sign", (-1.0, 1.0))
def test_recorded_road_preserves_direction_width_and_carriageway_separation(sign):
    metadata = {
        "upperLaneMarkings": [1.0, 4.0, 8.0],
        "lowerLaneMarkings": [20.0, 23.5, 27.5],
    }
    origin = sign * 5.0
    lanes, valid, edges = recording_lane_map(
        metadata, ego_global_y_m=origin, lateral_sign=sign
    )
    np.testing.assert_allclose(
        lanes[:, 0, 1], sign * np.array([-2.5, 1.0, 16.75, 20.5])
    )
    np.testing.assert_allclose(lanes[:, 0, 4], [3.0, 4.0, 3.5, 4.0])
    np.testing.assert_array_equal(edges, [[0, 1, 1], [1, 0, 1], [2, 3, 1], [3, 2, 1]])
    assert valid.all()


def _map(shift: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    lanes = np.zeros((3, 5, 6), np.float32)
    valid = np.ones((3, 5), bool)
    lanes[..., 0] = np.linspace(-40.0, 40.0, 5)
    lanes[:, :, 1] = np.asarray((-3.2, 0.0, 4.1))[:, None] + shift
    lanes[..., 2] = 1.0
    lanes[:, :, 4] = np.asarray((3.2, 3.6, 4.2))[:, None]
    return lanes, valid


def test_scene_geometry_preserves_nonstandard_lanes_and_adjacency() -> None:
    lanes, valid = _map()
    geometry = SceneGeometry.from_map_polylines(lanes, valid)
    assert np.allclose([geometry.center_y(i) for i in range(3)], (-3.2, 0.0, 4.1))
    assert geometry.adjacent_lane(1, -1) == 0
    assert geometry.adjacent_lane(1, 1) == 2
    assert geometry.adjacent_lane(0, -1) is None
    assert geometry.relative_lane(0, 2) == 2
    assert np.allclose(geometry.lateral_bounds, (-4.8, 6.2))


def test_scene_geometry_does_not_join_opposite_carriageways() -> None:
    centers = np.asarray((-8.43, -4.325, -0.22, 9.675, 13.735, 17.8))
    widths = np.asarray((4.32, 3.89, 4.32, 4.23, 3.89, 4.24))
    geometry = SceneGeometry.straight_template(
        lane_centers_y=centers, lane_widths=widths
    )
    assert geometry.adjacent_lane(2, -1) == 1
    assert geometry.adjacent_lane(2, 1) is None
    assert geometry.adjacent_lane(3, -1) is None
    assert geometry.adjacent_lane(3, 1) == 4

    lanes = np.zeros((len(centers), 4, 6), dtype=np.float64)
    lanes[..., 0] = np.linspace(-50.0, 50.0, 4)
    lanes[..., 1] = centers[:, None]
    lanes[..., 4] = widths[:, None]
    geometry = SceneGeometry.from_map_polylines(
        lanes, np.ones(lanes.shape[:2], dtype=bool)
    )
    assert geometry.adjacent_lane(2, -1) == 1
    assert geometry.adjacent_lane(2, 1) is None
    assert geometry.adjacent_lane(3, -1) is None
    assert geometry.adjacent_lane(3, 1) == 4


def test_highway_backend_uses_map_not_off_center_ego_and_real_dimensions() -> None:
    lanes, valid_map = _map(shift=7.0)
    geometry = SceneGeometry.from_map_polylines(lanes, valid_map)
    states = np.zeros((7, 6), np.float32)
    states[0, :4] = (0.0, 7.8, 20.0, 0.0)  # deliberately between lane centres
    states[1, :4] = (12.0, 7.0, 18.0, 0.0)
    valid = np.asarray((1, 1, 0, 0, 0, 0, 0), bool)
    dimensions = VehicleDimensions(
        np.asarray((5.2, 3.4, 4.8, 4.8, 4.8, 4.8, 4.8)),
        np.asarray((2.0, 1.6, 1.8, 1.8, 1.8, 1.8, 1.8)),
    )
    traffic = HighwayEnvTraffic()
    traffic.reset(
        states,
        valid,
        scene_geometry=geometry,
        vehicle_dimensions=dimensions,
    )
    assert traffic.scene_geometry is geometry
    assert traffic.ego.LENGTH == 5.2
    assert traffic.background[0].LENGTH == 3.4
    assert np.isclose(traffic.scene_geometry.center_y(1), 7.0)


def test_geometry_and_offroad_are_translation_invariant() -> None:
    for shift in (0.0, 23.5):
        lanes, valid_map = _map(shift)
        geometry = SceneGeometry.from_map_polylines(lanes, valid_map)
        positions = np.asarray(((0.0, shift), (0.0, shift + 6.25)))
        assert geometry.offroad(positions).tolist() == [False, True]
        assert geometry.offroad(
            np.asarray(((0.0, shift + 5.3),)),
            vehicle_width_m=np.asarray((2.0,)),
        ).item()


def test_boundary_clearance_distinguishes_small_overhang_from_center_outside():
    geometry = SceneGeometry.straight_template(lane_centers_y=np.array([0.0]))
    positions = np.array([[0, 0], [0, 0.95], [0, 2.0]])
    clearance = geometry.boundary_clearance(positions, vehicle_width_m=1.8)
    np.testing.assert_allclose(clearance, [0.9, -0.05, -1.1])
    assert geometry.offroad(positions, vehicle_width_m=1.8).tolist() == [
        False,
        True,
        True,
    ]
    assert geometry.offroad(positions, vehicle_width_m=1.8, margin_m=0.1).tolist() == [
        False,
        False,
        True,
    ]
    assert geometry.offroad(positions[:1], vehicle_width_m=1.8, margin_m=-1.0).item()
    shifted = SceneGeometry.straight_template(lane_centers_y=np.array([23.5]))
    positions[:, 1] += 23.5
    np.testing.assert_allclose(
        shifted.boundary_clearance(positions, vehicle_width_m=1.8), clearance
    )


def test_leader_gap_uses_pair_specific_half_lengths() -> None:
    states = torch.zeros(1, 7, 6)
    valid = torch.zeros(1, 7, dtype=torch.bool)
    valid[:, :3] = True
    states[0, 1, :3] = torch.tensor((0.0, 0.0, 20.0))
    states[0, 2, :3] = torch.tensor((10.0, 0.0, 18.0))
    lengths = torch.tensor(((4.8, 6.0, 2.0, 4.8, 4.8, 4.8, 4.8),))
    gap, _, _, leader = nearest_leader_observation(
        states,
        valid,
        vehicle_lengths_m=lengths,
        vehicle_widths_m=torch.full_like(lengths, 1.8),
    )
    assert leader[0, 0].item() == 2
    assert torch.isclose(gap[0, 0], torch.tensor(6.0))  # 10 - (6 + 2) / 2
