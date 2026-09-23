import numpy as np

from interactive_behavior_world_model.evaluation.geometry import (
    collision_matrix,
    oriented_box_corners,
    oriented_boxes_overlap,
    straight_road_offroad,
)


def test_true_size_boxes_overlap_and_edge_contact_does_not():
    a = oriented_box_corners(
        np.asarray([0.0, 0.0]), np.asarray(0.0), np.asarray(4.0), np.asarray(2.0)
    )
    b = oriented_box_corners(
        np.asarray([3.9, 0.0]), np.asarray(0.0), np.asarray(4.0), np.asarray(2.0)
    )
    edge = oriented_box_corners(
        np.asarray([4.0, 0.0]), np.asarray(0.0), np.asarray(4.0), np.asarray(2.0)
    )
    assert bool(oriented_boxes_overlap(a, b))
    assert not bool(oriented_boxes_overlap(a, edge))


def test_padding_never_collides():
    states = np.zeros((1, 3, 6), np.float32)
    states[..., 2] = 10.0
    states[0, 1, 0] = 1.0
    valid = np.asarray([[True, True, False]])
    matrix = collision_matrix(
        states, valid, np.asarray([4.0, 4.0, 10.0]), np.asarray([2.0, 2.0, 10.0])
    )
    assert matrix[0, 0, 1] and matrix[0, 1, 0]
    assert not matrix[0, 2].any() and not matrix[0, :, 2].any()


def test_straight_road_uses_vehicle_corners():
    states = np.zeros((1, 1, 6), np.float32)
    states[..., 2] = 10.0
    valid = np.ones((1, 1), bool)
    lines = np.zeros((1, 8, 6), np.float32)
    lines[..., 1] = 0.0
    lines[..., 4] = 4.0
    line_valid = np.ones((1, 8), bool)
    assert not straight_road_offroad(
        states, valid, np.asarray([4.0]), np.asarray([2.0]), lines, line_valid
    )[0, 0]
    states[..., 1] = 1.6
    assert straight_road_offroad(
        states, valid, np.asarray([4.0]), np.asarray([2.0]), lines, line_valid
    )[0, 0]
