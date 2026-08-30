"""Unit contracts for the A0 necessity gate."""

from __future__ import annotations

import numpy as np

from hierarchical_world_model.src.a0_necessity import ROLE_NAMES, role_masks


def test_role_masks_exclude_same_rear_and_identify_retained_roles() -> None:
    states = np.zeros((1, 80, 7, 6), np.float32)
    valid = np.ones((1, 80, 7), bool)
    states[:, :, 0, 2] = 20.0
    states[:, :, 1, 0] = 20.0
    # Agent index 2 is ``same_rear`` and is invalidated by the shared scope.
    states[:, :, 4, 0] = -15.0
    states[:, :, 4, 1] = 3.6
    states[:, :, 6, 0] = -15.0
    states[:, :, 6, 1] = -3.6
    states[:, :, 3, 0] = 10.0
    states[:, :, 3, 1] = 3.6
    states[:, 30:, 3, 1] = 0.0
    states[:, :, 5, 0] = -12.0
    states[:, :, 5, 1] = 3.6
    states[:, 50:, 0, 1] = 3.6
    masks = role_masks(states, valid, time_index=20, future_frames=40)
    assert tuple(masks) == ROLE_NAMES
    assert masks["same_lane_front"][0, 0]
    assert masks["adjacent_rear_left"][0, 3]
    assert masks["adjacent_rear_right"][0, 5]
    assert masks["cut_in_or_committed"][0, 2]
    assert masks["target_lane_follower"][0, 4]
    assert not masks["same_lane_front"][0, 1]


def test_future_labels_do_not_change_current_geometry_roles() -> None:
    states = np.zeros((1, 80, 7, 6), np.float32)
    valid = np.ones((1, 80, 7), bool)
    states[:, :, 1, 0] = 20.0
    states[:, :, 3, 0] = 10.0
    states[:, :, 3, 1] = 3.6
    before = role_masks(states, valid, time_index=20, future_frames=40)
    states[:, 40:, 3, 1] = 0.0
    after = role_masks(states, valid, time_index=20, future_frames=40)
    assert np.array_equal(before["same_lane_front"], after["same_lane_front"])
    assert np.array_equal(before["adjacent_rear_left"], after["adjacent_rear_left"])
    assert not before["cut_in_or_committed"][0, 2]
    assert after["cut_in_or_committed"][0, 2]
