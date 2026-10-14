"""Physical identity, original-track coverage and causal-neighbor data checks."""

import numpy as np
import pandas as pd

from process_highD.src.loader import HighDRecording

from hierarchical_world_model.src.stochastic_drivers.raw_data import (
    canonical_lateral,
    extract_complete_events,
    lane_labels,
    lane_side_data,
    nearest_neighbors,
    recording_splits,
)


def test_frozen_recording_splits_cover_all_sixty_without_overlap():
    splits = recording_splits()
    assert [len(splits[name]) for name in ("train", "val", "test")] == [42, 9, 9]
    recordings = [recording for values in splits.values() for recording in values]
    assert sorted(recordings) == list(range(1, 61))


def test_physical_neighbors_cover_full_scene_and_separate_direction_and_frame():
    states = pd.DataFrame(
        {
            "x": [10, 30, 20, 5, 35, 25, 12],
            "lane": [0, 0, 0, 1, 1, 0, 0],
            "frame": [5, 5, 5, 5, 5, 5, 10],
            "direction": [1, 1, 1, 1, 1, 2, 1],
        }
    )
    front, rear, target_front, target_rear, available = nearest_neighbors(
        states, {1: np.array([2, 6]), 2: np.array([10, 14])}
    )
    assert front.tolist() == [2, -1, 1, 4, -1, -1, -1]
    assert rear[2] == 0
    assert target_front[2, 1] == 4
    assert target_rear[2, 1] == 3
    assert available[0].tolist() == [False, True]
    assert available[3].tolist() == [True, False]
    assert target_front[6, 1] == -1


def label_track(onset):
    frames = np.arange(350)
    velocity = np.where((frames >= onset) & (frames < onset + 90), 1.0, 0.0)
    y = 1.8 + np.cumsum(velocity) * 0.04
    return pd.DataFrame(
        {
            "y": y,
            "yVelocity": velocity,
            "laneId": np.where(y < 3.6, 2, 3),
            "_abnormal": False,
        },
        index=frames,
    )


def test_one_second_onset_uses_extra_frames_for_confirmation_only():
    labels, observed, _ = lane_labels(label_track(25), {2: 1.8, 3: 5.4})
    assert observed[0]
    assert labels[0] == 2
    labels, _, _ = lane_labels(label_track(26), {2: 1.8, 3: 5.4})
    assert labels[0] == 0
    track = label_track(10)
    track.loc[17, "_abnormal"] = True
    _, observed, _ = lane_labels(track, {2: 1.8, 3: 5.4})
    assert not observed[0]


def test_centering_episode_is_keep_even_when_later_real_change_has_same_direction():
    track = label_track(100)
    track.loc[10:19, "yVelocity"] = 1.0
    track["y"] = 1.8 + np.cumsum(track.yVelocity) * 0.04
    track["laneId"] = np.where(track.y < 3.6, 2, 3)
    labels, observed, audit = lane_labels(track, {2: 1.8, 3: 5.4})
    assert observed[0] and labels[0] == 0
    assert audit["onset_cues"][0] and audit["no_cross"][0]
    assert not audit["cross_confirmed"][0]


def test_confirmed_lane_start_matches_both_canonical_driving_directions():
    for sign, expected in ((1, 2), (-1, 1)):
        track = label_track(10)
        track[["y", "yVelocity"]] *= sign
        labels, observed, audit = lane_labels(track, {2: sign * 1.8, 3: sign * 5.4})
        assert observed[0] and labels[0] == expected
        assert audit["cross_confirmed"][0]


def test_unconfirmed_truncated_or_discontinuous_episode_is_censored():
    track = label_track(10)
    labels, observed, audit = lane_labels(track.iloc[:40], {2: 1.8, 3: 5.4})
    assert labels[0] == 0 and not observed[0] and audit["truncated_future"][0]
    for removed in ([40], np.arange(40, 261)):
        labels, observed, audit = lane_labels(track.drop(removed), {2: 1.8, 3: 5.4})
        assert labels[0] == 0 and not observed[0] and audit["observation_gap"][0]
    quiet = track.iloc[:35].copy()
    quiet["yVelocity"] = 0
    quiet["y"] = 1.8
    labels, observed, _ = lane_labels(quiet, {2: 1.8, 3: 5.4})
    assert observed[0] and labels[0] == 0


def test_first_opposite_crossing_cannot_be_confirmed_by_later_return():
    track = label_track(350)
    track["yVelocity"] = 0.0
    track.loc[10:19, "yVelocity"] = 1.0
    track.loc[60:149, "yVelocity"] = -1.0
    track.loc[160:249, "yVelocity"] = 1.0
    track["y"] = 5.4 + np.cumsum(track.yVelocity) * 0.04
    track["laneId"] = np.floor(track.y / 3.6).astype(int) + 2
    labels, observed, audit = lane_labels(track, {2: 1.8, 3: 5.4, 4: 9.0})
    assert observed[0] and labels[0] == 0
    assert audit["opposite_cross"][0] and audit["multiple_crosses"][0]


def test_first_aligned_crossing_stays_valid_with_multiple_later_changes():
    track = label_track(10)
    track.loc[160:249, "yVelocity"] = -1.0
    track["y"] = 1.8 + np.cumsum(track.yVelocity) * 0.04
    track["laneId"] = np.floor(track.y / 3.6).astype(int) + 2
    labels, observed, audit = lane_labels(track, {2: 1.8, 3: 5.4})
    assert observed[0] and labels[0] == 2 and audit["multiple_crosses"][0]


def test_nonadjacent_or_unmapped_transition_is_not_a_lane_choice():
    for target, centers in ((4, {2: 1.8, 3: 5.4, 4: 9.0}), (3, {2: 1.8})):
        track = label_track(10)
        track.loc[track.laneId == 3, "laneId"] = target
        labels, observed, audit = lane_labels(track, centers)
        assert labels[0] == 0 and not observed[0] and audit["nonadjacent_transition"][0]


def test_episode_unfinished_at_ten_second_horizon_is_not_a_keep_label():
    track = label_track(350)
    track.loc[10:280, "yVelocity"] = 0.16
    track["y"] = 1.8 + np.cumsum(track.yVelocity) * 0.04
    track["laneId"] = np.where(track.y < 3.6, 2, 3)
    labels, observed, audit = lane_labels(track, {2: 1.8, 3: 5.4})
    assert labels[0] == 0 and not observed[0]
    assert audit["horizon_unresolved"][0]
    assert not audit["truncated_future"][0]


def complete_recording(gap=False):
    frames = np.arange(1, 276)
    u = np.clip((frames - 50) / 150, 0, 1)
    y = 1.8 + 3.6 * (10 * u**3 - 15 * u**4 + 6 * u**5)
    vy = np.gradient(y, 0.04)
    ay = np.gradient(vy, 0.04)
    track = pd.DataFrame(
        {
            "id": 7,
            "frame": frames,
            "y": y,
            "xVelocity": 25.0,
            "yVelocity": vy,
            "yAcceleration": ay,
            "_abnormal": False,
        }
    )
    if gap:
        track = track[track.frame != 75]
    track = track.set_index(["id", "frame"])
    meta = pd.DataFrame(
        {
            "id": [7],
            "drivingDirection": [2],
            "class": ["Car"],
            "width": [4.5],
            "height": [1.8],
        }
    ).set_index("id")
    return HighDRecording(1, track, meta, {"frameRate": 25})


def test_complete_raw_event_retains_pre_post_history_and_does_not_bridge_gaps():
    geometry = {
        "direction_2_lanes": [2, 3],
        "lanes": {2: {"center": 1.8}, 3: {"center": 5.4}},
    }
    events, states, offsets, _ = extract_complete_events(
        complete_recording(), "val", geometry
    )
    assert len(events) == 1
    event = events.iloc[0]
    assert event.recording_id == 1 and event.vehicle_id == 7 and event.split == "val"
    assert offsets[-1] == event.end_frame - event.start_frame + 51
    assert len(states) == offsets[-1]
    assert len(states) * 0.04 > 5.96
    assert event.start_frame < event.cross_frame < event.end_frame
    broken, _, _, _ = extract_complete_events(
        complete_recording(gap=True), "val", geometry
    )
    assert len(broken) == 0


def test_direction_two_mirroring_matches_world_lateral_actions():
    recording = complete_recording()
    geometry = {
        "direction_2_lanes": [2, 3],
        "lanes": {2: {"center": 1.8}, 3: {"center": 5.4}},
    }
    original = recording.tracks.copy()
    canonical_lateral(recording, geometry)
    np.testing.assert_allclose(recording.tracks.y, -original.y)
    np.testing.assert_allclose(recording.tracks.yVelocity, -original.yVelocity)
    events, states, _, _ = extract_complete_events(recording, "val", geometry)
    assert events.iloc[0].direction == -1
    assert events.iloc[0].target_y_m < events.iloc[0].source_y_m
    assert states[len(states) // 2, 2] < 0


def test_mobil_compact_neighbors_include_old_and_new_follower_benefits():
    data = {
        "recording": np.array([1]),
        "vehicle": np.array([3]),
        "frame": np.array([50]),
        "own_speed": np.array([20.0]),
        "own_length": np.array([4.0]),
        "current_front_gap": np.array([20.0]),
        "current_front_speed": np.array([18.0]),
        "current_rear_gap": np.array([10.0]),
        "current_rear_speed": np.array([22.0]),
        "target_front_gap": np.array([[50.0, 1e6]]),
        "target_front_speed": np.array([[22.0, 20.0]]),
        "target_rear_gap": np.array([[25.0, 1e6]]),
        "target_rear_speed": np.array([[21.0, 20.0]]),
        "available": np.array([[True, False]]),
        "side_label": np.array([[True, False]]),
        "features": np.zeros((1, 2, 8)),
    }
    values = lane_side_data(data, np.array([33, 2, 1.3, 1.2, 2]))
    assert len(values["frame"]) == 1
    assert values["side"][0] == -1
    assert values["label"][0]
    assert values["mobil_social"][0] != values["mobil_own"][0]
