import numpy as np

from hierarchical_world_model.src.lane_event_metrics import (
    lane_change_events,
    summarize_lane_change_events,
)


def lane_scene(*, edge_started: bool = False):
    frames = 150
    states = np.zeros((1, frames, 7, 6), dtype=np.float32)
    valid = np.ones((1, frames, 7), dtype=bool)
    maps = np.zeros((1, 2, 1, 2), dtype=np.float32)
    maps[0, 1, 0, 1] = 4.0
    map_valid = np.ones((1, 2, 1), dtype=bool)

    start = 0 if edge_started else 20
    stop = start + 100
    frame = np.arange(frames)
    y = np.clip((frame - start) * 4.0 / 100.0, 0.0, 4.0)
    vy = np.zeros(frames, dtype=np.float32)
    vy[start:stop] = 1.0
    states[0, :, 1, 0] = frame * 1.2
    states[0, :, 1, 1] = y
    states[0, :, 1, 2] = 30.0
    states[0, :, 1, 3] = vy
    return states, valid, maps, map_valid


def test_pooled_detector_measures_complete_lane_motion_and_censors_window_edges():
    states, valid, maps, map_valid = lane_scene()
    events = lane_change_events(states, valid, maps, map_valid)

    assert len(events) == 1
    assert 3.9 <= events[0]["duration_s"] <= 4.3
    assert events[0]["peak_lateral_speed_mps"] == 1.0
    summary = summarize_lane_change_events(events)
    assert summary["count"] == 1
    assert summary["duration_q05_q50_q95_s"] == [events[0]["duration_s"]] * 3

    edge_states, edge_valid, edge_maps, edge_map_valid = lane_scene(edge_started=True)
    assert lane_change_events(edge_states, edge_valid, edge_maps, edge_map_valid) == []
