import numpy as np
from PIL import Image

from hierarchical_world_model.src.npc_experiments.animations import (
    render_movie,
    select_examples,
    select_lane_example,
    vehicle_polygons,
)


def test_density_selection_uses_only_initial_state_not_outcome_or_future():
    cohorts = {
        str(index): {
            "valid": np.arange(9)[None] < index,
            "bounds": np.array([0, 300]),
            "lane_centers": np.array([0, 4, 8]),
        }
        for index in range(1, 10)
    }
    selected = select_examples(cohorts)
    assert [key for key, _ in selected] == ["2", "5", "8"]
    for cohort in cohorts.values():
        cohort["states"] = np.full((100, 9, 6), np.nan)
        cohort["completed_changes"] = np.full(9, 999)
    assert select_examples(cohorts) == selected


def test_vehicle_render_preserves_physical_dimensions_and_heading():
    states = np.array([[10, 4, 30, 0, 0, 0], [20, 8, 0, 30, 0, 0]], float)
    dimensions = np.array([[5, 2], [12, 2.5]])
    polygons = vehicle_polygons(states, dimensions)
    np.testing.assert_allclose(polygons.mean(1), states[:, :2])
    np.testing.assert_allclose(
        np.linalg.norm(polygons[:, 1] - polygons[:, 0], axis=1), dimensions[:, 0]
    )
    np.testing.assert_allclose(
        np.linalg.norm(polygons[:, 2] - polygons[:, 1], axis=1), dimensions[:, 1]
    )
    np.testing.assert_allclose(polygons[1, 1] - polygons[1, 0], [0, 12], atol=1e-12)


def test_lane_example_uses_all_roads_and_only_complete_autonomous_maneuvers(tmp_path):
    import pandas as pd

    from hierarchical_world_model.src.core.utils import save_json

    events, execution = [], {}
    for index, duration in enumerate((3.0, 4.0, 5.0)):
        key = f"{index + 1:02d}_1"
        event = {
            "source": "model",
            "repeat": 0,
            "vehicle": 117,
            "start_frame": 50,
            "duration": duration,
            "observed_complete": True,
        }
        events.append(
            pd.DataFrame(
                [
                    event,
                    {**event, "source": "real", "duration": 100},
                    {**event, "repeat": 1, "duration": 100},
                    {**event, "vehicle": 118, "duration": 100},
                    {**event, "observed_complete": False, "duration": 100},
                ]
            ).assign(cohort=key)
        )
        execution[key] = {
            "runs": [{"repeat": 0, "lane_starts": [{"vehicle": 117, "tick": 48}]}]
        }
    pd.concat(events).to_csv(tmp_path / "events.csv", index=False)
    save_json(execution, tmp_path / "runs.json")
    (key, event), count = select_lane_example(tmp_path)
    assert count == 3
    assert key == "02_1" and event["duration"] == 4.0
    assert event["autonomous_start"] == {"vehicle": 117, "tick": 48}


def test_gif_uses_executed_frames_and_declares_playback_time(tmp_path):
    states = np.zeros((26, 1, 6), np.float32)
    states[:, 0, 0] = 30 + np.arange(26)
    states[:, 0, 1] = 4
    states[:, 0, 2] = 25
    cohort = {
        "recording_id": np.asarray(4),
        "direction": np.asarray(1),
        "bounds": np.array([0, 330]),
        "vehicle_ids": np.array([117]),
        "dimensions": np.array([[5, 2]]),
        "lane_centers": np.array([0, 4, 8]),
        "lane_widths": np.full(3, 4),
    }
    result = render_movie(
        cohort,
        states,
        np.ones((26, 1), bool),
        {
            "entry_tick": np.array([0]),
            "exit_tick": np.array([-1]),
            "queue_vehicles": np.zeros(26, int),
        },
        tmp_path / "traffic.gif",
        "Autonomous traffic",
    )
    assert result["first_frame"] == 0 and result["last_frame"] == 25
    assert result["simulation_seconds"] == 1.0
    assert result["playback_speed"] == 1
    assert result["fps"] == 25
    assert result["frame_count"] == 26
    assert result["format_reference"] == "results/evt/playbacks/"
    assert {path.name for path in tmp_path.iterdir()} == {"traffic.gif"}
    assert "preview" not in result
    with Image.open(tmp_path / "traffic.gif") as movie:
        assert movie.size == (1440, 360)
        assert movie.n_frames == result["frame_count"]
        duration = 0
        for frame in range(movie.n_frames):
            movie.seek(frame)
            assert movie.info["duration"] == 40
            duration += movie.info["duration"]
        assert duration == round(result["gif_duration_seconds"] * 1000)
