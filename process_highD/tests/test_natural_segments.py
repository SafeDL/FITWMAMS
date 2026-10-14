from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from process_highD.scripts.fit_evt import DEFAULT_CONFIG_PATH as EVT_CONFIG
from process_highD.scripts.extract_highd_natural_segments import (
    DEFAULT_CONFIG_PATH as EXTRACT_CONFIG,
)
from process_highD.scripts.play_highd_natural_events import (
    DEFAULT_CONFIG_PATH as PLAYBACK_CONFIG,
)
from process_highD.scripts.prepare_highd_sequences import CONFIG as SEQUENCE_CONFIG
from process_highD.src.natural_segments import (
    NaturalSegmentOptions,
    _audit_lateral_events,
    validate_lateral_integrity,
)


def test_config_layout_keeps_data_paths() -> None:
    root = Path(__file__).resolve().parents[2]
    config_path = root / "process_highD/config.yaml"
    assert EXTRACT_CONFIG == PLAYBACK_CONFIG == config_path
    assert EVT_CONFIG == SEQUENCE_CONFIG == config_path

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert (config_path.parent / config["paths"]["raw_dir"]).resolve() == (
        root / "highD_dataset/Matlab/data"
    )
    assert (config_path.parent / config["paths"]["output_dir"]).resolve() == (
        root / "results/evt"
    )
    assert (
        SEQUENCE_CONFIG.parent
        / config["paths"]["highd_preprocessing_config"]
    ).resolve() == config_path
    cache_dir = SEQUENCE_CONFIG.parent / config["paths"]["sequence_cache_dir"]
    assert cache_dir.resolve() == root / (
        "results/highd_shared_training_data/highd_sequence_cache"
    )


def _vehicle(vehicle_id: int, lane: np.ndarray, x: float) -> dict:
    frames = np.arange(150, dtype=np.int64)
    return {
        "id": vehicle_id,
        "frames": frames,
        "initial": 0,
        "final": 149,
        "continuous": True,
        "frame_to_pos": None,
        "x": np.full(150, x, dtype=np.float32),
        "lane": lane.astype(np.int16),
        "vy_left": np.zeros(150, dtype=np.float32),
        "abnormal": np.zeros(150, dtype=bool),
        "class": "Car",
    }


def _audit(crossing: int, *, source_lane: int = 2) -> dict:
    ego_lane = np.full(150, 3, dtype=np.int16)
    target_lane = np.full(150, source_lane, dtype=np.int16)
    target_lane[crossing:] = 3
    vehicles = {
        0: _vehicle(0, ego_lane, 0.0),
        1: _vehicle(1, target_lane, 20.0),
    }
    frame_index = {frame: np.asarray((0, 1), dtype=np.int64) for frame in range(150)}
    lane_info = {
        "lanes": {
            1: {"direction": 1},
            2: {"direction": 1},
            3: {"direction": 1},
        },
        "direction_1_lanes": [1, 2, 3],
        "direction_2_lanes": [],
    }
    return _audit_lateral_events(
        ego_id=0,
        vehicle_ids=[0, 1],
        start_frame=0,
        options=NaturalSegmentOptions(),
        vehicles=vehicles,
        frame_index=frame_index,
        lane_info=lane_info,
    )


def test_complete_lane_change_is_labeled_as_strict_cutin() -> None:
    audit = _audit(50)
    assert audit["complete"]
    assert audit["num_lane_changes"] == 1
    assert audit["num_strict_cutins"] == 1
    assert audit["primary_cutin_cross_frame"] == 50


def test_lane_change_without_two_seconds_post_context_is_rejected() -> None:
    audit = _audit(120)
    assert not audit["complete"]
    assert audit["reject_reason"] == "insufficient_post_cross_context"


def test_non_adjacent_lane_change_is_rejected() -> None:
    audit = _audit(50, source_lane=1)
    assert not audit["complete"]
    assert audit["reject_reason"] == "non_adjacent_lane_change"


def test_downstream_contract_rejects_legacy_segments() -> None:
    with pytest.raises(RuntimeError, match="predates lateral-event"):
        validate_lateral_integrity(
            pd.DataFrame({"segment_id": ["old"]}),
            source="legacy.csv",
        )
    validate_lateral_integrity(
        pd.DataFrame({"lateral_event_complete": [1, 1]}),
        source="current.csv",
    )
