"""Tests for shared formal Flow-composition helpers."""

from __future__ import annotations

import numpy as np

from normalizing_flow.src.features import slot_feature_index
from world_model.src.core.flow_composition import (
    decode_flow_starts,
    write_flow_composition_report,
)
from world_model.src.core.utils import file_sha256


def test_shared_flow_start_decoder_and_report(tmp_path) -> None:
    features = np.zeros((2, 40), np.float32)
    features[:, :4] = ((20.0, 0.0, 0.0, 0.0), (22.0, 0.0, 0.0, 0.0))
    features[:, slot_feature_index("same_front", "rel_x_m")] = (15.0, 18.0)
    features[:, slot_feature_index("same_front", "rel_vx_mps")] = -3.0
    starts = {"features": features, "slot_mask": np.array(((True, False, False, False, False, False),) * 2)}
    states, valid, anchor, anchor_valid = decode_flow_starts(starts)
    assert np.allclose(states[:, 1, 2], (17.0, 19.0))
    assert valid[:, :2].all() and not valid[:, 2:].any()
    assert not anchor.any() and not anchor_valid.any()

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"shared-report")
    frames, background = 4, 6
    ego = np.zeros((2, frames, 6), np.float32)
    ego[..., 0] = np.arange(frames)[None, :]
    ego[..., 2] = 20.0
    generated = np.zeros((2, frames, background, 6), np.float32)
    generated[..., 0] = ego[..., None, 0] + 10.0 + np.arange(background)[None, None, :]
    generated[..., 2] = 15.0
    valid = np.ones((2, frames, background), bool)
    report = write_flow_composition_report(
        checkpoint=checkpoint,
        output_dir=tmp_path,
        protocol={"name": "test"},
        generated=generated,
        ego=ego,
        valid=valid,
        target=generated.copy(),
        target_ego=ego.copy(),
        target_valid=valid.copy(),
    )
    assert report["checkpoint"]["sha256"] == file_sha256(checkpoint)
    assert report["closed_loop_distribution"]["risk_variable_distribution"]["ttc_s"]["available"]
    assert (tmp_path / "flow_composition_evaluation.json").is_file()
