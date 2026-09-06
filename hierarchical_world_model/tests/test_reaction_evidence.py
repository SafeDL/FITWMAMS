from __future__ import annotations

import numpy as np
import pytest

from hierarchical_world_model.src.reaction_evidence import (
    ReactionEventReference, assert_split_isolation,
    build_reaction_event_reference, energy_score, event_window,
    recording_cluster_bootstrap,
)


def _arrays(rows: int = 3) -> dict[str, np.ndarray]:
    states = np.zeros((rows, 174, 7, 6), np.float32)
    valid = np.zeros((rows, 174, 7), bool)
    valid[:, 24:, :2] = True
    states[:, :, 0, 0] = 100.0
    states[:, :, 0, 2] = 20.0
    states[:, :, 1, 0] = 80.0
    states[:, :, 1, 2] = 21.0
    onset = 55
    states[:, onset:, 1, 2] += 0.08
    for frame in range(onset + 1, onset + 4):
        states[:, frame:, 0, 2] -= 0.04
    return {
        "agent_states": states,
        "agent_valid": valid,
        "agent_ids": np.asarray([[100 + row, 200 + row, -1, -1, -1, -1, -1] for row in range(rows)]),
        "row_index": np.arange(rows),
        "recording_id": np.arange(rows),
        "anchor_frame": np.full(rows, 1000),
    }


def test_events_use_real_vehicle_identity_sustained_onset_and_roundtrip(tmp_path):
    reference = build_reaction_event_reference(
        _arrays(), split="train", minimum_events=2, minimum_recordings=2,
    )
    assert len(reference.events.row_index) == 3
    assert reference.supported_cells
    assert np.all(reference.events.leader_id >= 100)
    assert np.all(reference.events.follower_id >= 200)
    assert np.all(reference.events.absolute_onset_frame == 1031)
    np.testing.assert_allclose(reference.events.initial_conditions[:, 4], 2.0, atol=1.0e-5)
    assert event_window(reference.events).shape == (3, 25, 6)
    reference.save(tmp_path)
    loaded = ReactionEventReference.load(tmp_path)
    np.testing.assert_array_equal(loaded.events.follower_id, reference.events.follower_id)


def test_short_brake_is_not_an_event_and_overlapping_windows_do_not_duplicate():
    arrays = _arrays(2)
    arrays["recording_id"][:] = 4
    arrays["agent_ids"][:] = np.asarray((10, 20, -1, -1, -1, -1, -1))
    duplicate = build_reaction_event_reference(arrays, split="train", minimum_events=1, minimum_recordings=1)
    assert len(duplicate.events.row_index) == 1
    short = _arrays(1)
    short["agent_states"][:, 57:, 0, 2] += 0.04
    short["agent_states"][:, 58:, 0, 2] += 0.04
    result = build_reaction_event_reference(short, split="train", minimum_events=1, minimum_recordings=1)
    assert not len(result.events.row_index)


def test_pair_events_are_merged_after_absolute_time_sorting():
    arrays = _arrays(2)
    arrays["recording_id"][:] = 4
    arrays["agent_ids"][:] = np.asarray((10, 20, -1, -1, -1, -1, -1))
    arrays["anchor_frame"][:] = (1100, 1000)
    reference = build_reaction_event_reference(
        arrays, split="train", minimum_events=1, minimum_recordings=1,
    )
    np.testing.assert_array_equal(
        reference.events.absolute_onset_frame, np.asarray((1031, 1131)),
    )


def test_split_isolation_rejects_shared_recordings():
    train = build_reaction_event_reference(_arrays(1), split="train", minimum_events=1, minimum_recordings=1)
    validation = build_reaction_event_reference(_arrays(1), split="validation", minimum_events=1, minimum_recordings=1)
    with pytest.raises(ValueError, match="recording"):
        assert_split_isolation(train, validation)


def test_held_out_support_is_inherited_from_training_cells():
    train = build_reaction_event_reference(
        _arrays(2), split="train", minimum_events=2, minimum_recordings=2,
    )
    validation_arrays = _arrays(1)
    validation_arrays["recording_id"] += 100
    validation = build_reaction_event_reference(
        validation_arrays, split="validation", minimum_events=100,
        minimum_recordings=5, supported_cells=train.supported_cells,
    )
    assert validation.supported_cells == train.supported_cells


def test_energy_score_and_recording_bootstrap_are_event_level():
    target = np.zeros((25, 2), np.float32)
    perfect = np.zeros((32, 25, 2), np.float32)
    shifted = perfect + 2.0
    assert energy_score(perfect, target) < energy_score(shifted, target)
    permutation = np.random.default_rng(4).permutation(32)
    assert energy_score(shifted, target) == pytest.approx(energy_score(shifted[permutation], target))
    result = recording_cluster_bootstrap(
        np.asarray((0.2, 0.1, -0.1, 0.4)), np.asarray((1, 1, 2, 3)), draws=100, seed=3,
    )
    assert result["events"] == 4 and result["recordings"] == 3
