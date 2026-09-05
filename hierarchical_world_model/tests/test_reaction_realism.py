from __future__ import annotations

import numpy as np

from hierarchical_world_model.src.reaction_realism import (
    ReactionRealismReference,
    SupportedEventPool,
    build_reaction_realism_reference,
    mloo_rewards,
    realism_metric,
)


def _event_arrays(rows: int = 3) -> tuple[dict[str, np.ndarray], np.ndarray]:
    states = np.zeros((rows, 174, 7, 6), np.float32)
    valid = np.zeros((rows, 174, 7), bool)
    valid[:, :, :2] = True
    states[:, :, 0, 0], states[:, :, 0, 2] = 100.0, 20.0
    states[:, :, 1, 0], states[:, :, 1, 2] = 80.0, 21.0
    # A causal 1 m/s² parent brake at frame 30.  The previous action is zero.
    states[:, 31:, 0, 2] = 19.96
    return {"agent_states": states, "agent_valid": valid}, np.arange(rows, dtype=np.int64)


def test_reference_mines_independent_ego_parent_support_events(tmp_path):
    arrays, rows = _event_arrays()
    reference = build_reaction_realism_reference(arrays, rows, minimum_events=2, window_frames=5)
    assert reference.supported_cells
    cell = reference.supported_cells[0]
    assert reference.event_counts[cell] == 3
    assert len(reference.events.row_index) == 3
    reference.save(tmp_path)
    loaded = ReactionRealismReference.load(tmp_path)
    assert loaded.supported_cells == reference.supported_cells
    np.testing.assert_array_equal(loaded.events.row_index, reference.events.row_index)


def test_evaluation_view_keeps_only_train_admitted_cells():
    arrays, rows = _event_arrays()
    reference = build_reaction_realism_reference(
        arrays, rows, minimum_events=1, window_frames=5,
    )
    cell = reference.supported_cells[0]
    view = reference.with_supported_cells((cell, 999), minimum_events=1)
    assert view.supported_cells == (cell,)
    assert view.distributions is reference.distributions
    assert view.minimum_events == 1


def test_support_count_is_independent_sequence_count_and_event_cell_uses_no_future_values():
    arrays, _ = _event_arrays(rows=3)
    # Duplicate source-row identifiers represent the same recording and must
    # not inflate a support threshold.
    duplicated_rows = np.asarray((7, 7, 8), np.int64)
    duplicate_reference = build_reaction_realism_reference(
        arrays, duplicated_rows, minimum_events=3, window_frames=5,
    )
    assert not duplicate_reference.supported_cells
    first = build_reaction_realism_reference(arrays, np.arange(3), minimum_events=1, window_frames=5)
    # Alter only a post-event follower state. The event's support bin/onset is
    # unchanged; only the offline reference distribution is allowed to vary.
    changed = {name: value.copy() for name, value in arrays.items()}
    changed["agent_states"][:, 34:, 1, 2] += 3.0
    second = build_reaction_realism_reference(changed, np.arange(3), minimum_events=1, window_frames=5)
    np.testing.assert_array_equal(first.events.cell, second.events.cell)
    np.testing.assert_array_equal(first.events.onset_step, second.events.onset_step)


def test_realism_metric_and_mloo_are_permutation_invariant():
    target = np.zeros((20, 6), np.float32)
    reference = ReactionRealismReference(
        distributions={0: target}, scales={0: np.ones(6, np.float32)}, event_counts={0: 20},
        supported_cells=(0,), events=SupportedEventPool(
            np.empty(0, np.int64), np.empty(0, np.int16), np.empty(0, np.int8), np.empty(0, np.int16),
        ), source_rows_sha256="test", window_frames=5, minimum_events=1,
    )
    perfect = np.zeros((3, 5, 6), np.float32)
    shifted = perfect + 2.0
    assert realism_metric(perfect, reference, 0)[0] > realism_metric(shifted, reference, 0)[0]
    trajectories = perfect.copy(); trajectories[2] += 4.0
    reward, _ = mloo_rewards(trajectories, reference, 0)
    assert abs(float(reward.sum())) < 1.e-5
    assert reward[2] < 0.0
    permutation = np.asarray((2, 0, 1))
    shuffled, _ = mloo_rewards(trajectories[permutation], reference, 0)
    np.testing.assert_allclose(shuffled, reward[permutation])
