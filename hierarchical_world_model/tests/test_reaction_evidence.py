from __future__ import annotations

import numpy as np
import pytest

from hierarchical_world_model.src.reaction_evidence import (
    ReactionEventReference, ReactionEvents, assert_split_isolation,
    build_reaction_event_reference, energy_score, event_window,
    recording_cluster_bootstrap,
)
from hierarchical_world_model.scripts.archive.reaction_policy.evaluate import (
    _factual_noninferiority, _paired_failures, event_selection, mechanism_events,
)
from hierarchical_world_model.scripts.archive.reaction_policy.validate import (
    evaluate as evaluate_acceptance,
)
from hierarchical_world_model.src.reaction_training import ReactionRollout


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


def test_evaluation_selection_reports_each_filter_and_balances_recordings():
    count = 12
    reference = ReactionEventReference(
        split="validation",
        events=ReactionEvents(
            row_index=np.arange(count),
            recording_id=np.repeat((1, 2, 3), 4),
            leader_id=np.arange(count) + 100,
            follower_id=np.arange(count) + 200,
            absolute_onset_frame=np.arange(count) + 1000,
            local_onset_frame=np.full(count, 40),
            leader_slot=np.asarray((0,) * 11 + (1,)),
            follower_slot=np.full(count, 2),
            cell=np.asarray((2,) * 10 + (1, 2)),
            initial_conditions=np.column_stack((
                np.arange(count), np.ones((count, 4)),
            )).astype(np.float32),
            trajectory=np.zeros((count, 100, 6), np.float32),
        ),
        supported_cells=(2,), event_counts={1: 1, 2: 11},
        recording_counts={1: 1, 2: 3},
    )
    selected, audit = event_selection(reference, np.arange(10), limit=6)
    assert audit == {
        "all_events": 12,
        "supported_events": 11,
        "ego_leader_events": 11,
        "supported_ego_leader_events": 10,
        "mapped_scene_events": 10,
        "evaluated_events": 6,
        "unique_scene_rows": 6,
        "recordings": 3,
        "events_per_recording": {"1": 2, "2": 2, "3": 2},
    }
    mechanism = mechanism_events(reference, np.arange(10))
    assert len(mechanism) == 6
    assert set(reference.events.recording_id[mechanism]) == {1, 2, 3}


def _rollout(*, crash: bool, role: int = 1) -> ReactionRollout:
    diagnostics = {
        name: np.zeros((1, 2, 6), dtype=dtype)
        for name, dtype in (
            ("alpha", np.float32), ("active", bool),
            ("rule_action_ax", np.float32), ("influence_authority", np.float32),
            ("influence_role", np.int64), ("influence_parent", np.int64),
            ("influence_predicted_ttc_s", np.float32),
            ("desired_action_ax", np.float32),
        )
    }
    diagnostics["active"][:, :, 1] = True
    diagnostics["influence_role"][:, :, 1] = role
    crashed = np.zeros((1, 2, 7), bool)
    if crash:
        crashed[0, 1, (0, 2)] = True
    return ReactionRollout(
        states=np.zeros((1, 2, 7, 6), np.float32),
        background_actions=np.zeros((1, 2, 6, 2), np.float32),
        base_background_actions=np.zeros((1, 2, 6, 2), np.float32),
        ego_actions=np.zeros((1, 2, 2), np.float32),
        controller_diagnostics=diagnostics,
        collision=crashed.any(-1), crashed=crashed,
    )


def test_paired_failure_attributes_execution_and_records_telemetry():
    telemetry = []
    failures = _paired_failures(
        "constant_brake_6", np.asarray((42,)),
        {
            "calibrated_residual": _rollout(crash=True),
            "a2_transfer": _rollout(crash=False),
            "idm_only": _rollout(crash=False),
        },
        _rollout(crash=False), telemetry,
    )
    assert failures[0]["cause"] == "execution_jerk_limited"
    assert failures[0]["causal_slots"] == [2]
    assert telemetry and telemetry[0]["row_index"] == 42


def test_factual_gate_uses_hiqr_and_evidence_requirement_is_shared():
    assert _factual_noninferiority({
        "frozen_hiqr": {"ade_m": 1.0, "fde_m": 1.0, "p95_m": 1.0},
        "a2_transfer": {"ade_m": 10.0, "fde_m": 10.0, "p95_m": 10.0},
        "calibrated_residual": {"ade_m": 1.01, "fde_m": 1.01, "p95_m": 1.01},
    })
    report = {
        "factual": {"calibrated_residual": {"noninferior": True}},
        "held_out_events": {
            "events": 100, "recordings": 5,
            "arms": {"calibrated_supervised": {"energy_score_mean": 10.0}},
            "paired_energy_score": {
                "a2_transfer_minus_calibrated_residual": {"lcb95": 1.0},
                "calibrated_supervised_minus_calibrated_residual": {"lcb95": 0.0},
            },
            "paired_diagnostics": {
                "gap": {"ci95_high": 0.0, "allowed_degradation": 0.0},
            },
            "paired_rear_collision": {"a2_transfer": {"ci95_high": 0.0}},
        },
        "physical_ood": {
            "calibrated_residual": {"valid": True, "jerk_limiter_failed": False},
            "failure_analysis": {"strict_causal_regressions": 0},
        },
    }
    assert evaluate_acceptance(report, stage="supervised")["accepted"]
    report["held_out_events"]["events"] = 99
    result = evaluate_acceptance(report, stage="supervised")
    assert not result["gates"]["human_evidence_sufficient"]
    assert not result["accepted"]
