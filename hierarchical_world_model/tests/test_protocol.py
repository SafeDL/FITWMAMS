"""Protocol gate contract tests."""

from __future__ import annotations

from hierarchical_world_model.src.protocol import (
    check_ams_readiness_gate,
    check_formal_manifest_gate,
    ACCEPTANCE_GATES,
)


def test_ams_readiness_gate_requires_readiness_and_finite_interfaces():
    readiness = {
        "formal_checkpoint_config_match": True,
        "world_serialization_exact": True,
        "same_world_same_ads_exact": True,
        "snapshot_restore_exact": True,
        "branch_changes_ego_trajectory": True,
        "evt_score_monotone_on_calibration_probe": True,
        "finite_state_rate": 1.0,
        "finite_evt_score_rate": 1.0,
    }
    assert check_ams_readiness_gate(readiness)

    degraded = dict(readiness)
    degraded["finite_evt_score_rate"] = 0.99
    assert not check_ams_readiness_gate(degraded)


def test_formal_manifest_gate_checks_contract_version_and_required_fields():
    manifest = {
        "protocol_version": ACCEPTANCE_GATES["protocol_version"],
        "checkpoint_sha256": "cafebabe",
        "code_commit": "123",
        "worktree_clean_at_start": True,
    }
    assert check_formal_manifest_gate(manifest)

    wrong = dict(manifest)
    wrong["protocol_version"] = "legacy"
    assert not check_formal_manifest_gate(wrong)
