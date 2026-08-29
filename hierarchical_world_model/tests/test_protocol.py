"""Protocol gate contract tests."""

from __future__ import annotations

from hierarchical_world_model.src.protocol import (
    check_ams_readiness_gate,
    check_formal_manifest_gate,
    check_sampled_end_to_end_gate,
    ACCEPTANCE_GATES,
    SAMPLED_END_TO_END,
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
        "protocol": ACCEPTANCE_GATES["protocol"],
        "checkpoint_sha256": "cafebabe",
        "code_commit": "123",
        "worktree_clean_at_start": True,
    }
    assert check_formal_manifest_gate(manifest)

    wrong = dict(manifest)
    wrong["protocol"] = "incompatible_protocol"
    assert not check_formal_manifest_gate(wrong)


def test_sampled_gate_does_not_claim_replay_or_crn_evidence():
    worlds = SAMPLED_END_TO_END["worlds"]
    sampled = {
        "worlds": worlds,
        "response_steps": SAMPLED_END_TO_END["response_steps"],
        "ADS_conditioned_sampled_world_risk": {
            "hold_current": {"finite_state_rate": 1.0, "risk_mean": 0.1},
            "idm": {"finite_state_rate": 1.0, "risk_mean": 0.2},
        },
        "sampled_K_to_diffusion_nonpaired_fidelity": {"k_adherence": {}},
        "paired_failure_table": {
            "both_safe": 1,
            "idm_only_failure": 1,
            "hold_only_failure": 1,
            "both_failure": 1,
        },
        "paired_world_risk": {
            "R_hold": [0.0] * worlds,
            "R_IDM": [0.0] * worlds,
            "Delta_R_IDM_minus_hold": [0.0] * worlds,
        },
        "provenance": {"code_commit": "abc", "release_tag": "v1"},
    }
    assert check_sampled_end_to_end_gate(sampled)
