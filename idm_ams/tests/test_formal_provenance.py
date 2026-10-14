"""Formal IDM runner provenance preflight contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch")

from idm_ams.src import world_subset_runner
from idm_ams.src import trafficbots_runner


def test_shared_world_preflight_allows_dirty_worktree(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        world_subset_runner,
        "_run_provenance",
        lambda config: {"repository_commit": "abc", "worktree_dirty": True},
    )
    monkeypatch.setattr(
        world_subset_runner,
        "_world_config",
        lambda *a: (
            {"paths": {"evaluation_checkpoint": str(tmp_path / "shared.pt")}},
            tmp_path / "world.yaml",
        ),
    )
    monkeypatch.setattr(
        world_subset_runner,
        "load_selected_components",
        lambda *a: (
            {
                "selected_arm": "rules_recovery",
                "reference_recovery_enabled": True,
                "source": "shared",
            },
            {},
        ),
    )
    monkeypatch.setattr(
        world_subset_runner,
        "_require_tail_flow_quality",
        lambda *a: {"tail_flow_quality_gates_passed": False},
    )
    result = world_subset_runner._require_formal_provenance({}, tmp_path)
    assert result["worktree_dirty"]
    assert result["reference_recovery_enabled"]
    assert not result["tail_flow_quality_gates_passed"]


def test_formal_trafficbots_runner_rejects_a_dirty_worktree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        trafficbots_runner,
        "_run_provenance",
        lambda config: {"repository_commit": "abc", "worktree_dirty": True},
    )
    with pytest.raises(
        RuntimeError,
        match="formal TrafficBots IDM evaluation requires a clean worktree",
    ):
        trafficbots_runner._require_trafficbots_provenance({}, Path("."))


def test_trafficbots_runner_rejects_historical_masked_checkpoint_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    traffic_config = tmp_path / "trafficbots.yaml"
    traffic_config.write_text(
        yaml.safe_dump(
            {
                "evaluation_scope": {
                    "schema": "highd_follower_excluded_v1",
                    "excluded_background_slots": ["same_rear"],
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        trafficbots_runner,
        "_run_provenance",
        lambda config: {"repository_commit": "abc", "worktree_dirty": False},
    )
    config = {"paths": {"trafficbots_config": str(traffic_config)}}
    with pytest.raises(ValueError, match="evaluation_scope schema is not supported"):
        trafficbots_runner._require_trafficbots_provenance(config, tmp_path)


def test_shared_world_preflight_rejects_mixed_components(tmp_path, monkeypatch):
    monkeypatch.setattr(
        world_subset_runner,
        "_run_provenance",
        lambda config: {"repository_commit": "abc", "worktree_dirty": False},
    )
    monkeypatch.setattr(
        world_subset_runner,
        "_world_config",
        lambda *a: (
            {"paths": {"evaluation_checkpoint": str(tmp_path / "old.pt")}},
            tmp_path / "world.yaml",
        ),
    )

    def reject(checkpoint):
        raise ValueError("world components differ from the published shared chain")

    monkeypatch.setattr(world_subset_runner, "load_selected_components", reject)
    with pytest.raises(ValueError, match="published shared chain"):
        world_subset_runner._require_formal_provenance({}, tmp_path)


def test_tail_flow_distribution_deviation_is_reported_not_rejected(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tail_flow/checkpoints/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"tail flow")
    (checkpoint.parent.parent / "manifest.json").write_text(
        json.dumps(
            {
                "population": "evt_tail",
                "all_quality_gates_passed": False,
            }
        ),
        encoding="utf-8",
    )
    config = {"paths": {"flow_checkpoint": str(checkpoint)}}

    result = world_subset_runner._require_tail_flow_quality(config, tmp_path)
    assert not result["tail_flow_quality_gates_passed"]


def test_generated_world_still_requires_tail_flow(tmp_path):
    checkpoint = tmp_path / "flow/checkpoints/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"flow")
    (checkpoint.parent.parent / "manifest.json").write_text(
        json.dumps({"population": "natural", "all_quality_gates_passed": True})
    )
    with pytest.raises(RuntimeError, match="EVT-tail Flow"):
        world_subset_runner._require_tail_flow_quality(
            {"paths": {"flow_checkpoint": str(checkpoint)}}, tmp_path
        )
