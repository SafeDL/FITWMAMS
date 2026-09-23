"""Formal IDM runner provenance preflight contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch")

from idm_ams.src import world_subset_runner
from idm_ams.src import trafficbots_runner


def test_formal_runner_rejects_a_dirty_worktree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        world_subset_runner,
        "_run_provenance",
        lambda config: {"repository_commit": "abc", "worktree_dirty": True},
    )
    with pytest.raises(
        RuntimeError, match="formal IDM evaluation requires a clean worktree"
    ):
        world_subset_runner._require_formal_provenance({}, Path("."))


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


def test_formal_runner_binds_manifest_to_the_current_world_artifact(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "final_world_model.pt"
    checkpoint.write_bytes(b"frozen world model")
    manifest = checkpoint.with_name("final_model_manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "checkpoint_sha256": world_subset_runner.file_sha256(checkpoint),
                "code_commit": "abc",
                "release_tag": "v1.0.0",
                "worktree_clean_at_start": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        world_subset_runner,
        "_run_provenance",
        lambda config: {"repository_commit": "abc", "worktree_dirty": False},
    )
    monkeypatch.setattr(
        world_subset_runner,
        "_world_config",
        lambda config, config_dir: (
            {"paths": {"evaluation_checkpoint": str(checkpoint)}},
            tmp_path / "release.yaml",
        ),
    )
    monkeypatch.setattr(
        world_subset_runner,
        "release_provenance",
        lambda **kwargs: {
            "code_commit": "abc",
            "release_tag": "v1.0.0",
            "worktree_clean_at_start": True,
        },
    )

    provenance = world_subset_runner._require_formal_provenance({}, tmp_path)

    assert provenance[
        "formal_artifact_checkpoint_sha256"
    ] == world_subset_runner.file_sha256(checkpoint)
    assert provenance["formal_manifest_release_tag"] == "v1.0.0"
