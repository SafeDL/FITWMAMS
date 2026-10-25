import json
import pickle
from copy import deepcopy
from zipfile import ZipFile

import numpy as np
import pytest

from ads.audit import check_archive_execution_labels, check_dqn_replay, check_records
from ads.common.instances import json_hash


def historical_archive():
    failure = {"safety_failure": True, "branch_id": "root.0", "root_sample_id": "r"}
    success = dict(failure, safety_failure=False)
    snapshots = [
        {"old_failure": [failure], "old_success": [success]},
        {"old_failure": [success], "new_failure": [failure]},
    ]
    archive = {
        "entries": {
            "old_failure": success,
            "old_success": success,
            "new_failure": failure,
        },
        "domains": {"root.0": ["old_failure", "new_failure"]},
        "neighbours": {"old_failure": "old_success", "new_failure": "old_failure"},
    }
    return archive, snapshots


def test_archive_audit_accepts_repaired_failure_becoming_later_success_neighbour():
    archive, snapshots = historical_archive()
    check_archive_execution_labels(archive, snapshots)


def test_common_partition_annotations_do_not_invalidate_executed_archives():
    archive, snapshots = historical_archive()
    for snapshot in snapshots:
        for rows in snapshot.values():
            for row in rows:
                row.update(common_branch_id="common.0", common_partition_resolved=True)
    check_archive_execution_labels(archive, snapshots)


@pytest.mark.parametrize("corruption", ["domain", "duplicate", "unexecuted"])
def test_archive_audit_rejects_fabricated_historical_roles(corruption):
    archive, snapshots = historical_archive()
    if corruption == "domain":
        archive["domains"] = {"root.1": ["old_failure"]}
        message = "failure domain label"
    elif corruption == "duplicate":
        archive["domains"]["root.0"].append("old_failure")
        message = "duplicate instances"
    else:
        archive["entries"]["old_failure"] = dict(
            archive["entries"]["old_failure"], invented_result=True
        )
        message = "absent from real execution"
    with pytest.raises(ValueError, match=message):
        check_archive_execution_labels(archive, snapshots)


def test_archive_audit_rejects_boundary_observed_only_across_different_policies():
    archive, snapshots = historical_archive()
    snapshots[0]["old_success"][0] = dict(
        snapshots[0]["old_success"][0], safety_failure=True
    )
    snapshots[1]["old_success"] = [archive["entries"]["old_success"]]
    with pytest.raises(ValueError, match="one policy snapshot"):
        check_archive_execution_labels(archive, snapshots)


def write_replay(directory, *, experiences, warmup=102):
    from gymnasium import spaces
    from stable_baselines3.common.buffers import ReplayBuffer

    replay = ReplayBuffer(
        32, spaces.Box(-1, 1, (49,), dtype=np.float32), spaces.Discrete(5), device="cpu"
    )
    for index in range(experiences):
        replay.add(
            np.zeros((1, 49), np.float32),
            np.ones((1, 49), np.float32),
            np.asarray([index % 5]),
            np.asarray([1.0]),
            np.asarray([False]),
            [{}],
        )
    with ZipFile(directory / "policy.zip", "w") as archive:
        archive.writestr(
            "data", json.dumps({"learning_starts": warmup, "buffer_size": 32})
        )
    with (directory / "replay_buffer.pkl").open("wb") as stream:
        pickle.dump(replay, stream)


def test_dqn_audit_rejects_old_replay_or_missing_second_round_experience(tmp_path):
    write_replay(tmp_path, experiences=8)
    check_dqn_replay(tmp_path, baseline_steps=100, round_index=1, steps=8)
    with pytest.raises(ValueError, match="new-world repair experience"):
        check_dqn_replay(tmp_path, baseline_steps=100, round_index=2, steps=8)
    write_replay(tmp_path, experiences=16)
    check_dqn_replay(tmp_path, baseline_steps=100, round_index=2, steps=8)
    write_replay(tmp_path, experiences=32)
    with pytest.raises(ValueError, match="new-world repair experience"):
        check_dqn_replay(tmp_path, baseline_steps=100, round_index=1, steps=8)


def test_dqn_audit_rejects_warmup_without_cumulative_offset(tmp_path):
    write_replay(tmp_path, experiences=8, warmup=2)
    with pytest.raises(ValueError, match="cumulative baseline"):
        check_dqn_replay(tmp_path, baseline_steps=100, round_index=1, steps=8)


def test_audit_rejects_changed_world_pairing_and_unsafe_label(instance):
    row = {
        "event_id": instance.instance_hash,
        "instance_hash": instance.instance_hash,
        "root_sample_id": instance.root_sample_id,
        "assets_hash": json_hash(instance.assets),
        "purpose": "final_test",
        "safety_failure": False,
        "collision": False,
        "offroad": False,
        "stalled": True,
        "numerical_valid": True,
        "progress_m": 0,
        "evt_score": 1,
        "event_risk": 0.5,
        "elapsed_s": 1,
        "critical_frame": 10,
        "physics_steps": 149,
    }
    options = {
        "assets_hash": row["assets_hash"],
        "cohort": {instance.instance_hash: instance},
    }
    check_records([row], **options)  # Stalling alone remains a task outcome.
    changed = deepcopy(row)
    changed["collision"] = True
    with pytest.raises(ValueError, match="safety label"):
        check_records([changed], **options)
    changed = deepcopy(row)
    changed["root_sample_id"] = "different-exogenous-realization"
    with pytest.raises(ValueError, match="concrete IID instance"):
        check_records([changed], **options)
    with pytest.raises(ValueError, match="duplicate instances"):
        check_records([row, row], **options)
    changed = deepcopy(row)
    changed["physics_steps"] = 150
    with pytest.raises(ValueError, match="physical-step"):
        check_records([changed], **options)
