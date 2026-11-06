"""Resume causal policy and controller state without rebasing world time."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ads.common.environment import PolicyEnvironment
from ads.common.expert import Candidate
from ads.common.instances import UsageLedger
from ads.common.randomness import PolicyRandomContext
from ads.common.setup import REPO_ROOT
from ads.common.world import FrozenWorldFactory
from ads.diffusion_planner.aggregation import collect_query, execute_suffix
from ads.diffusion_planner.label_tracking import check as check_labels
from ads.nuplan.teacher import demonstration
from ads.tests.conftest import SmallWorld


class SnapshotWorld(SmallWorld):
    def snapshot(self):
        return self.states.clone(), self.reference_index

    def restore(self, snapshot):
        self.states, self.reference_index = snapshot[0].clone(), snapshot[1]


def test_policy_snapshot_replays_controller_history_and_random_stream(instance):
    env = PolicyEnvironment("diffusion_planner", [instance], lambda e: SnapshotWorld())
    env.reset()
    action = np.zeros((80, 3), np.float32)
    action[:, 0] = np.arange(1, 81) * 2
    env.step(action)
    policy = SimpleNamespace(
        random_context=PolicyRandomContext("diffusion_planner"), decision=1
    )
    snapshot = env.snapshot(policy)
    expected_observation = deepcopy(env._observation())
    env.step(action)
    expected_states = np.array(env.physics_states)
    expected_controls = np.array(env.physics_controls)
    env.step(action)
    policy.decision = 20
    restored = env.restore(snapshot, policy)
    assert policy.decision == 1
    assert env.world.reference_index == 5
    np.testing.assert_array_equal(restored["history"], expected_observation["history"])
    env.step(action)
    np.testing.assert_array_equal(env.physics_states, expected_states)
    np.testing.assert_array_equal(env.physics_controls, expected_controls)
    with pytest.raises(ValueError, match="another algorithm"):
        env.restore({**snapshot, "algorithm": "carl"})


def test_suffix_labels_use_query_history_and_only_successful_future(instance):
    states = np.tile(instance.event.initial_states, (150, 1, 1))
    states[:, :, :2] += np.arange(150)[:, None, None] * 0.04 * states[:, :, 2:4]
    frame = 50
    sample = demonstration(instance, states, "diffusion_planner", frames=[frame])[0]
    assert sample["frame_id"] == frame
    assert sample["instance_id"] == instance.instance_hash
    assert sample["observation"]["history_valid"][:, 0].all()
    assert sample["future_valid"][0].sum() == 39
    np.testing.assert_array_equal(sample["target"][~sample["future_valid"]], 0)


def test_collected_offset_replays_policy_randomness_and_world_prefix(
    instance, evt, tmp_path
):
    states = instance.event.initial_states.copy()
    states[1, 2] = 5
    value = replace(instance, event=replace(instance.event, initial_states=states))
    value = replace(value, event=replace(value.event, event_id=value.instance_hash))

    class StochasticPlanner:
        device = "cpu"

        def reset(self, context):
            self.random_context, self.decision = context, 0

        def predict(self, observation):
            plan = np.zeros((80, 3), np.float32)
            plan[:, 0] = np.arange(1, 81) * 2
            plan[:, 1] = torch.randn(()).item() * 0.01
            self.decision += 1
            return plan, None

    class FailingWorld(SnapshotWorld):
        def advance_response(self, action):
            result = super().advance_response(action)
            if self.reference_index == 100:
                result["crashed"][0, 0] = True
                result["collision"][0] = True
            return result

    snapshot, report = collect_query(
        value,
        StochasticPlanner(),
        lambda e: FailingWorld(),
        evt,
        UsageLedger(tmp_path / "usage.sqlite"),
    )
    assert snapshot is not None
    assert report["snapshot_replay"] == {"success": True, "physics_steps": 5}
    assert snapshot["policy"]["decision"] == report["query_frame"] // 5


def test_frozen_world_snapshot_keeps_absolute_time_and_npc_response(instance, tmp_path):
    torch.set_num_threads(1)
    factory = FrozenWorldFactory(REPO_ROOT / "hierarchical_world_model/config.yaml")
    env = PolicyEnvironment("diffusion_planner", [instance], factory)
    env.reset()
    action = np.zeros((80, 3), np.float32)
    action[:, 0] = np.arange(1, 81) * 2
    env.step(action)
    env.step(action)
    snapshot = env.snapshot()
    env.step(action)
    expected = np.array(env.physics_states)
    env.restore(snapshot)
    assert env.world.reference_index == 10
    env.step(action)
    np.testing.assert_array_equal(env.physics_states, expected)
    lane = instance.event.geometry.nearest_lane(env._states()[0, :2])
    suffix = execute_suffix(instance, factory, snapshot, Candidate(lane, 20))
    np.testing.assert_array_equal(
        suffix.states[:11], snapshot["runtime"]["physics_states"]
    )
    assert len(suffix.states) == 150
    assert suffix.success
    tracking = check_labels(
        instance,
        suffix.states,
        factory,
        UsageLedger(tmp_path / "usage.sqlite"),
        snapshot=snapshot,
    )
    assert tracking["success"]
    assert tracking["physics_steps"] == 139
    assert not tracking["training_labels_extrapolated"]


def test_last_decisions_have_partial_labels_without_fabricating_future(instance):
    states = np.tile(instance.event.initial_states, (150, 1, 1))
    samples = demonstration(instance, states, "diffusion_planner")
    last = samples[-1]
    assert last["frame_id"] == 145
    assert last["future_valid"][0].sum() == 1
    np.testing.assert_array_equal(last["target"][~last["future_valid"]], 0)


def test_query_selection_uses_current_failures_and_prioritizes_success_regression():
    from ads.diffusion_planner.aggregation import select_query_records

    records = [
        {
            "instance_hash": name,
            "root_sample_id": name,
            "safety_failure": failure,
            "critical_frame": frame,
            "event_risk": risk,
            "descriptor": [risk, 0, 0, 0],
        }
        for name, failure, frame, risk in (
            ("regression", True, 80, 1),
            ("persistent", True, 70, 10),
            ("fixed", False, 70, 20),
            ("early", True, 20, 30),
        )
    ]
    original = {
        r["instance_hash"]: {**r, "safety_failure": r["instance_hash"] != "regression"}
        for r in records
    }
    selected = select_query_records(records, original)
    assert selected[0]["instance_hash"] == "regression"
    assert {r["instance_hash"] for r in selected} == {"regression", "persistent"}
    assert select_query_records([], original) == []
