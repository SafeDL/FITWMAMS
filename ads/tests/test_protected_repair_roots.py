"""Dangerous validation is observed by search but excluded from all learning."""

from dataclasses import replace

import numpy as np
import pytest

from ads.common.demonstrations import DemonstrationCache
from ads.common.instances import UsageLedger
from ads.diffusion_planner.sampling import BalancedSampler
from ads.diffusion_planner.sources import executed_success_neighbours


def test_search_ancestor_can_be_held_out_before_repair(instance, tmp_path):
    ledger = UsageLedger(tmp_path / "usage.sqlite")
    ledger.register(instance, "search")
    ledger.protect_roots([instance.root_sample_id])
    ledger.register(instance, "development")
    child = replace(instance, parent_id=instance.instance_hash)
    with pytest.raises(ValueError, match="Protected"):
        ledger.register(child, "repair")
    cache = DemonstrationCache(
        tmp_path / "experts",
        None,
        algorithm="diffusion_planner",
        candidate_budget=128,
        ledger=ledger,
    )
    with pytest.raises(ValueError, match="Protected"):
        cache.get(child)
    with pytest.raises(ValueError, match="Protected"):
        BalancedSampler(
            [{"scene_id": child.root_sample_id}],
            0,
            protected_roots=[instance.root_sample_id],
        )


def test_ancestor_cannot_be_relabelled_validation_after_learning(instance, tmp_path):
    ledger = UsageLedger(tmp_path / "usage.sqlite")
    ledger.register(instance, "repair")
    with pytest.raises(ValueError, match="already used for repair"):
        ledger.protect_roots([instance.root_sample_id])


@pytest.mark.parametrize("failure_is_child", [False, True])
def test_joint_neighbours_depend_on_executed_structure(
    source, tmp_path, failure_is_child
):
    parent = source.sample(1)[0]
    child = source.mutation(
        parent,
        parent.event.exogenous_state.mutate("c0_base_latent", beta=0.2, seed=1),
        "joint",
    )

    def records_for(value):
        return {
            parent.instance_hash: {
                "instance_hash": parent.instance_hash,
                "parent_id": None,
                "safety_failure": not failure_is_child,
                "execution_kind": "initial_prior",
            },
            value.instance_hash: {
                "instance_hash": value.instance_hash,
                "parent_id": parent.instance_hash,
                "safety_failure": failure_is_child,
                "execution_kind": "conditional_proposal",
            },
        }

    def paths_for(value):
        paths = {}
        for instance in (parent, value):
            path = tmp_path / (instance.instance_hash + ".npz")
            if not path.exists():
                instance.save(path)
            paths[instance.instance_hash] = path
        return paths

    failure, success = (child, parent) if failure_is_child else (parent, child)
    assert executed_success_neighbours(records_for(child), paths_for(child)) == {
        failure.instance_hash: success.instance_hash
    }
    valid = child.event.valid.copy()
    valid[1] = False
    changed = replace(child, event=replace(child.event, valid=valid))
    changed = replace(
        changed, event=replace(changed.event, event_id=changed.instance_hash)
    )
    assert not np.array_equal(changed.event.valid, parent.event.valid)
    assert executed_success_neighbours(records_for(changed), paths_for(changed)) == {}
