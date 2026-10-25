from dataclasses import replace

import numpy as np
import pytest
from conftest import SmallSampler

from ads.common.instances import (
    GeneratedInstance,
    GeneratedSource,
    InstanceStore,
    UsageLedger,
)
from ads.common.repair import RepairEnvironment


def test_complete_roundtrip_and_tamper_detection(instance, tmp_path):
    path = tmp_path / "instance.npz"
    instance.save(path)
    restored = GeneratedInstance.load(path)
    assert restored.instance_hash == instance.instance_hash
    assert restored.event.event_id == instance.event.event_id
    for name, value in instance.arrays().items():
        np.testing.assert_array_equal(value, restored.arrays()[name])
    restored.event.soft_reference[0, 0, 0] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        restored.validate()
    with pytest.raises(FileExistsError):
        instance.save(path)


def test_generation_is_independent_of_materialisation_batch_size():
    a = GeneratedSource(SmallSampler(), {}, master_seed=5, purpose="search")
    b = GeneratedSource(SmallSampler(), {}, master_seed=5, purpose="search")
    batch = a.sample(3)
    individual = [b.sample(1)[0] for _ in range(3)]
    assert [s.instance_hash for s in batch] == [s.instance_hash for s in individual]
    assert len({s.root_sample_id for s in batch}) == 3


def test_test_usage_rejected_by_identity_and_root(instance, tmp_path, world_factory):
    ledger = UsageLedger(tmp_path / "usage.sqlite")
    ledger.register(instance, "search")
    with pytest.raises(ValueError, match="leakage"):
        ledger.register(replace(instance, purpose="final_test"), "final_test")
    independent = GeneratedSource(
        SmallSampler(), {}, master_seed=5, purpose="final_test"
    ).sample(1)[0]
    ledger.register(independent, "final_test")
    with pytest.raises(ValueError, match="final-test"):
        ledger.register(independent, "repair")
    with pytest.raises(ValueError, match="final-test"):
        RepairEnvironment(
            "dqn", {"normal": [(independent, "normal")]}, world_factory, round_index=0
        )


def test_store_deduplicates_complete_content(instance, tmp_path):
    store = InstanceStore(
        tmp_path / "store",
        purpose="repair",
        ledger=UsageLedger(tmp_path / "usage.sqlite"),
    )
    store.add(instance)
    store.add(instance)
    assert len(store) == 1
    assert store[0].instance_hash == instance.instance_hash
