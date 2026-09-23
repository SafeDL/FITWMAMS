from __future__ import annotations

import json

import numpy as np

from reproduction.models.trafficbots.data import TrafficBotsCausalDataset


def _write_causal_cache(root) -> None:
    count = 3
    (root / "manifest.json").write_text(json.dumps({
        "cache_format": "npc_interaction_causal174_v1", "num_scenarios": count,
    }), encoding="utf-8")
    states = np.zeros((count, 174, 7, 6), np.float32)
    valid = np.ones((count, 174, 7), bool)
    states[..., 2] = 10.0
    # Make C0 directly distinguishable from the preceding history.
    states[:, 24, :, 0] = np.arange(count, dtype=np.float32)[:, None] + 100.0
    np.save(root / "agent_states.npy", states)
    np.save(root / "agent_valid.npy", valid)
    np.save(root / "target_actions_highd.npy", np.zeros((count, 149, 6, 2), np.float32))
    maps = np.zeros((count, 8, 8, 6), np.float32)
    maps[:, 0, :, 0] = np.arange(8, dtype=np.float32)
    np.save(root / "map_polylines.npy", maps)
    map_valid = np.zeros((count, 8, 8), bool)
    map_valid[:, 0] = True
    np.save(root / "map_polyline_valid.npy", map_valid)
    np.savez_compressed(
        root / "scenario_metadata.npz",
        split_index=np.asarray((0, 1, 2), np.int8),
        is_evt_tail=np.asarray((False, True, False)),
        sequence_id=np.asarray(("train", "validation", "test")),
    )


def test_strict_causal_dataset_uses_real_c0_and_split_metadata(tmp_path) -> None:
    _write_causal_cache(tmp_path)
    train = TrafficBotsCausalDataset(tmp_path, "train", seed=7)
    validation = TrafficBotsCausalDataset(tmp_path, "val", seed=7)
    test = TrafficBotsCausalDataset(tmp_path, "test", seed=7)
    assert len(train) == len(validation) == len(test) == 1
    sample = train[0]
    assert sample["sequence_id"] == "train"
    assert sample["canonical/states"].shape == (150, 7, 6)
    assert sample["canonical/valid"].shape == (150, 7)
    assert sample["canonical/actions_highd"].shape == (149, 6, 2)
    # The model C0 is causal-cache index 24, never padded legacy history.
    assert np.allclose(sample["canonical/states"].numpy()[0, :, 0], 100.0)
    assert bool(validation[0]["is_evt_tail"])
    assert test[0]["sequence_id"] == "test"
