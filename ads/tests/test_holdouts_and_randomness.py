from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from ads.common.instances import GeneratedInstance, UsageLedger
from ads.common.natural import empirical_manifest, recorded_history
from ads.common.randomness import PolicyRandomContext
from ads.nuplan.features import sample_history


def recorded_vehicles(ego_id, gap=35.0):
    frames = np.arange(274)
    return {
        vehicle_id: {
            "frames": frames,
            "initial": 0,
            "final": 273,
            "continuous": True,
            "x": origin + frames * 0.04 * speed,
            "y_left": np.zeros(274),
            "vx": np.full(274, speed),
            "vy_left": np.zeros(274),
            "ax": np.zeros(274),
            "ay_left": np.zeros(274),
            "abnormal": np.zeros(274, bool),
        }
        for vehicle_id, origin, speed in (
            (ego_id, 100.0, 20.0),
            (ego_id + 1, 100.0 + gap, 19.0),
        )
    }


def test_recorded_history_uses_available_past_and_survives_serialization(
    instance, tmp_path
):
    vehicles = recorded_vehicles(1)
    row = pd.Series({"ego_id": 1, "same_front_id": 2, "window_start_frame": 100})
    current = instance.event.initial_states.copy()
    current[1, 0] = 31.0
    event = recorded_history(
        vehicles, row, replace(instance.event, initial_states=current)
    )
    np.testing.assert_allclose(event.initial_history[[0, -1], 0, 0], [-40, 0])
    assert event.initial_history_valid[:, :2].all()
    for vehicle in vehicles.values():
        vehicle["x"][101:] = 999999
    repeated = recorded_history(vehicles, row, event)
    np.testing.assert_array_equal(event.initial_history, repeated.initial_history)
    empirical = replace(instance, event=event, source_kind="empirical_natural")
    empirical = replace(
        empirical, event=replace(event, event_id=empirical.instance_hash)
    )
    path = tmp_path / "recorded.npz"
    empirical.save(path)
    restored = GeneratedInstance.load(path)
    np.testing.assert_array_equal(restored.event.initial_history, event.initial_history)
    history, valid = sample_history(
        current[None],
        0,
        initial_history=restored.event.initial_history,
        initial_history_valid=restored.event.initial_history_valid,
    )
    assert valid[:, :2].all()
    np.testing.assert_allclose(history[[0, -1], 0, 0], [-40, 0])


def test_real_history_survives_hash_and_serialization(instance, tmp_path):
    history = np.tile(instance.event.initial_states, (25, 1, 1))
    history[:, :, :2] -= (
        np.arange(24, -1, -1)[:, None, None] * 0.04 * history[:, :, 2:4]
    )
    valid = np.tile(instance.event.valid, (25, 1))
    event = replace(
        instance.event,
        initial_history=history,
        initial_history_valid=valid,
        source_event_id="natural_source",
    )
    empirical = replace(
        instance,
        event=event,
        source_kind="empirical_natural",
        source_metadata={"recording_id": 1},
    )
    empirical = replace(
        empirical, event=replace(event, event_id=empirical.instance_hash)
    )
    path = tmp_path / "empirical.npz"
    empirical.save(path)
    restored = GeneratedInstance.load(path)
    np.testing.assert_array_equal(restored.event.initial_history, history)
    np.testing.assert_array_equal(restored.event.initial_history_valid, valid)
    past, mask = sample_history(
        history[-1:], 0, initial_history=history, initial_history_valid=valid
    )
    assert mask[:, 0].sum() == 10  # .96 seconds actual past, never fabricated 2 s.
    np.testing.assert_allclose(past[-1], history[-1])
    restored.event.initial_history_valid[0, 0] = False
    with pytest.raises(ValueError, match="hash mismatch"):
        restored.validate()


def test_policy_streams_do_not_depend_on_scenario_during_search():
    one = PolicyRandomContext("flow_planner", instance_hash="a")
    two = PolicyRandomContext("flow_planner", instance_hash="b")
    assert one.identity == two.identity
    with one.step(7, use="flow_noise"):
        expected = (torch.randn(4), np.random.randn(4))
    with two.step(7, use="flow_noise"):
        actual = (torch.randn(4), np.random.randn(4))
    torch.testing.assert_close(expected[0], actual[0], rtol=0, atol=0)
    np.testing.assert_array_equal(expected[1], actual[1])
    assert one.seed_for(7, "flow_noise") != one.seed_for(7, "other_use")
    assert one.seed_for(7) != one.seed_for(8)
    final = PolicyRandomContext("flow_planner", mode="final_test", instance_hash="a")
    other = replace(final, instance_hash="b")
    train = replace(one, mode="training")
    assert len({one.identity, final.identity, other.identity, train.identity}) == 4


def test_final_lineage_is_rejected_even_if_child_changes_its_purpose(
    instance, tmp_path
):
    ledger = UsageLedger(tmp_path / "usage.sqlite")
    final = replace(instance, purpose="final_test")
    ledger.register(final, "final_test")
    child = replace(instance, purpose="repair", parent_id=instance.instance_hash)
    with pytest.raises(ValueError, match="leakage"):
        ledger.register(child, "repair")


def test_empirical_sampling_keeps_natural_strata_and_tail_overlap():
    frame = pd.DataFrame(
        {
            "segment_id": [f"s{i}" for i in range(120)],
            "recording_id": [1] * 80 + [2] * 40,
            "ego_driving_direction": [1] * 60 + [2] * 60,
            "ego_id": range(120),
            "window_start_frame": np.arange(120) * 200,
            "window_end_frame": np.arange(120) * 200 + 149,
        }
    )
    arrays = {
        "segment_id": frame.segment_id.to_numpy(),
        "split_index": np.r_[np.zeros(20), np.full(100, 2)],
        "is_evt_tail": np.ones(120, bool),
    }
    chosen = empirical_manifest(frame, arrays, count=50)
    assert chosen.segment_id.nunique() == 50
    assert chosen.source_used_by_tail_flow.all()  # dangerous sources are not removed.
    assert chosen.recording_id.value_counts().to_dict() == {1: 30, 2: 20}
    assert set(chosen.segment_id) <= set(frame.segment_id[20:])
    pd.testing.assert_frame_equal(chosen, empirical_manifest(frame, arrays, count=50))


def test_natural_flow_statistics_exclude_test_without_fitting():
    import json
    from pathlib import Path

    from normalizing_flow.src.train import _mask_log_prob, natural_fit_population

    # Small synthetic copies of the existing schema, with extreme held-out values.
    schema = json.loads(
        Path(
            "results/highd_shared_training_data/scenario_conditions/dataset_schema.json"
        ).read_text()
    )
    with np.load(
        "results/highd_shared_training_data/scenario_conditions/dataset.npz"
    ) as source:
        arrays = {
            key: source[key][:6].copy()
            for key in (
                "features",
                "feature_valid",
                "trajectory_constraint",
                "trajectory_constraint_valid",
                "mask_pattern",
                "vehicle_dimensions",
                "slot_mask",
            )
        }
    arrays.update(
        split_index=np.array([0, 0, 1, 1, 2, 2]),
        segment_id=np.array([f"synthetic_{i}" for i in range(6)]),
    )
    original_schema = __import__("copy").deepcopy(schema)
    _, first, rows = natural_fit_population(arrays, schema)
    changed = {key: value.copy() for key, value in arrays.items()}
    changed["features"][4:] = 100000
    changed["trajectory_constraint"][4:] = 100000
    changed["mask_pattern"][4:] = 63
    changed["vehicle_dimensions"][4:] = 99
    _, second, _ = natural_fit_population(changed, schema)
    assert first["normalization"] == second["normalization"]
    assert (
        first["long_horizon_constraint"]["normalization"]
        == second["long_horizon_constraint"]["normalization"]
    )
    torch.testing.assert_close(
        _mask_log_prob(arrays, rows), _mask_log_prob(changed, rows), rtol=0, atol=0
    )
    assert schema == original_schema
    assert first["fit_event_ids"] == [
        "synthetic_0",
        "synthetic_1",
        "synthetic_2",
        "synthetic_3",
    ]


def test_entirely_empty_supervision_skips_update():
    from types import SimpleNamespace

    from ads.nuplan.train import train
    from ads.plant2.train import train as plant_train

    for learner in (train, plant_train):
        report = learner(SimpleNamespace(), {}, updates=1000, seed=20260926)
        assert report["gradient_updates"] == 0
        assert report["skipped"] == "no_qualified_demonstrations"


def test_modified_descendant_root_does_not_bypass_final_parent(instance, tmp_path):
    ledger = UsageLedger(tmp_path / "usage.sqlite")
    ledger.register(replace(instance, purpose="final_test"), "final_test")
    # A new content identity and root still point at a known final-test parent.
    event = replace(instance.event, initial_states=instance.event.initial_states + 0.01)
    child = replace(
        instance,
        event=event,
        root_sample_id="changed_root",
        parent_id=instance.instance_hash,
    )
    child = replace(child, event=replace(event, event_id=child.instance_hash))
    with pytest.raises(ValueError, match="leakage"):
        ledger.register(child, "repair")


def test_three_cohorts_materialize_from_their_frozen_synthetic_sources(
    instance, source, tmp_path, monkeypatch
):
    import json
    from types import SimpleNamespace

    from ads.common.cohorts import (
        final_cohort,
        prepare_evaluation_assets,
        verify_evaluation_assets,
    )
    from ads.common.evaluation_sampling import COHORTS, final_sampling_spec
    from ads.common.instances import json_hash

    cache = tmp_path / "sequence_cache"
    cache.mkdir()
    ids = np.array([f"synthetic_{index}" for index in range(4)])
    states = np.tile(instance.event.initial_states, (4, 174, 1, 1))
    anchors = np.array([0, 100, 0, 100])
    states[:, :, 1, 0] -= anchors[:, None] * 0.04
    states[..., :2] += (
        np.arange(-24, 150)[None, :, None, None] * 0.04 * states[..., 2:4]
    )
    arrays = {
        "sequence_id": ids,
        "agent_states": states,
        "agent_valid": np.tile(instance.event.valid, (4, 174, 1)),
        "map_polylines": np.tile(instance.event.map_polylines, (4, 1, 1, 1)),
        "map_polyline_valid": np.tile(instance.event.map_polyline_valid, (4, 1, 1)),
        "vehicle_dimensions": np.tile(instance.event.vehicle_dimensions, (4, 1, 1)),
    }
    for name, value in arrays.items():
        np.save(cache / (name + ".npy"), value)
    pd.DataFrame(
        {
            "segment_id": ids,
            "recording_id": [1, 1, 2, 2],
            "ego_driving_direction": [1, 1, 2, 2],
            "ego_id": [10, 10, 20, 20],
            "same_front_id": [11, 11, 21, 21],
            "window_start_frame": [0, 100, 0, 100],
            "window_end_frame": [173, 273, 173, 273],
        }
    ).to_csv(tmp_path / "natural.csv", index=False)
    np.savez(
        tmp_path / "dataset.npz",
        segment_id=ids,
        split_index=np.full(4, 2),
        is_evt_tail=np.array([True, False, False, True]),
    )
    (tmp_path / "natural_flow.pt").write_text("synthetic natural asset; no fit")
    import yaml

    config = tmp_path / "process_highD/config.yaml"
    config.parent.mkdir()
    config.write_text(yaml.safe_dump({"paths": {"raw_dir": "../history_raw"}}))
    raw = tmp_path / "history_raw"
    raw.mkdir()
    for recording in (1, 2):
        for name in ("tracks", "tracksMeta", "recordingMeta"):
            (raw / f"{recording:02d}_{name}.csv").write_text("synthetic history source")
    monkeypatch.setattr(
        "ads.common.natural.prepare_recording", lambda raw, recording, config: recording
    )
    monkeypatch.setattr(
        "ads.common.natural._build_vehicle_cache",
        lambda recording: recorded_vehicles(
            recording * 10, gap=float(instance.event.initial_states[1, 0])
        ),
    )
    natural_assets = {**source.assets, "flow": "independent-natural-flow"}
    monkeypatch.setattr("ads.common.cohorts.REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "ads.common.cohorts.load_world",
        lambda *args, **kwargs: (source.sampler, None, None, natural_assets),
    )
    campaign = SimpleNamespace(
        output=tmp_path / "synthetic_campaign",
        sampler=source.sampler,
        assets=source.assets,
        ledger=UsageLedger(tmp_path / "usage.sqlite"),
        config={
            "master_seed": 20260926,
            "runtime": {"generation_batch_size": 1},
            "evaluation": {
                "instances": 2,
                "natural_dataset": "dataset.npz",
                "natural_segments": "natural.csv",
                "sequence_cache": "sequence_cache",
                "natural_flow_checkpoint": "natural_flow.pt",
            },
        },
    )
    evaluation_assets = prepare_evaluation_assets(campaign)
    verify_evaluation_assets(evaluation_assets)
    protocol = {"settings": campaign.config, "assets": campaign.assets}
    (campaign.output / "protocol.json").write_text(json.dumps(protocol))
    frozen = {"evaluation_assets": evaluation_assets}
    assert set(final_sampling_spec(protocol, evaluation_assets)["cohorts"]) == set(
        COHORTS
    )
    cohorts = {name: final_cohort(campaign, name, frozen) for name in COHORTS}
    hashes, roots = set(), set()
    for name, worlds in cohorts.items():
        assert len(worlds) == 2
        assert all(world.source_kind == name for world in worlds)
        identities = {world.instance_hash for world in worlds}
        ancestors = {world.root_sample_id for world in worlds}
        assert not hashes & identities and not roots & ancestors
        hashes.update(identities)
        roots.update(ancestors)
        assert identities == {
            world.instance_hash for world in final_cohort(campaign, name, frozen)
        }
        with pytest.raises(ValueError, match="final-test"):
            campaign.ledger.register(worlds[0], "repair")
    for world in cohorts["empirical_natural"]:
        index = list(ids).index(world.event.source_event_id)
        expected = np.tile(states[index, 24], (51, 1, 1))
        expected[..., :2] += (
            np.arange(-50, 1)[:, None, None] * 0.04 * expected[..., 2:4]
        )
        valid = np.tile(instance.event.valid, (51, 1))
        valid[np.arange(-50, 1) + anchors[index] < 0] = False
        expected[~valid] = 0
        np.testing.assert_allclose(world.event.initial_history, expected, atol=1e-5)
        np.testing.assert_array_equal(world.event.initial_history_valid, valid)
        np.testing.assert_array_equal(
            world.event.soft_reference, states[index, 25:, 1:, :2]
        )
        assert "association_group" in world.source_metadata
    assert json_hash(cohorts["tail_generated"][0].assets) != json_hash(
        cohorts["natural_generated"][0].assets
    )
    (tmp_path / "natural_flow.pt").write_text("changed source")
    with pytest.raises(ValueError, match="source changed"):
        verify_evaluation_assets(evaluation_assets)
