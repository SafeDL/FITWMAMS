"""CaRL retention uses complete successful executions and real control history."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
import torch

from ads.carl.materials import (
    RecordingCaRLPolicy,
    capture_instance,
    load_keep_samples,
    load_native_samples,
)
from ads.carl.policy import CaRLPolicy
from ads.common.evaluation import WorldEvaluator
from ads.common.instances import UsageLedger


class FixedAction(torch.nn.Module):
    def forward(self, observations, deterministic=True):
        action = torch.tensor((-0.1, 0.001), device=observations["measurements"].device)
        return (action.expand(len(observations["measurements"]), 2),)


@pytest.fixture
def natural(instance):
    return replace(instance, purpose="repair", source_kind="natural_generated")


def test_recording_preserves_execution_and_captures_actual_measurements(
    natural, world_factory, evt, tmp_path
):
    recorder = RecordingCaRLPolicy(FixedAction())
    plain = CaRLPolicy(FixedAction())
    recorded = WorldEvaluator(recorder, "carl", world_factory, evt)(natural)
    original = WorldEvaluator(plain, "carl", world_factory, evt)(natural)
    np.testing.assert_array_equal(recorded.states, original.states)
    np.testing.assert_array_equal(recorded.controls, original.controls)
    saved, executions = capture_instance(
        recorder,
        natural,
        world_factory,
        evt,
        tmp_path / "keep",
        policy_hash="official",
        ledger=UsageLedger(tmp_path / "usage.sqlite"),
    )
    assert executions == 1 and saved["qualified_keep"]
    assert saved["decision_observations"] == 30
    with np.load(saved["observations_path"], allow_pickle=False) as data:
        np.testing.assert_array_equal(data["decision_physics_steps"], np.arange(30) * 5)
        for key in ("bev_semantics", "measurements", "value_measurements"):
            np.testing.assert_array_equal(
                data[key], np.stack([o[key] for o in recorder.observations])
            )
        np.testing.assert_array_equal(data["measurements"][0, :2], (0, 0))
        np.testing.assert_array_equal(
            data["measurements"][1, :2], np.asarray((-0.1, 0.001), np.float32)
        )
        assert data["measurements"][1, 6] > 0
        assert data["measurements"][1, 7] > 0
    report = tmp_path / "materials.json"
    report.write_text(json.dumps({"policy_sha256": "official", "executions": [saved]}))
    samples = load_keep_samples(report, official_hash="official")
    assert len(samples) == 30
    assert {s["scene_id"] for s in samples} == {natural.root_sample_id}
    assert all(
        set(s["observation"]) == {"bev_semantics", "measurements", "value_measurements"}
        for s in samples
    )

    def forbidden(event):
        pytest.fail("A cached execution was repeated")

    cached, new = capture_instance(
        recorder,
        natural,
        forbidden,
        evt,
        tmp_path / "keep",
        policy_hash="official",
        ledger=UsageLedger(tmp_path / "usage.sqlite"),
    )
    assert new == 0 and cached == saved
    with pytest.raises(ValueError, match="another policy"):
        capture_instance(
            recorder,
            natural,
            forbidden,
            evt,
            tmp_path / "keep",
            policy_hash="different",
            ledger=UsageLedger(tmp_path / "usage.sqlite"),
        )


def test_failed_execution_does_not_supply_its_safe_prefix(
    natural, world_factory, evt, tmp_path
):
    def failing_world(event):
        world = world_factory(event)
        advance = world.advance_response

        def step(action):
            result = advance(action)
            if world.reference_index >= 6:
                result["collision"][0] = True
                result["crashed"][0, 0] = True
            return result

        world.advance_response = step
        return world

    saved, _ = capture_instance(
        RecordingCaRLPolicy(FixedAction()),
        natural,
        failing_world,
        evt,
        tmp_path,
        policy_hash="official",
        ledger=UsageLedger(tmp_path / "usage.sqlite"),
    )
    assert saved["record"]["collision"] and not saved["qualified_keep"]
    assert saved["decision_observations"] == 2
    assert not list(tmp_path.glob("*.npz"))
    report = tmp_path / "materials.json"
    report.write_text(json.dumps({"policy_sha256": "official", "executions": [saved]}))
    assert load_keep_samples(report, official_hash="official") == []


@pytest.mark.parametrize("purpose", ["development", "final_test"])
def test_evaluation_instances_cannot_supply_retention(
    natural, world_factory, evt, tmp_path, purpose
):
    with pytest.raises(ValueError, match="training instances"):
        capture_instance(
            RecordingCaRLPolicy(FixedAction()),
            replace(natural, purpose=purpose),
            world_factory,
            evt,
            tmp_path,
            policy_hash="official",
            ledger=UsageLedger(tmp_path / "usage.sqlite"),
        )


def test_empirical_test_and_native_development_are_not_retention(
    natural, world_factory, evt, tmp_path
):
    empirical = replace(
        natural,
        source_kind="empirical_natural",
        source_metadata={"source_split": "test"},
    )
    with pytest.raises(ValueError, match="Empirical Test"):
        capture_instance(
            RecordingCaRLPolicy(FixedAction()),
            empirical,
            world_factory,
            evt,
            tmp_path,
            policy_hash="official",
            ledger=UsageLedger(tmp_path / "usage.sqlite"),
        )
    path = tmp_path / "native.json"
    path.write_text(
        json.dumps(
            {
                "status": "passed",
                "scenes": {"replay": [{"source": {"purpose": "development"}}]},
            }
        )
    )
    with pytest.raises(ValueError, match="development data"):
        load_native_samples(path)


@pytest.fixture
def native_preparation(tmp_path, monkeypatch):
    from ads.carl import prepare_data
    from ads.common.instances import file_hash, json_hash

    def row(log, city, purpose, token):
        return {
            "log": log,
            "city": city,
            "purpose": purpose,
            "token": token,
            "session": log,
        }

    invalid = row("a", "boston", "replay", "invalid")
    singapore = row("b", "singapore", "replay", "valid")
    replacement = row("a", "boston", "replay", "replacement")
    development = row("c", "pittsburgh", "development", "development")
    source = {
        "scenes": {
            "replay": {
                "selected": [invalid, singapore],
                "reserve": [invalid, replacement],
            },
            "development": {"selected": [development], "reserve": []},
        }
    }
    monkeypatch.setattr(prepare_data, "ROOT", tmp_path)
    monkeypatch.setattr(prepare_data, "COUNTS", {"replay": 2, "development": 1})
    monkeypatch.setattr(prepare_data, "select_scenes", lambda: source)
    monkeypatch.setattr(prepare_data, "native_source_fingerprint", lambda: "official")
    monkeypatch.setattr(
        prepare_data,
        "ProcessPoolExecutor",
        lambda max_workers, mp_context: ThreadPoolExecutor(max_workers),
    )
    calls = []

    def encode(row, directory):
        calls.append((row["log"], row["token"]))
        if row["token"] == "invalid":
            raise ValueError("Invalid native data")
        identity = json_hash([row["log"], row["token"]])
        path = directory / (identity + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            bev_semantics=np.zeros((9, 256, 256), np.uint8),
            measurements=np.zeros(10, np.float32),
            value_measurements=np.ones(4, np.float32),
        )
        return {
            "scene_id": identity,
            "path": str(path),
            "sha256": file_hash(path),
            "source": row,
        }

    monkeypatch.setattr(prepare_data, "encode_scene", encode)
    return prepare_data, calls


def test_native_preparation_preserves_city_quota_and_training_boundary(
    native_preparation,
):
    preparation, calls = native_preparation
    report = preparation.prepare()
    assert calls.count(("a", "invalid")) == 1
    assert len(report["excluded"]) == 1
    assert report["coverage"]["replay"]["cities"] == {"boston": 1, "singapore": 1}
    assert report["coverage"]["development"]["scenes"] == 1
    material = load_native_samples(preparation.ROOT / "carl/materials.json")
    assert len(material) == 2
    assert {row["scene_id"] for row in material} == {
        row["scene_id"] for row in report["scenes"]["replay"]
    }


def test_complete_native_cache_is_not_encoded_again(native_preparation, monkeypatch):
    preparation, calls = native_preparation
    original = preparation.prepare()

    def forbidden(*args, **kwargs):
        pytest.fail("A verified native cache was encoded again")

    monkeypatch.setattr(preparation, "encode_scene", forbidden)
    assert preparation.prepare() == original
    assert len(calls) == 4


def test_native_cache_requires_its_frozen_encoder(native_preparation, monkeypatch):
    preparation, _ = native_preparation
    preparation.prepare()
    monkeypatch.setattr(preparation, "native_source_fingerprint", lambda: "changed")
    with pytest.raises(ValueError, match="protocol changed"):
        preparation.prepare()


@pytest.fixture
def highway_preparation(tmp_path, monkeypatch):
    from ads.carl import materials

    root = tmp_path / "carl"
    common = tmp_path / "common.json"
    assets = {"tail": {"world": "frozen"}, "natural": {"world": "natural"}}
    common.write_text(
        json.dumps(
            {
                "keep": ["first", "last"],
                "development": {
                    "tail_generated": ["tail"],
                    "natural_generated": ["natural"],
                },
                "assets": assets,
            }
        )
    )
    official = tmp_path / "official.pth"
    official.write_bytes(b"official")
    settings = {
        "master_seed": 20260926,
        "runtime": {"torch_threads": 1, "device": "cpu", "world_device": "cpu"},
        "capability_preserving_repair": {
            "keep_candidates": 2,
            "development_tail": 1,
            "development_natural": 1,
        },
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(settings))
    monkeypatch.setattr(materials, "ROOT", root)
    monkeypatch.setattr(materials, "COMMON_INPUTS", common)
    monkeypatch.setattr(materials, "OFFICIAL", official)
    monkeypatch.setattr(
        materials,
        "load_world",
        lambda settings, distribution="tail_generated": (
            None,
            None,
            None,
            assets["natural" if distribution == "natural_generated" else "tail"],
        ),
    )
    monkeypatch.setattr(
        materials,
        "ProcessPoolExecutor",
        lambda **kwargs: ThreadPoolExecutor(kwargs["max_workers"]),
    )
    captured, validated = [], []

    def capture(path):
        captured.append(path)
        return {"qualified_keep": path == "last", "record": {"elapsed_s": 0.1}}, 1

    def evaluate(paths, policy, destination, settings, **kwargs):
        validated.append((list(paths), kwargs["algorithm"], kwargs["purpose"]))
        return [{"cache_hit": False} for path in paths]

    monkeypatch.setattr(materials, "capture_path", capture)
    monkeypatch.setattr(materials, "evaluate_paths", evaluate)
    return materials, config, captured, validated


def test_highway_preparation_uses_all_candidates_and_carl_validation(
    highway_preparation,
):
    preparation, config, captured, validated = highway_preparation
    report = preparation.prepare_highway(config)
    assert sorted(captured) == ["first", "last"]
    assert report["recorded_keep_executions"] == 2
    assert report["new_keep_executions_this_invocation"] == 2
    assert report["qualified_keep_instances"] == 1
    assert validated == [
        (["tail"], "carl", "development"),
        (["natural"], "carl", "development"),
    ]
    assert all(
        row["recorded_executions"] == 1 for row in report["development"].values()
    )


def test_highway_preparation_rejects_a_changed_world(highway_preparation, monkeypatch):
    preparation, config, captured, _ = highway_preparation
    monkeypatch.setattr(
        preparation,
        "load_world",
        lambda *a, **kw: (None, None, None, {"world": "changed"}),
    )
    with pytest.raises(ValueError, match="different frozen worlds"):
        preparation.prepare_highway(config)
    assert captured == []
