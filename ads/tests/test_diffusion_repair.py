"""DP repair invariants using the official model and synthetic causal scenes."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from ads.common.demonstrations import DemonstrationCache
from ads.common.instances import UsageLedger
from ads.common.setup import load_policy
from ads.diffusion_planner.repair import (
    TRAINABLE_PREFIXES,
    configure_local_updates,
    retention_loss,
    train,
    training_mode,
)
from ads.nuplan.assets import official_path
from ads.nuplan.teacher import demonstration


@pytest.fixture
def policy():
    torch.set_num_threads(1)
    return load_policy(
        "diffusion_planner", official_path("diffusion_planner"), device="cpu"
    )


def material(instance):
    states = np.tile(instance.event.initial_states, (150, 1, 1))
    states[:, :, :2] += np.arange(150)[:, None, None] * 0.04 * states[:, :, 2:4]
    sample = demonstration(instance, states, "diffusion_planner")[0]
    sample.update(
        scene_id=instance.instance_hash, domain="synthetic", label_source="expert"
    )
    return sample


def test_initial_optimizer_state_and_raw_match_official_deployment(policy):
    assert policy.optimizer_state is None
    assert policy.repair_state is None
    for name, value in policy.network.state_dict().items():
        assert torch.equal(value, policy.training_state[name])
        assert torch.equal(value, policy.ema_state[name])


def test_fixed_teacher_constraint_has_live_local_gradients(policy, instance):
    teacher = load_policy(
        "diffusion_planner", official_path("diffusion_planner"), device="cpu"
    )
    teacher.network.eval().requires_grad_(False)
    names = configure_local_updates(policy)
    assert all(name.startswith(TRAINABLE_PREFIXES) for name in names)
    sample = material(instance)
    unchanged = retention_loss(policy, teacher, [sample])
    assert unchanged.item() < 1e-10  # no-grad attention can select a different kernel
    assert unchanged.requires_grad
    with torch.no_grad():
        policy.network.decoder.decoder.dit.final_layer.proj[-1].bias.add_(0.01)
    changed = retention_loss(policy, teacher, [sample])
    assert changed.item() > 0
    changed.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in policy.network.parameters()
        if p.requires_grad
    )
    assert all(p.grad is None for p in teacher.network.parameters())
    assert all(
        p.grad is None for p in policy.network.parameters() if not p.requires_grad
    )
    training_mode(policy, True)
    assert not policy.network.encoder.training
    assert not policy.network.decoder.decoder.dit.route_encoder.training
    assert policy.network.decoder.decoder.dit.blocks[2].training


def test_mixed_repair_preserves_frozen_state_and_restores_native_ema(
    policy, instance, tmp_path
):
    sample = material(instance)
    banks = {
        name: [deepcopy(sample)] for name in ("native", "keep", "normal", "new_failure")
    }
    captured = []

    def save(model, update):
        model.save(tmp_path / "policy.pth")
        captured.append(update)

    report = train(policy, banks, updates=2, seed=20260926, checkpoint=save)
    assert captured == [2]
    assert report["frozen_state_unchanged"]
    assert report["category_samples"]["native"] == 64
    assert report["category_samples"]["keep"] == 32
    assert (
        report["category_samples"]["normal"] + report["category_samples"]["new_failure"]
        == 32
    )
    restored = load_policy("diffusion_planner", tmp_path / "policy.pth", device="cpu")
    assert restored.repair_state == policy.repair_state
    for name, value in policy.network.state_dict().items():
        assert torch.equal(value, restored.network.state_dict()[name])
        assert torch.equal(policy.training_state[name], restored.training_state[name])
    assert restored.optimizer_state is not None
    policy.export_native(tmp_path / "native/model.pth")
    native = load_policy(
        "diffusion_planner", tmp_path / "native/model.pth", device="cpu"
    )
    for name, value in policy.network.state_dict().items():
        assert torch.equal(value, native.network.state_dict()[name])
    assert native.optimizer_state is None


def test_successful_execution_is_retained_without_native_future_supervision(
    policy, instance, monkeypatch
):
    from ads.diffusion_planner import repair

    sample = material(instance)
    banks = {
        name: [deepcopy(sample)] for name in ("native", "keep", "normal", "new_failure")
    }
    for name, samples in banks.items():
        samples[0]["domain"] = name
    banks["normal"][0]["label_source"] = "policy_success"
    bias = policy.network.decoder.decoder.dit.final_layer.proj[-1].bias
    name = next(
        name for name, value in policy.network.named_parameters() if value is bias
    )
    with torch.no_grad():
        bias.add_(0.01)
    deployment = bias.detach().clone()
    policy.ema_state[name] = deployment.clone()
    policy.training_state[name] = deployment.clone() + 0.2
    supervised, retained = [], []

    def native_loss(student, samples):
        supervised.extend(samples)
        return (
            student.network.decoder.decoder.dit.final_layer.proj[-1]
            .bias.square()
            .mean()
        )

    def keep_loss(student, teacher, samples):
        retained.extend(samples)
        assert all(not p.requires_grad for p in teacher.network.parameters())
        if any(s["label_source"] == "policy_success" for s in samples):
            assert torch.equal(
                teacher.network.decoder.decoder.dit.final_layer.proj[-1].bias,
                deployment,
            )
        return (
            student.network.decoder.decoder.dit.final_layer.proj[-1]
            .bias.square()
            .mean()
        )

    monkeypatch.setattr(repair, "diffusion_loss", native_loss)
    monkeypatch.setattr(repair, "retention_loss", keep_loss)
    report = train(policy, banks, updates=1, seed=20260926)
    assert {s["domain"] for s in supervised} == {"native", "new_failure"}
    assert {s["domain"] for s in retained} == {"native", "keep", "normal"}
    assert report["objective_samples"]["original_retention"] == 48
    assert report["objective_samples"]["successful_retention"] > 0
    assert report["successful_reference"] == "round_start_deployment"


def test_only_successful_material_does_not_trigger_a_repair(policy, instance):
    sample = material(instance)
    sample["label_source"] = "policy_success"
    original = deepcopy(policy.network.state_dict())
    report = train(policy, {"normal": [sample]}, updates=1, seed=20260926)
    assert report["gradient_updates"] == 0
    assert report["skipped"] == "no_qualified_demonstrations"
    assert all(
        torch.equal(value, original[name])
        for name, value in policy.network.state_dict().items()
    )


def test_validation_instances_cannot_become_learning_material(instance, tmp_path):
    validation = replace(instance, purpose="development")
    ledger = UsageLedger(tmp_path / "usage.sqlite")
    ledger.register(validation, "development")
    for purpose in ("repair", "search", "final_test"):
        with pytest.raises(ValueError, match="development evaluation"):
            ledger.register(validation, purpose)
    cache = DemonstrationCache(
        tmp_path / "expert", None, algorithm="diffusion_planner", candidate_budget=128
    )
    with pytest.raises(ValueError, match="development"):
        cache.get(validation)


def test_pilot_and_formal_learning_streams_are_disjoint():
    from ads.common.evaluation_sampling import source_stream

    values = {
        source_stream({"run_kind": kind}, "protocol", purpose, "shared_prior", 0)
        for kind in ("pilot", "formal")
        for purpose in ("search", "repair", "development")
    }
    assert len(values) == 6


def test_expert_prefers_original_intent_over_higher_progress(instance):
    from ads.common.expert import Candidate, TeacherExecution, expert_rank

    states = np.tile(instance.event.initial_states, (150, 1, 1))
    controls = np.zeros((149, 2), np.float32)
    lane = instance.event.geometry.nearest_lane(states[0, 0, :2])
    normal = TeacherExecution(Candidate(lane, 20), True, 1, 120, states, controls)
    faster = replace(normal, candidate=Candidate(lane, 24), reward=100, progress=144)
    assert expert_rank(normal, instance.event) > expert_rank(faster, instance.event)


def test_highway_admission_checks_collision_and_task_retention():
    from ads.common.admission import safe_task_change

    before = [
        {"collision": False, "offroad": False, "stalled": False, "progress_m": 100.0}
    ]
    assert all(safe_task_change(before, before).values())
    assert not safe_task_change(before, [{**before[0], "collision": True}])["collision"]
    assert not safe_task_change(before, [{**before[0], "progress_m": 97.9}])["progress"]
    assert safe_task_change(before, [{**before[0], "progress_m": 98.0}])["progress"]


def test_world_history_window_preserves_full_policy_history(instance):
    from types import SimpleNamespace

    from conftest import SmallWorld

    from ads.common.environment import PolicyEnvironment

    class HistoryWorld(SmallWorld):
        cfg = SimpleNamespace(history_frames=25)

        def reset(self, *args, **kwargs):
            assert kwargs["initial_history"].shape == (1, 25, 7, 6)
            assert kwargs["initial_history_valid"].shape == (1, 25, 7)
            return super().reset(*args, **kwargs)

    history = np.tile(instance.event.initial_states, (51, 1, 1))
    valid = np.tile(instance.event.valid, (51, 1))
    empirical = replace(
        instance,
        event=replace(
            instance.event, initial_history=history, initial_history_valid=valid
        ),
        source_kind="empirical_natural",
    )
    empirical = replace(
        empirical, event=replace(empirical.event, event_id=empirical.instance_hash)
    )
    env = PolicyEnvironment(
        "diffusion_planner", [empirical], lambda event: HistoryWorld()
    )
    observation, _ = env.reset()
    assert len(empirical.event.initial_history) == 51
    assert observation["history_valid"][:, 0].all()
