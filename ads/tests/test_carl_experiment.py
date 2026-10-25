"""Campaign integration keeps a rejected local PPO candidate out of deployment."""

import json
from types import SimpleNamespace

import pytest
import torch

from ads.carl import experiment
from ads.carl.policy import CaRLPolicy
from ads.common.instances import UsageLedger, file_hash
from ads.common.repair import RepairEnvironment
from ads.common.reporting import write_json
from ads.risk_stratified_repair import Campaign


def prepare_materials(monkeypatch, tmp_path, instance):
    root = tmp_path / "materials"
    inputs = root / "instances.json"
    assets = {"tail": instance.assets, "natural": {"natural": "fixed"}}
    write_json(inputs, {"assets": assets, "development": {}})
    write_json(root / "protocol.json", {"assets": assets})
    write_json(
        root / "highway.json",
        {
            "status": "passed",
            "protocol_sha256": file_hash(root / "protocol.json"),
            "input_manifest_sha256": file_hash(inputs),
            "policy_sha256": file_hash(experiment.OFFICIAL),
        },
    )
    monkeypatch.setattr(experiment, "ROOT", root)
    monkeypatch.setattr(experiment, "COMMON_INPUTS", inputs)
    return root


def test_campaign_runs_new_ppo_and_restores_rejected_base(
    instance, world_factory, tmp_path, monkeypatch
):
    torch.set_num_threads(1)
    root = prepare_materials(monkeypatch, tmp_path, instance)
    banks = {"normal": [(instance, "prior")]}
    env = RepairEnvironment("carl", banks, world_factory, round_index=0)
    observation, _ = env.reset()
    samples = [{"scene_id": instance.root_sample_id, "observation": observation}]
    monkeypatch.setattr(experiment, "load_native_samples", lambda path: samples)
    monkeypatch.setattr(experiment, "load_keep_samples", lambda *a, **k: samples)
    native = root / "native.json"
    native.write_text("synthetic native material")
    campaign = Campaign.__new__(Campaign)
    campaign.device = "cpu"
    campaign.factory = world_factory
    campaign.assets = instance.assets
    campaign.ledger = UsageLedger(tmp_path / "usage.sqlite")
    campaign.protocol_hash = "synthetic-protocol"
    campaign.config = {
        "run_kind": "pilot",
        "repair": {"rl_steps": 8},
        "capability_preserving_repair": {
            "native_materials": str(native),
            "retention_coefficient": 1.0,
            "minimum_progress_ratio": 0.98,
        },
        "carl": {
            "rollout_steps": 4,
            "batch_size": 4,
            "micro_batch_size": 16,
            "epochs": 1,
            "learning_rate": 2.5e-5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.1,
            "value_coefficient": 0.5,
            "entropy_coefficient": 0.01,
        },
    }
    base = tmp_path / "baseline/policy.pth"
    initial = CaRLPolicy.load(experiment.OFFICIAL)
    initial.save(base)
    checked = []

    def reject(candidates, previous, owner, directory, **options):
        assert owner is campaign and previous == base
        assert [step for step, _ in candidates] == [8]
        assert options["algorithm"] == "carl"
        assert options["training_instances"] == [instance.instance_hash]
        trained = CaRLPolicy.load(candidates[0][1])
        assert trained.repair_state["decision_steps"] == 8
        assert trained.optimizer_state is not None
        assert any(
            not torch.equal(value, trained.network.state_dict()[name])
            for name, value in initial.network.state_dict().items()
        )
        checked.append(candidates[0][1])
        outcome = {"accepted_update": False, "selected_policy": str(base)}
        write_json(directory / "admission.json", outcome)
        return base, outcome

    monkeypatch.setattr(experiment, "admission", reject)
    directory = tmp_path / "guided/round_1"
    deployed = campaign.train(
        "carl", directory, banks, base=base, seed=20260926, round_index=0
    )
    restored = CaRLPolicy.load(deployed)
    for name, value in initial.network.state_dict().items():
        assert torch.equal(value, restored.network.state_dict()[name])
    assert restored.optimizer_state is None and restored.repair_state is None
    report = json.loads((directory / "training.json").read_text())
    assert checked and report["learning"]["actual_steps"] == 8
    assert not report["learning"]["admission"]["accepted_update"]
    assert sum(report["learning"]["event_decisions"].values()) == 8
    assert report["learning"]["retention_materials"]["native_observations"] == 1
    assert (
        json.loads((directory / "optimization.json").read_text())["actual_steps"] == 8
    )


def test_material_identity_is_checked_before_any_ppo_update(
    instance, tmp_path, monkeypatch
):
    root = prepare_materials(monkeypatch, tmp_path, instance)
    inputs = json.loads(experiment.COMMON_INPUTS.read_text())
    inputs["assets"]["tail"] = {"different": "world"}
    write_json(experiment.COMMON_INPUTS, inputs)
    campaign = SimpleNamespace(assets=instance.assets)
    with pytest.raises(ValueError, match="protocols differ"):
        experiment.repair_and_select(
            campaign,
            None,
            None,
            base=None,
            directory=root / "round",
            steps=20480,
            seed=20260926,
            progress=None,
        )
