"""CaRL pilot uses fresh optimization, paired development and native handoff."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from ads.carl import pilot
from ads.carl.policy import CaRLPolicy
from ads.common.instances import UsageLedger
from ads.common.reporting import write_json


@pytest.mark.parametrize("resumed", [False, True])
@pytest.mark.parametrize("accepted", [False, True])
def test_pilot_keeps_engineering_weights_out_of_two_round_repair(
    tmp_path, monkeypatch, resumed, accepted
):
    settings = {
        "algorithms": ["carl"],
        "methods": ["guided"],
        "run_kind": "pilot",
        "master_seed": 20260926,
        "rounds": 2,
    }
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(settings))
    materials = tmp_path / "materials"
    monkeypatch.setattr(pilot, "ROOT", materials)
    calls = []

    def prepare(path):
        assert path == config
        calls.append("prepare")
        write_json(materials / "highway.json", {"status": "passed"})

    monkeypatch.setattr(pilot, "prepare_highway", prepare)
    output = tmp_path / "pilot"
    first = output / "carl/20260926/guided/round_1"
    if resumed:
        write_json(materials / "highway.json", {"status": "passed"})
        write_json(first / "training.json", {"already_complete": True})

    class Campaign:
        def __init__(self, path):
            assert path == config
            self.config, self.output, self.device = settings, output, "cpu"

        def verify_runtime(self):
            calls.append("runtime")

        def baseline(self, algorithm, seed):
            assert algorithm == "carl" and seed == 20260926
            path = output / "baseline.pth"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"official")
            return path

        def run_branch(self, algorithm, seed, method, initial, **options):
            assert (algorithm, seed, method) == ("carl", 20260926, "guided")
            assert initial.read_bytes() == b"official"
            if options:
                assert options == {"through_round": 1, "search_only": True}
                calls.append("search_first")
                return first / "search_summary.json"
            calls.append("two_rounds")
            for number in (1, 2):
                write_json(
                    first.parent / f"round_{number}/admission.json",
                    {"accepted_update": accepted and number == 1},
                )
            final = first.parent / "round_2/policy.pth"
            final.write_bytes(b"enhanced" if accepted else b"official")
            return final

    monkeypatch.setattr(pilot, "Campaign", Campaign)

    def engineering(campaign, initial, seed):
        calls.append("engineering")
        target = campaign.output / "engineering.pth"
        target.write_bytes(b"engineering-only")

    monkeypatch.setattr(pilot, "engineering", engineering)
    final_models = []

    def load(algorithm, path, **options):
        assert algorithm == "carl" and options["device"] == "cpu"
        final_models.append(Path(path).read_bytes())
        return object()

    monkeypatch.setattr(pilot, "load_policy", load)

    def export(model, path):
        assert path == output / "handoff/model_best.pth"
        return {"native_export": str(path), "native_export_readback_identical": True}

    monkeypatch.setattr(pilot, "export_and_verify", export)
    report = pilot.run(config)
    assert final_models == [b"enhanced" if accepted else b"official"]
    assert report["status"] == (
        "highway_pilot_passed" if accepted else "no_accepted_enhancement"
    )
    assert report["rounds"] == 2 and len(report["admission_reports"]) == 2
    assert not report["final_cohorts_generated"]
    assert not report["formal_experiment_completed"]
    assert report["nuplan_validation"] == "manual_by_user"
    assert calls == (
        ["runtime", "two_rounds"]
        if resumed
        else ["prepare", "runtime", "search_first", "engineering", "two_rounds"]
    )
    assert not (output / "frozen_policies.json").exists()
    assert not (output / "instances").exists()


def test_engineering_uses_eight_keep_scenes_and_fixed_development_subset(
    source, tmp_path, monkeypatch
):
    rows = source.sample(20)
    banks = {
        "normal": [(instance, "prior") for instance in rows[:10]],
        "new_failure": [(instance, "branch") for instance in rows[10:]],
    }
    monkeypatch.setattr(pilot, "load_curriculum", lambda *a: (banks, []))
    native = [{"scene_id": f"native-{i}"} for i in range(10)]
    keep = [{"scene_id": f"keep-{i}"} for i in range(10) for _ in range(2)]
    monkeypatch.setattr(pilot, "load_native_samples", lambda path: native)
    monkeypatch.setattr(pilot, "load_keep_samples", lambda *a, **k: keep)
    inputs = tmp_path / "instances.json"
    write_json(
        inputs,
        {
            "development": {
                name: list(range(20))
                for name in ("tail_generated", "natural_generated")
            }
        },
    )
    monkeypatch.setattr(pilot, "COMMON_INPUTS", inputs)

    class Model:
        def save(self, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"engineering")

    monkeypatch.setattr(pilot, "load_policy", lambda *a, **k: Model())
    monkeypatch.setattr(
        pilot,
        "export_and_verify",
        lambda *a: {"native_export_readback_identical": True},
    )
    seen = []

    def train(model, env, **options):
        assert len(env.banks["normal"]) == len(env.banks["new_failure"]) == 8
        assert options["steps"] == 1024
        assert len(options["retention"]["native"]) == 8
        chosen = options["retention"]["keep"]
        assert len(chosen) == 16
        assert len({sample["scene_id"] for sample in chosen}) == 8
        seen.append("train")
        return {"actual_steps": 1024, "frozen_state_identical": True}

    monkeypatch.setattr(pilot, "train", train)

    def evaluate(paths, policy, directory, config, **options):
        assert paths == list(range(16))
        assert options["algorithm"] == "carl"
        assert policy.read_bytes() == b"engineering"
        seen.append("development")
        return [{"cache_hit": False, "record": {"collision": False}} for _ in paths]

    monkeypatch.setattr(pilot, "evaluate_paths", evaluate)
    campaign = SimpleNamespace(
        output=tmp_path,
        device="cpu",
        ledger=UsageLedger(tmp_path / "usage.sqlite"),
        factory=lambda *a: pytest.fail("Stub engineering must not run a world"),
        config={
            "carl": {},
            "capability_preserving_repair": {
                "native_materials": "native.json",
                "retention_coefficient": 1.0,
            },
        },
    )
    initial = tmp_path / "initial.pth"
    initial.write_bytes(b"official")
    report = pilot.engineering(campaign, initial, 20260926)
    assert seen == ["train", "development", "development"]
    assert report["native_retention_scenes"] == report["highway_retention_scenes"] == 8
    assert report["development"]["tail_generated"]["new_executions"] == 16
    assert pilot.engineering(campaign, initial, 20260926) == report
    assert len(seen) == 3
    initial.write_bytes(b"another official")
    with pytest.raises(ValueError, match="another initial"):
        pilot.engineering(campaign, initial, 20260926)


def test_trained_head_exports_with_identical_native_state(tmp_path):
    torch.set_num_threads(1)
    model = CaRLPolicy.load(pilot.OFFICIAL)
    with torch.no_grad():
        model.network.dist_mu[0].bias.add_(0.1)
    path = tmp_path / "native/model_best.pth"
    report = pilot.export_and_verify(model, path)
    assert report["native_export_readback_identical"]
    assert report["native_state_tensors"] == len(model.network.state_dict())
    assert report["native_export_sha256"]
