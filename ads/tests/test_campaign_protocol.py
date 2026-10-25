"""Exercise persisted campaign boundaries with synthetic scenes and stub learners."""

import json
from dataclasses import asdict

import pytest

from ads.audit import check_archive_execution_labels
from ads.common.curriculum import persist_curriculum
from ads.common.instances import UsageLedger, file_hash
from ads.common.reporting import write_json
from ads.risk_stratified_repair import METHODS, Campaign
from ads.search.archive import FailureArchive
from ads.tasks import POLICY_SUFFIX, SUPERVISED_ALGORITHMS
from ads.tests.test_search import evaluate, small_config


def test_restart_training_reuses_complete_search_without_world_execution(
    instance, tmp_path, monkeypatch
):
    campaign = Campaign.__new__(Campaign)
    campaign.output = tmp_path
    campaign.config = {"search": {"budget": 5000}}
    campaign.assets = instance.assets
    campaign.ledger = UsageLedger(tmp_path / "usage.sqlite")
    baseline = tmp_path / "dtpp/20260926/baseline"
    baseline.mkdir(parents=True)
    initial = baseline / "policy.pth"
    initial.write_bytes(b"initial policy")
    write_json(baseline / "training.json", {"policy_sha256": file_hash(initial)})
    directory = tmp_path / "dtpp/20260926/guided/round_1"
    persist_curriculum(directory, {"normal": [(instance, "prior")]}, campaign.ledger)
    original = file_hash(directory / "inputs/curriculum.json")
    (directory / "search.jsonl").write_text("{}\n")
    write_json(directory / "search_summary.json", {"evaluations": 1})
    FailureArchive().save(directory / "archive")

    def forbidden(*args, **kwargs):
        raise AssertionError("training restart must not execute the world or search")

    monkeypatch.setattr("ads.risk_stratified_repair.SearchEngine", forbidden)
    monkeypatch.setattr("ads.risk_stratified_repair.WorldEvaluator", forbidden)
    monkeypatch.setattr("ads.risk_stratified_repair.load_policy", forbidden)
    calls = []

    def train(algorithm, path, banks, **options):
        assert algorithm == "dtpp" and path == directory
        assert options["base"] == initial and options["round_index"] == 0
        assert not options["teacher_history"] and options["replay"] is None
        assert banks["normal"][0][0].instance_hash == instance.instance_hash
        assert persist_curriculum(path, banks, campaign.ledger) == original
        calls.append(path)
        return path / "policy.pth"

    monkeypatch.setattr(campaign, "train", train)
    assert (
        campaign.train_saved_search("dtpp", 20260926, "guided", 1)
        == directory / "policy.pth"
    )
    assert calls == [directory]
    write_json(
        directory / "search_summary.json",
        {"evaluations": 1, "policy_sha256": "another policy"},
    )
    with pytest.raises(ValueError, match="another policy"):
        campaign.train_saved_search("dtpp", 20260926, "guided", 1)


@pytest.mark.parametrize("algorithm", tuple(POLICY_SUFFIX))
@pytest.mark.parametrize("search_first", (False, True))
def test_two_round_campaign_resume_preserves_protocol(
    algorithm, search_first, source, tmp_path, monkeypatch
):
    campaign = Campaign.__new__(Campaign)
    campaign.output = tmp_path
    campaign.config = {
        "rounds": 2,
        "master_seed": 20260926,
        "runtime": {"generation_batch_size": 4},
        "repair": {"normal_instances": 4, "plant_new_teacher_instances": 32},
        "search": asdict(small_config(budget=60, replay_limit=2)),
    }
    campaign.protocol_hash = "synthetic-protocol"
    campaign.sampler, campaign.assets = source.sampler, source.assets
    campaign.ledger = UsageLedger(tmp_path / "usage.sqlite")
    campaign.device, campaign.factory, campaign.evt = "cpu", None, None

    def evaluator(instance):
        result = evaluate(instance)
        result.record.update(
            instance_hash=instance.instance_hash,
            root_sample_id=instance.root_sample_id,
        )
        return result

    monkeypatch.setattr(
        "ads.risk_stratified_repair.load_policy", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        "ads.risk_stratified_repair.WorldEvaluator", lambda *args, **kwargs: evaluator
    )
    calls = []

    def save(path, contents):
        path.parent.mkdir(parents=True, exist_ok=True)
        if algorithm == "str2":
            path.mkdir()
            (path / "synthetic-weights").write_text(contents)
        else:
            path.write_text(contents)

    def learner(algorithm, directory, banks, **options):
        persist_curriculum(
            directory,
            banks,
            campaign.ledger,
            teacher_history=options["teacher_history"],
        )
        round_index = options["round_index"]
        if algorithm == "dqn":
            assert (options["replay"] is None) == (round_index == 0)
            if round_index:
                assert options["replay"].is_file()
            (directory / "replay_buffer.pkl").write_bytes(b"synthetic replay")
        if round_index:
            assert options["base"].parent.name == "round_1"
            assert banks["history"] or not previous_archive_ids[directory.parent.name]
        else:
            assert options["base"] == initial
            assert not banks.get("history")
        if algorithm in SUPERVISED_ALGORITHMS:
            assert "new_failure" in banks
            chosen = banks.get("new_failure", [])[:1]
            selection = []
            for instance, branch in chosen:
                instance.save(
                    directory / "teacher_inputs" / (instance.instance_hash + ".npz")
                )
                selection.append(
                    {"instance_hash": instance.instance_hash, "domain": branch}
                )
            write_json(directory / "teacher_selection.json", selection)
            if round_index:
                assert (
                    len(options["teacher_history"])
                    == first_teacher_count[directory.parent.name]
                )
            else:
                first_teacher_count[directory.parent.name] = len(selection)
        checkpoint = campaign.checkpoint(directory, algorithm)
        save(checkpoint, f"{algorithm}:{directory.parent.name}:{round_index}")
        write_json(
            directory / "training.json", {"policy_sha256": file_hash(checkpoint)}
        )
        calls.append((directory.parent.name, round_index))
        return checkpoint

    monkeypatch.setattr(campaign, "train", learner)
    baseline = tmp_path / algorithm / "20260926" / "baseline"
    initial = campaign.checkpoint(baseline, algorithm)
    save(initial, "same initial policy")
    write_json(baseline / "training.json", {"policy_sha256": file_hash(initial)})
    previous_archive_ids, first_teacher_count, initial_samples = {}, {}, {}
    for method in METHODS:
        directory = tmp_path / algorithm / "20260926" / method
        if search_first:
            before = len(calls)
            prepared = campaign.run_branch(
                algorithm,
                20260926,
                method,
                initial,
                through_round=1,
                search_only=True,
            )
            assert prepared == directory / "round_1/search_summary.json"
            assert len(calls) == before
            assert not (directory / "round_1/training.json").exists()
            journal = file_hash(directory / "round_1/search.jsonl")
            curriculum = file_hash(directory / "round_1/inputs/curriculum.json")
            with monkeypatch.context() as preparation:

                def forbidden(*args, **kwargs):
                    raise AssertionError("prepared search must not execute again")

                preparation.setattr(
                    "ads.risk_stratified_repair.SearchEngine", forbidden
                )
                preparation.setattr(
                    "ads.risk_stratified_repair.WorldEvaluator", forbidden
                )
                assert (
                    campaign.run_branch(
                        algorithm,
                        20260926,
                        method,
                        initial,
                        through_round=1,
                        search_only=True,
                    )
                    == prepared
                )
                campaign.run_branch(
                    algorithm, 20260926, method, initial, through_round=1
                )
            assert file_hash(directory / "round_1/search.jsonl") == journal
            assert file_hash(directory / "round_1/inputs/curriculum.json") == curriculum
        else:
            campaign.run_branch(algorithm, 20260926, method, initial, through_round=1)
        archive = json.loads((directory / "round_1/archive/archive.json").read_text())
        previous_archive_ids[method] = set(archive["entries"])
        campaign.run_branch(algorithm, 20260926, method, initial, through_round=2)
        complete = campaign.run_branch(algorithm, 20260926, method, initial)
        assert complete.parent.name == "round_2"
        snapshots = []
        for number in (1, 2):
            round_path = directory / f"round_{number}"
            rows = [
                json.loads(line)
                for line in (round_path / "search.jsonl").read_text().splitlines()
            ]
            summary = json.loads((round_path / "search_summary.json").read_text())
            snapshot = {}
            for row in rows:
                snapshot.setdefault(row["instance_hash"], []).append(row)
            snapshots.append(snapshot)
            check_archive_execution_labels(
                json.loads((round_path / "archive/archive.json").read_text()),
                snapshots,
            )
            assert len(rows) == sum(summary["execution_costs"].values()) <= 60
            if number == 1:
                initial_samples[method] = [
                    row["instance_hash"]
                    for row in rows
                    if row["execution_kind"] == "initial_prior"
                ]
            else:
                retained = json.loads(
                    (round_path / "archive/archive.json").read_text()
                )["retained"]
                assert set(retained) == previous_archive_ids[method]
                assert summary["execution_costs"].get("archive_replay", 0) <= 2
            manifest = json.loads((round_path / "inputs/curriculum.json").read_text())
            assert all(
                entry["instance_hash"]
                not in {
                    row.get("instance_hash")
                    for row in rows
                    if row["execution_kind"] == "branch_check"
                }
                for group in manifest["categories"].values()
                for entry in group
            )
    assert calls == [
        (method, round_index) for method in METHODS for round_index in (0, 1)
    ]
    assert (
        initial_samples["monte_carlo"]
        == initial_samples["subset"]
        == initial_samples["guided"]
    )


@pytest.mark.parametrize("operation", (None, "--search-only", "--training-only"))
def test_worker_records_search_preparation_separately_from_training(
    operation, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from ads import worker

    directory = tmp_path / "str2/20260926/guided/round_1"
    for name in (
        "search_summary.json",
        "inputs/curriculum.json",
        "archive/archive.json",
    ):
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {})
    called = []

    def resume(*identity):
        called.append(("train", identity))
        return directory / "policy"

    def branch(algorithm, seed, method, initial, **options):
        assert initial == tmp_path / "str2/20260926/baseline/policy"
        assert options == {
            "through_round": 1,
            "search_only": operation == "--search-only",
        }
        called.append(
            (
                "prepare" if options["search_only"] else "branch",
                (algorithm, seed, method, 1),
            )
        )
        return directory / (
            "search_summary.json" if options["search_only"] else "policy"
        )

    campaign = SimpleNamespace(
        output=tmp_path,
        config={"algorithms": ["str2"], "seeds": [20260926]},
        verify_runtime=lambda: None,
        checkpoint=lambda path, algorithm: path / "policy",
        train_saved_search=resume,
        run_branch=branch,
    )
    monkeypatch.setattr(worker, "Campaign", lambda path: campaign)
    command = [
        "worker",
        "repair",
        "str2",
        "20260926",
        "--method",
        "guided",
        "--round",
        "1",
    ]
    if operation:
        command.append(operation)
    monkeypatch.setattr("sys.argv", command)
    worker.main()
    status = json.loads((directory / "unit_status.json").read_text())
    expected_operation = {
        None: "branch",
        "--search-only": "prepare",
        "--training-only": "train",
    }[operation]
    assert called == [
        (
            expected_operation,
            ("str2", 20260926, "guided", 1),
        )
    ]
    assert status["status"] == (
        "search_complete" if operation == "--search-only" else "complete"
    )
