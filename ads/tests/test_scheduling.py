import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from ads.common.instances import file_hash, json_hash
from ads.risk_stratified_repair import Campaign
from ads.run_campaign import Scheduler
from ads.tasks import Task, experiment_graph


def test_fresh_campaign_starts_baseline_without_a_policy_freeze(tmp_path, monkeypatch):
    import pytest

    config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/risk_stratified_repair.yaml").read_text()
    )
    config["output_dir"] = str(tmp_path / "out")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    scheduler = Scheduler(path)
    assert len(scheduler.graph) == 72
    assert all(not scheduler.done(task) for task in scheduler.graph)

    def launch(task):
        assert task.stage == "baseline"
        raise RuntimeError("initial policy import launched")

    monkeypatch.setattr(scheduler, "launch", launch)
    with pytest.raises(RuntimeError, match="initial policy import launched"):
        scheduler.run()


def test_str2_repair_has_exclusive_gpu_admission():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.workers = 3
    ordinary = Task("repair", "diffusion_planner", 20260926, "guided", 1)
    str2 = Task("repair", "str2", 20260926, "guided", 1)
    scheduler.running = {}
    assert scheduler.can_launch(str2)
    scheduler.running = {ordinary: 1}
    assert scheduler.can_launch(Task("repair", "dqn", 20260926, "guided", 1))
    assert not scheduler.can_launch(str2)
    scheduler.running = {str2: 2}
    assert not scheduler.can_launch(ordinary)


def test_str2_inference_limit_covers_evaluation_and_diagnostics():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.workers = 12
    initial = Task("iid", "str2", 20260926, "initial")
    guided = Task("iid", "str2", 20260926, "guided")
    diagnostic = Task("diagnostic", "str2", 20260926, "monte_carlo")
    scheduler.running = {initial: 1, guided: 2}
    assert scheduler.can_launch(diagnostic)
    scheduler.running[diagnostic] = 3
    assert not scheduler.can_launch(Task("iid", "str2", 20260926, "subset"))
    assert not scheduler.can_launch(Task("diagnostic", "str2", 20260926, "subset"))
    assert scheduler.can_launch(Task("iid", "dtpp", 20260926, "guided"))
    scheduler.running.pop(initial)
    assert scheduler.can_launch(Task("iid", "str2", 20260926, "subset"))


def test_prepared_search_is_pending_training_not_failed_or_complete(
    tmp_path, monkeypatch
):
    import os
    from types import SimpleNamespace

    import pytest

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.output = tmp_path
    scheduler.workers = 2
    prepared = Task("repair", "str2", 20260926, "guided", 1)
    ordinary = Task("repair", "dtpp", 20260926, "guided", 2)
    directory = prepared.directory(tmp_path)
    for name in (
        "search_summary.json",
        "inputs/curriculum.json",
        "archive/archive.json",
    ):
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    (directory / "unit_status.json").write_text(
        json.dumps(
            {
                "status": "search_complete",
                "search_only": True,
                "pid": 0,
            }
        )
    )
    assert scheduler.search_prepared(prepared)
    assert not scheduler.training_done(prepared)
    scheduler.graph = {prepared: set(), ordinary: set()}
    scheduler.running = {prepared: 0, ordinary: os.getpid()}
    scheduler.children = {prepared: SimpleNamespace(poll=lambda: 0)}
    scheduler.failed = set()
    monkeypatch.setattr(scheduler, "done", lambda task: False)

    def forbidden(task):
        raise AssertionError("STR2 optimization must wait for the other worker")

    def tick(seconds):
        raise RuntimeError("one scheduler tick")

    monkeypatch.setattr(scheduler, "launch", forbidden)
    monkeypatch.setattr("ads.run_campaign.time.sleep", tick)
    with pytest.raises(RuntimeError, match="one scheduler tick"):
        scheduler.run()
    assert not scheduler.failed
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["completed_units"] == 0
    assert status["running"] == {ordinary.key: os.getpid()}


def test_final_iid_gate_depends_on_all_registered_training():
    config = {
        "algorithms": ["dqn", "ppo", "carl", "plant2"],
        "seeds": [1, 2, 3],
        "methods": ["monte_carlo", "subset", "guided"],
        "rounds": 2,
    }
    graph = experiment_graph(config)
    assert len(graph) == 170
    freeze = Task("freeze")
    assert len(graph[freeze]) == 36
    reached = set()
    pending = list(graph[freeze])
    while pending:
        task = pending.pop()
        if task not in reached:
            reached.add(task)
            pending.extend(graph[task])
    assert len(reached) == 84
    assert all(graph[t] == {freeze} for t in graph if t.stage == "iid")
    assert len(graph[Task("report")]) == 84


def test_incomplete_freeze_cannot_open_the_formal_evaluation_gate(tmp_path):
    import pytest

    campaign = object.__new__(Campaign)
    campaign.output = tmp_path
    campaign.protocol_hash = "registered"
    campaign.config = {
        "algorithms": ["dqn"],
        "seeds": [1, 2],
        "methods": ["guided"],
        "rounds": 2,
        "run_kind": "formal",
    }
    path = tmp_path / "frozen_policies.json"
    path.write_text(json.dumps({"protocol_hash": "registered", "policies": {}}))
    with pytest.raises(ValueError, match="complete registered policy set"):
        campaign.verify_freeze()
    policies = {
        f"dqn/{seed}/{arm}": {} for seed in (1, 2) for arm in ("initial", "guided")
    }
    path.write_text(json.dumps({"protocol_hash": "registered", "policies": policies}))
    with pytest.raises(ValueError, match="every registered repair checkpoint"):
        campaign.verify_freeze()


def test_status_file_does_not_substitute_for_completed_training(tmp_path):
    config = {
        "output_dir": str(tmp_path / "out"),
        "algorithms": ["dqn"],
        "seeds": [1],
        "methods": ["guided"],
        "rounds": 2,
        "baseline": {"rl_steps": 16, "development_instances": 2},
        "repair": {"rl_steps": 8},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    scheduler = Scheduler(path)
    task = Task("baseline", "dqn", 1)
    directory = task.directory(scheduler.output)
    directory.mkdir(parents=True)
    (directory / "unit_status.json").write_text(json.dumps({"status": "complete"}))
    assert not scheduler.done(task)
    protocol = {"settings": config}
    (scheduler.output / "protocol.json").write_text(json.dumps(protocol))
    checkpoint = directory / "policy.zip"
    checkpoint.write_bytes(b"checkpoint")
    inputs = directory / "inputs/curriculum.json"
    inputs.parent.mkdir()
    inputs.write_text("{}")
    meta = {
        "policy_sha256": file_hash(checkpoint),
        "protocol_hash": json_hash(protocol),
        "curriculum_sha256": file_hash(inputs),
        "learning": {"actual_steps": 0},
    }
    (directory / "training.json").write_text(json.dumps(meta))
    (directory / "development.json").write_text("[{},{}]")
    assert scheduler.done(task)
    checkpoint.write_bytes(b"changed")
    import pytest

    with pytest.raises(ValueError, match="checkpoint hash"):
        scheduler.done(task)


def test_concurrent_pool_creation_uses_one_exact_sample_set(tmp_path, monkeypatch):
    from conftest import SmallSampler

    sampler = SmallSampler()
    monkeypatch.setattr(
        "ads.risk_stratified_repair.load_world",
        lambda settings: (sampler, None, None, {"world": "frozen"}),
    )
    config = {
        "output_dir": str(tmp_path / "out"),
        "master_seed": 1,
        "algorithms": ["dqn"],
        "methods": ["guided"],
        "rounds": 2,
        "runtime": {"torch_threads": 1, "device": "cpu", "generation_batch_size": 2},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    campaign = Campaign(path)
    with ThreadPoolExecutor(2) as executor:
        results = list(
            executor.map(
                lambda _: campaign.pool("repair", "shared", 4, "shared"), range(2)
            )
        )
    assert [i.instance_hash for i in results[0]] == [
        i.instance_hash for i in results[1]
    ]
    records = (campaign.output / "generation.jsonl").read_text().splitlines()
    assert len(records) == 1 and json.loads(records[0])["new_instances"] == 4


def test_dead_worker_restarts_incomplete_task(tmp_path, monkeypatch):
    import pytest

    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "output_dir": str(tmp_path / "out"),
                "algorithms": ["dqn"],
                "seeds": [1],
                "methods": ["guided"],
                "rounds": 2,
            }
        )
    )
    scheduler = Scheduler(path)
    task = Task("baseline", "dqn", 1)
    scheduler.graph = {task: set()}
    directory = task.directory(scheduler.output)
    directory.mkdir(parents=True)
    (directory / "unit_status.json").write_text(
        json.dumps(
            {
                "pid": 999999999,
                "status": "running",
            }
        )
    )
    monkeypatch.setattr(scheduler, "done", lambda task: False)

    def relaunched(task):
        raise RuntimeError("replacement worker launched")

    monkeypatch.setattr(scheduler, "launch", relaunched)
    with pytest.raises(RuntimeError, match="replacement worker launched"):
        scheduler.run()
    assert not scheduler.failed


def test_parallel_expert_cache_does_not_duplicate_simulation(
    instance, tmp_path, monkeypatch
):
    from ads.common.demonstrations import DemonstrationCache
    from ads.common.expert import Candidate, TeacherExecution

    calls = []

    def execute(*args, **kwargs):
        calls.append(1)
        time.sleep(0.03)
        return TeacherExecution(
            Candidate(1, 20), True, 1, 10, np.zeros((2, 7, 6)), np.zeros((1, 2))
        ), {"candidates": 8, "qualified_demonstration": True, "elapsed_s": 0.03}

    monkeypatch.setattr("ads.common.expert.successful_expert", execute)
    monkeypatch.setattr(
        "ads.plant2.demonstrations.demonstration", lambda *args: ([{}], 0)
    )
    cache = DemonstrationCache(
        tmp_path, None, algorithm="plant2", cfg=None, candidate_budget=8
    )
    with ThreadPoolExecutor(2) as executor:
        results = list(executor.map(lambda _: cache.get(instance), range(2)))
    assert len(calls) == 1
    assert sum(report["new_candidates"] for samples, report in results) == 8


def test_five_supervised_planners_share_the_physical_expert_cache(
    instance, tmp_path, monkeypatch
):
    from ads.common.demonstrations import DemonstrationCache
    from ads.common.expert import Candidate, TeacherExecution

    calls = []

    def execute(*args, **kwargs):
        calls.append(1)
        states = np.tile(instance.event.initial_states, (150, 1, 1))
        states[:, :, :2] += np.arange(150)[:, None, None] * 0.04 * states[:, :, 2:4]
        return TeacherExecution(
            Candidate(1, 20), True, 1, 120, states, np.zeros((149, 2))
        ), {
            "instance_hash": instance.instance_hash,
            "candidates": 8,
            "qualified_demonstration": True,
            "elapsed_s": 0.01,
        }

    monkeypatch.setattr("ads.common.expert.successful_expert", execute)
    monkeypatch.setattr(
        "ads.plant2.demonstrations.demonstration", lambda *args: ([{}], 0)
    )
    reports = []
    for algorithm in ("plant2", "diffusion_planner", "dtpp", "flow_planner", "str2"):
        cache = DemonstrationCache(
            tmp_path, None, algorithm=algorithm, candidate_budget=8
        )
        samples, report = cache.get(instance)
        assert samples
        reports.append(report)
    assert len(calls) == 1
    assert len({report["cache_path"] for report in reports}) == 1
    assert sum(report["new_candidates"] for report in reports) == 8
