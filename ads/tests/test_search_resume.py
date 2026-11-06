"""Search slots count separately; shared validation reuses completed executions."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from ads.common.evaluation import EpisodeResult
from ads.common.instances import file_hash
from ads.diffusion_planner.search_first import ReplayableEvaluator
from ads.tests.test_search import evaluate


def test_repeated_input_still_costs_two_slots_and_resume_does_not_repeat(
    instance, tmp_path
):
    calls = []

    def run(value):
        calls.append(value.instance_hash)
        result = evaluate(value)
        result.record["instance_hash"] = value.instance_hash
        return result

    current = ReplayableEvaluator(run, tmp_path, "initial")
    current(instance)
    current(instance)
    assert len(calls) == 2
    resumed = ReplayableEvaluator(run, tmp_path, "initial")
    first, second = resumed(instance), resumed(instance)
    assert first.record == second.record
    assert len(calls) == 2


def test_reused_prior_refreshes_metrics_without_changing_cached_execution(
    source, instance, evt, tmp_path, monkeypatch
):
    from ads.diffusion_planner import search_first

    historical = tmp_path / "historical/search"
    historical.mkdir(parents=True)
    directory = historical.parent / "common_prior"
    executions = directory / "executions"
    executions.mkdir(parents=True)
    instance.save(directory / "search_instances" / (instance.instance_hash + ".npz"))
    (historical / "protocol.json").write_text(json.dumps({"assets": instance.assets}))
    initial = tmp_path / "initial.pth"
    initial.write_bytes(b"official initial policy")
    (tmp_path / "evt.json").write_text("{}")
    states = np.stack([instance.event.initial_states] * 2)
    states[1, :, 0] += states[0, :, 2] * 0.04
    record = {
        "instance_hash": instance.instance_hash,
        "event_risk": 999,
        "critical_frame": 999,
        "collision": False,
        "offroad": False,
        "stalled": False,
        "task_reward": 0.5,
        "elapsed_s": 0.2,
        "policy_stream_id": "fixed-policy-stream",
        "policy_random_mode": "search",
    }
    replay = ReplayableEvaluator(
        lambda value: EpisodeResult(value, record, states, np.zeros((1, 2)), []),
        executions,
        file_hash(initial),
    )
    replay(instance)
    metadata = executions / "00001.json"
    original_metadata = metadata.read_bytes()
    original_trace = metadata.with_suffix(".npz").read_bytes()
    controller = SimpleNamespace(
        driver_correction=None,
        mobil_onset_calibration=None,
        mobil_parameter_variation=False,
        lane_style=None,
    )
    monkeypatch.setattr(search_first, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(search_first, "official_path", lambda name: initial)
    monkeypatch.setattr(search_first.GPDTailModel, "from_dict", lambda value: evt)
    monkeypatch.setattr(
        search_first,
        "FrozenWorldFactory",
        lambda path: lambda event: SimpleNamespace(controller=controller),
    )
    results = search_first.reused_prior(
        {
            "historical_run": "historical",
            "master_seed": 123,
            "search": {"initial_samples": 1},
            "world_config": "unused.yaml",
            "evt_model": "evt.json",
        },
        source,
        instance.assets,
    )
    result = results[0]
    assert 0 <= result.score < 10
    assert result.record["critical_frame"] < len(states)
    assert not result.failure
    assert result.record["policy_stream_id"] == record["policy_stream_id"]
    assert result.record["policy_random_mode"] == record["policy_random_mode"]
    assert result.record["cached_prior_source"] == {
        "metadata_path": str(metadata.resolve()),
        "metadata_sha256": file_hash(metadata),
        "trace_sha256": file_hash(metadata.with_suffix(".npz")),
    }
    np.testing.assert_array_equal(result.states, states)
    assert source.index == 1
    assert metadata.read_bytes() == original_metadata
    assert metadata.with_suffix(".npz").read_bytes() == original_trace


def initialize_synthetic_replay(directory):
    from ads.diffusion_planner import search_first

    def evaluator(value):
        result = evaluate(value)
        result.record["instance_hash"] = value.instance_hash
        return result

    search_first._search_replay = ReplayableEvaluator(evaluator, directory, "initial")


def test_parallel_slots_survive_worker_reordering_and_resume(instance, tmp_path):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(
        max_workers=2,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_synthetic_replay,
        initargs=(tmp_path,),
    ) as executor:
        current = ReplayableEvaluator(None, tmp_path, "initial", executor=executor)
        results = current.many([instance] * 4)
        assert len(results) == current.calls == current.new_calls == 4
        assert current.cache_hits == 0
        assert len(list(tmp_path.glob("*.json"))) == 4
        resumed = ReplayableEvaluator(None, tmp_path, "initial", executor=executor)
        assert [r.record for r in resumed.many([instance] * 4)] == [
            r.record for r in results
        ]
        assert resumed.cache_hits == 4 and resumed.new_calls == 0


def test_concurrent_validation_calls_share_one_execution(
    instance, tmp_path, monkeypatch
):
    from ads.common import batch_evaluation

    source = tmp_path / "instance.npz"
    instance.save(source)
    policy = tmp_path / "policy.pth"
    policy.write_bytes(b"test policy")
    directory = tmp_path / "development"
    calls = []
    barrier = threading.Barrier(2)

    def evaluate(value):
        calls.append(value.instance_hash)
        time.sleep(0.1)
        return SimpleNamespace(
            record={
                "instance_hash": value.instance_hash,
                "collision": False,
                "offroad": False,
                "stalled": False,
                "elapsed_s": 0.1,
                "progress_m": 100.0,
            }
        )

    def initialize(*args):
        batch_evaluation._policy_hash = file_hash(policy)
        batch_evaluation._evaluator = evaluate

    def pool(**kwargs):
        kwargs.pop("mp_context")
        return ThreadPoolExecutor(**kwargs)

    def request(_):
        barrier.wait(timeout=5)
        return batch_evaluation.evaluate_paths(
            [source], policy, directory, {}, workers=1
        )

    monkeypatch.setattr(batch_evaluation, "initialize", initialize)
    monkeypatch.setattr(batch_evaluation, "ProcessPoolExecutor", pool)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(request, range(2)))
    assert calls == [instance.instance_hash]
    assert sorted(result[0]["cache_hit"] for result in results) == [False, True]
    assert results[0][0]["record"] == results[1][0]["record"]
    assert json.loads((directory / "summary.json").read_text())["instances"] == 1


@pytest.fixture
def search_launcher(tmp_path, monkeypatch):
    from ads.diffusion_planner import search_first

    monkeypatch.setattr(search_first, "REPO_ROOT", tmp_path)
    design = tmp_path / "doc/search_budget_feedback.md"
    design.parent.mkdir()
    design.write_text("Current search design\n")
    world = tmp_path / "hierarchical_world_model/src/driver.py"
    world.parent.mkdir(parents=True)
    world.write_text("FROZEN_DRIVER = True\n")
    output = tmp_path / "output"
    output.mkdir()
    commands = []

    def spawn(command, **kwargs):
        commands.append(command)
        assert kwargs["cwd"] == tmp_path
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(search_first.subprocess, "Popen", spawn)
    monkeypatch.setattr(
        search_first.psutil,
        "Process",
        lambda pid: SimpleNamespace(create_time=lambda: 1.0),
    )
    monkeypatch.setattr(search_first.psutil, "pid_exists", lambda pid: False)
    settings = {
        "reuse_search": "separate_search",
        "methods": ["guided"],
        "master_seed": 20260926,
        "search": {"budget": 5000, "initial_samples": 1000},
    }
    return search_first, settings, output, commands


def test_search_worker_uses_parent_settings_and_current_design(
    search_launcher, monkeypatch
):
    module, settings, output, commands = search_launcher
    module.launch_search(settings, output)
    assert len(commands) == 1
    configuration = output.parent / settings["reuse_search"] / "worker_settings.json"
    assert json.loads(configuration.read_text()) == settings
    assert commands[0][4:] == [
        "--config",
        str(configuration.resolve()),
        "--method",
        "guided",
    ]
    snapshot = configuration.parent / "source_snapshot"
    assert (snapshot / configuration.name).read_bytes() == configuration.read_bytes()
    assert (
        snapshot / "doc/search_budget_feedback.md"
    ).read_text() == "Current search design\n"
    received = []
    monkeypatch.setattr(
        module, "run", lambda current, method: received.append((current, method))
    )
    module.main(commands[0][4:])
    assert received == [(settings, "guided")]


def test_search_resume_rejects_changed_worker_settings(search_launcher):
    module, settings, output, commands = search_launcher
    module.launch_search(settings, output)
    configuration = output.parent / settings["reuse_search"] / "worker_settings.json"
    frozen = configuration.read_bytes()
    changed = {**settings, "search": {**settings["search"], "budget": 4000}}
    with pytest.raises(ValueError, match="Frozen search worker settings changed"):
        module.launch_search(changed, output)
    assert len(commands) == 1
    assert configuration.read_bytes() == frozen


def test_search_entry_requires_explicit_configuration():
    from ads.diffusion_planner import search_first

    with pytest.raises(SystemExit) as error:
        search_first.main(["--method", "guided"])
    assert error.value.code == 2
