"""One bounded correction pass at states actually reached by the repaired planner."""

import json
import time
from dataclasses import asdict, replace

import numpy as np
import torch

from ads.common.environment import PolicyEnvironment
from ads.common.evaluation import summarise_trace
from ads.common.expert import (
    CandidateController,
    TeacherExecution,
    coarse_candidates,
    expert_rank,
    refine_candidates,
)
from ads.common.instances import file_hash, json_hash
from ads.common.randomness import PolicyRandomContext
from ads.common.reporting import write_json
from ads.diffusion_planner.label_tracking import check as check_labels
from ads.nuplan.teacher import demonstration


def select_query_records(records, original):
    failures = {
        row["instance_hash"]: row
        for row in records
        if row["safety_failure"] and row["critical_frame"] >= 30
    }
    return sorted(
        failures.values(),
        key=lambda row: (
            original[row["instance_hash"]]["safety_failure"],
            row["root_sample_id"],
            -row["event_risk"],
            row["instance_hash"],
        ),
    )


class SuffixEnvironment(PolicyEnvironment):
    def _observation(self):
        return np.zeros(1, np.float32)


def execute_suffix(instance, factory, snapshot, candidate):
    env = SuffixEnvironment(
        "diffusion_planner", [instance], factory, random_events=False
    )
    env.reset()
    env.restore(snapshot)
    start = int(env.world.reference_index)
    current = replace(instance.event, initial_states=env._states().copy())
    env.controller = CandidateController(current, candidate)
    terminated, truncated, reward_sum = False, False, 0.0
    info = {"stalled": False}
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = env.step(0)
        reward_sum += reward
    states = np.asarray(env.physics_states)
    return TeacherExecution(
        candidate,
        not terminated and not info["stalled"],
        reward_sum,
        float(states[-1, 0, 0] - states[start, 0, 0]),
        states,
        np.asarray(env.physics_controls).reshape(-1, 2),
    )


def suffix_expert(instance, factory, snapshot, *, budget=128):
    if not 1 <= budget <= 128:
        raise ValueError("Suffix expert budget must be between 1 and 128")
    current = replace(
        instance.event,
        initial_states=np.asarray(snapshot["runtime"]["physics_states"][-1]),
    )
    start = int(snapshot["world"].reference_index)
    executions, tried = [], set()
    for candidate in coarse_candidates(current)[: min(64, budget)]:
        executions.append(execute_suffix(instance, factory, snapshot, candidate))
        tried.add(candidate)
    rank = lambda value: expert_rank(value, current)
    leaders = sorted(executions, key=rank, reverse=True)[:8]
    coarse_count = len(executions)
    for candidate in refine_candidates([e.candidate for e in leaders]):
        if len(executions) >= budget:
            break
        if candidate not in tried:
            executions.append(execute_suffix(instance, factory, snapshot, candidate))
            tried.add(candidate)
    successful = [e for e in executions if e.success]
    return max(successful, key=rank) if successful else None, {
        "candidates": len(executions),
        "coarse_candidates": coarse_count,
        "successful_candidates": len(successful),
        "qualified_demonstration": bool(successful),
        "query_frame": start,
        "suffix_physics_steps": sum(len(e.states) - 1 - start for e in executions),
    }


def collect_query(instance, policy, factory, evt, ledger, *, seed=20260926):
    ledger.register(instance, "repair")
    context = PolicyRandomContext(
        "diffusion_planner", seed, "search", instance.instance_hash
    )
    policy.reset(context)
    env = PolicyEnvironment(
        "diffusion_planner",
        [instance],
        factory,
        purpose="repair",
        ledger=ledger,
        random_events=False,
    )
    observation, info = env.reset()
    terminated, truncated = env.initial_failure, False
    snapshots, reward_sum = {}, 0.0
    started = time.perf_counter()
    while not (terminated or truncated):
        snapshots[int(env.world.reference_index)] = env.snapshot(policy)
        with context.step(policy.decision, policy.device):
            action, _ = policy.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        reward_sum += reward
    result = summarise_trace(
        instance,
        env.physics_states,
        env.physics_controls,
        env.physics_diagnostics,
        info,
        evt_model=evt,
        reward=reward_sum,
        elapsed_s=time.perf_counter() - started,
    )
    if not result.failure:
        return None, {
            "collection": result.record,
            "skipped": "current_policy_already_safe",
        }
    target = int(result.record["critical_frame"]) - 25
    eligible = [
        frame
        for frame in snapshots
        if 5 <= frame <= target and frame < len(env.physics_states) - 25
    ]
    if not eligible:
        return None, {
            "collection": result.record,
            "skipped": "no_safe_offset_query_with_observed_future",
        }
    frame = max(eligible)
    snapshot = snapshots[frame]
    expected = np.asarray(env.physics_states)[frame + 1 : frame + 6].copy()
    observation = env.restore(snapshot, policy)
    with context.step(policy.decision, policy.device):
        action, _ = policy.predict(observation)
    env.step(action)
    actual = np.asarray(env.physics_states)[frame + 1 :]
    if not np.array_equal(actual, expected):
        raise AssertionError(
            "Restored query changed its fixed-policy prefix or NPC random time index"
        )
    return snapshot, {
        "collection": result.record,
        "query_frame": frame,
        "snapshot_replay": {"success": True, "physics_steps": len(actual)},
    }


def correct_query(
    instance, snapshot, factory, directory, *, policy_hash, ledger, budget=128
):
    ledger.register(instance, "repair")
    directory.mkdir(parents=True, exist_ok=True)
    start = int(snapshot["world"].reference_index)
    key = json_hash(
        {
            "instance": instance.instance_hash,
            "policy": policy_hash,
            "frame": start,
            "assets": instance.assets,
            "budget": budget,
            "rule": "native_suffix_intent_v1",
        }
    )
    report_path, trace_path = directory / (key + ".json"), directory / (key + ".npz")
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if file_hash(trace_path) != report["trajectory_sha256"]:
            raise ValueError("Expert suffix trajectory changed")
        with np.load(trace_path, allow_pickle=False) as data:
            states = data["states"]
    else:
        started = time.perf_counter()
        execution, report = suffix_expert(instance, factory, snapshot, budget=budget)
        states = execution.states if execution else np.empty((0, 7, 6), np.float32)
        expert_success = report["qualified_demonstration"]
        tracking = (
            check_labels(instance, states, factory, ledger, snapshot=snapshot)
            if expert_success
            else None
        )
        report.update(
            expert_success=expert_success,
            label_tracking=tracking,
            qualified_demonstration=bool(tracking and tracking["success"]),
        )
        np.savez_compressed(
            trace_path,
            states=states,
            controls=execution.controls if execution else np.empty((0, 2), np.float32),
        )
        snapshot_path = directory / (key + ".snapshot.pt")
        torch.save(snapshot, snapshot_path)
        report.update(
            instance_hash=instance.instance_hash,
            root_sample_id=instance.root_sample_id,
            policy_sha256=policy_hash,
            assets_hash=json_hash(instance.assets),
            snapshot_path=str(snapshot_path),
            snapshot_sha256=file_hash(snapshot_path),
            trajectory_path=str(trace_path),
            trajectory_sha256=file_hash(trace_path),
            candidate=asdict(execution.candidate) if execution else None,
            elapsed_s=time.perf_counter() - started,
        )
        write_json(report_path, report)
    samples = (
        demonstration(
            instance,
            states,
            "diffusion_planner",
            frames=range(start, len(states) - 3, 5),
        )
        if report["qualified_demonstration"]
        else []
    )
    for sample in samples:
        sample.update(
            scene_id=instance.root_sample_id,
            instance_id=instance.instance_hash + "@query=" + str(start),
            query_id=key,
            domain="correction",
            label_source="expert",
            purpose="repair",
            critical_window=sample["frame_id"] <= start + 25,
        )
    return samples, report
