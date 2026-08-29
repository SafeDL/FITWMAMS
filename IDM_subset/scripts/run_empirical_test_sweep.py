#!/usr/bin/env python3
"""Run every held-out highD context once under fixed ``K_GT`` and IDM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from IDM_subset.src.world_subset_runner import (  # noqa: E402
    _build_evaluator,
    _failure_target,
    _resolve_path,
    _run_provenance,
)
from world_model.src.core.evaluation_scope import scoped_canonical_trajectory  # noqa: E402
from world_model.src.core.highd_metrics import semantic_cutin_agents  # noqa: E402
from world_model.src.core.sequential_dataset import load_sequential_dataset  # noqa: E402
from world_model.src.core.utils import load_yaml, save_json  # noqa: E402

CONFIG = ROOT / "IDM_subset/configs/world_subset_idm.yaml"


def _fixed_context_worlds(
    sampler,
    start: int,
    count: int,
    total: int,
    seed: int,
    response_steps: int,
):
    """Generate one reproducible response draw for each exact context index."""
    world = sampler.sample_world_exogenous(
        count, seed=int(seed) + int(start), response_steps=int(response_steps)
    )
    values = world.as_dict()
    values["test_row_uniform"] = (
        (np.arange(int(start), int(start) + int(count), dtype=np.float64) + 0.5)
        / float(total)
    )
    return world.replace_arrays(values)


def run(config: dict, config_path: Path) -> Path:
    if config.get("test_space", {}).get("kind") != "empirical_test_fixed_k_gt":
        raise ValueError("the exhaustive sweep requires test_space=empirical_test_fixed_k_gt")
    settings = config.get("test_sweep", {})
    if int(settings.get("response_draws_per_context", 1)) != 1:
        raise ValueError("only one fixed CRN response draw per test context is supported")
    evaluator, provenance = _build_evaluator(config, config_path.parent)
    sampler = evaluator.sampler
    available_contexts = len(sampler.rows)
    max_contexts = int(settings.get("max_contexts", 0))
    total = (
        available_contexts
        if max_contexts <= 0
        else min(max_contexts, available_contexts)
    )
    output = _resolve_path(settings["output_dir"], config_path.parent)
    output.mkdir(parents=True, exist_ok=True)
    failure_dir = output / "failure_cases"
    failure_dir.mkdir(parents=True, exist_ok=True)
    target = _failure_target(evaluator.evt_model, config)
    arrays, _ = load_sequential_dataset("results/highd_shared_training_data/highd_sequence_cache")
    context_rows = np.asarray(sampler.rows[:total], dtype=np.int64)
    raw_states = np.asarray(arrays["agent_states"])[context_rows, 24:174]
    raw_valid = np.asarray(arrays["agent_valid"])[context_rows, 24:174]
    raw_states, raw_valid = scoped_canonical_trajectory(raw_states, raw_valid)
    cutin = semantic_cutin_agents(raw_states, raw_valid).any(axis=1)
    batch_size = int(config.get("runtime", {}).get("batch_size", evaluator.batch_size))
    score = np.empty(total, np.float64)
    risk = np.empty(total, np.float32)
    collision = np.empty(total, bool)
    gap = np.empty(total, np.float32)
    valid = np.empty(total, bool)
    failures = []
    seed = int(settings["seed"])
    for start in range(0, total, batch_size):
        count = min(batch_size, total - start)
        world = _fixed_context_worlds(
            sampler,
            start,
            count,
            total,
            seed,
            response_steps=evaluator.steps,
        )
        result = evaluator.evaluate(world)
        stop = start + count
        score[start:stop] = result.evt_score
        risk[start:stop] = result.event_risk
        collision[start:stop] = result.collision
        gap[start:stop] = result.min_gap_m
        valid[start:stop] = result.numerical_valid
        for local in np.flatnonzero(result.evt_score >= target["evt_score_threshold"]):
            index = start + int(local)
            case_id = f"test_sweep_failure_{len(failures) + 1:04d}"
            path = failure_dir / f"{case_id}.npz"
            world.select(slice(int(local), int(local) + 1)).save(path)
            failures.append(
                {
                    "case_id": case_id,
                    "world_exogenous_state": str(path),
                    "test_context_index": int(index),
                    "cache_row": int(sampler.rows[index]),
                    "sequence_id": str(arrays["sequence_id"][sampler.rows[index]]),
                    "strict_cutin_context": bool(cutin[index]),
                    "evt_score": float(result.evt_score[local]),
                    "event_risk": float(result.event_risk[local]),
                    "collision": bool(result.collision[local]),
                    "min_gap_m": float(result.min_gap_m[local]),
                }
            )
    failure = score >= target["evt_score_threshold"]
    np.savez_compressed(
        output / "test_context_sweep.npz",
        test_context_index=np.arange(total, dtype=np.int64),
        cache_row=context_rows,
        strict_cutin_context=cutin,
        evt_score=score,
        event_risk=risk,
        collision=collision,
        min_gap_m=gap,
        numerical_valid=valid,
    )
    save_json(failures, output / "test_sweep_failure_cases.json")
    summary = {
        "schema": "hierarchical_empirical_fixed_k_gt_test_sweep_v1",
        "formal": False,
        "test_space": sampler.context_contract,
        "response_draws_per_context": 1,
        "response_randomness": "one fixed common-random-number draw per context",
        "failure_event": target,
        "test_contexts": int(total),
        "available_test_contexts": int(available_contexts),
        "strict_cutin_contexts": int(cutin.sum()),
        "failure_count": int(failure.sum()),
        "failure_fraction_one_draw": float(failure.mean()),
        "cutin_failure_count": int((failure & cutin).sum()),
        "collision_count": int(collision.sum()),
        "numerical_valid_fraction": float(valid.mean()),
        "failure_cases": str(output / "test_sweep_failure_cases.json"),
        "provenance": {**provenance, **_run_provenance(config)},
        "interpretation": (
            "exhaustive empirical test-context sweep; not a Monte-Carlo or AMS "
            "probability estimator because each context has one fixed response draw"
        ),
    }
    path = output / "test_context_sweep_summary.json"
    save_json(summary, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    path = args.config.resolve()
    print(run(load_yaml(path), path))


if __name__ == "__main__":
    main()
