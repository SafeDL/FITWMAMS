"""Subset-simulation and Monte-Carlo runners for TrafficBots + IDM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tools.evt import load_evt_model
from tools.idm_ego import load_idm_ego_config
from world_model.src.core.utils import (
    file_sha256,
    load_json,
    load_yaml,
    save_json,
    select_device,
)
from world_model.src.core.evaluation_scope import (
    evaluation_scope_contract,
    require_evaluation_scope,
    require_scoped_evt_model,
)
from world_model.trafficbots.config import load_config as load_trafficbots_config

from .trafficbots_evaluator import TrafficBotsIDMWorldEvaluator
from .trafficbots_randomness import TrafficBotsExogenousState
from .trafficbots_world import (
    TRAFFICBOTS_IDM_DYNAMICS_CONTRACT,
    TrafficBotsInitialSampler,
    build_trafficbots_idm_world,
)
from .world_evaluator import WorldEvaluation
from .world_subset_runner import (
    _failure_target,
    _level_rows,
    _monte_carlo_uncertainty,
    _resolve_path,
    _run_provenance,
    _save_final_population,
    _save_top_cases,
    _subset_uncertainty,
    _write_csv,
)
from .world_subset_simulation import run_world_subset_simulation


def _require_trafficbots_provenance(
    config: dict[str, Any], config_dir: Path
) -> dict[str, Any]:
    run = _run_provenance(config)
    if run["worktree_dirty"]:
        raise RuntimeError("formal TrafficBots IDM evaluation requires a clean worktree")
    checkpoint = _resolve_path(config["paths"]["trafficbots_checkpoint"], config_dir)
    acceptance = _resolve_path(config["paths"]["trafficbots_acceptance"], config_dir)
    audit = _resolve_path(config["paths"]["trafficbots_audit"], config_dir)
    if not checkpoint.is_file() or not acceptance.is_file() or not audit.is_file():
        raise FileNotFoundError("TrafficBots checkpoint, acceptance and audit are required")
    acceptance_values = load_json(acceptance)
    audit_values = load_json(audit)
    if not acceptance_values.get("all_passed"):
        raise RuntimeError("TrafficBots full acceptance did not pass")
    if not audit_values.get("method_identity_passed"):
        raise RuntimeError("TrafficBots method-identity audit did not pass")
    declared = audit_values.get("provenance", {}).get("checkpoint_sha256")
    actual = file_sha256(checkpoint)
    if declared != actual:
        raise RuntimeError("TrafficBots audit does not match the selected checkpoint")
    return {
        **run,
        "trafficbots_checkpoint_sha256": actual,
        "trafficbots_acceptance_sha256": file_sha256(acceptance),
        "trafficbots_audit_sha256": file_sha256(audit),
    }


def _build_evaluator(
    config: dict[str, Any], config_dir: Path
) -> tuple[TrafficBotsIDMWorldEvaluator, dict[str, Any]]:
    require_evaluation_scope(config)
    backend = str(config["simulation"].get("execution_backend", "local_highway_env"))
    if backend != "local_highway_env":
        raise ValueError("TrafficBots IDM evaluation requires local_highway_env")
    repo_root = Path(__file__).resolve().parents[2]
    device = select_device(str(config.get("runtime", {}).get("device", "auto")))
    common_path = _resolve_path(config["paths"]["common_world_config"], config_dir)
    common = load_yaml(common_path)
    traffic_config_path = _resolve_path(
        config["paths"]["trafficbots_config"], config_dir
    )
    traffic_config = load_trafficbots_config(traffic_config_path)
    checkpoint = _resolve_path(config["paths"]["trafficbots_checkpoint"], config_dir)
    idm_path = _resolve_path(config["paths"]["idm_ego_config"], config_dir)
    evt_path = _resolve_path(config["paths"]["evt_model"], config_dir)
    require_scoped_evt_model(evt_path)
    flow_checkpoint = _resolve_path(common["paths"]["flow_checkpoint"], repo_root)
    flow_output = _resolve_path(common["paths"]["flow_output_dir"], repo_root)
    initial_sampler = TrafficBotsInitialSampler(
        flow_checkpoint=flow_checkpoint,
        flow_output_dir=flow_output,
        repo_root=repo_root,
        device=device,
    )
    idm_config = load_idm_ego_config(idm_path)
    world = build_trafficbots_idm_world(
        config=traffic_config,
        checkpoint=checkpoint,
        idm_config=idm_config,
        device=device,
    )
    evaluator = TrafficBotsIDMWorldEvaluator(
        initial_sampler,
        world,
        load_evt_model(evt_path),
        steps=int(config["simulation"].get("steps", 149)),
        batch_size=int(config["runtime"].get("batch_size", 16)),
    )
    return evaluator, {
        "device": str(device),
        "execution_backend": backend,
        "background_dynamics_contract": TRAFFICBOTS_IDM_DYNAMICS_CONTRACT,
        "initial_scene_prior": "shared_external_Flow_p(M,C0); no K enters TrafficBots",
        "common_world_config": str(common_path),
        "common_world_config_sha256": file_sha256(common_path),
        "flow_checkpoint": str(flow_checkpoint),
        "flow_checkpoint_sha256": file_sha256(flow_checkpoint),
        "trafficbots_config": str(traffic_config_path),
        "trafficbots_config_sha256": file_sha256(traffic_config_path),
        "trafficbots_checkpoint": str(checkpoint),
        "trafficbots_checkpoint_sha256": file_sha256(checkpoint),
        "evt_model": str(evt_path),
        "evt_model_sha256": file_sha256(evt_path),
        "idm_ego_config": str(idm_path),
        "idm_ego_config_sha256": file_sha256(idm_path),
        "evaluation_scope": evaluation_scope_contract(),
        "test_space": config.get("test_space", {"kind": "full_prior"}),
    }


def _sample_worlds(count: int, seed: int) -> TrafficBotsExogenousState:
    return TrafficBotsExogenousState.sample(count, seed=seed, latent_dim=16)


def run_trafficbots_subset_from_config(
    config: dict[str, Any], config_dir: Path, *, formal: bool = True
) -> Path:
    formal_provenance = (
        _require_trafficbots_provenance(config, config_dir)
        if formal
        else _run_provenance(config)
    )
    output_dir = _resolve_path(config["subset_simulation"]["output_dir"], config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator, provenance = _build_evaluator(config, config_dir)
    target = _failure_target(evaluator.evt_model, config)
    settings = config["subset_simulation"]
    result = run_world_subset_simulation(
        evaluator.evaluate,
        num_samples=int(settings["num_samples"]),
        p0=float(settings["p0"]),
        max_levels=int(settings["max_levels"]),
        failure_threshold=float(target["evt_score_threshold"]),
        response_steps=int(config["simulation"].get("steps", 149)),
        scene_dim=0,
        agent_dim=16,
        mutation_blocks=tuple(settings["mutation_blocks"]),
        pcn_beta=float(settings["pcn_beta"]),
        mcmc_steps=int(settings["mcmc_steps"]),
        seed=int(settings["seed"]),
        sample_worlds=_sample_worlds,
    )
    _save_final_population(result, output_dir)
    top_cases = _save_top_cases(
        result.levels[-1].worlds,
        result.levels[-1].evaluation,
        evaluator,
        output_dir,
        prefix="trafficbots_subset_final",
        top_k=int(config["output"].get("top_cases", 10)),
    )
    rows = _level_rows(result)
    _write_csv(output_dir / "world_subset_level_stats.csv", rows)
    save_json(top_cases, output_dir / "world_subset_top_cases.json")
    summary = {
        "schema": "trafficbots_highway_env_idm_subset_simulation",
        "world_model_id": "trafficbots",
        "world_model": "TrafficBots V1.5-HighD",
        "estimator": "adaptive_multilevel_splitting_pcn_subset_simulation",
        "formal": bool(formal),
        "probability": result.probability,
        "final_failure_fraction": result.final_failure_fraction,
        "failure_event": target,
        "stop_reason": result.stop_reason,
        "num_levels": len(result.levels),
        "simulation_counts": {
            "world_evaluations": result.total_evaluations,
            "proposal_evaluations": result.proposal_evaluations,
            "stored_final_population": result.levels[-1].worlds.batch_size,
        },
        "mutation_kernel": {
            "type": "Gaussian pCN plus independent uniform refresh",
            "blocks": list(settings["mutation_blocks"]),
            "pcn_beta": float(settings["pcn_beta"]),
            "mcmc_steps": int(settings["mcmc_steps"]),
        },
        "world_prior": "p_ext(M,C0) * N(Z;0,I) * U(destination_base)",
        "evaluation_contract": {
            "population_scope": evaluation_scope_contract(),
            "steps": int(config["simulation"].get("steps", 149)),
            "dt_s": 0.04,
            "execution_backend": config["simulation"].get(
                "execution_backend", "local_highway_env"
            ),
            "ego_controller": "native HighwayEnv IDMVehicle",
            "background_world_model": "trafficbots",
            "metric_scope": "IDM ego trajectory risk under generated background response",
            "test_space": provenance["test_space"],
        },
        "level_statistics": rows,
        "uncertainty": _subset_uncertainty(result, float(settings["p0"])),
        "provenance": {**provenance, **formal_provenance},
    }
    path = output_dir / "world_subset_summary.json"
    save_json(summary, path)
    return path


def run_trafficbots_monte_carlo_from_config(
    config: dict[str, Any], config_dir: Path, *, formal: bool = True
) -> Path:
    formal_provenance = (
        _require_trafficbots_provenance(config, config_dir) if formal else {}
    )
    output_dir = _resolve_path(config["monte_carlo"]["output_dir"], config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator, provenance = _build_evaluator(config, config_dir)
    target = _failure_target(evaluator.evt_model, config)
    settings = config["monte_carlo"]
    total = int(settings["num_samples"])
    if total < 1:
        raise ValueError("monte_carlo.num_samples must be positive")
    batch_size = int(settings.get("sample_batch_size", evaluator.batch_size))
    rng = np.random.default_rng(int(settings["seed"]))
    scores: list[np.ndarray] = []
    risks: list[np.ndarray] = []
    collisions: list[np.ndarray] = []
    gaps: list[np.ndarray] = []
    numerical: list[np.ndarray] = []
    worlds: list[TrafficBotsExogenousState] = []
    for start in range(0, total, batch_size):
        count = min(batch_size, total - start)
        batch = _sample_worlds(
            count, int(rng.integers(0, np.iinfo(np.int64).max))
        )
        evaluation = evaluator.evaluate(batch)
        worlds.append(batch)
        scores.append(evaluation.evt_score)
        risks.append(evaluation.event_risk)
        collisions.append(evaluation.collision)
        gaps.append(evaluation.min_gap_m)
        numerical.append(evaluation.numerical_valid)
    population = TrafficBotsExogenousState.concatenate(worlds)
    evaluation = WorldEvaluation(
        evt_score=np.concatenate(scores),
        event_risk=np.concatenate(risks),
        collision=np.concatenate(collisions),
        min_gap_m=np.concatenate(gaps),
        numerical_valid=np.concatenate(numerical),
    )
    failure = evaluation.evt_score >= float(target["evt_score_threshold"])
    probability = float(failure.mean())
    top_cases = _save_top_cases(
        population,
        evaluation,
        evaluator,
        output_dir,
        prefix="trafficbots_monte_carlo",
        top_k=int(config["output"].get("top_cases", 10)),
    )
    _write_csv(
        output_dir / "world_monte_carlo_stats.csv",
        [{
            "num_samples": total,
            "failure_count": int(failure.sum()),
            "probability": probability,
            "score_mean": float(evaluation.evt_score.mean()),
            "score_max": float(evaluation.evt_score.max()),
            "risk_mean": float(evaluation.event_risk.mean()),
            "risk_max": float(evaluation.event_risk.max()),
            "collision_fraction": float(evaluation.collision.mean()),
            "numerical_valid_fraction": float(evaluation.numerical_valid.mean()),
        }],
    )
    save_json(top_cases, output_dir / "world_monte_carlo_top_cases.json")
    summary = {
        "schema": "trafficbots_highway_env_idm_monte_carlo",
        "world_model_id": "trafficbots",
        "world_model": "TrafficBots V1.5-HighD",
        "estimator": "independent_monte_carlo",
        "formal": bool(formal),
        "probability": probability,
        "failure_count": int(failure.sum()),
        "failure_event": target,
        "simulation_counts": {"world_evaluations": total},
        "world_prior": "p_ext(M,C0) * N(Z;0,I) * U(destination_base)",
        "evaluation_contract": {
            "population_scope": evaluation_scope_contract(),
            "steps": int(config["simulation"].get("steps", 149)),
            "dt_s": 0.04,
            "execution_backend": config["simulation"].get(
                "execution_backend", "local_highway_env"
            ),
            "ego_controller": "native HighwayEnv IDMVehicle",
            "background_world_model": "trafficbots",
            "metric_scope": "IDM ego trajectory risk under generated background response",
            "test_space": provenance["test_space"],
        },
        "numerical_valid_fraction": float(evaluation.numerical_valid.mean()),
        "collision_fraction": float(evaluation.collision.mean()),
        "evt_score_summary": {
            "mean": float(evaluation.evt_score.mean()),
            "p95": float(np.quantile(evaluation.evt_score, 0.95)),
            "max": float(evaluation.evt_score.max()),
        },
        "uncertainty": _monte_carlo_uncertainty(probability, total),
        "provenance": {**provenance, **formal_provenance},
    }
    path = output_dir / "world_monte_carlo_summary.json"
    save_json(summary, path)
    return path
