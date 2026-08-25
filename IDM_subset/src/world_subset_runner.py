"""Current-world subset simulation and independent Monte Carlo runners."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from hierarchical_traffic_world_model.src.composition import HierarchicalWorldSampler
from hierarchical_traffic_world_model.src.highway_env_execution import (
    HIGHWAY_ENV_HIQR_DYNAMICS_CONTRACT,
)
from hierarchical_traffic_world_model.src.world_randomness import WorldExogenousState
from tools.evt import GPDTailModel, load_evt_model
from tools.idm_ego import load_idm_ego_config
from world_model.src.core.utils import file_sha256, load_yaml, save_json, select_device

from .idm_policy import HighwayEnvIDMPolicy
from .world_evaluator import CurrentWorldEvaluator, WorldEvaluation
from .world_subset_simulation import WorldSubsetResult, run_world_subset_simulation


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_provenance(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            text=True,
        ).strip()
    )
    encoded = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "repository_commit": commit,
        "worktree_dirty": dirty,
        "run_config_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _world_config(config: dict[str, Any], config_dir: Path) -> tuple[dict[str, Any], Path]:
    path = _resolve_path(config["paths"]["world_model_config"], config_dir)
    return load_yaml(path), path


def _build_evaluator(
    config: dict[str, Any],
    config_dir: Path,
) -> tuple[CurrentWorldEvaluator, dict[str, Any]]:
    backend = str(config["simulation"].get("execution_backend", "local_highway_env"))
    if backend != "local_highway_env":
        raise ValueError("formal IDM evaluation requires simulation.execution_backend=local_highway_env")
    world_config, world_config_path = _world_config(config, config_dir)
    world_paths = world_config["paths"]
    device = select_device(str(config.get("runtime", {}).get("device", "auto")))
    sampler = HierarchicalWorldSampler(
        flow_checkpoint=world_paths["flow_checkpoint"],
        flow_output_dir=world_paths["flow_output_dir"],
        diffusion_checkpoint=world_paths["diffusion_checkpoint"],
        diffusion_contract=world_paths["diffusion_contract"],
        response_checkpoint=world_paths["evaluation_checkpoint"],
        # ``config_dir`` is now ``IDM_subset/configs``; derive the repository
        # root from this maintained module instead of depending on config depth.
        repo_root=Path(__file__).resolve().parents[2],
        device=device,
        ddim_steps=int(config["simulation"].get("ddim_steps", 20)),
    )
    idm_path = _resolve_path(config["paths"]["idm_ego_config"], config_dir)
    policy = HighwayEnvIDMPolicy.from_dict(load_idm_ego_config(idm_path))
    evt_path = _resolve_path(config["paths"]["evt_model"], config_dir)
    evaluator = CurrentWorldEvaluator(
        sampler,
        policy,
        load_evt_model(evt_path),
        steps=int(config["simulation"].get("steps", 149)),
        batch_size=int(config["runtime"].get("batch_size", 32)),
    )
    manifest_path = Path(world_paths["evaluation_checkpoint"]).with_name(
        "final_model_manifest.json"
    )
    return evaluator, {
        "device": str(device),
        "execution_backend": backend,
        "hiqr_vehicle_dynamics_contract": HIGHWAY_ENV_HIQR_DYNAMICS_CONTRACT,
        "world_model_config": str(world_config_path),
        "world_model_config_sha256": file_sha256(world_config_path),
        "flow_checkpoint": str(world_paths["flow_checkpoint"]),
        "flow_checkpoint_sha256": file_sha256(world_paths["flow_checkpoint"]),
        "diffusion_checkpoint": str(world_paths["diffusion_checkpoint"]),
        "diffusion_checkpoint_sha256": file_sha256(world_paths["diffusion_checkpoint"]),
        "response_checkpoint": str(world_paths["evaluation_checkpoint"]),
        "response_checkpoint_sha256": file_sha256(world_paths["evaluation_checkpoint"]),
        "final_model_manifest": str(manifest_path),
        "final_model_manifest_sha256": file_sha256(manifest_path),
        "evt_model": str(evt_path),
        "evt_model_sha256": file_sha256(evt_path),
        "idm_ego_config": str(idm_path),
        "idm_ego_config_sha256": file_sha256(idm_path),
    }


def _failure_target(evt: GPDTailModel, config: dict[str, Any]) -> dict[str, float | int]:
    settings = config["failure_event"]
    return_period = int(settings.get("return_period", 100))
    if return_period <= 1:
        raise ValueError("failure_event.return_period must exceed one")
    risk_threshold = float(evt.return_level(return_period))
    return {
        "return_period": return_period,
        "event_risk_threshold": risk_threshold,
        "evt_score_threshold": float(evt.score(risk_threshold)),
    }


def _world_dimensions(evaluator: CurrentWorldEvaluator) -> dict[str, int]:
    cfg = evaluator.sampler.response.cfg
    return {
        "response_steps": evaluator.steps,
        "scene_dim": int(cfg.scene_latent_dim),
        "agent_dim": int(cfg.agent_latent_dim),
    }


def _level_rows(result: WorldSubsetResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level in result.levels:
        evaluation = level.evaluation
        rows.append(
            {
                "level": level.level,
                "num_samples": int(evaluation.evt_score.size),
                "subset_threshold": level.threshold,
                "score_mean": float(evaluation.evt_score.mean()),
                "score_p90": float(np.quantile(evaluation.evt_score, 0.90)),
                "score_p95": float(np.quantile(evaluation.evt_score, 0.95)),
                "score_max": float(evaluation.evt_score.max()),
                "risk_mean": float(evaluation.event_risk.mean()),
                "risk_max": float(evaluation.event_risk.max()),
                "failure_fraction": float(
                    (evaluation.evt_score >= result.failure_threshold).mean()
                ),
                "collision_fraction": float(evaluation.collision.mean()),
                "numerical_valid_fraction": float(evaluation.numerical_valid.mean()),
                "proposal_acceptance_rate": (
                    float(level.proposal_acceptance_rate)
                    if np.isfinite(level.proposal_acceptance_rate)
                    else None
                ),
                "chain_moved_fraction": (
                    float(level.accepted.mean())
                    if np.isfinite(level.proposal_acceptance_rate)
                    else None
                ),
            }
        )
    return rows


def _save_case(
    world: WorldExogenousState,
    evaluation: WorldEvaluation,
    case_index: int,
    output_dir: Path,
    *,
    prefix: str,
) -> dict[str, Any]:
    if world.batch_size != 1 or evaluation.evt_score.shape != (1,):
        raise ValueError("saved cases must contain exactly one replayed world")
    case_id = f"{prefix}_{case_index:04d}"
    path = output_dir / "failure_cases" / f"{case_id}.npz"
    world.save(path)
    return {
        "case_id": case_id,
        "world_exogenous_state": str(path),
        "evt_score": float(evaluation.evt_score[0]),
        "event_risk": float(evaluation.event_risk[0]),
        "collision": bool(evaluation.collision[0]),
        "min_gap_m": float(evaluation.min_gap_m[0]),
        "numerical_valid": bool(evaluation.numerical_valid[0]),
    }


def _save_top_cases(
    worlds: WorldExogenousState,
    evaluation: WorldEvaluation,
    evaluator: CurrentWorldEvaluator,
    output_dir: Path,
    *,
    prefix: str,
    top_k: int,
) -> list[dict[str, Any]]:
    order = np.argsort(evaluation.evt_score)[::-1][: int(top_k)]
    cases: list[dict[str, Any]] = []
    for rank, index in enumerate(order, start=1):
        # CUDA kernels may differ by a few ULPs between a population batch and
        # a single replay.  Persist the single-world score with the case so
        # the stored state has one exact, self-contained replay contract.
        world = worlds.select(slice(int(index), int(index) + 1))
        replay = evaluator.evaluate(world)
        cases.append(_save_case(world, replay, rank, output_dir, prefix=prefix))
    return cases


def _save_final_population(result: WorldSubsetResult, output_dir: Path) -> None:
    final = result.levels[-1]
    np.savez_compressed(
        output_dir / "world_subset_final_population.npz",
        **final.worlds.as_dict(),
        evt_score=final.evaluation.evt_score,
        event_risk=final.evaluation.event_risk,
        collision=final.evaluation.collision,
        min_gap_m=final.evaluation.min_gap_m,
        numerical_valid=final.evaluation.numerical_valid,
    )


def _subset_uncertainty(result: WorldSubsetResult, p0: float) -> dict[str, float | str]:
    fraction = result.final_failure_fraction
    n = result.levels[-1].evaluation.evt_score.size
    scale = float(p0) ** max(len(result.levels) - 1, 0)
    standard_error = scale * np.sqrt(fraction * (1.0 - fraction) / max(n, 1))
    return {
        "probability_standard_error": float(standard_error),
        "probability_ci95_lower": float(
            max(0.0, result.probability - 1.96 * standard_error)
        ),
        "probability_ci95_upper": float(
            min(1.0, result.probability + 1.96 * standard_error)
        ),
        "uncertainty_method": (
            "final-level binomial approximation; does not account for MCMC "
            "correlation"
        ),
    }


def run_subset_from_config(config: dict[str, Any], config_dir: Path) -> Path:
    """Run pCN subset simulation under the complete current world prior."""
    output_dir = _resolve_path(config["subset_simulation"]["output_dir"], config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator, provenance = _build_evaluator(config, config_dir)
    target = _failure_target(evaluator.evt_model, config)
    dimensions = _world_dimensions(evaluator)
    settings = config["subset_simulation"]
    result = run_world_subset_simulation(
        evaluator.evaluate,
        num_samples=int(settings["num_samples"]),
        p0=float(settings["p0"]),
        max_levels=int(settings["max_levels"]),
        failure_threshold=float(target["evt_score_threshold"]),
        response_steps=dimensions["response_steps"],
        scene_dim=dimensions["scene_dim"],
        agent_dim=dimensions["agent_dim"],
        mutation_blocks=tuple(settings["mutation_blocks"]),
        pcn_beta=float(settings["pcn_beta"]),
        mcmc_steps=int(settings["mcmc_steps"]),
        seed=int(settings["seed"]),
    )
    _save_final_population(result, output_dir)
    top_cases = _save_top_cases(
        result.levels[-1].worlds,
        result.levels[-1].evaluation,
        evaluator,
        output_dir,
        prefix="subset_final",
        top_k=int(config["output"].get("top_cases", 10)),
    )
    level_rows = _level_rows(result)
    _write_csv(output_dir / "world_subset_level_stats.csv", level_rows)
    save_json(top_cases, output_dir / "world_subset_top_cases.json")
    summary = {
        "schema": "highway_env_idm_subset_simulation_v3",
        "estimator": "adaptive_multilevel_splitting_pcn_subset_simulation",
        "probability": result.probability,
        "final_failure_fraction": result.final_failure_fraction,
        "failure_event": target,
        "stop_reason": result.stop_reason,
        "num_levels": len(result.levels),
        "simulation_counts": {
            "world_evaluations": result.total_evaluations,
            "proposal_evaluations": result.proposal_evaluations,
            "stored_final_population": int(result.levels[-1].worlds.batch_size),
        },
        "mutation_kernel": {
            "type": "prior-reversible joint block pCN sweep plus categorical refresh",
            "blocks": list(settings["mutation_blocks"]),
            "pcn_beta": float(settings["pcn_beta"]),
            "mcmc_steps": int(settings["mcmc_steps"]),
        },
        "world_prior": "p(M)p(C0|M)p(K|C0,M) * p(z_diff) * p(z_response)",
        "dimensions": dimensions,
        "level_statistics": level_rows,
        "uncertainty": _subset_uncertainty(result, float(settings["p0"])),
        "provenance": {**provenance, **_run_provenance(config)},
    }
    path = output_dir / "world_subset_summary.json"
    save_json(summary, path)
    return path


def _monte_carlo_uncertainty(probability: float, count: int) -> dict[str, float]:
    standard_error = np.sqrt(probability * (1.0 - probability) / max(count, 1))
    return {
        "probability_standard_error": float(standard_error),
        "probability_ci95_lower": float(max(0.0, probability - 1.96 * standard_error)),
        "probability_ci95_upper": float(min(1.0, probability + 1.96 * standard_error)),
    }


def run_monte_carlo_from_config(config: dict[str, Any], config_dir: Path) -> Path:
    """Run independent prior samples for the current-world probability baseline."""
    output_dir = _resolve_path(config["monte_carlo"]["output_dir"], config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator, provenance = _build_evaluator(config, config_dir)
    target = _failure_target(evaluator.evt_model, config)
    dimensions = _world_dimensions(evaluator)
    settings = config["monte_carlo"]
    total = int(settings["num_samples"])
    if total < 1:
        raise ValueError("monte_carlo.num_samples must be positive")
    rng = np.random.default_rng(int(settings["seed"]))
    batch_size = int(settings.get("sample_batch_size", evaluator.batch_size))
    scores: list[np.ndarray] = []
    risks: list[np.ndarray] = []
    collisions: list[np.ndarray] = []
    gaps: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    worlds: list[WorldExogenousState] = []
    for start in range(0, total, batch_size):
        count = min(batch_size, total - start)
        batch = WorldExogenousState.sample(
            count,
            seed=int(rng.integers(0, np.iinfo(np.int64).max)),
            response_steps=dimensions["response_steps"],
            scene_dim=dimensions["scene_dim"],
            agent_dim=dimensions["agent_dim"],
        )
        evaluation = evaluator.evaluate(batch)
        scores.append(evaluation.evt_score)
        risks.append(evaluation.event_risk)
        collisions.append(evaluation.collision)
        gaps.append(evaluation.min_gap_m)
        valid.append(evaluation.numerical_valid)
        worlds.append(batch)
    population = WorldExogenousState.concatenate(worlds)
    evaluation = WorldEvaluation(
        evt_score=np.concatenate(scores),
        event_risk=np.concatenate(risks),
        collision=np.concatenate(collisions),
        min_gap_m=np.concatenate(gaps),
        numerical_valid=np.concatenate(valid),
    )
    failure = evaluation.evt_score >= float(target["evt_score_threshold"])
    probability = float(failure.mean())
    top_cases = _save_top_cases(
        population,
        evaluation,
        evaluator,
        output_dir,
        prefix="monte_carlo",
        top_k=int(config["output"].get("top_cases", 10)),
    )
    _write_csv(
        output_dir / "world_monte_carlo_stats.csv",
        [
            {
                "num_samples": total,
                "failure_count": int(failure.sum()),
                "probability": probability,
                "score_mean": float(evaluation.evt_score.mean()),
                "score_max": float(evaluation.evt_score.max()),
                "risk_mean": float(evaluation.event_risk.mean()),
                "risk_max": float(evaluation.event_risk.max()),
                "collision_fraction": float(evaluation.collision.mean()),
                "numerical_valid_fraction": float(evaluation.numerical_valid.mean()),
            }
        ],
    )
    save_json(top_cases, output_dir / "world_monte_carlo_top_cases.json")
    summary = {
        "schema": "highway_env_idm_monte_carlo_v3",
        "estimator": "independent_monte_carlo",
        "probability": probability,
        "failure_count": int(failure.sum()),
        "failure_event": target,
        "simulation_counts": {"world_evaluations": total},
        "world_prior": "p(M)p(C0|M)p(K|C0,M) * p(z_diff) * p(z_response)",
        "dimensions": dimensions,
        "numerical_valid_fraction": float(evaluation.numerical_valid.mean()),
        "collision_fraction": float(evaluation.collision.mean()),
        "evt_score_summary": {
            "mean": float(evaluation.evt_score.mean()),
            "p95": float(np.quantile(evaluation.evt_score, 0.95)),
            "max": float(evaluation.evt_score.max()),
        },
        "uncertainty": _monte_carlo_uncertainty(probability, total),
        "provenance": {**provenance, **_run_provenance(config)},
    }
    path = output_dir / "world_monte_carlo_summary.json"
    save_json(summary, path)
    return path
