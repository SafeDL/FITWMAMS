"""Current-world subset simulation and independent Monte Carlo runners."""

from __future__ import annotations

import csv
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from hierarchical_world_model.src.composition import (
    DEFAULT_WORLD_SELECTION,
    HierarchicalWorldSampler,
    load_selected_components,
)
from hierarchical_world_model.src.empirical_context import EmpiricalKContextSampler
from hierarchical_world_model.src.highway import (
    HIGHWAY_ENV_HIQR_DYNAMICS_CONTRACT,
)
from tools.evt import GPDTailModel, load_evt_model
from tools.idm_ego import load_idm_ego_config
from diffusion.src.data import load_data_bundle, split_rows
from hierarchical_world_model.src.core.utils import (
    load_json,
    load_yaml,
    save_json,
    select_device,
)
from hierarchical_world_model.src.core.evaluation_scope import (
    evaluation_scope_contract,
    require_evaluation_scope,
    require_scoped_evt_model,
)

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
    return {
        "repository_commit": commit,
        "worktree_dirty": dirty,
    }


def _world_config(
    config: dict[str, Any], config_dir: Path
) -> tuple[dict[str, Any], Path]:
    path = _resolve_path(config["paths"]["world_model_config"], config_dir)
    return load_yaml(path), path


def _require_tail_flow_quality(
    world_config: dict[str, Any], repo_root: Path
) -> dict[str, Any]:
    flow_checkpoint = _resolve_path(world_config["paths"]["flow_checkpoint"], repo_root)
    manifest_path = flow_checkpoint.parent.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"formal tail-Flow quality manifest is missing: {manifest_path}"
        )
    manifest = load_json(manifest_path)
    if manifest.get("population") != "evt_tail":
        raise RuntimeError(
            "formal generated-world evaluation requires an EVT-tail Flow"
        )
    return {
        "tail_flow_quality_manifest": str(manifest_path),
        "tail_flow_quality_gates_passed": bool(manifest["all_quality_gates_passed"]),
        "tail_flow_quality_scope": "distribution checks are reported, not a safety certification or an all-metrics admission gate",
    }


def _require_formal_provenance(
    config: dict[str, Any], config_dir: Path
) -> dict[str, Any]:
    """Require the selected component chain, not a clean Git release."""
    run_provenance = _run_provenance(config)
    world_config, _ = _world_config(config, config_dir)
    root = Path(__file__).resolve().parents[2]
    checkpoint_path = _resolve_path(
        world_config["paths"]["evaluation_checkpoint"],
        root,
    )
    selection, _ = load_selected_components(checkpoint_path)
    return {
        **run_provenance,
        **_require_tail_flow_quality(world_config, root),
        "world_selection": str(DEFAULT_WORLD_SELECTION),
        "selected_arm": selection["selected_arm"],
        "reference_recovery_enabled": selection["reference_recovery_enabled"],
        "source": selection["source"],
    }


def _build_evaluator(
    config: dict[str, Any],
    config_dir: Path,
) -> tuple[CurrentWorldEvaluator, dict[str, Any]]:
    require_evaluation_scope(config)
    backend = str(config["simulation"].get("execution_backend", "local_highway_env"))
    if backend != "local_highway_env":
        raise ValueError(
            "formal IDM evaluation requires simulation.execution_backend=local_highway_env"
        )
    world_config, world_config_path = _world_config(config, config_dir)
    world_paths = world_config["paths"]
    device = select_device(str(config.get("runtime", {}).get("device", "auto")))
    base_sampler = HierarchicalWorldSampler(
        flow_checkpoint=world_paths["flow_checkpoint"],
        diffusion_checkpoint=world_paths["diffusion_checkpoint"],
        diffusion_contract=world_paths["diffusion_contract"],
        response_checkpoint=world_paths["evaluation_checkpoint"],
        # ``config_dir`` is ``idm_ams/configs``; derive the repository
        # root from this maintained module instead of depending on config depth.
        repo_root=Path(__file__).resolve().parents[2],
        device=device,
        ddim_steps=int(config["simulation"].get("ddim_steps", 20)),
    )
    idm_path = _resolve_path(config["paths"]["idm_ego_config"], config_dir)
    policy = HighwayEnvIDMPolicy.from_dict(load_idm_ego_config(idm_path))
    evt_path = _resolve_path(config["paths"]["evt_model"], config_dir)
    require_scoped_evt_model(evt_path)
    test_space = config.get("test_space", {})
    kind = str(test_space.get("kind", "full_prior"))
    if kind == "empirical_test_fixed_k_gt":
        # Release paths are repository-relative so the immutable cache used by
        # the frozen artifact is resolved from the repository root, not from
        # ``hierarchical_world_model/config``.
        bundle = load_data_bundle(world_config, Path(__file__).resolve().parents[2])
        split = str(test_space.get("split", "test"))
        if split != "test":
            raise ValueError(
                "the fixed-K ADS protocol must use the held-out test split"
            )
        sampler = EmpiricalKContextSampler(
            base_sampler,
            bundle,
            split_rows(bundle.arrays, split, seed=0),
        )
    elif kind == "full_prior":
        sampler = base_sampler
    else:
        raise ValueError(f"unsupported factual HiQR test_space.kind={kind!r}")
    evaluator = CurrentWorldEvaluator(
        sampler,
        policy,
        load_evt_model(evt_path),
        steps=int(config["simulation"].get("steps", 149)),
        batch_size=int(config["runtime"].get("batch_size", 32)),
    )
    return evaluator, {
        "device": str(device),
        "execution_backend": backend,
        "hiqr_vehicle_dynamics_contract": HIGHWAY_ENV_HIQR_DYNAMICS_CONTRACT,
        "world_model_config": str(world_config_path),
        "world_model_config_mtime_ns": world_config_path.stat().st_mtime_ns,
        "flow_checkpoint": str(world_paths["flow_checkpoint"]),
        "flow_checkpoint_mtime_ns": Path(world_paths["flow_checkpoint"])
        .stat()
        .st_mtime_ns,
        "diffusion_checkpoint": str(world_paths["diffusion_checkpoint"]),
        "diffusion_checkpoint_mtime_ns": Path(world_paths["diffusion_checkpoint"])
        .stat()
        .st_mtime_ns,
        "response_checkpoint": str(world_paths["evaluation_checkpoint"]),
        "response_checkpoint_mtime_ns": Path(world_paths["evaluation_checkpoint"])
        .stat()
        .st_mtime_ns,
        "world_selection": str(DEFAULT_WORLD_SELECTION),
        "world_selection_mtime_ns": DEFAULT_WORLD_SELECTION.stat().st_mtime_ns,
        "selected_components": base_sampler.selection,
        "evt_model": str(evt_path),
        "evt_model_mtime_ns": evt_path.stat().st_mtime_ns,
        "idm_ego_config": str(idm_path),
        "idm_ego_config_mtime_ns": idm_path.stat().st_mtime_ns,
        "evaluation_scope": evaluation_scope_contract(),
        "test_space": (
            sampler.context_contract
            if isinstance(sampler, EmpiricalKContextSampler)
            else {"test_space": "full_prior"}
        ),
    }


def _failure_target(
    evt: GPDTailModel, config: dict[str, Any]
) -> dict[str, float | int]:
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
    world: Any,
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
    worlds: Any,
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


def run_subset_from_config(
    config: dict[str, Any], config_dir: Path, *, formal: bool = True
) -> Path:
    """Run pCN subset simulation under the complete current world prior.

    Runs record their configuration and selected components. A dirty worktree
    alone does not invalidate the selected model or its evaluation.
    """
    formal_provenance = (
        _require_formal_provenance(config, config_dir)
        if formal
        else _run_provenance(config)
    )
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
        sample_worlds=lambda count, seed: evaluator.sampler.sample_world_exogenous(
            count, seed=seed, response_steps=dimensions["response_steps"]
        ),
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
        "schema": "highway_env_idm_subset_simulation",
        "world_model_id": "factual_hiqr",
        "world_model": "EVT-tail Flow–Diffusion–responsive HiQR world",
        "estimator": "adaptive_multilevel_splitting_pcn_subset_simulation",
        "formal": bool(formal),
        "probability": result.probability,
        "probability_scope": "conditional_on_evt_tail_world_prior",
        "probability_definition": (
            "P(closed-loop R_SEI reaches the EVT return level | EVT-tail world)"
        ),
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
        "world_prior": (
            "empirical_test(C0,M,K_GT) * p(z_diff) * p(z_response)"
            if getattr(evaluator.sampler, "test_space", "full_prior")
            == "empirical_test_fixed_k_gt"
            else ("p_tail(M,C0,K) * p(z_diff) * p(z_response)")
        ),
        "evaluation_contract": {
            "population_scope": evaluation_scope_contract(),
            "steps": int(config["simulation"].get("steps", 149)),
            "dt_s": 0.04,
            "execution_backend": config["simulation"].get(
                "execution_backend", "local_highway_env"
            ),
            "ego_controller": "native HighwayEnv IDMVehicle",
            "background_world_model": "responsive_hiqr_with_online_ma_idm",
            "metric_scope": "IDM ego trajectory risk under generated background response",
            "test_space": provenance["test_space"],
        },
        "dimensions": dimensions,
        "level_statistics": level_rows,
        "uncertainty": _subset_uncertainty(result, float(settings["p0"])),
        "provenance": {**provenance, **formal_provenance},
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


def run_monte_carlo_from_config(
    config: dict[str, Any], config_dir: Path, *, formal: bool = True
) -> Path:
    """Run independent prior samples for the current-world probability baseline."""
    formal_provenance = (
        _require_formal_provenance(config, config_dir)
        if formal
        else _run_provenance(config)
    )
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
    worlds: list[Any] = []
    for start in range(0, total, batch_size):
        count = min(batch_size, total - start)
        batch = evaluator.sampler.sample_world_exogenous(
            count,
            seed=int(rng.integers(0, np.iinfo(np.int64).max)),
            response_steps=dimensions["response_steps"],
        )
        evaluation = evaluator.evaluate(batch)
        scores.append(evaluation.evt_score)
        risks.append(evaluation.event_risk)
        collisions.append(evaluation.collision)
        gaps.append(evaluation.min_gap_m)
        valid.append(evaluation.numerical_valid)
        worlds.append(batch)
    population = type(worlds[0]).concatenate(worlds)
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
        "schema": "highway_env_idm_monte_carlo",
        "world_model_id": "factual_hiqr",
        "world_model": "EVT-tail Flow–Diffusion–responsive HiQR world",
        "estimator": "independent_monte_carlo",
        "formal": bool(formal),
        "probability": probability,
        "probability_scope": "conditional_on_evt_tail_world_prior",
        "probability_definition": (
            "P(closed-loop R_SEI reaches the EVT return level | EVT-tail world)"
        ),
        "failure_count": int(failure.sum()),
        "failure_event": target,
        "simulation_counts": {"world_evaluations": total},
        "world_prior": (
            "empirical_test(C0,M,K_GT) * p(z_diff) * p(z_response)"
            if getattr(evaluator.sampler, "test_space", "full_prior")
            == "empirical_test_fixed_k_gt"
            else ("p_tail(M,C0,K) * p(z_diff) * p(z_response)")
        ),
        "evaluation_contract": {
            "population_scope": evaluation_scope_contract(),
            "steps": int(config["simulation"].get("steps", 149)),
            "dt_s": 0.04,
            "execution_backend": config["simulation"].get(
                "execution_backend", "local_highway_env"
            ),
            "ego_controller": "native HighwayEnv IDMVehicle",
            "background_world_model": "responsive_hiqr_with_online_ma_idm",
            "metric_scope": "IDM ego trajectory risk under generated background response",
            "test_space": provenance["test_space"],
        },
        "dimensions": dimensions,
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
