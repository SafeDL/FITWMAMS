#!/usr/bin/env python3
"""Shared subset simulation implementation."""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.utils import load_json, save_json
from process_highD.src.idm_ego import load_idm_ego_config
from tools.io import resolve_path, write_csv
from PPO_subset.src.closed_loop_runner import (
    ClosedLoopCutInRunner,
    ClosedLoopFollowingRunner,
)
from PPO_subset.src.context_distribution import (
    CUTIN_DISTRIBUTION_SOURCE,
    FOLLOWING_DISTRIBUTION_SOURCE,
    TailContextDistribution,
    load_tail_context_distribution,
)
from PPO_subset.src.frozen_diffusion_sampler import FrozenDiffusionSampler
from PPO_subset.src.latent_evaluator import LatentMpcEpisodeEvaluator
from PPO_subset.src.subset_simulation import (
    SubsetLevel,
    run_subset_simulation,
)
from PPO_subset.src.evt_target import resolve_evt_failure_threshold


logger = logging.getLogger(__name__)
KM_PER_MILE = 1.609344
SOURCE_INDEPENDENT_TAIL_PEAK = "highd_independent_tail_peak"
TAIL_DISTRIBUTION_SOURCE_TYPES = {
    SOURCE_INDEPENDENT_TAIL_PEAK,
    FOLLOWING_DISTRIBUTION_SOURCE,
    CUTIN_DISTRIBUTION_SOURCE,
}
_WORKER_EVALUATOR: LatentMpcEpisodeEvaluator | None = None


def _worker_init(torch_num_threads: int) -> None:
    if torch_num_threads <= 0:
        return
    try:
        import torch

        torch.set_num_threads(int(torch_num_threads))
        torch.set_num_interop_threads(1)
    except Exception as exc:  # 防御性工作进程初始化设置
        logger.warning("Could not set worker torch threads: %s", exc)


def _is_cuda_out_of_memory(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def _progress_interval(total: int) -> int:
    return max(1, min(max(1000, total // 100), max(1, total)))


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _log_progress(label: str, done: int, total: int, start_time: float) -> None:
    elapsed = max(time.monotonic() - start_time, 1.0e-9)
    rate = float(done) / elapsed
    remaining = max(int(total) - int(done), 0)
    eta = float(remaining) / rate if rate > 0.0 else float("inf")
    logger.info(
        "%s progress %d/%d (%.1f%%) rate=%.2f samples/s elapsed=%s eta=%s",
        label,
        done,
        total,
        100.0 * float(done) / max(float(total), 1.0),
        rate,
        _format_duration(elapsed),
        _format_duration(eta) if np.isfinite(eta) else "unknown",
    )


def _worker_evaluate_task(
    task: tuple[int, int, np.ndarray],
) -> tuple[int, Any]:
    if _WORKER_EVALUATOR is None:
        raise RuntimeError("Multiprocessing worker evaluator is not initialized")
    sample_idx, context_index, latent = task
    return int(sample_idx), _WORKER_EVALUATOR.evaluate(
        int(context_index),
        np.asarray(latent, dtype=np.float32),
    )


class _MultiprocessPopulationEvaluator:
    """Evaluate independent level populations in forked CPU workers."""

    def __init__(
        self,
        evaluator: LatentMpcEpisodeEvaluator,
        *,
        num_workers: int,
        chunksize: int,
        worker_torch_num_threads: int,
    ) -> None:
        self.evaluator = evaluator
        self.num_workers = int(num_workers)
        self.chunksize = max(1, int(chunksize))
        self.worker_torch_num_threads = int(worker_torch_num_threads)
        self.pool: mp.pool.Pool | None = None

    def __enter__(self) -> "_MultiprocessPopulationEvaluator":
        global _WORKER_EVALUATOR
        _WORKER_EVALUATOR = self.evaluator
        context = mp.get_context("fork")
        self.pool = context.Pool(
            processes=self.num_workers,
            initializer=_worker_init,
            initargs=(self.worker_torch_num_threads,),
        )
        logger.info(
            "Enabled multiprocessing population evaluation workers=%d chunksize=%d",
            self.num_workers,
            self.chunksize,
        )
        return self

    def __exit__(self, exc_type, exc, _traceback) -> None:
        global _WORKER_EVALUATOR
        if self.pool is not None:
            if exc_type is None:
                self.pool.close()
            else:
                self.pool.terminate()
            self.pool.join()
            self.pool = None
        _WORKER_EVALUATOR = None

    def evaluate_many(
        self,
        context_indices: np.ndarray,
        latents: np.ndarray,
        level: int,
    ) -> tuple[np.ndarray, list[np.ndarray], list[dict[str, float]], list]:
        if self.pool is None:
            raise RuntimeError("Multiprocessing population evaluator is not active")
        total = int(latents.shape[0])
        scores = np.zeros(total, dtype=np.float64)
        actions: list[np.ndarray | None] = [None] * total
        metrics: list[dict[str, float] | None] = [None] * total
        traces: list[list[dict[str, float]] | None] = [None] * total
        tasks = (
            (idx, int(context_indices[idx]), latents[idx].copy())
            for idx in range(total)
        )
        interval = _progress_interval(total)
        start_time = time.monotonic()
        done = 0
        for sample_idx, result in self.pool.imap_unordered(
            _worker_evaluate_task,
            tasks,
            chunksize=self.chunksize,
        ):
            scores[sample_idx] = float(result.score)
            actions[sample_idx] = result.actions.astype(np.float32, copy=True)
            item_metrics = dict(result.metrics)
            item_metrics["context_index"] = float(context_indices[sample_idx])
            metrics[sample_idx] = item_metrics
            traces[sample_idx] = list(result.trace)
            done += 1
            if done == total or done % interval == 0:
                _log_progress(f"Subset level {level}", done, total, start_time)

        return (
            scores,
            [item for item in actions if item is not None],
            [item for item in metrics if item is not None],
            [item for item in traces if item is not None],
        )


class _BatchedPopulationEvaluator:
    """Decode diffusion plans in batches, then run closed-loop rollouts."""

    def __init__(
        self,
        evaluator: LatentMpcEpisodeEvaluator,
        *,
        batch_size: int,
    ) -> None:
        self.evaluator = evaluator
        self.batch_size = max(1, int(batch_size))

    def __enter__(self) -> "_BatchedPopulationEvaluator":
        logger.info(
            "Enabled batched population diffusion decoding batch_size=%d",
            self.batch_size,
        )
        return self

    def __exit__(self, exc_type, exc, _traceback) -> None:
        return None

    def _decode_plans_with_retry(
        self,
        context_indices: np.ndarray,
        latents: np.ndarray,
    ) -> list[np.ndarray]:
        plans: list[np.ndarray] = []
        start = 0
        batch_size = min(self.batch_size, int(latents.shape[0]))
        while start < int(latents.shape[0]):
            end = min(start + batch_size, int(latents.shape[0]))
            try:
                plans.extend(
                    self.evaluator.decode_plans(
                        context_indices[start:end],
                        latents[start:end],
                        batch_size=batch_size,
                    )
                )
                start = end
            except RuntimeError as exc:
                if batch_size <= 1 or not _is_cuda_out_of_memory(exc):
                    raise
                next_batch_size = max(1, batch_size // 2)
                logger.warning(
                    (
                        "CUDA out of memory while decoding population batch; "
                        "reducing population_batch_size from %d to %d"
                    ),
                    batch_size,
                    next_batch_size,
                )
                batch_size = next_batch_size
                self.batch_size = min(self.batch_size, batch_size)
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception as cache_exc:
                    logger.debug("Could not clear CUDA cache after OOM: %s", cache_exc)
        return plans

    def evaluate_many(
        self,
        context_indices: np.ndarray,
        latents: np.ndarray,
        level: int,
    ) -> tuple[np.ndarray, list[np.ndarray], list[dict[str, float]], list]:
        total = int(latents.shape[0])
        scores = np.zeros(total, dtype=np.float64)
        actions: list[np.ndarray] = []
        metrics: list[dict[str, float]] = []
        traces: list[list[dict[str, float]]] = []
        interval = _progress_interval(total)
        start_time = time.monotonic()
        done = 0
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            plans = self._decode_plans_with_retry(
                context_indices[start:end],
                latents[start:end],
            )
            for offset, plan in enumerate(plans):
                idx = start + offset
                context_index = int(context_indices[idx])
                result = self.evaluator.evaluate_decoded_plan(context_index, plan)
                scores[idx] = float(result.score)
                actions.append(result.actions.astype(np.float32, copy=True))
                item_metrics = dict(result.metrics)
                item_metrics["context_index"] = float(context_index)
                metrics.append(item_metrics)
                traces.append(list(result.trace))
                done += 1
                if done == total or done % interval == 0:
                    _log_progress(
                        f"Subset level {level} decoded-plan",
                        done,
                        total,
                        start_time,
                    )
        return scores, actions, metrics, traces


def _paths(config: dict[str, Any], base: Path) -> dict[str, Path]:
    paths = config.get("paths", {})
    required = ("tail_context_path", "condition_distribution_path", "evt_model_path")
    missing = [key for key in required if key not in paths]
    if missing:
        raise KeyError(f"Config paths is missing required keys: {missing}")
    output_value = config.get("subset_simulation", {}).get("output_dir")
    if not output_value:
        raise KeyError("Config subset_simulation.output_dir is required")
    resolved = {
        "tail_contexts": resolve_path(paths["tail_context_path"], base),
        "condition_distribution": resolve_path(
            paths["condition_distribution_path"],
            base,
        ),
        "evt_model": resolve_path(paths["evt_model_path"], base),
        "output_dir": resolve_path(str(output_value), base),
    }
    if "exposure_summary_path" in paths:
        resolved["exposure_summary"] = resolve_path(
            paths["exposure_summary_path"],
            base,
        )
    return resolved


def _input_paths_summary(
    config: dict[str, Any],
    base: Path,
    paths: dict[str, Path],
    sampler: FrozenDiffusionSampler,
) -> dict[str, Any]:
    configured = config.get("paths", {})
    idm_config = configured.get("idm_ego_config_path") or config.get(
        "idm_ego_config_path"
    )
    payload: dict[str, Any] = {
        "natural_dataset_dir": str(getattr(sampler, "natural_dataset_dir", "")),
        "diffusion_checkpoint": str(getattr(sampler, "checkpoint_path", "")),
        "tail_context_path": str(paths["tail_contexts"]),
        "condition_distribution_path": str(paths["condition_distribution"]),
        "evt_model_path": str(paths["evt_model"]),
    }
    if "exposure_summary" in paths:
        payload["exposure_summary_path"] = str(paths["exposure_summary"])
    if idm_config:
        payload["idm_ego_config_path"] = str(resolve_path(str(idm_config), base))
    return payload


def _context_sampling_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("context_sampling", {}) or {})


def _context_provider_rows(contexts: Any) -> list[dict[str, Any]]:
    if isinstance(contexts, TailContextDistribution):
        return list(contexts.base_rows)
    return list(contexts)


def _context_source_types(contexts: Any, *, event_type: str) -> set[str]:
    if isinstance(contexts, TailContextDistribution):
        return {
            CUTIN_DISTRIBUTION_SOURCE
            if event_type == "cut_in"
            else FOLLOWING_DISTRIBUTION_SOURCE
        }
    return {str(context.get("source_type", "")) for context in contexts}


def _load_contexts(
    path: Path,
    distribution_path: Path,
    config: dict[str, Any],
    *,
    event_type: str,
) -> Any:
    sampling_cfg = _context_sampling_config(config)
    target_fps = float(config.get("sampling", {}).get("target_fps", 25.0))
    return load_tail_context_distribution(
        path,
        distribution_path,
        event_type=event_type,
        seed=int(sampling_cfg.get("seed", config.get("training", {}).get("seed", 42))),
        population_size=int(sampling_cfg.get("population_size", 2_147_483_647)),
        dt=1.0 / max(target_fps, 1.0e-6),
    )


def _apply_shared_idm_ego_config(
    config: dict[str, Any],
    config_dir: Path,
    *,
    event_type: str,
) -> None:
    configured = config.get("idm_ego_config_path") or config.get("paths", {}).get(
        "idm_ego_config_path"
    )
    if not configured:
        return
    shared = load_idm_ego_config(
        resolve_path(str(configured), config_dir),
        event_type=event_type,
    )
    config["idm_ego"] = {**dict(config.get("idm_ego", {}) or {}), **shared}
    env_cfg = config.setdefault("env", {})
    ego_response_cfg = config.setdefault("ego_response", {})
    if "target_speed" in shared:
        target_speed = shared["target_speed"]
        target_speed_text = str(target_speed).lower()
        if target_speed is None or target_speed_text in {"initial", "context"}:
            env_cfg["ego_target_speed"] = "context"
        else:
            env_cfg["ego_target_speed"] = float(target_speed)
    if "speed_limit" in shared:
        env_cfg["speed_limit"] = float(shared["speed_limit"])
    if "lanes_count" in shared:
        env_cfg["lanes_count"] = int(shared["lanes_count"])
    if "enable_lane_change" in shared:
        ego_response_cfg["enable_lane_change"] = bool(shared["enable_lane_change"])


def _validate_context_schema(
    contexts: Any,
    sampler: FrozenDiffusionSampler,
    context_path: Path,
) -> None:
    if not contexts:
        raise ValueError(f"Subset tail context file is empty: {context_path}")
    states = np.asarray(contexts[0]["initial_states"], dtype=np.float32)
    conditions = np.asarray(contexts[0]["scenario_conditions"], dtype=np.float32)
    if states.shape != (2, 6):
        raise ValueError(
            "Subset initial_states must have shape [num_actors, state_features], "
            f"got {tuple(states.shape)} in {context_path}"
        )
    cfg = sampler.prior.model.denoiser.cfg
    expected_dim = int(cfg.scenario_condition_dim)
    if conditions.ndim != 1 or int(conditions.shape[0]) != expected_dim:
        raise ValueError(
            "Subset tail context scenario condition schema does not match the "
            f"diffusion checkpoint: got {tuple(conditions.shape)}, expected "
            f"({expected_dim},). Context file: {context_path}. Rebuild the "
            "highD event cache and tail contexts with the current settings: "
            "python process_highD/scripts/extract_highd_events.py && "
            "python process_highD/scripts/select_following_tail_contexts.py or "
            "python process_highD/scripts/select_cutin_tail_contexts.py"
        )
    if isinstance(contexts, TailContextDistribution):
        logger.info(
            (
                "Validated tail context distribution from %s with "
                "empirical_base_contexts=%d scenario_condition_dim=%d"
            ),
            context_path,
            len(contexts.base_rows),
            expected_dim,
        )
    else:
        logger.info(
            "Loaded %d subset contexts from %s with scenario_condition_dim=%d",
            len(contexts),
            context_path,
            expected_dim,
        )


def _multiprocess_population_evaluator(
    evaluator: LatentMpcEpisodeEvaluator,
    sampler: FrozenDiffusionSampler,
    config: dict[str, Any],
) -> _MultiprocessPopulationEvaluator | _BatchedPopulationEvaluator | None:
    parallel_cfg = config.get("parallel", {})
    subset_cfg = config.get("subset_simulation", {})
    device_type = str(getattr(sampler.prior.device, "type", sampler.prior.device))
    if device_type == "cuda":
        batch_size = int(
            parallel_cfg.get(
                "population_batch_size",
                subset_cfg.get("population_batch_size", 128),
            )
        )
        if batch_size <= 1:
            return None
        return _BatchedPopulationEvaluator(evaluator, batch_size=batch_size)
    num_workers = int(
        parallel_cfg.get(
            "population_num_workers",
            subset_cfg.get("population_num_workers", 1),
        )
    )
    cpu_count = max(1, int(os.cpu_count() or 1))
    max_workers = int(parallel_cfg.get("population_max_workers", min(4, cpu_count)))
    num_workers = max(1, min(num_workers, cpu_count, max_workers))
    if num_workers <= 1:
        return None
    return _MultiprocessPopulationEvaluator(
        evaluator,
        num_workers=num_workers,
        chunksize=int(parallel_cfg.get("population_chunksize", 1)),
        worker_torch_num_threads=int(
            parallel_cfg.get("worker_torch_num_threads", 1)
        ),
    )


def _evt_failure_threshold(
    path: Path,
    config: dict[str, Any],
    *,
    config_dir: Path | None = None,
    exposure_summary_path: Path | None = None,
) -> tuple[float, dict[str, Any]]:
    return resolve_evt_failure_threshold(
        path,
        config,
        config_dir=config_dir,
        exposure_summary_path=exposure_summary_path,
    )


def _metric_array(
    levels: list[SubsetLevel],
    key: str,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for level in levels:
        values = [float(item.get(key, np.nan)) for item in level.metrics]
        rows.append(np.asarray(values, dtype=np.float32))
    return np.stack(rows, axis=0)


def _actions_array(levels: list[SubsetLevel]) -> tuple[np.ndarray, np.ndarray]:
    max_steps = max(
        int(action.shape[0]) for level in levels for action in level.actions
    )
    max_dim = max(int(action.shape[1]) for level in levels for action in level.actions)
    shape = (len(levels), len(levels[0].actions), max_steps, max_dim)
    actions = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape[:3], dtype=np.float32)
    for level_idx, level in enumerate(levels):
        for sample_idx, action in enumerate(level.actions):
            steps = int(action.shape[0])
            dim = int(action.shape[1])
            actions[level_idx, sample_idx, :steps, :dim] = action
            mask[level_idx, sample_idx, :steps] = 1.0
    return actions, mask


def _context_metric_array(
    levels: list[SubsetLevel],
    contexts: Any,
    key: str,
    *,
    default: float = np.nan,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for level in levels:
        values: list[float] = []
        for context_index in level.context_indices:
            context = contexts[int(context_index)]
            try:
                values.append(float(context.get(key, default)))
            except (TypeError, ValueError):
                values.append(float(default))
        rows.append(np.asarray(values, dtype=np.float32))
    return np.stack(rows, axis=0)


def _context_stack_array(
    levels: list[SubsetLevel],
    contexts: Any,
    key: str,
    *,
    dtype: np.dtype,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for level in levels:
        rows.append(
            np.stack(
                [
                    np.asarray(contexts[int(context_index)][key], dtype=dtype)
                    for context_index in level.context_indices
                ],
                axis=0,
            )
        )
    return np.stack(rows, axis=0)


def _save_samples(
    result,
    output_dir: Path,
    contexts: Any | None = None,
) -> None:
    levels = result.levels
    actions, action_mask = _actions_array(levels)
    payload: dict[str, np.ndarray] = {
        "context_indices": np.stack(
            [level.context_indices for level in levels],
            axis=0,
        ),
        "latents": np.stack([level.latents for level in levels], axis=0),
        "scores": np.stack([level.scores for level in levels], axis=0),
        "failure_threshold": np.asarray(
            float(result.failure_threshold),
            dtype=np.float32,
        ),
        "thresholds": np.asarray(
            [level.threshold for level in levels],
            dtype=np.float32,
        ),
        "acceptance_rate": np.asarray(
            [level.acceptance_rate for level in levels],
            dtype=np.float32,
        ),
        "accepted_mask": np.stack([level.accepted for level in levels], axis=0),
        "actions": actions,
        "action_mask": action_mask,
        "collision": _metric_array(levels, "collision"),
        "min_gap": _metric_array(levels, "min_gap"),
        "min_ttc": _metric_array(levels, "min_ttc"),
        "physical_feasible": _metric_array(levels, "physical_feasible"),
        "y_long": _metric_array(levels, "y_long"),
        "y_cutin": _metric_array(levels, "y_cutin"),
        "is_cutin": _metric_array(levels, "is_cutin"),
        "evt_tail_probability": _metric_array(levels, "evt_tail_probability"),
    }
    if contexts is not None:
        payload.update(
            {
                "scenario_conditions": _context_stack_array(
                    levels,
                    contexts,
                    "scenario_conditions",
                    dtype=np.float32,
                ),
                "initial_states": _context_stack_array(
                    levels,
                    contexts,
                    "initial_states",
                    dtype=np.float32,
                ),
                "ego_length": _context_metric_array(levels, contexts, "ego_length"),
                "adv_length": _context_metric_array(levels, contexts, "adv_length"),
                "recording_id": _context_metric_array(
                    levels,
                    contexts,
                    "recording_id",
                ),
                "ego_id": _context_metric_array(levels, contexts, "ego_id"),
                "target_id": _context_metric_array(levels, contexts, "target_id"),
                "anchor_frame": _context_metric_array(
                    levels,
                    contexts,
                    "anchor_frame",
                ),
                "context_anchor_frame": _context_metric_array(
                    levels,
                    contexts,
                    "context_anchor_frame",
                ),
                "risk_start_index": _context_metric_array(
                    levels,
                    contexts,
                    "risk_start_index",
                ),
                "cross_frame": _context_metric_array(levels, contexts, "cross_frame"),
                "cutin_start_frame": _context_metric_array(
                    levels,
                    contexts,
                    "cutin_start_frame",
                ),
            }
        )
    np.savez_compressed(
        output_dir / "latent_subset_samples.npz",
        **payload,
    )


def _top_cases(
    result,
    contexts: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    metric_keys = (
        "collision",
        "near_collision",
        "min_gap",
        "min_ttc",
        "risk_score",
        "y_long",
        "y_cutin",
        "evt_tail_probability",
        "physical_feasible",
        "cutin_safety_risk_score",
        "cutin_gap",
        "cutin_ttc",
        "cutin_time_headway",
        "cutin_lateral_time_gap",
        "max_post_cutin_drac",
        "min_abs_lateral_offset",
        "final_abs_lateral_offset",
        "max_lateral_approach_speed",
        "lateral_overlap_fraction",
        "is_cutin",
        "is_front_cutin",
    )
    for level in result.levels:
        for idx, score in enumerate(level.scores):
            context_index = int(level.context_indices[idx])
            context = contexts[context_index]
            cases.append(
                {
                    "level": int(level.level),
                    "sample_index": int(idx),
                    "context_index": context_index,
                    "recording_id": context.get("recording_id"),
                    "event_id": context.get("event_id"),
                    "score": float(score),
                    "metrics": {
                        key: float(level.metrics[idx][key])
                        for key in metric_keys
                        if key in level.metrics[idx]
                    },
                }
            )
    cases.sort(key=lambda item: float(item["score"]), reverse=True)
    return cases[:top_k]


def _probability_uncertainty(
    result,
    *,
    num_samples: int,
    p0: float,
) -> dict[str, float]:
    level_power = max(len(result.levels) - 1, 0)
    scale = float(p0) ** level_power
    q = float(result.final_failure_fraction)
    n = max(int(num_samples), 1)
    conditional_se = float(np.sqrt(max(q * (1.0 - q), 0.0) / n))
    se = scale * conditional_se
    probability = float(result.probability)
    lower = max(0.0, probability - 1.96 * se)
    upper = min(1.0, probability + 1.96 * se)
    rel = float(se / probability) if probability > 0.0 else float("inf")
    return {
        "probability_standard_error": float(se),
        "probability_ci95_lower": float(lower),
        "probability_ci95_upper": float(upper),
        "conditional_final_fraction_standard_error": conditional_se,
        "relative_standard_error": rel,
        "uncertainty_method": (
            "binomial final-level approximation; ignores MCMC correlation"
        ),
    }


def _uniqueness_stats(
    context_indices: np.ndarray,
    latents: np.ndarray,
) -> dict[str, float]:
    num_samples = max(int(latents.shape[0]), 1)
    context_counter: dict[int, int] = {}
    state_counter: dict[tuple[int, bytes], int] = {}
    for idx in range(int(latents.shape[0])):
        context = int(context_indices[idx])
        state = (context, np.ascontiguousarray(latents[idx]).tobytes())
        context_counter[context] = context_counter.get(context, 0) + 1
        state_counter[state] = state_counter.get(state, 0) + 1
    context_counts = np.asarray(list(context_counter.values()), dtype=np.int64)
    state_counts = np.asarray(list(state_counter.values()), dtype=np.int64)
    return {
        "unique_contexts": float(len(context_counter)),
        "largest_context_count": float(np.max(context_counts)),
        "largest_context_share": float(np.max(context_counts) / num_samples),
        "unique_states": float(len(state_counter)),
        "largest_state_count": float(np.max(state_counts)),
        "largest_state_share": float(np.max(state_counts) / num_samples),
    }


def _level_stats(
    result,
    failure_threshold: float,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for level in result.levels:
        scores = np.asarray(level.scores, dtype=np.float64)
        uniqueness = _uniqueness_stats(level.context_indices, level.latents)
        rows.append(
            {
                "level": float(level.level),
                "num_samples": float(len(scores)),
                "score_min": float(np.min(scores)),
                "score_mean": float(np.mean(scores)),
                "score_std": float(np.std(scores)),
                "score_p50": float(np.quantile(scores, 0.50)),
                "score_p90": float(np.quantile(scores, 0.90)),
                "score_p95": float(np.quantile(scores, 0.95)),
                "score_max": float(np.max(scores)),
                "subset_threshold": float(level.threshold),
                "failure_fraction": float(np.mean(scores >= float(failure_threshold))),
                "acceptance_rate": float(level.acceptance_rate),
                **uniqueness,
            }
        )
    return rows


def _subset_simulation_counts(result) -> dict[str, int]:
    levels = result.levels
    if not levels:
        return {
            "closed_loop_evaluations": int(getattr(result, "total_evaluations", 0)),
            "proposal_evaluations": int(getattr(result, "proposal_evaluations", 0)),
            "stored_level_samples": 0,
            "num_levels": 0,
            "unique_context_indices_all_levels": 0,
            "unique_context_indices_final_level": 0,
        }
    all_contexts = np.concatenate([level.context_indices for level in levels], axis=0)
    return {
        "closed_loop_evaluations": int(
            getattr(result, "total_evaluations", int(all_contexts.shape[0]))
        ),
        "proposal_evaluations": int(getattr(result, "proposal_evaluations", 0)),
        "stored_level_samples": int(all_contexts.shape[0]),
        "num_levels": int(len(levels)),
        "unique_context_indices_all_levels": int(np.unique(all_contexts).shape[0]),
        "unique_context_indices_final_level": int(
            np.unique(levels[-1].context_indices).shape[0]
        ),
    }


def _monte_carlo_simulation_counts(context_indices: np.ndarray) -> dict[str, int]:
    return {
        "closed_loop_evaluations": int(context_indices.shape[0]),
        "stored_samples": int(context_indices.shape[0]),
        "unique_context_indices": int(np.unique(context_indices).shape[0]),
    }


def _input_space_summary(evaluator: LatentMpcEpisodeEvaluator) -> dict[str, int | list[int]]:
    latent_shape = tuple(int(item) for item in evaluator.latent_shape)
    latent_dimension = int(np.prod(latent_shape, dtype=np.int64))
    cfg = evaluator.sampler.prior.model.denoiser.cfg
    scenario_condition_dimension = int(cfg.scenario_condition_dim)
    return {
        "scenario_condition_dimension": scenario_condition_dimension,
        "diffusion_noise_shape": list(latent_shape),
        "diffusion_noise_dimension": latent_dimension,
        "joint_condition_noise_dimension": int(
            scenario_condition_dimension + latent_dimension
        ),
    }


def _policy_summary(config: dict[str, Any]) -> dict[str, Any]:
    policy_cfg = dict(config.get("ppo_policy", {}) or {})
    policy_type = str(policy_cfg.get("policy_type", "ppo")).lower()
    return {
        "policy_name": str(policy_cfg.get("name", f"PPO-{policy_type.upper()}")),
        "policy_type": policy_type,
        "backend": str(policy_cfg.get("backend", "stable_baselines3")),
        "checkpoint_path": str(policy_cfg.get("checkpoint_path", "")),
        "deterministic": bool(policy_cfg.get("deterministic", True)),
    }


def _reliability_thresholds(
    config: dict[str, Any],
    *,
    num_contexts: int,
    num_samples: int,
) -> dict[str, float]:
    subset_cfg = config.get("subset_simulation", {})
    min_context_absolute = int(subset_cfg.get("reliability_min_unique_contexts", 10))
    min_unique_contexts = min(int(num_contexts), min_context_absolute)
    min_state_fraction = float(
        subset_cfg.get("reliability_min_unique_state_fraction", 0.50)
    )
    min_unique_states = max(1, int(np.ceil(min_state_fraction * num_samples)))
    return {
        "min_unique_contexts": float(min_unique_contexts),
        "min_unique_states": float(min_unique_states),
        "max_largest_context_share": float(
            subset_cfg.get("reliability_max_largest_context_share", 0.30)
        ),
        "max_largest_state_share": float(
            subset_cfg.get("reliability_max_largest_state_share", 0.10)
        ),
        "min_acceptance_rate": float(
            subset_cfg.get("reliability_min_acceptance_rate", 0.10)
        ),
    }


def _reliability_assessment(
    level_stats: list[dict[str, float]],
    config: dict[str, Any],
    *,
    num_contexts: int,
    num_samples: int,
) -> dict[str, Any]:
    thresholds = _reliability_thresholds(
        config,
        num_contexts=num_contexts,
        num_samples=num_samples,
    )
    if not level_stats:
        return {
            "status": "fail",
            "reason": ["no subset levels were produced"],
            "thresholds": thresholds,
        }
    final = dict(level_stats[-1])
    failures: list[str] = []
    warnings: list[str] = []

    if final["unique_contexts"] < thresholds["min_unique_contexts"]:
        failures.append(
            "unique_contexts "
            f"{final['unique_contexts']:.0f} < {thresholds['min_unique_contexts']:.0f}"
        )
    if final["unique_states"] < thresholds["min_unique_states"]:
        failures.append(
            "unique_states "
            f"{final['unique_states']:.0f} < {thresholds['min_unique_states']:.0f}"
        )
    if final["largest_context_share"] > thresholds["max_largest_context_share"]:
        failures.append(
            "largest_context_share "
            f"{final['largest_context_share']:.3f} > "
            f"{thresholds['max_largest_context_share']:.3f}"
        )
    if final["largest_state_share"] > thresholds["max_largest_state_share"]:
        failures.append(
            "largest_state_share "
            f"{final['largest_state_share']:.3f} > "
            f"{thresholds['max_largest_state_share']:.3f}"
        )

    acceptance_level: dict[str, float] | None = None
    transition_rows = level_stats[:-1] if len(level_stats) > 1 else []
    for row in reversed(transition_rows):
        candidate = float(row.get("acceptance_rate", np.nan))
        if np.isfinite(candidate):
            acceptance_level = row
            break
    acceptance = (
        float(acceptance_level["acceptance_rate"])
        if acceptance_level is not None
        else float("nan")
    )
    if np.isfinite(acceptance):
        if acceptance < thresholds["min_acceptance_rate"]:
            failures.append(
                "acceptance_rate "
                f"{acceptance:.3f} < {thresholds['min_acceptance_rate']:.3f}"
            )
    elif len(level_stats) > 1:
        warnings.append("no transition acceptance_rate is available")

    status = "fail" if failures else ("warning" if warnings else "pass")
    return {
        "status": status,
        "failures": failures,
        "warnings": warnings,
        "thresholds": thresholds,
        "assessed_level": int(final.get("level", -1)),
        "observed": {
            "unique_contexts": final.get("unique_contexts"),
            "unique_states": final.get("unique_states"),
            "largest_context_share": final.get("largest_context_share"),
            "largest_state_share": final.get("largest_state_share"),
            "acceptance_rate": acceptance,
            "acceptance_rate_level": (
                acceptance_level.get("level") if acceptance_level is not None else None
            ),
            "final_level_acceptance_rate": final.get("acceptance_rate"),
        },
    }


def _context_tail_thresholds(contexts: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for context in _context_provider_rows(contexts):
        for key in ("tail_threshold", "evt_tail_threshold_u"):
            if key not in context:
                continue
            try:
                value = float(context[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
            break
    return values


def _context_collision_levels(contexts: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for context in _context_provider_rows(contexts):
        if "collision_critical_level" not in context:
            continue
        try:
            value = float(context["collision_critical_level"])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def _apply_evt_target_to_context_metadata(
    contexts: Any,
    evt_target: dict[str, Any],
) -> None:
    if str(evt_target.get("evt_target_mode")) != "collision_critical_level":
        return
    try:
        level = float(evt_target.get("evt_return_level_target", np.nan))
    except (TypeError, ValueError):
        return
    if not np.isfinite(level):
        return
    rows = getattr(contexts, "context_rows", None)
    if rows is None:
        rows = contexts
    for context in rows:
        context["collision_critical_level"] = level


def _mileage_return_period(
    result,
    contexts: list[dict[str, Any]],
    config: dict[str, Any],
    evt_target: dict[str, float],
    reliability: dict[str, Any],
    probability_estimate_kind: str,
    exposure_summary_path: Path | None,
) -> dict[str, Any]:
    cfg = config.get("mileage_return_period", {})
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return {"enabled": False}

    strictness_failures: list[str] = []
    exposure: dict[str, Any] = {}
    if exposure_summary_path is None:
        strictness_failures.append("paths.exposure_summary_path is not configured")
    elif not exposure_summary_path.exists():
        strictness_failures.append(f"exposure_summary_missing={exposure_summary_path}")
    else:
        exposure = load_json(exposure_summary_path)

    event_type = str(config.get("event", {}).get("event_type", "following"))
    is_cutin = event_type == "cut_in"
    expected_denominator = "all_vehicle_km"
    primary_label = "highD all-vehicle"
    risk_label = "Y_cutin_sim" if is_cutin else "Y_long_sim"
    total_km = float(exposure.get("all_vehicle_km", 0.0))
    total_miles = float(exposure.get("all_vehicle_miles", total_km / KM_PER_MILE))
    total_hours = float(exposure.get("all_vehicle_hours", 0.0))
    all_vehicle_miles = float(exposure.get("all_vehicle_miles", 0.0))
    all_vehicle_hours = float(exposure.get("all_vehicle_hours", 0.0))
    if total_km <= 0.0:
        strictness_failures.append("total_exposure_km <= 0")
    if str(exposure.get("exposure_denominator", "")) != expected_denominator:
        strictness_failures.append(
            f"exposure_denominator!={expected_denominator}"
        )

    probability = float(result.probability)
    if not np.isfinite(probability) or probability < 0.0:
        strictness_failures.append("subset_probability is not finite and non-negative")

    tail_rate_per_mile = float(exposure.get("tail_peak_rate_per_mile", 0.0))
    tail_rate_per_hour = float(exposure.get("tail_peak_rate_per_hour", 0.0))
    tail_rate_per_all_vehicle_mile = float(
        exposure.get(
            "tail_peak_rate_per_all_vehicle_mile",
            tail_rate_per_mile if is_cutin else 0.0,
        )
    )
    tail_rate_per_all_vehicle_hour = float(
        exposure.get(
            "tail_peak_rate_per_all_vehicle_hour",
            tail_rate_per_hour if is_cutin else 0.0,
        )
    )

    def _periods(rate_per_mile: float, rate_per_hour: float) -> dict[str, float]:
        intensity_per_mile = float(max(rate_per_mile, 0.0) * probability)
        intensity_per_hour = float(max(rate_per_hour, 0.0) * probability)
        return {
            "intensity_per_mile": intensity_per_mile,
            "return_period_miles": (
                float(1.0 / intensity_per_mile)
                if intensity_per_mile > 0.0
                else float("inf")
            ),
            "intensity_per_km": float(intensity_per_mile / KM_PER_MILE),
            "return_period_km": (
                float(KM_PER_MILE / intensity_per_mile)
                if intensity_per_mile > 0.0
                else float("inf")
            ),
            "intensity_per_hour": intensity_per_hour,
            "return_period_hours": (
                float(1.0 / intensity_per_hour)
                if intensity_per_hour > 0.0
                else float("inf")
            ),
        }

    primary_periods = _periods(tail_rate_per_mile, tail_rate_per_hour)
    all_vehicle_periods = _periods(
        tail_rate_per_all_vehicle_mile,
        tail_rate_per_all_vehicle_hour,
    )

    def _ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0.0 or not np.isfinite(denominator):
            return float("nan")
        return float(numerator / denominator)

    target_mode = str(evt_target.get("evt_target_mode", "return_period"))
    human_reference: dict[str, Any] | None = None
    all_vehicle_highd_intensity_mile = float("nan")
    all_vehicle_highd_intensity_km = float("nan")
    all_vehicle_highd_intensity_hour = float("nan")
    all_vehicle_highd_return_miles = float("nan")
    all_vehicle_highd_return_km = float("nan")
    all_vehicle_highd_return_hours = float("nan")
    if target_mode == "collision_critical_level":
        highd_intensity_mile = float(
            exposure.get("highd_safety_critical_intensity_per_mile", np.nan)
        )
        highd_intensity_km = float(
            exposure.get(
                "highd_safety_critical_intensity_per_km",
                highd_intensity_mile / KM_PER_MILE,
            )
        )
        highd_intensity_hour = float(
            exposure.get("highd_safety_critical_intensity_per_hour", np.nan)
        )
        highd_return_miles = float(
            exposure.get("highd_safety_critical_return_period_miles", np.nan)
        )
        highd_return_km = float(
            exposure.get(
                "highd_safety_critical_return_period_km",
                highd_return_miles * KM_PER_MILE,
            )
        )
        highd_return_hours = float(
            exposure.get("highd_safety_critical_return_period_hours", np.nan)
        )
        human_reference = {
            "interpretation": (
                f"Human highD {event_type} reference at the same EVT "
                f"safety-critical level and {expected_denominator} denominator."
            ),
            "highd_safety_critical_intensity_per_mile": highd_intensity_mile,
            "highd_safety_critical_intensity_per_km": highd_intensity_km,
            "highd_safety_critical_intensity_per_hour": highd_intensity_hour,
            "highd_safety_critical_return_period_miles": highd_return_miles,
            "highd_safety_critical_return_period_km": highd_return_km,
            "highd_safety_critical_return_period_hours": highd_return_hours,
            "ads_to_highd_intensity_ratio_per_mile": _ratio(
                primary_periods["intensity_per_mile"],
                highd_intensity_mile,
            ),
            "ads_to_highd_intensity_ratio_per_hour": _ratio(
                primary_periods["intensity_per_hour"],
                highd_intensity_hour,
            ),
            "ads_return_period_over_highd_return_period_miles": _ratio(
                primary_periods["return_period_miles"],
                highd_return_miles,
            ),
            "ads_return_period_over_highd_return_period_hours": _ratio(
                primary_periods["return_period_hours"],
                highd_return_hours,
            ),
        }
        all_vehicle_highd_intensity_mile = float(
            highd_intensity_mile
            if is_cutin
            else exposure.get(
                "safety_critical_intensity_per_all_vehicle_mile",
                np.nan,
            )
        )
        all_vehicle_highd_intensity_km = float(
            exposure.get(
                "safety_critical_intensity_per_all_vehicle_km",
                all_vehicle_highd_intensity_mile / KM_PER_MILE,
            )
        )
        all_vehicle_highd_intensity_hour = float(
            highd_intensity_hour
            if is_cutin
            else exposure.get(
                "safety_critical_intensity_per_all_vehicle_hour",
                np.nan,
            )
        )
        all_vehicle_highd_return_miles = float(
            highd_return_miles
            if is_cutin
            else exposure.get(
                "safety_critical_return_period_all_vehicle_miles",
                np.nan,
            )
        )
        all_vehicle_highd_return_km = float(
            exposure.get(
                "safety_critical_return_period_all_vehicle_km",
                all_vehicle_highd_return_miles * KM_PER_MILE,
            )
        )
        all_vehicle_highd_return_hours = float(
            highd_return_hours
            if is_cutin
            else exposure.get(
                "safety_critical_return_period_all_vehicle_hours",
                np.nan,
            )
        )

    if bool(cfg.get("require_tail_threshold_match", True)):
        exposure_u = exposure.get("evt_tail_threshold_u")
        if exposure_u is None:
            strictness_failures.append("exposure evt_tail_threshold_u is missing")
        else:
            exposure_u = float(exposure_u)
            evt_model_u = float(evt_target.get("evt_model_u", np.nan))
            tol = float(cfg.get("tail_threshold_abs_tol", 1.0e-6))
            if not np.isfinite(evt_model_u) or abs(exposure_u - evt_model_u) > tol:
                strictness_failures.append(
                    "exposure evt_tail_threshold_u does not match subset EVT model u"
                )
            context_thresholds = _context_tail_thresholds(contexts)
            if not context_thresholds:
                strictness_failures.append("tail context threshold metadata is missing")
            elif max(abs(value - exposure_u) for value in context_thresholds) > tol:
                strictness_failures.append(
                    "tail context threshold does not match exposure evt_tail_threshold_u"
                )
    collision_level = evt_target.get("evt_return_level_target")
    if target_mode == "collision_critical_level":
        exposure_collision = exposure.get("collision_critical_level")
        tol = float(cfg.get("tail_threshold_abs_tol", 1.0e-6))
        if exposure_collision is None:
            strictness_failures.append("exposure collision_critical_level is missing")
        elif abs(float(exposure_collision) - float(collision_level)) > tol:
            strictness_failures.append(
                "exposure collision_critical_level does not match subset target"
            )
        context_collision = _context_collision_levels(contexts)
        if not context_collision:
            strictness_failures.append(
                "tail context collision critical metadata is missing"
            )
        elif (
            max(abs(value - float(collision_level)) for value in context_collision)
            > tol
        ):
            strictness_failures.append(
                "tail context collision critical level does not match subset target"
            )

    if bool(cfg.get("require_independent_peak_contexts", True)):
        source_types = _context_source_types(contexts, event_type=event_type)
        if (
            not source_types
            or not source_types.issubset(TAIL_DISTRIBUTION_SOURCE_TYPES)
        ):
            strictness_failures.append(
                "tail_context_source!="
                "independent_tail_peak_or_tail_feature_distribution "
                f"({sorted(source_types)})"
            )

    if bool(cfg.get("require_subset_reliability_pass", True)):
        if reliability.get("status") != "pass":
            strictness_failures.append(
                f"subset_reliability_status={reliability.get('status')}"
            )
    if probability_estimate_kind != "standard_subset_estimate":
        strictness_failures.append(
            f"probability_estimate_kind={probability_estimate_kind}"
        )

    return {
        "enabled": True,
        "event_type": event_type,
        "risk_label": risk_label,
        "primary_exposure_label": primary_label,
        "exposure_denominator": expected_denominator,
        "exposure_summary_path": (
            str(exposure_summary_path) if exposure_summary_path is not None else None
        ),
        "ads_exceedance_probability_conditional": probability,
        "tail_peak_rate_per_mile": tail_rate_per_mile,
        "ads_extreme_risk_intensity_per_mile": primary_periods[
            "intensity_per_mile"
        ],
        "ads_safety_critical_intensity_per_mile": primary_periods[
            "intensity_per_mile"
        ],
        "ads_return_period_miles": primary_periods["return_period_miles"],
        "ads_safety_critical_return_period_miles": primary_periods[
            "return_period_miles"
        ],
        "ads_extreme_risk_intensity_per_km": primary_periods[
            "intensity_per_km"
        ],
        "ads_safety_critical_intensity_per_km": primary_periods[
            "intensity_per_km"
        ],
        "ads_return_period_km": primary_periods["return_period_km"],
        "ads_safety_critical_return_period_km": primary_periods[
            "return_period_km"
        ],
        "tail_peak_rate_per_hour": tail_rate_per_hour,
        "ads_extreme_risk_intensity_per_hour": primary_periods[
            "intensity_per_hour"
        ],
        "ads_safety_critical_intensity_per_hour": primary_periods[
            "intensity_per_hour"
        ],
        "ads_return_period_hours": primary_periods["return_period_hours"],
        "ads_safety_critical_return_period_hours": primary_periods[
            "return_period_hours"
        ],
        "all_highd_vehicle_background": {
            "total_all_vehicle_miles": all_vehicle_miles,
            "total_all_vehicle_km": float(
                exposure.get("all_vehicle_km", all_vehicle_miles * KM_PER_MILE)
            ),
            "total_all_vehicle_hours": all_vehicle_hours,
            "following_ego_mile_fraction_of_all_vehicle_miles": float(
                exposure.get("ego_mile_fraction_of_all_vehicle", 0.0)
            ),
            "tail_peak_rate_per_all_vehicle_mile": tail_rate_per_all_vehicle_mile,
            "tail_peak_rate_per_all_vehicle_km": float(
                exposure.get(
                    "tail_peak_rate_per_all_vehicle_km",
                    tail_rate_per_all_vehicle_mile / KM_PER_MILE,
                )
            ),
            "tail_peak_rate_per_all_vehicle_hour": (
                tail_rate_per_all_vehicle_hour
            ),
            "ads_extreme_risk_intensity_per_all_vehicle_mile": (
                all_vehicle_periods["intensity_per_mile"]
            ),
            "ads_safety_critical_intensity_per_all_vehicle_mile": (
                all_vehicle_periods["intensity_per_mile"]
            ),
            "ads_return_period_all_vehicle_miles": all_vehicle_periods[
                "return_period_miles"
            ],
            "ads_safety_critical_return_period_all_vehicle_miles": (
                all_vehicle_periods["return_period_miles"]
            ),
            "ads_extreme_risk_intensity_per_all_vehicle_km": (
                all_vehicle_periods["intensity_per_km"]
            ),
            "ads_safety_critical_intensity_per_all_vehicle_km": (
                all_vehicle_periods["intensity_per_km"]
            ),
            "ads_return_period_all_vehicle_km": all_vehicle_periods[
                "return_period_km"
            ],
            "ads_safety_critical_return_period_all_vehicle_km": (
                all_vehicle_periods["return_period_km"]
            ),
            "ads_extreme_risk_intensity_per_all_vehicle_hour": (
                all_vehicle_periods["intensity_per_hour"]
            ),
            "ads_safety_critical_intensity_per_all_vehicle_hour": (
                all_vehicle_periods["intensity_per_hour"]
            ),
            "ads_return_period_all_vehicle_hours": all_vehicle_periods[
                "return_period_hours"
            ],
            "ads_safety_critical_return_period_all_vehicle_hours": (
                all_vehicle_periods["return_period_hours"]
            ),
            "highd_safety_critical_intensity_per_all_vehicle_mile": (
                all_vehicle_highd_intensity_mile
            ),
            "highd_safety_critical_intensity_per_all_vehicle_km": (
                all_vehicle_highd_intensity_km
            ),
            "highd_safety_critical_intensity_per_all_vehicle_hour": (
                all_vehicle_highd_intensity_hour
            ),
            "highd_safety_critical_return_period_all_vehicle_miles": (
                all_vehicle_highd_return_miles
            ),
            "highd_safety_critical_return_period_all_vehicle_km": (
                all_vehicle_highd_return_km
            ),
            "highd_safety_critical_return_period_all_vehicle_hours": (
                all_vehicle_highd_return_hours
            ),
            "ads_to_highd_intensity_ratio_per_all_vehicle_km": _ratio(
                all_vehicle_periods["intensity_per_km"],
                all_vehicle_highd_intensity_km,
            ),
            "ads_return_period_over_highd_return_period_all_vehicle_km": _ratio(
                all_vehicle_periods["return_period_km"],
                all_vehicle_highd_return_km,
            ),
        },
        "human_highd_reference": human_reference,
        "evt_target_mode": target_mode,
        "collision_critical_level": (
            float(collision_level)
            if target_mode == "collision_critical_level"
            else None
        ),
        "ads_safety_critical_intensity_per_mile_at_x_c": (
            primary_periods["intensity_per_mile"]
            if target_mode == "collision_critical_level"
            else None
        ),
        "ads_safety_critical_return_period_miles_at_x_c": (
            primary_periods["return_period_miles"]
            if target_mode == "collision_critical_level"
            else None
        ),
        "ads_safety_critical_intensity_per_km_at_x_c": (
            primary_periods["intensity_per_km"]
            if target_mode == "collision_critical_level"
            else None
        ),
        "ads_safety_critical_return_period_km_at_x_c": (
            primary_periods["return_period_km"]
            if target_mode == "collision_critical_level"
            else None
        ),
        "ads_safety_critical_intensity_per_hour_at_x_c": (
            primary_periods["intensity_per_hour"]
            if target_mode == "collision_critical_level"
            else None
        ),
        "ads_safety_critical_return_period_hours_at_x_c": (
            primary_periods["return_period_hours"]
            if target_mode == "collision_critical_level"
            else None
        ),
        "primary_exposure_miles": total_miles,
        "primary_exposure_hours": total_hours,
        "following_ego_miles": float(exposure.get("following_ego_miles", 0.0)),
        "following_ego_hours": float(exposure.get("following_ego_hours", 0.0)),
        "all_vehicle_miles": all_vehicle_miles,
        "all_vehicle_hours": all_vehicle_hours,
        "num_independent_tail_peaks": int(
            exposure.get("num_independent_tail_peaks", 0)
        ),
        "strict_mileage_interpretation": not strictness_failures,
        "strictness_failures": strictness_failures,
    }


def _write_level_stats(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    write_csv(path, rows)


def _save_monte_carlo_samples(
    output_dir: Path,
    *,
    context_indices: np.ndarray,
    latents: np.ndarray,
    scores: np.ndarray,
    metrics: list[dict[str, float]],
    failure_threshold: float,
) -> None:
    def metric_array(key: str) -> np.ndarray:
        return np.asarray(
            [float(item.get(key, np.nan)) for item in metrics],
            dtype=np.float32,
        )

    payload: dict[str, np.ndarray] = {
        "context_indices": np.asarray(context_indices, dtype=np.int64),
        "latents": np.asarray(latents, dtype=np.float32),
        "scores": np.asarray(scores, dtype=np.float32),
        "failure_mask": (np.asarray(scores) >= float(failure_threshold)).astype(
            np.float32,
        ),
        "collision": metric_array("collision"),
        "min_gap": metric_array("min_gap"),
        "min_ttc": metric_array("min_ttc"),
        "physical_feasible": metric_array("physical_feasible"),
        "y_long": metric_array("y_long"),
        "y_cutin": metric_array("y_cutin"),
        "is_cutin": metric_array("is_cutin"),
        "evt_tail_probability": metric_array("evt_tail_probability"),
    }
    np.savez_compressed(output_dir / "latent_monte_carlo_samples.npz", **payload)


def _monte_carlo_top_cases(
    contexts: Any,
    context_indices: np.ndarray,
    scores: np.ndarray,
    metrics: list[dict[str, float]],
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    metric_keys = (
        "collision",
        "near_collision",
        "min_gap",
        "min_ttc",
        "risk_score",
        "y_long",
        "y_cutin",
        "evt_tail_probability",
        "physical_feasible",
        "cutin_safety_risk_score",
        "cutin_gap",
        "cutin_ttc",
        "cutin_time_headway",
        "cutin_lateral_time_gap",
        "max_post_cutin_drac",
        "min_abs_lateral_offset",
        "final_abs_lateral_offset",
        "max_lateral_approach_speed",
        "lateral_overlap_fraction",
        "is_cutin",
        "is_front_cutin",
    )
    rows: list[dict[str, Any]] = []
    for sample_idx, score in enumerate(scores):
        context_index = int(context_indices[sample_idx])
        context = contexts[context_index]
        rows.append(
            {
                "sample_index": int(sample_idx),
                "context_index": context_index,
                "recording_id": context.get("recording_id"),
                "event_id": context.get("event_id"),
                "score": float(score),
                "metrics": {
                    key: float(metrics[sample_idx][key])
                    for key in metric_keys
                    if key in metrics[sample_idx]
                },
            }
        )
    rows.sort(key=lambda item: float(item["score"]), reverse=True)
    return rows[:top_k]


def _monte_carlo_stats(
    scores: np.ndarray,
    metrics: list[dict[str, float]],
    failure_threshold: float,
) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    failures = scores >= float(failure_threshold)
    n = max(int(scores.size), 1)
    probability = float(np.mean(failures))
    se = float(np.sqrt(max(probability * (1.0 - probability), 0.0) / n))

    def metric_mean(key: str) -> float:
        values = np.asarray(
            [float(item.get(key, np.nan)) for item in metrics],
            dtype=np.float64,
        )
        return float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")

    return {
        "num_samples": float(n),
        "probability": probability,
        "probability_standard_error": se,
        "probability_ci95_lower": max(0.0, probability - 1.96 * se),
        "probability_ci95_upper": min(1.0, probability + 1.96 * se),
        "failure_count": float(np.sum(failures)),
        "failure_fraction": probability,
        "score_min": float(np.min(scores)),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_p50": float(np.quantile(scores, 0.50)),
        "score_p90": float(np.quantile(scores, 0.90)),
        "score_p95": float(np.quantile(scores, 0.95)),
        "score_p99": float(np.quantile(scores, 0.99)),
        "score_max": float(np.max(scores)),
        "collision_rate": metric_mean("collision"),
        "near_collision_rate": metric_mean("near_collision"),
        "semantic_cutin_rate": metric_mean("is_cutin"),
        "front_cutin_rate": metric_mean("is_front_cutin"),
        "physical_feasible_rate": metric_mean("physical_feasible"),
    }


def _log_mileage_return_period(
    summary: dict[str, Any],
    result: Any,
    logger: logging.Logger,
) -> None:
    """将里程回报周期指标打印到控制台。"""
    mileage = summary.get("mileage_return_period", {})
    if not mileage.get("enabled"):
        return

    probability = float(result.probability)
    target_mode = str(mileage.get("evt_target_mode", "return_period"))
    collision_level = mileage.get("collision_critical_level")
    strict = bool(mileage.get("strict_mileage_interpretation", False))
    strict_note = "" if strict else " [非严格解释]"
    risk_label = str(mileage.get("risk_label", "Y_long_sim"))
    primary_label = str(mileage.get("primary_exposure_label", "following ego"))
    event_type = str(mileage.get("event_type", "following"))

    logger.info("=" * 72)
    logger.info("📊 里程回报周期 (Mileage Return Period) 分析%s", strict_note)
    logger.info("=" * 72)
    logger.info(
        "  子集概率 P(%s > threshold | tail peak): %.6g",
        risk_label,
        probability,
    )
    logger.info(
        "  尾部峰值率:        %.4f /mile | %.4f /km | %.2f /hour",
        mileage["tail_peak_rate_per_mile"],
        mileage.get(
            "tail_peak_rate_per_km",
            mileage["tail_peak_rate_per_mile"] / 1.609344,
        ),
        mileage.get("tail_peak_rate_per_hour", float("nan")),
    )
    logger.info("  ── %s 里程 ──", primary_label)
    logger.info(
        "  safety-critical 强度: %.4g /mile | %.4g /km | %.4g /hour",
        mileage["ads_extreme_risk_intensity_per_mile"],
        mileage["ads_extreme_risk_intensity_per_km"],
        mileage["ads_extreme_risk_intensity_per_hour"],
    )
    logger.info(
        "  回报周期:          %.1f miles | %.1f km | %.1f hours",
        mileage["ads_return_period_miles"],
        mileage["ads_return_period_km"],
        mileage["ads_return_period_hours"],
    )

    all_bg = mileage.get("all_highd_vehicle_background", {})
    primary_uses_all_vehicle = str(mileage.get("exposure_denominator", "")) in {
        "all_vehicle_km",
        "all_vehicle_miles",
    }
    if all_bg and not primary_uses_all_vehicle:
        logger.info("  ── all highD vehicle 背景 ──")
        logger.info(
            "  following ego 占全车比例: %.3f",
            float(
                all_bg.get(
                    "following_ego_mile_fraction_of_all_vehicle_miles",
                    0.0,
                )
            )
        )
        logger.info(
            "  尾部峰值率(全车):  %.4f /mile | %.4f /km | %.2f /hour",
            all_bg["tail_peak_rate_per_all_vehicle_mile"],
            all_bg["tail_peak_rate_per_all_vehicle_km"],
            all_bg["tail_peak_rate_per_all_vehicle_hour"],
        )
        logger.info(
            "  safety-critical 强度(全车): %.4g /mile | %.4g /km | %.4g /hour",
            all_bg["ads_extreme_risk_intensity_per_all_vehicle_mile"],
            all_bg["ads_extreme_risk_intensity_per_all_vehicle_km"],
            all_bg["ads_extreme_risk_intensity_per_all_vehicle_hour"],
        )
        logger.info(
            "  回报周期(全车):    %.1f miles | %.1f km | %.1f hours",
            all_bg["ads_return_period_all_vehicle_miles"],
            all_bg["ads_return_period_all_vehicle_km"],
            all_bg["ads_return_period_all_vehicle_hours"],
        )

    if target_mode == "collision_critical_level" and collision_level is not None:
        logger.info("  ── safety-critical 阈值等效 ──")
        logger.info(
            "  目标阈值:          collision_critical_level = %.6g；"
            "上方 ADS 强度/回报周期即该阈值结果",
            float(collision_level),
        )
        human = mileage.get("human_highd_reference") or {}
        if human:
            logger.info("  ── highD 人类驾驶基线对比 ──")
            logger.info(
                "  highD safety-critical 强度: %.4g /mile | %.4g /km | %.4g /hour",
                float(
                    human.get(
                        "highd_safety_critical_intensity_per_mile",
                        float("nan"),
                    )
                ),
                float(
                    human.get("highd_safety_critical_intensity_per_km", float("nan"))
                ),
                float(
                    human.get(
                        "highd_safety_critical_intensity_per_hour",
                        float("nan"),
                    )
                ),
            )
            logger.info(
                "  highD safety-critical 回报周期: %.1f miles | %.1f km | %.1f hours",
                float(
                    human.get(
                        "highd_safety_critical_return_period_miles",
                        float("nan"),
                    )
                ),
                float(
                    human.get(
                        "highd_safety_critical_return_period_km",
                        float("nan"),
                    )
                ),
                float(
                    human.get(
                        "highd_safety_critical_return_period_hours",
                        float("nan"),
                    )
                ),
            )
            logger.info(
                "  ADS/highD强度比:   %.3f /mile | %.3f /hour",
                float(
                    human.get("ads_to_highd_intensity_ratio_per_mile", float("nan"))
                ),
                float(
                    human.get("ads_to_highd_intensity_ratio_per_hour", float("nan"))
                ),
            )

    failures = mileage.get("strictness_failures", [])
    if failures:
        logger.warning(
            "  严格性检查未通过 (%d): %s",
            len(failures),
            "; ".join(str(f) for f in failures),
        )
    logger.info("=" * 72)


def _finite_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0 or not np.isfinite(denominator):
        return float("nan")
    return float(numerator / denominator)


def _global_risk_exposure_comparison(summary: dict[str, Any]) -> dict[str, Any]:
    """Map the tail-conditional subset estimate to all-vehicle km exposure."""
    mileage = dict(summary.get("mileage_return_period", {}) or {})
    if not mileage.get("enabled"):
        return {
            "enabled": False,
            "reason": "mileage_return_period is disabled",
        }

    all_vehicle = dict(mileage.get("all_highd_vehicle_background", {}) or {})
    probability = float(
        mileage.get(
            "ads_exceedance_probability_conditional",
            summary.get("probability", float("nan")),
        )
    )
    tail_rate_km = float(all_vehicle["tail_peak_rate_per_all_vehicle_km"])
    ads_intensity_km = float(
        all_vehicle["ads_safety_critical_intensity_per_all_vehicle_km"]
    )
    ads_return_km = float(
        all_vehicle["ads_safety_critical_return_period_all_vehicle_km"]
    )
    highd_intensity_km = float(
        all_vehicle["highd_safety_critical_intensity_per_all_vehicle_km"]
    )
    highd_return_km = float(
        all_vehicle["highd_safety_critical_return_period_all_vehicle_km"]
    )
    return {
        "enabled": True,
        "event_type": summary.get("event_type"),
        "failure_event": summary.get("failure_event"),
        "probability_target": summary.get("probability_target"),
        "subset_probability_domain": "highD tail scenario-condition distribution",
        "global_mapping_formula": (
            "ads_safety_critical_intensity_per_all_vehicle_km = "
            "P_ads_failure_given_tail_test_space "
            "* highD_tail_peak_rate_per_all_vehicle_km"
        ),
        "exposure_denominator": "all_vehicle_km",
        "total_all_vehicle_km": float(all_vehicle["total_all_vehicle_km"]),
        "tail_conditional_failure_probability": probability,
        "tail_peak_rate_per_all_vehicle_km": tail_rate_km,
        "ads_safety_critical_intensity_per_all_vehicle_km": ads_intensity_km,
        "ads_safety_critical_return_period_all_vehicle_km": ads_return_km,
        "highd_safety_critical_intensity_per_all_vehicle_km": highd_intensity_km,
        "highd_safety_critical_return_period_all_vehicle_km": highd_return_km,
        "ads_to_highd_intensity_ratio_per_all_vehicle_km": _finite_ratio(
            ads_intensity_km,
            highd_intensity_km,
        ),
        "ads_return_period_over_highd_return_period_all_vehicle_km": _finite_ratio(
            ads_return_km,
            highd_return_km,
        ),
        "strict_global_exposure_interpretation": bool(
            mileage.get("strict_mileage_interpretation", False)
        ),
        "strictness_failures": list(mileage.get("strictness_failures", []) or []),
        "exposure_summary_path": mileage.get("exposure_summary_path"),
    }


def _write_global_risk_exposure_comparison(
    output_dir: Path,
    comparison: dict[str, Any],
) -> None:
    save_json(comparison, output_dir / "global_risk_exposure_comparison.json")
    scalar_row = {
        key: value
        for key, value in comparison.items()
        if not isinstance(value, (dict, list, tuple))
    }
    write_csv(output_dir / "global_risk_exposure_comparison.csv", [scalar_row])


def _summary(
    result,
    contexts: list[dict[str, Any]],
    config: dict[str, Any],
    failure_threshold: float,
    evt_target: dict[str, float],
    level_stats: list[dict[str, float]],
    figures: dict[str, str],
    exposure_summary_path: Path | None,
    input_paths: dict[str, Any],
    simulation_counts: dict[str, int],
    input_space: dict[str, int | list[int]],
) -> dict[str, Any]:
    subset_cfg = config.get("subset_simulation", {})
    uncertainty = _probability_uncertainty(
        result,
        num_samples=subset_cfg.get("num_samples", 100),
        p0=subset_cfg.get("p0", 0.1),
    )
    reliability = _reliability_assessment(
        level_stats,
        config,
        num_contexts=len(contexts),
        num_samples=subset_cfg.get("num_samples", 100),
    )
    target_mode = str(evt_target.get("evt_target_mode", "return_period"))
    event_type = str(config.get("event", {}).get("event_type", "following"))
    source_types = _context_source_types(contexts, event_type=event_type)
    risk_label = "Y_cutin_sim" if event_type == "cut_in" else "Y_long_sim"
    if source_types == {SOURCE_INDEPENDENT_TAIL_PEAK}:
        if target_mode == "collision_critical_level":
            probability_target = (
                f"P_context,z({risk_label} > x_c | o in highD independent tail peaks)"
            )
        else:
            probability_target = (
                f"P_context,z({risk_label} > z_m | o in highD independent tail peaks)"
            )
    elif source_types and source_types.issubset(TAIL_DISTRIBUTION_SOURCE_TYPES):
        if target_mode == "collision_critical_level":
            probability_target = (
                f"P_context,z({risk_label} > x_c | "
                "o sampled from highD tail scenario-condition distribution)"
            )
        else:
            probability_target = (
                f"P_context,z({risk_label} > z_m | "
                "o sampled from highD tail scenario-condition distribution)"
            )
    elif source_types == {"highd_event_tail"}:
        probability_target = (
            f"P_context,z({risk_label} > z_m | o in highD tail contexts)"
        )
    else:
        probability_target = f"P_context,z({risk_label} > z_m | configured contexts)"
    strict_probability = reliability.get("status") == "pass"
    if strict_probability:
        probability_estimate_kind = "standard_subset_estimate"
    else:
        probability_estimate_kind = "low_reliability_standard_estimate"
    mileage_return_period = _mileage_return_period(
        result,
        contexts,
        config,
        evt_target,
        reliability,
        probability_estimate_kind,
        exposure_summary_path,
    )
    return_period = int(evt_target.get("evt_return_period", 100))
    if target_mode == "collision_critical_level":
        failure_event = (
            f"{risk_label} > x_c "
            f"({float(evt_target['evt_return_level_target']):.6g})"
        )
    else:
        failure_event = f"{risk_label} > z{return_period}"
    summary = {
        "event_type": event_type,
        "probability": float(result.probability),
        **uncertainty,
        "policy": _policy_summary(config),
        "input_paths": input_paths,
        "input_space": input_space,
        "simulation_counts": simulation_counts,
        "reliability": reliability,
        "probability_target": probability_target,
        "probability_estimate_kind": probability_estimate_kind,
        "strict_probability_interpretation": strict_probability,
        "mileage_return_period": mileage_return_period,
        "failure_event": failure_event,
        "score_space": config.get("evt", {}).get("score_space", "evt"),
        **evt_target,
        "failure_threshold": failure_threshold,
        "final_failure_fraction": result.final_failure_fraction,
        "thresholds": [level.threshold for level in result.levels],
        "acceptance_rates": [level.acceptance_rate for level in result.levels],
        "level_stats": level_stats,
        "figures": figures,
        "num_levels": len(result.levels),
        "context_sampling_mode": "process_highd_tail_distribution",
        "num_samples": subset_cfg.get("num_samples", 100),
        "p0": subset_cfg.get("p0", 0.1),
        "proposal_std": subset_cfg.get("proposal_std", 0.35),
        "context_refresh_prob": subset_cfg.get("context_refresh_prob", 0.1),
        "mh_retries_per_sample": subset_cfg.get("mh_retries_per_sample", 4),
        "max_levels": int(subset_cfg.get("max_levels", 8)),
        "adaptive_stop_enabled": bool(
            subset_cfg.get("adaptive_stop_enabled", False)
        ),
        "adaptive_stop_min_failure_count": int(
            subset_cfg.get("adaptive_stop_min_failure_count", 20)
        ),
        "adaptive_stop_min_levels": int(
            subset_cfg.get("adaptive_stop_min_levels", 2)
        ),
        "stop_reason": str(getattr(result, "stop_reason", "")),
        "stop_level": int(getattr(result, "stop_level", len(result.levels) - 1)),
        "episode_steps": int(
            config.get(
                "_effective_episode_steps",
                config.get("env", {}).get("episode_steps", 200),
            )
        ),
        "execution_mode": "fixed_horizon",
    }
    return summary


def run_subset_from_config(
    config: dict[str, Any],
    config_dir: Path,
    *,
    expected_event_type: str | None = None,
) -> Path:
    base = Path(config_dir)
    paths = _paths(config, base)
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    failure_threshold, evt_target = _evt_failure_threshold(
        paths["evt_model"],
        config,
        config_dir=base,
        exposure_summary_path=paths.get("exposure_summary"),
    )
    sampler = FrozenDiffusionSampler.from_config(config, config_dir=base)
    event_type = str(
        sampler.prior.schema.get(
            "event_type",
            config.get("event", {}).get("event_type", "following"),
        )
    )
    if expected_event_type is not None and event_type != expected_event_type:
        raise ValueError(
            f"Expected {expected_event_type} diffusion prior/config, got {event_type}"
        )
    contexts = _load_contexts(
        paths["tail_contexts"],
        paths["condition_distribution"],
        config,
        event_type=event_type,
    )
    _apply_evt_target_to_context_metadata(contexts, evt_target)
    _validate_context_schema(contexts, sampler, paths["tail_contexts"])
    _apply_shared_idm_ego_config(config, base, event_type=event_type)
    if event_type == "cut_in":
        runner = ClosedLoopCutInRunner(sampler, config)
    else:
        runner = ClosedLoopFollowingRunner(sampler, config)
    evaluator = LatentMpcEpisodeEvaluator(
        sampler,
        runner,
        contexts,
        config,
        inference_steps=int(
            config.get("sampling", {}).get("eval_diffusion_steps", 100)
        ),
    )
    config["_effective_episode_steps"] = int(evaluator.episode_steps)
    subset_cfg = config.get("subset_simulation", {})
    target_label = (
        "x_c"
        if str(evt_target.get("evt_target_mode")) == "collision_critical_level"
        else "z_m"
    )
    logger.info(
        (
            "Running mixed-context subset simulation joint_space="
            "tail_condition_distribution_x_diffusion_noise "
            "samples=%d p0=%.3f max_levels=%d threshold=%.6f "
            "%s=%.6f latent_shape=%s proposal_std=%.3f "
            "context_refresh_prob=%.3f mh_retries=%d"
        ),
        subset_cfg.get("num_samples", 100),
        subset_cfg.get("p0", 0.1),
        subset_cfg.get("max_levels", 8),
        failure_threshold,
        target_label,
        evt_target["evt_return_level_target"],
        evaluator.latent_shape,
        subset_cfg.get("proposal_std", 0.35),
        subset_cfg.get("context_refresh_prob", 0.1),
        subset_cfg.get("mh_retries_per_sample", 4),
    )
    population_evaluator = _multiprocess_population_evaluator(
        evaluator,
        sampler,
        config,
    )
    with ExitStack() as stack:
        evaluate_many = None
        if population_evaluator is not None:
            evaluate_many = stack.enter_context(population_evaluator).evaluate_many
        result = run_subset_simulation(
            evaluator.evaluate,
            context_count=evaluator.context_count,
            latent_shape=evaluator.latent_shape,
            num_samples=subset_cfg.get("num_samples", 100),
            p0=subset_cfg.get("p0", 0.1),
            max_levels=subset_cfg.get("max_levels", 8),
            proposal_std=subset_cfg.get("proposal_std", 0.35),
            context_refresh_prob=subset_cfg.get("context_refresh_prob", 0.1),
            failure_threshold=failure_threshold,
            seed=config.get("training", {}).get("seed", 42),
            mh_retries_per_sample=subset_cfg.get("mh_retries_per_sample", 4),
            evaluate_many=evaluate_many,
            adaptive_stop_enabled=bool(
                subset_cfg.get("adaptive_stop_enabled", False)
            ),
            adaptive_stop_min_failure_count=int(
                subset_cfg.get("adaptive_stop_min_failure_count", 20)
            ),
            adaptive_stop_min_levels=int(
                subset_cfg.get("adaptive_stop_min_levels", 2)
            ),
        )
    _save_samples(result, output_dir, contexts)
    level_stats = _level_stats(result, failure_threshold)
    _write_level_stats(output_dir / "latent_subset_level_stats.csv", level_stats)
    reliability = _reliability_assessment(
        level_stats,
        config,
        num_contexts=len(contexts),
        num_samples=int(subset_cfg.get("num_samples", 100)),
    )
    message = (
        "Subset reliability %s at level %d | unique_contexts=%.0f "
        "unique_states=%.0f largest_context_share=%.3f "
        "largest_state_share=%.3f acceptance_rate=%s"
    )
    observed = reliability.get("observed", {})
    acceptance = observed.get("acceptance_rate")
    acceptance_text = (
        f"{float(acceptance):.3f}"
        if isinstance(acceptance, (int, float)) and np.isfinite(float(acceptance))
        else "nan"
    )
    log_fn = logger.info if reliability["status"] == "pass" else logger.warning
    log_fn(
        message,
        reliability["status"],
        reliability.get("assessed_level", -1),
        float(observed.get("unique_contexts", np.nan)),
        float(observed.get("unique_states", np.nan)),
        float(observed.get("largest_context_share", np.nan)),
        float(observed.get("largest_state_share", np.nan)),
        acceptance_text,
    )
    if reliability.get("failures"):
        logger.warning(
            "Subset reliability failures: %s",
            "; ".join(str(item) for item in reliability["failures"]),
        )
    if reliability.get("warnings"):
        logger.warning(
            "Subset reliability warnings: %s",
            "; ".join(str(item) for item in reliability["warnings"]),
        )
    figures: dict[str, str] = {}
    summary = _summary(
        result,
        contexts,
        config,
        failure_threshold,
        evt_target,
        level_stats,
        figures,
        paths.get("exposure_summary"),
        _input_paths_summary(config, base, paths, sampler),
        _subset_simulation_counts(result),
        _input_space_summary(evaluator),
    )
    exposure_comparison = _global_risk_exposure_comparison(summary)
    summary["global_risk_exposure_comparison"] = exposure_comparison
    _write_global_risk_exposure_comparison(output_dir, exposure_comparison)
    save_json(summary, output_dir / "latent_subset_summary.json")
    counts = summary["simulation_counts"]
    logger.info(
        (
            "Subset actual simulated scenario count: "
            "closed_loop_evaluations=%d stored_level_samples=%d "
            "unique_context_indices_all_levels=%d "
            "unique_context_indices_final_level=%d"
        ),
        counts["closed_loop_evaluations"],
        counts["stored_level_samples"],
        counts["unique_context_indices_all_levels"],
        counts["unique_context_indices_final_level"],
    )

    # ── 里程回报周期控制台打印 ──
    _log_mileage_return_period(summary, result, logger)
    save_json(
        _top_cases(result, contexts),
        output_dir / "latent_subset_top_cases.json",
    )
    return output_dir / "latent_subset_summary.json"


def run_monte_carlo_from_config(
    config: dict[str, Any],
    config_dir: Path,
    *,
    expected_event_type: str | None = None,
) -> Path:
    base = Path(config_dir)
    paths = _paths(config, base)
    mc_cfg = config.get("monte_carlo", {})
    output_value = mc_cfg.get("output_dir")
    if output_value:
        output_dir = resolve_path(str(output_value), base)
    else:
        output_dir = paths["output_dir"] / "monte_carlo"
    output_dir.mkdir(parents=True, exist_ok=True)

    failure_threshold, evt_target = _evt_failure_threshold(
        paths["evt_model"],
        config,
        config_dir=base,
        exposure_summary_path=paths.get("exposure_summary"),
    )
    sampler = FrozenDiffusionSampler.from_config(config, config_dir=base)
    event_type = str(
        sampler.prior.schema.get(
            "event_type",
            config.get("event", {}).get("event_type", "following"),
        )
    )
    if expected_event_type is not None and event_type != expected_event_type:
        raise ValueError(
            f"Expected {expected_event_type} diffusion prior/config, got {event_type}"
        )
    contexts = _load_contexts(
        paths["tail_contexts"],
        paths["condition_distribution"],
        config,
        event_type=event_type,
    )
    _apply_evt_target_to_context_metadata(contexts, evt_target)
    _validate_context_schema(contexts, sampler, paths["tail_contexts"])
    _apply_shared_idm_ego_config(config, base, event_type=event_type)
    if event_type == "cut_in":
        runner = ClosedLoopCutInRunner(sampler, config)
    else:
        runner = ClosedLoopFollowingRunner(sampler, config)

    evaluator = LatentMpcEpisodeEvaluator(
        sampler,
        runner,
        contexts,
        config,
        inference_steps=int(
            config.get("sampling", {}).get("eval_diffusion_steps", 100)
        ),
    )
    config["_effective_episode_steps"] = int(evaluator.episode_steps)
    subset_cfg = config.get("subset_simulation", {})
    num_samples = int(mc_cfg.get("num_samples", subset_cfg.get("num_samples", 100)))
    if num_samples <= 0:
        raise ValueError("monte_carlo.num_samples must be positive")
    seed = int(mc_cfg.get("seed", config.get("training", {}).get("seed", 42)))
    rng = np.random.default_rng(seed)
    context_indices = rng.integers(
        0,
        int(evaluator.context_count),
        size=num_samples,
        dtype=np.int64,
    )
    latents = rng.standard_normal((num_samples, *evaluator.latent_shape)).astype(
        np.float32
    )
    logger.info(
        (
            "Running latent Monte Carlo baseline joint_space="
            "tail_condition_distribution_x_diffusion_noise "
            "samples=%d threshold=%.6f latent_shape=%s"
        ),
        num_samples,
        failure_threshold,
        evaluator.latent_shape,
    )
    population_evaluator = _multiprocess_population_evaluator(
        evaluator,
        sampler,
        config,
    )
    with ExitStack() as stack:
        if population_evaluator is not None:
            scores, _actions, metrics, _traces = stack.enter_context(
                population_evaluator
            ).evaluate_many(context_indices, latents, 0)
        else:
            scores, metrics = [], []
            interval = _progress_interval(num_samples)
            start_time = time.monotonic()
            for idx, latent in enumerate(latents):
                context_index = int(context_indices[idx])
                result = evaluator.evaluate(context_index, latent)
                scores.append(float(result.score))
                item_metrics = dict(result.metrics)
                item_metrics["context_index"] = float(context_index)
                metrics.append(item_metrics)
                done = idx + 1
                if done == num_samples or done % interval == 0:
                    _log_progress(
                        "Monte Carlo baseline",
                        done,
                        num_samples,
                        start_time,
                    )
            scores = np.asarray(scores, dtype=np.float64)

    stats = _monte_carlo_stats(scores, metrics, failure_threshold)
    figures: dict[str, str] = {}
    _save_monte_carlo_samples(
        output_dir,
        context_indices=context_indices,
        latents=latents,
        scores=scores,
        metrics=metrics,
        failure_threshold=failure_threshold,
    )
    write_csv(output_dir / "latent_monte_carlo_stats.csv", [stats])
    save_json(
        _monte_carlo_top_cases(contexts, context_indices, scores, metrics),
        output_dir / "latent_monte_carlo_top_cases.json",
    )

    risk_label = "Y_cutin_sim" if event_type == "cut_in" else "Y_long_sim"
    target_mode = str(evt_target.get("evt_target_mode", "return_period"))
    if target_mode == "collision_critical_level":
        failure_event = (
            f"{risk_label} > x_c "
            f"({float(evt_target['evt_return_level_target']):.6g})"
        )
        probability_event = f"{risk_label} > x_c"
    else:
        return_period = int(evt_target.get("evt_return_period", 100))
        failure_event = f"{risk_label} > z{return_period}"
        probability_event = f"{risk_label} > z_m"
    summary = {
        "estimator": "independent_monte_carlo",
        "event_type": event_type,
        "input_paths": _input_paths_summary(config, base, paths, sampler),
        "input_space": _input_space_summary(evaluator),
        "simulation_counts": _monte_carlo_simulation_counts(context_indices),
        "probability_target": (
            f"P_context,z({probability_event} | o sampled from highD tail "
            "scenario-condition distribution)"
        ),
        "probability": float(stats["probability"]),
        "probability_standard_error": float(stats["probability_standard_error"]),
        "probability_ci95_lower": float(stats["probability_ci95_lower"]),
        "probability_ci95_upper": float(stats["probability_ci95_upper"]),
        "policy": _policy_summary(config),
        "failure_event": failure_event,
        "score_space": config.get("evt", {}).get("score_space", "evt"),
        **evt_target,
        "failure_threshold": float(failure_threshold),
        "num_samples": int(num_samples),
        "seed": int(seed),
        "context_sampling_mode": "process_highd_tail_distribution",
        "latent_shape": list(evaluator.latent_shape),
        "latent_dimension": int(np.prod(evaluator.latent_shape, dtype=np.int64)),
        "scenario_condition_dimension": int(
            sampler.prior.model.denoiser.cfg.scenario_condition_dim
        ),
        "joint_condition_noise_dimension": int(
            np.prod(evaluator.latent_shape, dtype=np.int64)
            + sampler.prior.model.denoiser.cfg.scenario_condition_dim
        ),
        "episode_steps": int(config["_effective_episode_steps"]),
        "figures": figures,
        "stats": stats,
    }
    save_json(summary, output_dir / "latent_monte_carlo_summary.json")
    counts = summary["simulation_counts"]
    logger.info(
        (
            "Monte Carlo baseline finished probability %.8g failures=%.0f/%d "
            "closed_loop_evaluations=%d unique_context_indices=%d"
        ),
        stats["probability"],
        stats["failure_count"],
        num_samples,
        counts["closed_loop_evaluations"],
        counts["unique_context_indices"],
    )
    return output_dir / "latent_monte_carlo_summary.json"
