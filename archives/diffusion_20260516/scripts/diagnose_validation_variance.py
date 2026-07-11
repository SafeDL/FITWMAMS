#!/usr/bin/env python3
"""Diagnose validation metric variance for a fixed diffusion checkpoint."""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from diffusion.src.data import SPLIT_TO_INDEX
from diffusion.src.model import GaussianActionDiffusion, build_model_from_schema
from diffusion.src.utils import load_json, load_yaml, save_json, select_device, set_seed, setup_logging


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "diffusion_following.yaml"
DEFAULT_CHECKPOINT_PATH = "checkpoints/best.pt"
DEFAULT_DATASET_PATH = "dataset_normalized.npz"
DEFAULT_SCHEMA_PATH = "feature_schema.json"
DEFAULT_STATS_PATH = "normalization_stats.json"
DEFAULT_SPLIT = "val"
DEFAULT_MAX_BATCHES = 0
DEFAULT_MAX_SAMPLES = 1024
DEFAULT_NUM_REPEATS = 10
DEFAULT_FIXED_TIMESTEPS = "0,25,50,75,99"
DEFAULT_BATCH_SIZE = 0
DEFAULT_NUM_WORKERS = None
DEFAULT_DEVICE = None
DEFAULT_SEED = None
METRIC_KEYS = ("loss", "noise_mse", "x0_l1", "smooth")
logger = logging.getLogger(__name__)


def _resolve_output_dir(config: dict, config_dir: Path) -> Path:
    return (config_dir / config.get("paths", {}).get("output_dir", "../../../data/diffusion/following")).resolve()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _resolve_input_path(value: str | None, output_dir: Path, default: str) -> Path:
    raw = value or default
    path = Path(raw)
    if path.is_absolute():
        return path
    cwd_path = path.resolve()
    if cwd_path.exists():
        return cwd_path
    return (output_dir / path).resolve()


def _parse_timesteps(value: str | None, num_steps: int) -> list[int]:
    if value is None or str(value).strip() == "":
        raw = [0, 25, 50, 75, 99]
    else:
        raw = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    out: list[int] = []
    for t in raw:
        clipped = min(max(int(t), 0), int(num_steps) - 1)
        if clipped not in out:
            out.append(clipped)
    return out


def _make_subset_loader(
    arrays: dict[str, np.ndarray],
    split: str,
    batch_size: int,
    num_workers: int,
    max_samples: int,
) -> tuple[DataLoader, int]:
    mask_idx = np.where(arrays["split_index"] == SPLIT_TO_INDEX[split])[0]
    if len(mask_idx) == 0:
        raise RuntimeError(f"No samples for split={split}")
    if max_samples > 0:
        mask_idx = mask_idx[: int(max_samples)]

    relative = arrays.get("relative_history")
    if relative is None:
        relative = np.zeros((arrays["context_states"].shape[0], arrays["context_states"].shape[1], 6), dtype=np.float32)
    risk_condition = arrays.get("risk_condition")
    if risk_condition is None:
        risk_condition = arrays["risk"].reshape(-1, 1)

    tensors = (
        torch.from_numpy(arrays["context_states"][mask_idx]).float(),
        torch.from_numpy(arrays["context_features"][mask_idx]).float(),
        torch.from_numpy(relative[mask_idx]).float(),
        torch.from_numpy(risk_condition[mask_idx]).float(),
        torch.from_numpy(arrays["actions"][mask_idx]).float(),
    )
    loader = DataLoader(
        TensorDataset(*tensors),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=torch.cuda.is_available(),
    )
    return loader, int(len(mask_idx))


def _random_losses(
    model: GaussianActionDiffusion,
    actions: torch.Tensor,
    history: torch.Tensor,
    context: torch.Tensor,
    relative: torch.Tensor,
    risk_condition: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return model.p_losses(actions, history, context, relative, risk_condition)


def _fixed_timestep_losses(
    model: GaussianActionDiffusion,
    actions: torch.Tensor,
    history: torch.Tensor,
    context: torch.Tensor,
    relative: torch.Tensor,
    risk_condition: torch.Tensor,
    timestep: int,
) -> dict[str, torch.Tensor]:
    t = torch.full((actions.shape[0],), int(timestep), device=actions.device, dtype=torch.long)
    noise = torch.randn_like(actions)
    noisy = model.q_sample(actions, t, noise)
    pred = model.denoiser(noisy, t, history, context, relative, risk_condition)
    noise_mse = F.mse_loss(pred, noise)
    x0 = model.predict_start_from_noise(noisy, t, pred)
    x0_l1 = F.l1_loss(x0, actions)
    if x0.shape[1] > 1:
        smooth = torch.mean(torch.abs(x0[:, 1:] - x0[:, :-1]))
    else:
        smooth = torch.zeros((), device=actions.device, dtype=actions.dtype)
    loss = noise_mse + model.denoiser.cfg.x0_weight * x0_l1 + model.denoiser.cfg.smooth_weight * smooth
    return {
        "loss": loss,
        "noise_mse": noise_mse.detach(),
        "x0_l1": x0_l1.detach(),
        "smooth": smooth.detach(),
    }


@torch.no_grad()
def _evaluate_loader(
    model: GaussianActionDiffusion,
    loader: DataLoader,
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], dict[str, torch.Tensor]],
    max_batches: int,
) -> dict[str, Any]:
    totals: dict[str, float] = {}
    total_n = 0
    total_batches = 0
    for batch_idx, (history, context, relative, risk_condition, actions) in enumerate(loader, start=1):
        if max_batches > 0 and batch_idx > max_batches:
            break
        history = history.to(device, non_blocking=True)
        context = context.to(device, non_blocking=True)
        relative = relative.to(device, non_blocking=True)
        risk_condition = risk_condition.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        losses = loss_fn(actions, history, context, relative, risk_condition)
        n = int(actions.shape[0])
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * n
        total_n += n
        total_batches += 1
    if total_n == 0:
        raise RuntimeError("No validation samples were evaluated; check max_batches/max_samples.")
    metrics = {key: totals.get(key, 0.0) / total_n for key in METRIC_KEYS}
    metrics["num_samples"] = int(total_n)
    metrics["num_batches"] = int(total_batches)
    return metrics


def _metric_summary(values: list[float]) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    cv: float | None
    if abs(mean) > 1e-12:
        cv = float(std / abs(mean))
    else:
        cv = 0.0 if std == 0.0 else None
    return {
        "mean": mean,
        "std": std,
        "cv": cv,
        "var": float(np.var(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    return {metric: _metric_summary([float(run[metric]) for run in runs]) for metric in METRIC_KEYS}


def _write_csv(
    path: Path,
    random_runs: list[dict[str, Any]],
    random_summary: dict[str, dict[str, float | None]],
    fixed_runs: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_repeats = len(random_runs)
    fieldnames = [
        "mode",
        "timestep",
        "metric",
        "metric_mean",
        "metric_std",
        "metric_cv",
        "metric_var",
        "metric_min",
        "metric_max",
        "num_evals",
        "num_samples",
        "num_batches",
    ] + [f"repeat_{i}" for i in range(1, max_repeats + 1)]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metric in METRIC_KEYS:
            stats = random_summary[metric]
            row: dict[str, Any] = {
                "mode": "random",
                "timestep": "",
                "metric": metric,
                "metric_mean": stats["mean"],
                "metric_std": stats["std"],
                "metric_cv": stats["cv"],
                "metric_var": stats["var"],
                "metric_min": stats["min"],
                "metric_max": stats["max"],
                "num_evals": len(random_runs),
                "num_samples": random_runs[0]["num_samples"] if random_runs else 0,
                "num_batches": random_runs[0]["num_batches"] if random_runs else 0,
            }
            for i, run in enumerate(random_runs, start=1):
                row[f"repeat_{i}"] = run[metric]
            writer.writerow(row)
        for run in fixed_runs:
            for metric in METRIC_KEYS:
                value = float(run[metric])
                writer.writerow(
                    {
                        "mode": "fixed_timestep",
                        "timestep": int(run["timestep"]),
                        "metric": metric,
                        "metric_mean": value,
                        "metric_std": 0.0,
                        "metric_cv": 0.0,
                        "metric_var": 0.0,
                        "metric_min": value,
                        "metric_max": value,
                        "num_evals": 1,
                        "num_samples": int(run["num_samples"]),
                        "num_batches": int(run["num_batches"]),
                    }
                )


def diagnose(config: dict, config_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if int(args.num_repeats) < 1:
        raise ValueError("--num-repeats must be >= 1")
    if int(args.max_batches) < 0 or int(args.max_samples) < 0:
        raise ValueError("--max-batches and --max-samples must be >= 0")

    output_dir = _resolve_output_dir(config, config_dir)
    checkpoint_path = _resolve_input_path(args.checkpoint_path, output_dir, DEFAULT_CHECKPOINT_PATH)
    dataset_path = _resolve_input_path(args.dataset_path, output_dir, DEFAULT_DATASET_PATH)
    schema_path = _resolve_input_path(args.schema_path, output_dir, DEFAULT_SCHEMA_PATH)
    stats_path = _resolve_input_path(args.stats_path, output_dir, DEFAULT_STATS_PATH)
    result_dir = Path(args.output_dir).resolve() if args.output_dir else output_dir

    schema = load_json(schema_path)
    stats = load_json(stats_path)
    arrays = _load_npz(dataset_path)

    training_cfg = config.get("training", {})
    batch_size = int(args.batch_size or training_cfg.get("batch_size", 256))
    num_workers = int(args.num_workers if args.num_workers is not None else training_cfg.get("num_workers", 0))
    device = select_device(args.device or training_cfg.get("device", "auto"))
    model = build_model_from_schema(schema, config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    loader, selected_samples = _make_subset_loader(
        arrays,
        str(args.split),
        batch_size,
        num_workers,
        int(args.max_samples),
    )
    fixed_timesteps = _parse_timesteps(args.fixed_timesteps, model.num_steps)
    seed = int(args.seed if args.seed is not None else config.get("evaluation", {}).get("seed", training_cfg.get("seed", 42)))

    random_runs: list[dict[str, Any]] = []
    for repeat in range(1, int(args.num_repeats) + 1):
        set_seed(seed + repeat - 1)
        run = _evaluate_loader(
            model,
            loader,
            device,
            lambda actions, history, context, relative, risk: _random_losses(
                model, actions, history, context, relative, risk
            ),
            int(args.max_batches),
        )
        run["repeat"] = repeat
        random_runs.append(run)
        logger.info(
            "random repeat=%d loss=%.6f noise_mse=%.6f x0_l1=%.6f smooth=%.6f",
            repeat,
            run["loss"],
            run["noise_mse"],
            run["x0_l1"],
            run["smooth"],
        )

    fixed_runs: list[dict[str, Any]] = []
    for offset, timestep in enumerate(fixed_timesteps):
        set_seed(seed + 10_000 + offset)
        run = _evaluate_loader(
            model,
            loader,
            device,
            lambda actions, history, context, relative, risk, t=timestep: _fixed_timestep_losses(
                model, actions, history, context, relative, risk, t
            ),
            int(args.max_batches),
        )
        run["timestep"] = int(timestep)
        fixed_runs.append(run)
        logger.info(
            "fixed timestep=%d loss=%.6f noise_mse=%.6f x0_l1=%.6f smooth=%.6f",
            timestep,
            run["loss"],
            run["noise_mse"],
            run["x0_l1"],
            run["smooth"],
        )

    random_summary = _summarize_runs(random_runs)
    fixed_by_timestep = {}
    for run in fixed_runs:
        row = {metric: float(run[metric]) for metric in METRIC_KEYS}
        row["num_samples"] = int(run["num_samples"])
        row["num_batches"] = int(run["num_batches"])
        fixed_by_timestep[str(int(run["timestep"]))] = row
    fixed_flat_metrics = {
        f"{metric}_at_t{int(run['timestep'])}": float(run[metric])
        for run in fixed_runs
        for metric in METRIC_KEYS
    }
    summary: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "schema_path": str(schema_path),
        "normalization_stats_path": str(stats_path),
        "config_path": str(Path(args.config).resolve() if args.config else DEFAULT_CONFIG_PATH),
        "split": str(args.split),
        "selected_samples_before_max_batches": int(selected_samples),
        "max_batches": int(args.max_batches),
        "max_samples": int(args.max_samples),
        "batch_size": int(batch_size),
        "num_repeats": int(args.num_repeats),
        "seed": int(seed),
        "diffusion_steps": int(model.num_steps),
        "schema_action_representation": schema.get("action_representation"),
        "normalization_stats_keys": sorted(stats.keys()),
        "random_validation": {
            "runs": random_runs,
            "metrics": random_summary,
        },
        "fixed_timestep_validation": {
            "timesteps": fixed_timesteps,
            "runs": fixed_runs,
            "metrics_by_timestep": fixed_by_timestep,
            "flat_metrics": fixed_flat_metrics,
        },
    }

    summary_path = result_dir / "validation_variance_summary.json"
    csv_path = result_dir / "validation_variance.csv"
    save_json(summary, summary_path)
    _write_csv(csv_path, random_runs, random_summary, fixed_runs)
    logger.info("Wrote %s", summary_path)
    logger.info("Wrote %s", csv_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path.")
    parser.add_argument(
        "--checkpoint-path",
        default=DEFAULT_CHECKPOINT_PATH,
        help="Checkpoint .pt path. Relative paths are resolved from cwd if present, otherwise from config output_dir.",
    )
    parser.add_argument(
        "--dataset-path",
        default=DEFAULT_DATASET_PATH,
        help="dataset_normalized.npz path. Relative paths are resolved from cwd if present, otherwise from config output_dir.",
    )
    parser.add_argument(
        "--schema-path",
        default=DEFAULT_SCHEMA_PATH,
        help="feature_schema.json path. Relative paths are resolved from cwd if present, otherwise from config output_dir.",
    )
    parser.add_argument(
        "--stats-path",
        default=DEFAULT_STATS_PATH,
        help="normalization_stats.json path. Relative paths are resolved from cwd if present, otherwise from config output_dir.",
    )
    parser.add_argument("--output-dir", default=None, help="Directory for validation_variance_summary.json/csv; defaults to output_dir.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=sorted(SPLIT_TO_INDEX.keys()))
    parser.add_argument("--max-batches", type=int, default=DEFAULT_MAX_BATCHES, help="Stop after this many batches; 0 means all selected samples.")
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES, help="Use the first N split samples; 0 means all split samples.")
    parser.add_argument("--num-repeats", type=int, default=DEFAULT_NUM_REPEATS, help="Random timestep/noise validation repeats.")
    parser.add_argument(
        "--fixed-timesteps",
        default=DEFAULT_FIXED_TIMESTEPS,
        help="Comma-separated timesteps for fixed-t validation, clipped to diffusion steps.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Override training.batch_size; 0 uses config.")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Override training.num_workers.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="cpu/cuda/auto; defaults to training.device.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base seed for reproducible diagnostic repeats; defaults to config seed.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg_path = Path(args.config).resolve()
    diagnose(load_yaml(cfg_path), cfg_path.parent, args)


if __name__ == "__main__":
    main()
