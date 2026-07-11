"""Evaluation utilities for the naturalness discriminator."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diffusion.src.data import SPLIT_TO_INDEX
from diffusion.src.utils import load_json, save_json, select_device, set_seed

from .data import load_discriminator_dataset
from .model import build_discriminator_from_schema, naturalness_guidance_objective
from .train import _binary_metrics, _source_metrics

logger = logging.getLogger(__name__)


def _resolve_output_dir(config: dict, config_dir: str | Path | None) -> Path:
    base = Path(config_dir).resolve() if config_dir is not None else Path.cwd()
    return (base / config.get("paths", {}).get("output_dir", "../../../data/diffusion_natural/following/discriminator")).resolve()


def _resolve_natural_dir(config: dict, config_dir: str | Path | None) -> Path:
    base = Path(config_dir).resolve() if config_dir is not None else Path.cwd()
    return (base / config.get("paths", {}).get("natural_dataset_dir", "../../../data/diffusion_natural/following")).resolve()


def _resolve_checkpoint(path: str | None, output_dir: Path) -> Path:
    p = Path(path or "checkpoints/best_auc.pt")
    if p.is_absolute():
        return p
    cwd = p.resolve()
    if cwd.exists():
        return cwd
    return (output_dir / p).resolve()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


@torch.no_grad()
def _predict(model, arrays: dict[str, np.ndarray], split: str, batch_size: int, device: torch.device) -> dict[str, np.ndarray]:
    idx = np.where(arrays["split_index"] == SPLIT_TO_INDEX[split])[0]
    logits: list[np.ndarray] = []
    for start in range(0, len(idx), batch_size):
        sub = idx[start:start + batch_size]
        out = model(
            torch.from_numpy(arrays["context_states"][sub]).float().to(device),
            torch.from_numpy(arrays["context_features"][sub]).float().to(device),
            torch.from_numpy(arrays["relative_history"][sub]).float().to(device),
            torch.from_numpy(arrays["future_action_features"][sub]).float().to(device),
            torch.from_numpy(arrays["summary_features"][sub]).float().to(device),
        )
        logits.append(out.detach().cpu().numpy())
    logit = np.concatenate(logits, axis=0)
    return {
        "index": idx,
        "logits": logit,
        "scores": 1.0 / (1.0 + np.exp(-logit)),
        "labels": arrays["labels"][idx],
        "source_type": arrays["source_type"][idx].astype(str),
    }


def _write_source_csv(path: Path, labels: np.ndarray, scores: np.ndarray, sources: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_type", "count", "label_mean", "mean_score", "accept_rate", "reject_rate"])
        writer.writeheader()
        for source in sorted(np.unique(sources).tolist()):
            mask = sources == source
            writer.writerow(
                {
                    "source_type": source,
                    "count": int(np.sum(mask)),
                    "label_mean": float(np.mean(labels[mask])),
                    "mean_score": float(np.mean(scores[mask])),
                    "accept_rate": float(np.mean(scores[mask] >= 0.5)),
                    "reject_rate": float(np.mean(scores[mask] < 0.5)),
                }
            )


def _write_plots(
    output_dir: Path,
    config: dict,
    labels: np.ndarray,
    scores: np.ndarray,
    sources: np.ndarray,
    summary_features_raw: np.ndarray,
) -> list[str]:
    plot_dir = output_dir / str(config.get("evaluation", {}).get("plot_dir", "discriminator_plots"))
    plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
    except Exception as exc:  # noqa: BLE001
        logger.warning("Plot dependencies unavailable; skipping plots: %s", exc)
        return []
    written: list[Path] = []

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for source in sorted(np.unique(sources).tolist()):
        mask = sources == source
        ax.hist(scores[mask], bins=50, alpha=0.5, density=True, label=source)
    ax.set_xlabel("Dpsi score")
    ax.set_ylabel("density")
    ax.set_title("Score Distribution by Source")
    ax.legend(fontsize=8)
    path = plot_dir / "score_distribution_by_source.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    if len(np.unique(labels)) > 1:
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        RocCurveDisplay.from_predictions(labels, scores, ax=ax)
        path = plot_dir / "roc_curve.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        PrecisionRecallDisplay.from_predictions(labels, scores, ax=ax)
        path = plot_dir / "pr_curve.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    prob_true, prob_pred = calibration_curve(labels, scores, n_bins=10, strategy="quantile")
    ax.plot(prob_pred, prob_true, marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("mean predicted score")
    ax.set_ylabel("fraction positive")
    ax.set_title("Calibration")
    path = plot_dir / "calibration_curve.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    diagnostics = [
        ("score_vs_jerk.png", summary_features_raw[:, 5], "max_abs_jerk"),
        ("score_vs_min_ax.png", summary_features_raw[:, 0], "min_ax"),
        ("score_vs_speed_min.png", summary_features_raw[:, 6], "speed_min"),
    ]
    for filename, x, xlabel in diagnostics:
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        ax.scatter(x, scores, s=5, alpha=0.25, c=labels, cmap="coolwarm")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Dpsi score")
        path = plot_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)
    return [str(p) for p in written]


def _write_example_plots(output_dir: Path, config: dict, arrays: dict[str, np.ndarray], pred: dict[str, np.ndarray]) -> list[str]:
    plot_dir = output_dir / str(config.get("evaluation", {}).get("plot_dir", "discriminator_plots"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    written: list[Path] = []
    labels = pred["labels"]
    scores = pred["scores"]
    idx = pred["index"]
    candidates = [
        ("example_false_positive.png", np.where((labels < 0.5) & (scores >= 0.5))[0]),
        ("example_false_negative.png", np.where((labels > 0.5) & (scores < 0.5))[0]),
    ]
    for filename, positions in candidates:
        if len(positions) == 0:
            continue
        pos = int(positions[np.argmax(scores[positions]) if "positive" in filename else np.argmin(scores[positions])])
        sample_idx = idx[pos]
        feature = arrays["future_action_features"][sample_idx]
        fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
        ax.plot(feature)
        ax.set_title(f"{filename}: score={scores[pos]:.3f}, source={pred['source_type'][pos]}")
        ax.set_xlabel("future step")
        ax.set_ylabel("normalized feature")
        path = plot_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)
    return [str(p) for p in written]


def _differentiability_check(
    model,
    config: dict,
    schema: dict,
    discriminator_stats: dict,
    output_dir: Path,
    config_dir: str | Path | None,
    device: torch.device,
) -> dict[str, Any]:
    natural_dir = _resolve_natural_dir(config, config_dir)
    raw = _load_npz(natural_dir / "dataset.npz")
    norm = _load_npz(natural_dir / "dataset_normalized.npz")
    stage1_schema = load_json(natural_dir / "feature_schema.json")
    stage1_stats = load_json(natural_dir / "normalization_stats.json")
    split = str(config.get("evaluation", {}).get("split", "test"))
    idx = np.where(raw["split_index"] == SPLIT_TO_INDEX[split])[0]
    if len(idx) == 0:
        return {"passed": False, "reason": f"no Stage 1 samples for split={split}"}
    i = idx[: min(8, len(idx))]
    future_actions = torch.from_numpy(norm["actions"][i]).float().to(device)
    future_actions.requires_grad_(True)
    objective = naturalness_guidance_objective(
        model,
        torch.from_numpy(norm["context_states"][i]).float().to(device),
        torch.from_numpy(norm["context_features"][i]).float().to(device),
        torch.from_numpy(norm["relative_history"][i]).float().to(device),
        future_actions,
        ego_length=torch.from_numpy(raw["ego_length"][i]).float().to(device),
        adv_length=torch.from_numpy(raw["adv_length"][i]).float().to(device),
        schema=stage1_schema,
        config=config,
        discriminator_stats=discriminator_stats,
        stage1_stats=stage1_stats,
        inputs_normalized=True,
        actions_normalized=True,
    )
    loss = -objective.mean()
    model.zero_grad(set_to_none=True)
    loss.backward()
    passed = future_actions.grad is not None and bool(torch.isfinite(future_actions.grad).all())
    result = {
        "passed": bool(passed),
        "grad_abs_mean": float(future_actions.grad.abs().mean().detach().cpu()) if future_actions.grad is not None else 0.0,
        "num_checked": int(len(i)),
        "checkpoint_dir": str(output_dir / "checkpoints"),
    }
    return result


def evaluate_discriminator(
    config: dict,
    *,
    config_dir: str | Path | None = None,
    checkpoint: str | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    output_dir = _resolve_output_dir(config, config_dir)
    schema = load_json(output_dir / "discriminator_schema.json")
    stats = load_json(output_dir / "discriminator_stats.json")
    arrays = load_discriminator_dataset(output_dir)
    eval_cfg = config.get("evaluation", {})
    set_seed(int(eval_cfg.get("seed", config.get("training", {}).get("seed", 42))))
    device = select_device(config.get("training", {}).get("device", "auto"))
    model = build_discriminator_from_schema(schema, config).to(device)
    checkpoint_path = _resolve_checkpoint(checkpoint, output_dir)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    split_name = str(split or eval_cfg.get("split", "test"))
    pred = _predict(model, arrays, split_name, int(config.get("training", {}).get("batch_size", 256)), device)
    metrics = _binary_metrics(pred["labels"], pred["scores"])
    source_metrics = _source_metrics(pred["labels"], pred["scores"], pred["source_type"])
    summary_norm = stats["summary_features"]
    summary_raw = arrays["summary_features"][pred["index"]] * np.asarray(summary_norm["std"], dtype=np.float32) + np.asarray(summary_norm["mean"], dtype=np.float32)
    plots = _write_plots(output_dir, config, pred["labels"], pred["scores"], pred["source_type"], summary_raw)
    plots.extend(_write_example_plots(output_dir, config, arrays, pred))
    _write_source_csv(output_dir / "source_wise_metrics.csv", pred["labels"], pred["scores"], pred["source_type"])
    grad_check = _differentiability_check(model, config, schema, stats, output_dir, config_dir, device)
    summary = {
        "checkpoint": str(checkpoint_path),
        "split": split_name,
        "num_samples": int(len(pred["labels"])),
        "metrics": metrics,
        "source_wise_metrics": source_metrics,
        "differentiability_check": grad_check,
        "plots": plots,
    }
    save_json(summary, output_dir / "discriminator_eval_summary.json")
    return summary
