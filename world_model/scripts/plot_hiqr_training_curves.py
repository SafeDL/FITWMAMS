#!/usr/bin/env python3
"""Render a live HiQR-WM training figure from its TensorBoard event files.

The HiQR trainer deliberately writes TensorBoard scalars after every batch and
after every completed epoch.  This script merges resumed-run event files by
step (newer files win) so it can be run while training is still in progress.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.hiqr.config import HiQRWorldModelConfig  # noqa: E402


SYSTEM_CJK_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
STAGE_STYLE = {
    "start_warmup": ("START 预热（1 秒）", "#dceef8"),
    "closed_loop": ("闭环训练（3 秒）", "#e3f2df"),
    "full_hierarchical": ("全程分层交互（5.96 秒）", "#fbe8dc"),
}
LOSS_WEIGHTS = {
    "position": "position_weight",
    "velocity": "velocity_weight",
    "action": "action_weight",
    "plan_position": "plan_position_weight",
    "plan_action": "plan_action_weight",
    "continuation": "continuation_weight",
    "gate": "gate_weight",
    "interaction": "interaction_weight",
    "physical": "physical_weight",
    "jerk": "jerk_weight",
    "lane": "lane_weight",
    "gap_ttc": "gap_ttc_weight",
    "scene_kl": "scene_kl_weight",
    "agent_kl": "agent_kl_weight",
    "diversity_floor": "diversity_weight",
}


def _font_family() -> str:
    """Register the host CJK font because Conda Matplotlib may not index it."""
    if SYSTEM_CJK_FONT.is_file():
        font_manager.fontManager.addfont(str(SYSTEM_CJK_FONT))
        return font_manager.FontProperties(fname=SYSTEM_CJK_FONT).get_name()
    return "DejaVu Sans"


def _scalar_values(directory: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a scalar from every event file, preferring newer duplicates."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as error:
        raise RuntimeError("TensorBoard is required to read HiQR training logs") from error

    values: dict[int, float] = {}
    event_files = sorted(directory.glob("events.out.tfevents.*"), key=lambda path: path.stat().st_mtime)
    for path in event_files:
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        values.update({event.step: event.value for event in accumulator.Scalars(tag)})
    if not values:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    steps = np.asarray(sorted(values), dtype=np.int64)
    return steps, np.asarray([values[int(step)] for step in steps], dtype=np.float64)


def _value_map(directory: Path, tag: str) -> dict[int, float]:
    steps, values = _scalar_values(directory, tag)
    return dict(zip(steps.tolist(), values.tolist(), strict=True))


def _load_config(path: Path) -> dict[str, Any]:
    """Return the effective model and curriculum settings used for the run."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    model = asdict(HiQRWorldModelConfig())
    model.update(raw.get("model", {}))
    return {"model": model, "training": raw.get("training", {})}


def _validation_loss(
    directory: Path, epochs: np.ndarray, model_config: dict[str, Any]
) -> np.ndarray:
    """Reconstruct validation weighted loss from the logged raw objective terms."""
    term_maps = {
        term: _value_map(directory, f"epoch/validation/{term}")
        for term in LOSS_WEIGHTS
    }
    losses: list[float] = []
    for epoch in epochs:
        values = [
            model_config[weight_name] * term_maps[term].get(int(epoch), float("nan"))
            for term, weight_name in LOSS_WEIGHTS.items()
        ]
        losses.append(float(np.sum(values)) if all(np.isfinite(values)) else float("nan"))
    return np.asarray(losses, dtype=np.float64)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    return np.convolve(values, np.full(window, 1.0 / window), mode="valid")


def _stage_ranges(stages: list[dict[str, Any]]) -> list[tuple[str, float, float]]:
    ranges: list[tuple[str, float, float]] = []
    next_epoch = 1
    for stage in stages:
        epochs = int(stage["epochs"])
        if epochs > 0:
            ranges.append((str(stage["name"]), next_epoch - 0.5, next_epoch + epochs - 0.5))
        next_epoch += epochs
    return ranges


def _shade_stages(axis, ranges: list[tuple[str, float, float]]) -> None:
    for name, start, end in ranges:
        label, color = STAGE_STYLE.get(name, (name, "#eeeeee"))
        axis.axvspan(start, end, color=color, alpha=0.70, label=label)
        axis.axvline(end, color="#777777", linestyle="--", linewidth=0.8)


def _full_stage_epochs(
    rollout: dict[int, float], stages: list[dict[str, Any]]
) -> set[int]:
    """Find epochs evaluated at the longest curriculum rollout."""
    longest = max((float(stage.get("rollout_seconds", 0.0)) for stage in stages), default=0.0)
    return {
        epoch
        for epoch, seconds in rollout.items()
        if np.isclose(seconds, longest, rtol=0.0, atol=1e-5)
    }


def plot(
    tensorboard_dir: Path,
    config_path: Path,
    output: Path,
    window: int,
) -> None:
    config = _load_config(config_path)
    training = config["training"]
    stages = list(training.get("stages", []))
    ranges = _stage_ranges(stages)
    configured_steps, configured_values = _scalar_values(
        tensorboard_dir, "training/configured_epochs"
    )
    configured_epochs = int(configured_values[-1]) if len(configured_steps) else int(training.get("epochs", 0))

    epochs, train_loss = _scalar_values(tensorboard_dir, "epoch/train/loss")
    if not len(epochs):
        raise ValueError(f"no completed HiQR epochs found in {tensorboard_dir}")
    validation_loss = _validation_loss(tensorboard_dir, epochs, config["model"])
    fde = _value_map(tensorboard_dir, "selection/validation_fde_m")
    rollout = _value_map(tensorboard_dir, "training/rollout_seconds")
    batch_steps, batch_loss = _scalar_values(tensorboard_dir, "batch/train/loss")
    full_epochs = _full_stage_epochs(rollout, stages)
    full_fde = [(epoch, fde[epoch]) for epoch in sorted(full_epochs & fde.keys())]
    best_epoch, best_fde = min(full_fde, key=lambda item: item[1]) if full_fde else (None, None)

    plt.rcParams.update({"font.family": _font_family(), "axes.unicode_minus": False, "font.size": 11})
    figure, axes = plt.subplots(2, 2, figsize=(14.1, 8.61), dpi=180)
    current_step = int(batch_steps[-1]) if len(batch_steps) else 0
    figure.suptitle(
        f"HIQR 世界模型训练曲线（已完成 {int(epochs[-1])}/{configured_epochs} 个 epoch；最新 batch step {current_step:,}）",
        fontsize=18,
        fontweight="bold",
    )

    axis = axes[0, 0]
    _shade_stages(axis, ranges)
    axis.plot(epochs, train_loss, "o-", color="#1f77b4", label="训练加权损失")
    axis.plot(epochs, validation_loss, "o-", color="#8f24b9", label="验证加权损失（重构）")
    axis.set(title="按课程阶段的训练目标", xlabel="epoch", ylabel="加权目标")
    axis.legend(loc="best")
    axis.grid(alpha=0.25)

    axis = axes[0, 1]
    _shade_stages(axis, ranges)
    for stage, start, end in ranges:
        stage_points = [(epoch, fde[epoch]) for epoch in sorted(fde) if start <= epoch <= end]
        if not stage_points:
            continue
        stage_epochs, stage_fde = zip(*stage_points, strict=True)
        label = STAGE_STYLE.get(stage, (stage, ""))[0]
        axis.plot(stage_epochs, stage_fde, "o-", linewidth=1.8, label=label)
    if best_epoch is not None and best_fde is not None:
        axis.scatter(best_epoch, best_fde, marker="*", s=260, color="#d62728", zorder=5,
                     label=f"最佳 5.96 秒 FDE：{best_fde:.4f} m（epoch {best_epoch}）")
    axis.set(title="验证 FDE（不同课程阶段的时域独立）", xlabel="epoch", ylabel="FDE（m）")
    axis.legend(loc="upper left")
    axis.grid(alpha=0.25)

    axis = axes[1, 0]
    for term, label, color in (
        ("position", "位置", "#1f77b4"),
        ("interaction", "交互", "#d35428"),
        ("agent_kl", "Agent KL", "#2ca02c"),
        ("gap_ttc", "Gap/TTC", "#8f24b9"),
    ):
        values = _value_map(tensorboard_dir, f"epoch/validation/{term}")
        points = [(epoch, values[epoch]) for epoch in epochs if int(epoch) in values]
        if points:
            term_epochs, term_values = zip(*points, strict=True)
            axis.plot(term_epochs, term_values, "o-", linewidth=1.6, label=label, color=color)
    axis.set(title="关键验证子目标", xlabel="epoch", ylabel="原始损失项")
    axis.legend(loc="best")
    axis.grid(alpha=0.25)

    axis = axes[1, 1]
    axis.plot(batch_steps, batch_loss, color="#9fb3c8", alpha=0.35, linewidth=0.55, label="batch 损失")
    smooth = _rolling_mean(batch_loss, window)
    offset = max(window - 1, 0)
    axis.plot(batch_steps[offset:], smooth, color="#d35428", linewidth=1.8,
              label=f"{window}-batch 移动平均")
    axis.set(title=f"实时 batch 损失（{len(batch_steps):,} 条）", xlabel="全局 batch step", ylabel="加权目标")
    axis.legend(loc="best")
    axis.grid(alpha=0.25)

    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensorboard-dir", type=Path,
                        default=ROOT / "results/highd_world_model/hiqr_world_model/tensorboard")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "results/highd_world_model/hiqr_world_model/configs/highd_hiqr_world_model_gpu96.yaml")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/highd_world_model/hiqr_world_model/current_training_curves.png")
    parser.add_argument("--moving-average-window", type=int, default=256)
    args = parser.parse_args()
    if args.moving_average_window < 1:
        raise ValueError("--moving-average-window must be positive")
    plot(args.tensorboard_dir, args.config, args.output, args.moving_average_window)
    print(args.output)


if __name__ == "__main__":
    main()
