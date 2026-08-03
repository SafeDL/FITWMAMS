#!/usr/bin/env python3
"""Render the final QR-WM training figure from CSV and TensorBoard records."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STAGE_STYLE = {
    "buffer_warmup": ("缓存预热（1 秒）", "#dceef8"),
    "closed_loop": ("闭环训练（3 秒）", "#e3f2df"),
    "full_refinement": ("全程精炼（5 秒）", "#fbe8dc"),
}
SYSTEM_CJK_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def _font_family() -> str:
    """Register the host CJK font because Conda Matplotlib may not index it."""
    if SYSTEM_CJK_FONT.is_file():
        font_manager.fontManager.addfont(str(SYSTEM_CJK_FONT))
        return font_manager.FontProperties(fname=SYSTEM_CJK_FONT).get_name()
    return "DejaVu Sans"


def _rows(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    rows: list[dict[str, float | str]] = []
    for row in raw:
        parsed: dict[str, float | str] = {}
        for name, value in row.items():
            if name == "stage":
                parsed[name] = value
            else:
                try:
                    parsed[name] = float(value)
                except (TypeError, ValueError):
                    parsed[name] = float("nan")
        rows.append(parsed)
    return rows


def _batch_losses(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load every recorded batch loss, preferring the newest value for duplicate steps."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return np.empty(0), np.empty(0)
    values: dict[int, float] = {}
    for path in sorted(directory.glob("events.out.tfevents.*"), key=lambda item: item.stat().st_mtime):
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        if "batch/train/loss" not in accumulator.Tags().get("scalars", []):
            continue
        values.update({item.step: item.value for item in accumulator.Scalars("batch/train/loss")})
    if not values:
        return np.empty(0), np.empty(0)
    steps = np.asarray(sorted(values), dtype=np.int64)
    return steps, np.asarray([values[int(step)] for step in steps], dtype=np.float64)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    weights = np.full(window, 1.0 / window)
    return np.convolve(values, weights, mode="valid")


def _stage_ranges(rows: list[dict[str, float | str]]) -> list[tuple[str, float, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["stage"])].append(float(row["epoch"]))
    return [(name, min(epochs) - 0.5, max(epochs) + 0.5) for name, epochs in grouped.items()]


def _shade_stages(axis, ranges: list[tuple[str, float, float]]) -> None:
    for stage, start, end in ranges:
        label, color = STAGE_STYLE.get(stage, (stage, "#eeeeee"))
        axis.axvspan(start, end, color=color, alpha=0.7, label=label)
        axis.axvline(end, color="#777777", linestyle="--", linewidth=0.8)


def _stage_segments(rows: list[dict[str, float | str]]) -> dict[str, list[dict[str, float | str]]]:
    groups: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        groups[str(row["stage"])].append(row)
    return groups


def plot(history: Path, summary: Path, tensorboard_dir: Path, output: Path, window: int) -> None:
    rows = _rows(history)
    if not rows:
        raise ValueError(f"empty training history: {history}")
    summary_data = json.loads(summary.read_text(encoding="utf-8"))
    stages, ranges = _stage_segments(rows), _stage_ranges(rows)
    epochs = np.asarray([row["epoch"] for row in rows], dtype=float)
    train_loss = np.asarray([row["train_loss"] for row in rows], dtype=float)
    validation_loss = np.asarray([row["val_loss"] for row in rows], dtype=float)
    best_fde = float(summary_data["best_validation_fde"])
    full_rows = stages.get("full_refinement", [])
    if not full_rows:
        raise ValueError("training history has no full_refinement rows")
    best_row = min(full_rows, key=lambda row: float(row["selection_metric"]))
    batch_steps, batch_loss = _batch_losses(tensorboard_dir)

    plt.rcParams.update({"font.family": _font_family(), "axes.unicode_minus": False, "font.size": 11})
    figure, axes = plt.subplots(2, 2, figsize=(14.1, 8.61), dpi=180)
    figure.suptitle(
        f"QR 世界模型训练曲线（已完成 {int(summary_data['epochs_completed'])}/{int(summary_data['configured_epochs'])} 个 epoch）",
        fontsize=18,
        fontweight="bold",
    )

    axis = axes[0, 0]
    _shade_stages(axis, ranges)
    axis.plot(epochs, train_loss, "o-", color="#1f77b4", label="训练损失")
    axis.plot(epochs, validation_loss, "o-", color="#8f24b9", label="验证损失")
    axis.set(title="按课程阶段的训练目标", xlabel="epoch", ylabel="加权目标")
    axis.legend(loc="best")
    axis.grid(alpha=0.25)

    axis = axes[0, 1]
    _shade_stages(axis, ranges)
    for stage, stage_rows in stages.items():
        stage_epochs = [float(row["epoch"]) for row in stage_rows]
        fde = [float(row["selection_metric"]) for row in stage_rows]
        label = STAGE_STYLE.get(stage, (stage, ""))[0]
        axis.plot(stage_epochs, fde, "o-", linewidth=1.8, label=label)
    axis.scatter(float(best_row["epoch"]), best_fde, marker="*", s=260, color="#d62728", zorder=5,
                 label=f"最佳 5 秒 FDE：{best_fde:.4f} m（epoch {int(best_row['epoch'])}）")
    axis.set(title="验证 FDE（不同阶段的时域独立）", xlabel="epoch", ylabel="FDE（m）")
    axis.legend(loc="upper left")
    axis.grid(alpha=0.25)

    axis = axes[1, 0]
    refinement_epochs = np.asarray([row["epoch"] for row in full_rows], dtype=float)
    refinement_train = np.asarray([row["train_loss"] for row in full_rows], dtype=float)
    refinement_validation = np.asarray([row["val_loss"] for row in full_rows], dtype=float)
    refinement_fde = np.asarray([row["selection_metric"] for row in full_rows], dtype=float)
    loss_train, = axis.plot(refinement_epochs, refinement_train, "o-", color="#1f77b4", label="训练损失")
    loss_validation, = axis.plot(refinement_epochs, refinement_validation, "o-", color="#8f24b9", label="验证损失")
    fde_axis = axis.twinx()
    fde_line, = fde_axis.plot(refinement_epochs, refinement_fde, "s--", color="#d35428", label="验证 FDE")
    fde_axis.scatter(float(best_row["epoch"]), best_fde, marker="*", s=180, color="#d62728", zorder=5)
    axis.set(title="5 秒全程精炼阶段", xlabel="epoch", ylabel="加权目标")
    fde_axis.set_ylabel("FDE（m）", color="#d35428")
    axis.legend([loss_train, loss_validation, fde_line], [item.get_label() for item in (loss_train, loss_validation, fde_line)], loc="upper left")
    axis.grid(alpha=0.25)

    axis = axes[1, 1]
    if len(batch_steps):
        axis.plot(batch_steps, batch_loss, color="#9fb3c8", alpha=0.35, linewidth=0.55, label="batch 损失")
        smooth = _rolling_mean(batch_loss, window)
        offset = max(window - 1, 0)
        axis.plot(batch_steps[offset:], smooth, color="#d35428", linewidth=1.8, label=f"{window}-batch 移动平均")
        axis.set(title=f"TensorBoard 记录的 batch 损失（{len(batch_steps):,} 条）", xlabel="全局 batch step", ylabel="加权目标")
        axis.legend(loc="best")
    else:
        axis.text(0.5, 0.5, "未找到 batch TensorBoard 记录", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
    axis.grid(alpha=0.25)

    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=ROOT / "results/highd_world_model/qr_world_model/training_history.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "results/highd_world_model/qr_world_model/training_summary.json")
    parser.add_argument("--tensorboard-dir", type=Path, default=ROOT / "results/highd_world_model/qr_world_model/tensorboard")
    parser.add_argument("--output", type=Path, default=ROOT / "results/highd_world_model/qr_world_model/current_training_curves.png")
    parser.add_argument("--moving-average-window", type=int, default=128)
    args = parser.parse_args()
    if args.moving_average_window < 1:
        raise ValueError("--moving-average-window must be positive")
    plot(args.history, args.summary, args.tensorboard_dir, args.output, args.moving_average_window)
    print(args.output)


if __name__ == "__main__":
    main()
