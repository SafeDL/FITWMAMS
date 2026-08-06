#!/usr/bin/env python3
"""Render isolated HiQR-v2 TensorBoard curves while training is running."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


STAGES = {
    "start_warmup": ("START prior warm-up", "#dceef8"),
    "closed_loop": ("closed loop", "#e3f2df"),
    "full_prior_rollout": ("5.96 s prior rollout", "#fbe8dc"),
}
PREFIX = "hiqr_v2/"


def _scalars(directory: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    values: dict[int, float] = {}
    for path in sorted(
        directory.glob("events.out.tfevents.*"), key=lambda item: item.stat().st_mtime
    ):
        events = EventAccumulator(str(path), size_guidance={"scalars": 0})
        events.Reload()
        if tag in events.Tags().get("scalars", []):
            values.update({event.step: event.value for event in events.Scalars(tag)})
    steps = np.asarray(sorted(values), np.int64)
    return steps, np.asarray([values[int(step)] for step in steps], np.float64)


def _mapping(directory: Path, tag: str) -> dict[int, float]:
    step, value = _scalars(directory, tag)
    return dict(zip(step.tolist(), value.tolist(), strict=True))


def _shade(axis, stages: list[dict]) -> None:
    epoch = 1
    for stage in stages:
        width = int(stage["epochs"])
        name = str(stage["name"])
        label, color = STAGES.get(name, (name, "#eeeeee"))
        axis.axvspan(
            epoch - 0.5, epoch + width - 0.5, color=color, alpha=0.7, label=label
        )
        axis.axvline(epoch + width - 0.5, color="#777", linestyle="--", linewidth=0.8)
        epoch += width


def _series(axis, directory: Path, tag: str, label: str, color: str) -> None:
    values = _mapping(directory, tag)
    if values:
        step = np.asarray(sorted(values))
        axis.plot(
            step,
            [values[int(item)] for item in step],
            "o-",
            label=label,
            color=color,
            linewidth=1.6,
        )


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    return (
        values
        if len(values) < window
        else np.convolve(values, np.ones(window) / window, mode="valid")
    )


def plot(tensorboard: Path, config: Path, output: Path, window: int) -> None:
    raw = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    stages = list(raw.get("training", {}).get("stages", []))
    epochs, train = _scalars(tensorboard, PREFIX + "epoch/train/loss")
    if not len(epochs):
        raise ValueError(f"no completed HiQR-v2 epochs found in {tensorboard}")
    configured = _mapping(tensorboard, PREFIX + "training/configured_epochs")
    configured_epochs = (
        int(list(configured.values())[-1])
        if configured
        else int(raw["training"]["epochs"])
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 8.6), dpi=180)
    figure.suptitle(
        f"HiQR-v2 training ({int(epochs[-1])}/{configured_epochs} epochs)",
        fontsize=17,
        fontweight="bold",
    )

    axis = axes[0, 0]
    _shade(axis, stages)
    axis.plot(epochs, train, "o-", color="#1f77b4", label="train objective")
    _series(
        axis,
        tensorboard,
        PREFIX + "epoch/validation/loss",
        "prior validation objective",
        "#8f24b9",
    )
    axis.set(
        xlabel="epoch", ylabel="objective", title="Prior-driven training objective"
    )
    axis.legend()
    axis.grid(alpha=0.25)

    axis = axes[0, 1]
    _shade(axis, stages)
    fde = _mapping(tensorboard, PREFIX + "selection/validation_fde_m")
    seconds = _mapping(tensorboard, PREFIX + "training/rollout_seconds")
    full = [
        (epoch, fde[epoch])
        for epoch in sorted(fde)
        if np.isclose(seconds.get(epoch, -1), 5.96)
    ]
    if full:
        x, y = zip(*full, strict=True)
        axis.plot(x, y, "o-", color="#d35428", label="5.96 s deterministic prior FDE")
        best = min(full, key=lambda item: item[1])
        axis.scatter(
            *best, marker="*", s=220, color="#c0392b", label=f"best {best[1]:.3f} m"
        )
    axis.set(xlabel="epoch", ylabel="FDE (m)", title="Fixed-cohort selection metric")
    axis.legend()
    axis.grid(alpha=0.25)

    axis = axes[1, 0]
    for term, label, color in (
        ("position", "prior position", "#1f77b4"),
        ("posterior_position", "posterior auxiliary", "#2ca02c"),
        ("gate", "gate", "#d35428"),
        ("prior_agent_std", "residual std", "#8f24b9"),
    ):
        _series(axis, tensorboard, PREFIX + f"epoch/validation/{term}", label, color)
    axis.set(xlabel="epoch", ylabel="raw value", title="Attribution diagnostics")
    axis.legend()
    axis.grid(alpha=0.25)

    axis = axes[1, 1]
    step, loss = _scalars(tensorboard, PREFIX + "batch/train/loss")
    axis.plot(
        step, loss, color="#9fb3c8", alpha=0.35, linewidth=0.55, label="batch objective"
    )
    if len(loss):
        smooth = _smooth(loss, window)
        axis.plot(
            step[max(window - 1, 0) :],
            smooth,
            color="#d35428",
            linewidth=1.8,
            label=f"{window}-batch mean",
        )
    axis.set(xlabel="global batch", ylabel="objective", title="Live batch loss")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=ROOT / "results/highd_world_model/hiqr_v2_world_model/tensorboard",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "world_model/scripts/configs/highd_hiqr_v2_world_model.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results/highd_world_model/hiqr_v2_world_model/training_curves.png",
    )
    parser.add_argument("--moving-average-window", type=int, default=256)
    args = parser.parse_args()
    if args.moving_average_window < 1:
        raise ValueError("--moving-average-window must be positive")
    plot(args.tensorboard_dir, args.config, args.output, args.moving_average_window)
    print(args.output)


if __name__ == "__main__":
    main()
