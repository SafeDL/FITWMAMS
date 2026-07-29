"""Shared configuration materialization for world-model command-line tools."""
from __future__ import annotations

from copy import deepcopy
import csv
import logging
from pathlib import Path
from typing import Iterable

import yaml

from .utils import load_yaml, resolve_path

logger = logging.getLogger(__name__)

def materialize_config(
    template_path: Path,
    output_dir: Path,
    *,
    config_name: str,
    resolve_path_keys: Iterable[str] = (),
    drop_path_keys: Iterable[str] = (),
) -> tuple[dict, Path]:
    """Copy a config into an isolated output directory with resolved inputs."""
    template_path, output_dir = template_path.resolve(), output_dir.resolve()
    config = deepcopy(load_yaml(template_path))
    paths = dict(config.get("paths", {}))
    for key in drop_path_keys:
        paths.pop(key, None)
    for key in resolve_path_keys:
        paths[key] = str(resolve_path(paths[key], base=template_path.parent))
    paths["output_dir"] = str(output_dir)
    config["paths"] = paths
    config_path = output_dir / "configs" / config_name
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config, config_path


def plot_training_losses(history_path: Path, output_path: Path) -> Path | None:
    """Write a train/validation loss curve when a completed training history exists."""
    if not history_path.exists():
        logger.warning("training history not found; skipped loss plot: %s", history_path)
        return None
    with history_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    points = [
        (float(row["epoch"]), float(row["train_loss"]), float(row["val_loss"]))
        for row in rows if row.get("epoch") and row.get("train_loss") and row.get("val_loss")
    ]
    if not points:
        logger.warning("training history has no train_loss/val_loss rows: %s", history_path)
        return None
    import matplotlib.pyplot as plt

    epochs, train_loss, val_loss = zip(*points)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(epochs, train_loss, label="train_loss")
    axis.plot(epochs, val_loss, label="val_loss")
    axis.set(xlabel="epoch", ylabel="loss", title="Training and validation loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path
