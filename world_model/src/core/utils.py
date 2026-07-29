"""Small IO, path, device and reproducibility helpers."""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def resolve_path(value: str | Path, *, base: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (Path(base) / path).resolve()


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # noqa: BLE001 - torch may not be installed outside tread.
        return


def select_device(requested: str | None = None):
    import torch

    value = str(requested or "auto").lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def finite_mean_std(values: np.ndarray, valid: np.ndarray, min_std: float = 1.0e-3) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=np.float32)
    m = np.asarray(valid, dtype=bool) & np.isfinite(x)
    if x.ndim < 2:
        raise ValueError("finite_mean_std expects values with at least 2 dimensions")
    flat = x.reshape(-1, x.shape[-1])
    flat_valid = m.reshape(-1, x.shape[-1])
    mean = np.zeros(x.shape[-1], dtype=np.float32)
    std = np.ones(x.shape[-1], dtype=np.float32)
    for dim in range(x.shape[-1]):
        dim_values = flat[flat_valid[:, dim], dim]
        if len(dim_values) == 0:
            continue
        mean[dim] = float(np.mean(dim_values))
        dim_std = float(np.std(dim_values))
        std[dim] = dim_std if dim_std > min_std else 1.0
    return mean, std


def normalize_with_mask(values: np.ndarray, valid: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    out = np.zeros_like(x, dtype=np.float32)
    m = np.asarray(valid, dtype=bool)
    normed = (x - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    out[m] = normed[m]
    return out
