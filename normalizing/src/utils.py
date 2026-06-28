"""Small utilities for the highD tail normalizing-flow module."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=json_default)


def load_json(path: str | Path) -> Any:
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def resolve_path(value: str | Path, *, base: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (Path(base).resolve() / path).resolve()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ModuleNotFoundError:
        pass


def select_device(name: str = "auto"):
    import torch

    pref = str(name or "auto").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if pref != "auto":
        raise ValueError(f"Unsupported device setting: {name}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_local_nflows_to_path(repo_root: str | Path) -> Path:
    """Make the bundled nflows reference implementation importable."""
    import sys

    nflows_root = Path(repo_root).resolve() / "ref_code" / "nflows-master"
    if not nflows_root.exists():
        raise FileNotFoundError(f"Bundled nflows tree not found: {nflows_root}")
    if str(nflows_root) not in sys.path:
        sys.path.insert(0, str(nflows_root))
    return nflows_root


def repo_root_from_file(path: str | Path) -> Path:
    p = Path(path).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "process_highD").exists() and (candidate / "ref_code").exists():
            return candidate
    raise RuntimeError(f"Could not infer repository root from {path}")

