"""Small content-addressed provenance records for experiment inputs."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path
import platform

import gymnasium
import numpy as np
import pandas as pd
import scipy
import stable_baselines3
import torch


def sha256_file(path: str | Path) -> str:
    value = Path(path).resolve()
    digest = hashlib.sha256()
    with value.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_record(path: str | Path | None) -> dict[str, object] | None:
    """Describe an immutable file input by resolved path, size and digest."""
    if path is None:
        return None
    value = Path(path).resolve()
    if not value.is_file():
        raise FileNotFoundError(value)
    return {
        "path": str(value),
        "bytes": int(value.stat().st_size),
        "sha256": sha256_file(value),
    }


def software_versions() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": None if not torch.backends.cudnn.is_available() else str(torch.backends.cudnn.version()),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "gymnasium": gymnasium.__version__,
        "highway_env": version("highway-env"),
    }
