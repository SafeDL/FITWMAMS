"""Formal-release configuration and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

FORMAL_PROTOCOL = "hierarchical_world_model"
RANDOMNESS_NAMESPACE = {
    "world": "world_rng",
    "training": "training_rng",
    "evaluation": "evaluation_rng",
}
STAGED_TRAINING_GATES = {
    "probe_count": 128,
    "trajectory_diversity_min_m": 0.02,
    # Keep a non-degeneracy floor without making a historical diagnostic an
    # impossible stochastic-stage gate.
    "terminal_diversity_min_m": 0.02,
    "relative_ks_limit_ratio": 1.10,
    "base_factual_fde_fallback_weight": 0.25,
}
def long_horizon_constraint(flow_schema: dict[str, Any]) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Extract the declared Flow knot contract; never infer it from array length."""
    contract = flow_schema.get("long_horizon_constraint")
    if not isinstance(contract, dict):
        raise ValueError("Flow schema is missing long_horizon_constraint")
    if "knot_frames" not in contract or "knot_times_s" not in contract:
        raise ValueError("Flow schema must declare knot_frames and knot_times_s")
    frames = tuple(int(x) for x in contract["knot_frames"])
    times = tuple(float(x) for x in contract["knot_times_s"])
    if len(frames) != len(times) or not frames:
        raise ValueError("Flow knot frame/time contract is empty or inconsistent")
    return frames, times


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_protocol_config(path: str | Path) -> dict[str, Any]:
    """Load the sole formal config, resolving only repo-root logical paths."""
    config = deepcopy(load_yaml(path))
    paths = config.get("paths", {})
    root = repo_root()
    for name, value in paths.items():
        if value is None:
            continue
        candidate = Path(value)
        if candidate.is_absolute():
            raise ValueError(f"formal config paths.{name} must be repo-root-relative")
        paths[name] = str((root / candidate).resolve())
    config["paths"] = paths
    return config


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def logical_path(path: str | Path) -> str:
    value = Path(path).resolve()
    try:
        return str(value.relative_to(repo_root()))
    except ValueError as exc:
        raise ValueError(f"formal artifact lies outside repository: {value}") from exc


def release_provenance(*, release_tag: str | None = None, require_clean: bool = False) -> dict[str, Any]:
    root = repo_root()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    if require_clean and dirty:
        raise RuntimeError("formal release requires a clean worktree")
    if release_tag:
        tagged = subprocess.check_output(["git", "rev-list", "-n", "1", release_tag], cwd=root, text=True).strip()
        if tagged != commit:
            raise RuntimeError(f"release tag {release_tag!r} does not point at HEAD")
    return {"code_commit": commit, "release_tag": release_tag, "worktree_clean_at_start": not dirty}


def environment_provenance(lockfile: str | Path | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"python": sys.version, "platform": platform.platform()}
    try:
        import numpy
        import torch
        result.update({"numpy": numpy.__version__, "torch": torch.__version__, "cuda": torch.version.cuda})
    except ImportError:
        pass
    lock_path = Path(lockfile) if lockfile is not None else None
    if lock_path is not None and lock_path.is_file():
        result["dependency_lockfile"] = logical_path(lock_path)
        result["dependency_lockfile_sha256"] = file_sha256(lock_path)
    return result
