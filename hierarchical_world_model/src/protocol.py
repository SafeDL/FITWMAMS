"""Formal-release configuration and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import math
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
RANDOMNESS_ABLATION = {
    "samples": 16,
    "motion_seed_offset": 100_000,
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
RANDOMNESS_ABLATION_GATES = {
    "energy_improvement_min_fraction": 0.05,
    "trajectory_pairwise_min_m": 0.02,
    "terminal_pairwise_min_m": 0.05,
    "speed_ax_degradation_max_ratio": 0.10,
    "windowed_jerk_degradation_max_ratio": 0.10,
    "response_response_floor_min_movement": 0.001,
}
SAMPLED_END_TO_END = {
    "schema": "sampled_end_to_end",
    "worlds": 1024,
    "response_steps": 149,
    "knot_frames": (50, 100, 149),
    "knot_times_s": (2.0, 4.0, 5.96),
}
AMS_READINESS = {
    "worlds": 2,
    "steps": 64,
    "seed": 20260823,
}
AMS_READINESS_GATES = {
    "required_true": (
        "formal_checkpoint_config_match",
        "world_serialization_exact",
        "same_world_same_ads_exact",
        "snapshot_restore_exact",
        "branch_changes_ego_trajectory",
        "evt_score_monotone_on_calibration_probe",
    ),
    "finite_state_rate_min": 1.0,
    "finite_evt_score_rate_min": 1.0,
}
ACCEPTANCE_GATES = {
    "protocol": FORMAL_PROTOCOL,
    "factual_limits_m": {
        "ADE_m": 0.06,
        "FDE_m": 0.06,
        "P95_displacement_error_m": 0.12,
    },
    "intervention": {
        "direction_success_rate_min": 0.95,
        "dose_monotonicity_min": 0.95,
        "separation_non_decrease_max": 0.15,
        "separation_non_decrease_min": 0.90,
        "response_latency_min_s": 0.04,
    },
}


def check_ams_readiness_gate(
    readiness: dict[str, Any], *, gates: dict[str, Any] | None = None
) -> bool:
    """Return whether the AMS readiness report satisfies its required payload gate."""
    criteria = gates or AMS_READINESS_GATES
    required = criteria.get("required_true", ())
    if not all(bool(readiness.get(name)) for name in required):
        return False
    if float(readiness.get("finite_state_rate", 0.0)) < float(
        criteria["finite_state_rate_min"]
    ):
        return False
    if float(readiness.get("finite_evt_score_rate", 0.0)) < float(
        criteria["finite_evt_score_rate_min"]
    ):
        return False
    return True


def check_formal_manifest_gate(
    manifest: dict[str, Any],
    *,
    protocol: str | None = None,
) -> bool:
    """Return whether a final manifest declares a valid formal protocol contract."""
    target = protocol or ACCEPTANCE_GATES["protocol"]
    return (
        bool(manifest.get("protocol") == target)
        and bool(manifest.get("checkpoint_sha256"))
        and bool(manifest.get("code_commit"))
        and bool(manifest.get("worktree_clean_at_start"))
    )


def check_factual_fidelity_gate(metrics: dict[str, Any], *, limits: dict[str, float] | None = None) -> bool:
    """Return whether sampled factual metrics satisfy the formal acceptance limits."""
    budget = limits or ACCEPTANCE_GATES["factual_limits_m"]
    return all(metrics.get(key, float("inf")) <= value for key, value in budget.items())


def check_intervention_gate(
    effects: dict[str, Any], *, thresholds: dict[str, float] | None = None
) -> bool:
    """Return whether intervention metrics meet strict responsiveness gate."""
    table = thresholds or ACCEPTANCE_GATES["intervention"]
    return (
        effects["brake"]["direction_success_rate"] >= table["direction_success_rate_min"]
        and effects["accelerate"]["direction_success_rate"] >= table["direction_success_rate_min"]
        and effects["brake"]["dose_monotonicity_rate"] >= table["dose_monotonicity_min"]
        and effects["accelerate"]["dose_monotonicity_rate"] >= table["dose_monotonicity_min"]
        and effects["left"]["separation_non_decrease_rate"] >= table["separation_non_decrease_min"]
        and all(
            value["locality_ratio_far_to_near"] < table["separation_non_decrease_max"]
            for value in effects.values()
        )
        and all(
            value["response_latency_s"] >= table["response_latency_min_s"]
            for value in effects.values()
        )
    )


def check_sampled_end_to_end_gate(sampled: dict[str, Any], *, worlds: int = SAMPLED_END_TO_END["worlds"],
                                 response_steps: int = SAMPLED_END_TO_END["response_steps"]) -> bool:
    """Return whether sampled E2E report has the required minimum payload."""
    risks = sampled.get("ADS_conditioned_sampled_world_risk", {})
    k_adherence = sampled.get("sampled_K_to_diffusion_nonpaired_fidelity", {})
    paired = sampled.get("paired_failure_table", {})
    paired_world = sampled.get("paired_world_risk", {})
    provenance = sampled.get("provenance", {})
    finite_risk = all(
        all(isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in summary.values() if isinstance(value, (int, float)))
        for summary in risks.values()
    )
    return (
        sampled.get("worlds") == worlds
        and sampled.get("response_steps") == response_steps
        and set(risks) == {"hold_current", "idm"}
        and all(value["finite_state_rate"] == 1.0 for value in risks.values())
        and finite_risk
        and "k_adherence" in k_adherence
        and set(paired) == {"both_safe", "idm_only_failure", "hold_only_failure", "both_failure"}
        and all(len(paired_world.get(name, ())) == worlds for name in ("R_hold", "R_IDM", "Delta_R_IDM_minus_hold"))
        and bool(provenance.get("code_commit"))
        and bool(provenance.get("release_tag"))
    )


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
    highway = repo_root() / "HighwayEnv"
    if highway.exists():
        result["highway_env_tree"] = subprocess.check_output(
            ["git", "-C", str(highway), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip() if (highway / ".git").exists() else file_sha256(highway / "README.md")
        result["highway_env_version"] = "local-tree"
    lock_path = Path(lockfile) if lockfile is not None else None
    if lock_path is None:
        # This repository has no generated pip lock; the maintained local
        # HighwayEnv project manifest is the dependency contract used by the
        # release and is hashed as such.
        candidate = repo_root() / "HighwayEnv/pyproject.toml"
        if candidate.is_file():
            lock_path = candidate
    if lock_path is not None and lock_path.is_file():
        result["dependency_lockfile"] = logical_path(lock_path)
        result["dependency_lockfile_sha256"] = file_sha256(lock_path)
    return result
