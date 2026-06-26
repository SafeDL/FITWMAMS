"""Shared EVT target resolution for subset simulation workflows."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from diffusion.src.utils import load_json
from tools.evt import load_evt_model
from tools.io import resolve_path


def _configured_exposure_summary_path(
    config: dict[str, Any],
    config_dir: Path | None,
) -> Path | None:
    raw = config.get("paths", {}).get("exposure_summary_path")
    if not raw:
        return None
    base = Path(config_dir) if config_dir is not None else Path.cwd()
    return resolve_path(str(raw), base)


def _read_exposure_collision_level(path: Path) -> float:
    exposure = load_json(path)
    value = exposure.get("collision_critical_level")
    if value is None:
        raise KeyError(
            f"Exposure summary does not contain collision_critical_level: {path}"
        )
    level = float(value)
    if not np.isfinite(level):
        raise ValueError(f"Exposure collision critical level is not finite: {level}")
    return level


def _event_type(config: dict[str, Any]) -> str:
    return str(config.get("event", {}).get("event_type", "")).strip().lower()


def _exposure_collision_level(
    config: dict[str, Any],
    *,
    config_dir: Path | None,
    exposure_summary_path: Path | None,
) -> tuple[float, str, Path]:
    summary_path = exposure_summary_path or _configured_exposure_summary_path(
        config,
        config_dir,
    )
    if summary_path is None:
        raise KeyError(
            "evt.target_mode=collision_critical_level requires "
            "paths.exposure_summary_path"
        )
    level = _read_exposure_collision_level(summary_path)
    return level, "exposure_summary", summary_path


def resolve_collision_critical_level(
    config: dict[str, Any],
    *,
    config_dir: Path | None = None,
    exposure_summary_path: Path | None = None,
) -> tuple[float, str, Path | None]:
    """Resolve evt.collision_critical_level from config or exposure summary."""
    evt_cfg = config.setdefault("evt", {})
    raw = evt_cfg.get("collision_critical_level")
    if _event_type(config) == "cut_in":
        if raw is not None:
            raise ValueError(
                "cut-in collision_critical_level is fixed by the human-calibrated "
                "exposure summary; remove evt.collision_critical_level from config"
            )
        return _exposure_collision_level(
            config,
            config_dir=config_dir,
            exposure_summary_path=exposure_summary_path,
        )
    if raw is None:
        return _exposure_collision_level(
            config,
            config_dir=config_dir,
            exposure_summary_path=exposure_summary_path,
        )

    try:
        level = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("evt.collision_critical_level must be numeric") from exc
    if not np.isfinite(level):
        raise ValueError(f"evt.collision_critical_level is not finite: {level}")
    return level, "config", exposure_summary_path


def resolve_evt_failure_threshold(
    evt_model_path: Path,
    config: dict[str, Any],
    *,
    config_dir: Path | None = None,
    exposure_summary_path: Path | None = None,
) -> tuple[float, dict[str, float | str | None]]:
    """Resolve the target risk level and its EVT score-space threshold."""
    path = Path(evt_model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"EVT model is required before subset simulation: {path}"
        )

    evt_cfg = config.setdefault("evt", {})
    evt_cfg["model_path"] = str(path)
    evt_cfg["score_space"] = str(evt_cfg.get("score_space", "evt"))
    target_mode = str(evt_cfg.get("target_mode", "return_period"))
    return_period = int(evt_cfg.get("return_period", 100))
    model = load_evt_model(path)
    target_source: str | None = None
    target_summary_path: Path | None = None
    if target_mode == "collision_critical_level":
        z_target, target_source, target_summary_path = resolve_collision_critical_level(
            config,
            config_dir=config_dir,
            exposure_summary_path=exposure_summary_path,
        )
        evt_cfg["collision_critical_level"] = float(z_target)
    elif target_mode == "return_period":
        z_target = float(model.return_level(return_period))
    else:
        raise ValueError(f"Unsupported evt.target_mode: {target_mode}")

    failure_threshold = float(model.score(z_target))
    return failure_threshold, {
        "evt_target_mode": target_mode,
        "evt_return_period": float(return_period),
        "evt_return_level_target": float(z_target),
        "evt_return_level_target_source": target_source,
        "evt_exposure_summary_path": (
            str(target_summary_path) if target_summary_path is not None else None
        ),
        "evt_failure_threshold": failure_threshold,
        "evt_model_u": float(model.u),
        "evt_model_xi": float(model.xi),
        "evt_model_beta": float(model.beta),
        "evt_model_exceedance_rate": float(model.exceedance_rate),
    }
