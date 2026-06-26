#!/usr/bin/env python3
"""Shared final-level subset playback implementation."""
from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from tools.plot_style import configure_matplotlib

configure_matplotlib()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.utils import load_json, save_json
from process_highD.src.idm_ego import load_idm_ego_config
from process_highD.src.io_utils import load_config, resolve_data_path
from process_highD.src.loader import HighDRecording, load_recording
from process_highD.src.preprocess import (
    filter_abnormal_tracks,
    normalize_driving_direction,
    resample_recording,
)
from tools.io import load_npz, resolve_path
from PPO_subset.src.closed_loop_runner import (
    ClosedLoopCutInRunner,
    ClosedLoopFollowingRunner,
)
from PPO_subset.src.evt_target import resolve_evt_failure_threshold
from PPO_subset.src.frozen_diffusion_sampler import FrozenDiffusionSampler


DEFAULT_CONFIG_PATH = (
    ROOT
    / "PPO_subset"
    / "scripts"
    / "configs"
    / "latent_subset_following.yaml"
)
SCRIPT_DEFAULTS: dict[str, Any] = {
    "config": str(DEFAULT_CONFIG_PATH),
    "samples_path": None,
    "output_dir": None,
    "num_cases": 10,
    "random_seed": 42,
    "level": -1,
    "unique_test_scenarios": True,
    "view_width": 120.0,
    "vehicle_width": 2.0,
    "tail_steps": 50,
    "speed": 1.0,
    "render_gif": True,
    "render_background": True,
    "background_config_path": str(
        ROOT / "process_highD" / "scripts" / "configs" / "highd_default.yaml"
    ),
    "background_lane_width": 3.75,
    "background_neighbor_margin": 20.0,
    "log_level": "INFO",
}
_BACKGROUND_RECORDING_CACHE: dict[tuple[str, int], HighDRecording] = {}
logger = logging.getLogger(__name__)


def _paths(
    config: dict[str, Any],
    base: Path,
    *,
    samples_path: str | None,
    output_dir: str | None,
) -> dict[str, Path]:
    paths = config.get("paths", {})
    subset_cfg = config.get("subset_simulation", {})
    if "evt_model_path" not in paths:
        raise KeyError("Config paths.evt_model_path is required")
    if "output_dir" not in subset_cfg:
        raise KeyError("Config subset_simulation.output_dir is required")
    subset_output = resolve_path(str(subset_cfg["output_dir"]), base)
    samples = (
        resolve_path(samples_path, base)
        if samples_path
        else subset_output / "latent_subset_samples.npz"
    )
    out_dir = (
        resolve_path(output_dir, base)
        if output_dir
        else subset_output / "final_level_playbacks"
    )
    resolved = {
        "evt_model": resolve_path(paths["evt_model_path"], base),
        "samples": samples,
        "output_dir": out_dir,
    }
    if "exposure_summary_path" in paths:
        resolved["exposure_summary"] = resolve_path(
            str(paths["exposure_summary_path"]),
            base,
        )
    return resolved


def _clear_generated_playbacks(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for path in output_dir.glob("level*_rank*_sample*_ctx*"):
        if path.suffix.lower() in {".gif", ".png"}:
            path.unlink()
    manifest = output_dir / "final_level_playback_manifest.json"
    if manifest.exists():
        manifest.unlink()


def _level_index(samples: dict[str, np.ndarray], requested: int) -> int:
    levels = int(samples["scores"].shape[0])
    idx = requested if requested >= 0 else levels + requested
    if idx < 0 or idx >= levels:
        raise IndexError(f"level {requested} is out of range for {levels} levels")
    return int(idx)


def _sample_signature(
    samples: dict[str, np.ndarray],
    level_idx: int,
    sample_idx: int,
    *,
    steps: int,
) -> str:
    hasher = hashlib.sha256()
    for key in (
        "scenario_conditions",
        "initial_states",
        "latents",
        "actions",
        "action_mask",
    ):
        if key not in samples:
            continue
        value = np.asarray(samples[key][level_idx, sample_idx])
        if key in {"actions", "action_mask"}:
            value = value[:steps]
        value = np.ascontiguousarray(value)
        hasher.update(key.encode("utf-8"))
        hasher.update(str(value.shape).encode("utf-8"))
        hasher.update(str(value.dtype).encode("utf-8"))
        hasher.update(value.view(np.uint8))
    return hasher.hexdigest()


def _case_rows(
    samples: dict[str, np.ndarray],
    level_idx: int,
    *,
    num_cases: int,
    unique_test_scenarios: bool,
    failure_threshold: float,
    random_seed: int,
) -> list[dict[str, Any]]:
    if int(num_cases) <= 0:
        raise ValueError("num_cases must be positive for final-level playback")
    required_cached_keys = ("scenario_conditions", "initial_states")
    missing_cached = [key for key in required_cached_keys if key not in samples]
    if missing_cached:
        raise KeyError(
            "Final-level playback requires subset samples generated with cached "
            f"context fields; missing {missing_cached}. Re-run subset simulation."
        )
    scores = np.asarray(samples["scores"][level_idx], dtype=np.float64)
    order = np.argsort(scores)[::-1]
    candidates: list[dict[str, Any]] = []
    mask = samples.get("action_mask")
    seen_scenarios: set[str] = set()
    score_rank = 0
    for sample_idx in order:
        sample_idx = int(sample_idx)
        if float(scores[sample_idx]) < float(failure_threshold):
            continue
        context_index = int(samples["context_indices"][level_idx, sample_idx])
        if mask is None:
            steps = int(samples["actions"].shape[2])
        else:
            steps = int(np.sum(mask[level_idx, sample_idx] > 0.0))
        steps = max(steps, 1)
        scenario_key = _sample_signature(
            samples,
            level_idx,
            sample_idx,
            steps=steps,
        )
        if unique_test_scenarios and scenario_key in seen_scenarios:
            continue
        seen_scenarios.add(scenario_key)
        score_rank += 1
        candidates.append(
            {
                "rank": int(score_rank),
                "score_rank": int(score_rank),
                "level": int(level_idx),
                "sample_index": sample_idx,
                "context_index": context_index,
                "test_scenario_id": scenario_key[:16],
                "score": float(scores[sample_idx]),
                "threshold": float(samples["thresholds"][level_idx])
                if "thresholds" in samples
                else float("nan"),
                "accepted": float(samples["accepted_mask"][level_idx, sample_idx])
                if "accepted_mask" in samples
                else float("nan"),
                "failure_threshold": float(failure_threshold),
                "steps": int(steps),
                "actions": np.asarray(
                    samples["actions"][level_idx, sample_idx, :steps],
                    dtype=np.float32,
                ),
            }
        )
        for key in (
            "scenario_conditions",
            "initial_states",
            "ego_length",
            "adv_length",
            "recording_id",
            "ego_id",
            "target_id",
            "anchor_frame",
            "context_anchor_frame",
            "risk_start_index",
            "cross_frame",
            "cutin_start_frame",
        ):
            if key in samples:
                candidates[-1][key] = np.asarray(samples[key][level_idx, sample_idx])
    if not candidates:
        return []
    rng = np.random.default_rng(int(random_seed))
    selection_count = min(int(num_cases), len(candidates))
    selected_indices = rng.choice(
        len(candidates),
        size=selection_count,
        replace=False,
    )
    rows = [candidates[int(idx)] for idx in selected_indices]
    for selection_idx, row in enumerate(rows, start=1):
        row["rank"] = int(selection_idx)
    return rows


def _stored_failure_threshold(
    samples: dict[str, np.ndarray],
    paths: dict[str, Path],
) -> float:
    if "failure_threshold" in samples:
        raw = np.asarray(samples["failure_threshold"])
        value = float(raw.item() if raw.ndim == 0 else raw.reshape(-1)[0])
        if np.isfinite(value):
            return value
    summary_path = paths["samples"].with_name("latent_subset_summary.json")
    if summary_path.exists():
        value = float(load_json(summary_path).get("failure_threshold", np.nan))
        if np.isfinite(value):
            return value
    return float("nan")


def _failure_threshold(
    samples: dict[str, np.ndarray],
    paths: dict[str, Path],
    config: dict[str, Any],
    config_dir: Path,
) -> float:
    try:
        value, _target = resolve_evt_failure_threshold(
            paths["evt_model"],
            config,
            config_dir=config_dir,
            exposure_summary_path=paths.get("exposure_summary"),
        )
        if np.isfinite(value):
            stored = _stored_failure_threshold(samples, paths)
            if np.isfinite(stored) and abs(stored - value) > 1.0e-8:
                logger.info(
                    (
                        "Using current configured failure threshold %.6g "
                        "instead of stored sample threshold %.6g"
                    ),
                    value,
                    stored,
                )
            return float(value)
    except Exception as exc:
        logger.warning(
            "Could not resolve current configured failure threshold; "
            "falling back to stored subset threshold: %s",
            exc,
        )
    value = _stored_failure_threshold(samples, paths)
    if np.isfinite(value):
        return value
    raise KeyError(
        "Final-level playback requires failure_threshold in "
        "latent_subset_samples.npz or latent_subset_summary.json"
    )


def _display_ttc_label(item: dict[str, float]) -> str:
    if float(item.get("collision", 0.0)) > 0.0:
        return "collision"
    ttc = float(item.get("ttc", np.nan))
    if not np.isfinite(ttc) or ttc >= 999.0:
        return "n/a"
    if ttc > 60.0:
        return ">60s"
    return f"{ttc:.2f}s"


def _context_kinematics(context: dict[str, Any]) -> dict[str, float]:
    raw = np.asarray(context["initial_states"], dtype=np.float32)
    ego = raw[0]
    lead = raw[1]
    ego_speed = float(np.hypot(float(ego[2]), float(ego[3])))
    lead_speed = float(np.hypot(float(lead[2]), float(lead[3])))
    gap = float(
        lead[0]
        - ego[0]
        - 0.5
        * (
            float(context.get("ego_length", 4.8))
            + float(context.get("adv_length", 4.8))
        )
    )
    return {
        "context_initial_gap": gap,
        "context_initial_ego_speed": ego_speed,
        "context_initial_lead_speed": lead_speed,
        "context_initial_closing_speed": ego_speed - lead_speed,
    }


def _event_type_from_config(config: dict[str, Any]) -> str:
    return str(config.get("event", {}).get("event_type", "following"))


def _apply_shared_idm_ego_config(
    config: dict[str, Any],
    config_dir: Path,
    *,
    event_type: str,
) -> None:
    configured = config.get("idm_ego_config_path") or config.get("paths", {}).get(
        "idm_ego_config_path"
    )
    if not configured:
        return
    shared = load_idm_ego_config(
        resolve_path(str(configured), config_dir),
        event_type=event_type,
    )
    config["idm_ego"] = {**dict(config.get("idm_ego", {}) or {}), **shared}
    env_cfg = config.setdefault("env", {})
    ego_response_cfg = config.setdefault("ego_response", {})
    if "target_speed" in shared:
        env_cfg["ego_target_speed"] = float(shared["target_speed"])
    if "speed_limit" in shared:
        env_cfg["speed_limit"] = float(shared["speed_limit"])
    if "lanes_count" in shared:
        env_cfg["lanes_count"] = int(shared["lanes_count"])
    if "enable_lane_change" in shared:
        ego_response_cfg["enable_lane_change"] = bool(shared["enable_lane_change"])


def _make_runner(
    config: dict[str, Any],
    config_dir: Path,
) -> ClosedLoopFollowingRunner:
    sampler = FrozenDiffusionSampler.from_config(config, config_dir=config_dir).eval()
    event_type = str(
        sampler.prior.schema.get("event_type", _event_type_from_config(config))
    )
    if event_type == "cut_in":
        return ClosedLoopCutInRunner(sampler, config)
    return ClosedLoopFollowingRunner(sampler, config)


def _trace_array(trace: list[dict[str, float]], key: str) -> np.ndarray:
    return np.asarray(
        [float(item.get(key, np.nan)) for item in trace],
        dtype=np.float32,
    )


def _scalar_from_row(row: dict[str, Any], key: str) -> Any | None:
    if key not in row:
        return None
    raw = np.asarray(row[key])
    value = raw.item() if raw.ndim == 0 else raw.reshape(-1)[0]
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(numeric):
        return None
    return numeric


def _context_from_saved_sample(
    row: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild the exact context fields cached during subset simulation."""
    out: dict[str, Any] = {
        "scenario_conditions": np.asarray(
            row["scenario_conditions"],
            dtype=np.float32,
        ).copy(),
        "initial_states": np.asarray(row["initial_states"], dtype=np.float32).copy(),
        "event_id": f"subset_context_{int(row['context_index'])}",
        "source_type": "subset_cached_context",
    }
    for key in (
        "ego_length",
        "adv_length",
        "recording_id",
        "ego_id",
        "target_id",
        "anchor_frame",
        "context_anchor_frame",
        "cross_frame",
        "cutin_start_frame",
    ):
        value = _scalar_from_row(row, key)
        if value is None:
            continue
        integer_keys = {
            "recording_id",
            "ego_id",
            "target_id",
            "anchor_frame",
            "context_anchor_frame",
            "risk_start_index",
            "cross_frame",
            "cutin_start_frame",
        }
        if key in integer_keys:
            out[key] = int(round(float(value)))
        else:
            out[key] = float(value)
    if "risk_start_index" not in out:
        cross = out.get("cross_frame")
        anchor = out.get("anchor_frame")
        if cross is not None and anchor is not None:
            out["risk_start_index"] = max(0, int(cross) - int(anchor) - 1)
    if "risk_start_index" not in out:
        conditions = np.asarray(
            out["scenario_conditions"],
            dtype=np.float32,
        ).reshape(-1)
        if conditions.size >= 9 and np.isfinite(float(conditions[8])):
            dt = 1.0 / 25.0
            out["risk_start_index"] = max(
                0,
                int(round(float(conditions[8]) / dt)) - 1,
            )
    return out


def _load_background_recording(
    config_path_text: str | None,
    recording_id: int,
) -> HighDRecording | None:
    if not config_path_text:
        return None
    config_path = Path(config_path_text).resolve()
    if not config_path.exists():
        logger.warning("Background config not found: %s", config_path)
        return None
    cache_key = (str(config_path), int(recording_id))
    if cache_key in _BACKGROUND_RECORDING_CACHE:
        return _BACKGROUND_RECORDING_CACHE[cache_key]
    try:
        config = load_config(config_path)
        raw_dir = resolve_data_path(config["paths"]["raw_dir"], config_path)
        rec = load_recording(str(raw_dir), int(recording_id))
        rec = normalize_driving_direction(rec)
        rec = filter_abnormal_tracks(rec, config)
        target_fps = int(
            config.get("sampling", {}).get(
                "target_fps",
                rec.recording_meta.get("frameRate", 25),
            )
        )
        rec = resample_recording(rec, target_fps)
    except Exception as exc:
        logger.warning(
            "Could not load background recording %s from %s: %s",
            recording_id,
            config_path,
            exc,
        )
        return None
    _BACKGROUND_RECORDING_CACHE[cache_key] = rec
    return rec


def _anchor_frame_from_context(context: dict[str, Any]) -> int | None:
    for key in ("context_anchor_frame", "anchor_frame", "cross_frame"):
        value = context.get(key)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            return int(round(numeric))
    return None


def _lane_changing_background_ids(
    rec: HighDRecording,
    *,
    start_frame: int,
    end_frame: int,
    exclude_ids: set[int],
) -> set[int]:
    changing: set[int] = set()
    if end_frame < start_frame:
        return changing
    for vehicle_id in rec.vehicle_ids():
        vehicle_id = int(vehicle_id)
        if vehicle_id in exclude_ids:
            continue
        try:
            track = rec.get_vehicle_track(vehicle_id)
        except KeyError:
            continue
        if "laneId" not in track.columns:
            continue
        window = track[
            (track.index >= int(start_frame)) & (track.index <= int(end_frame))
        ]
        if window.empty:
            continue
        if int(window["laneId"].dropna().nunique()) > 1:
            changing.add(vehicle_id)
    return changing


def _background_reference(
    context: dict[str, Any],
    trace: list[dict[str, float]],
) -> dict[str, Any] | None:
    if not trace or not bool(SCRIPT_DEFAULTS["render_background"]):
        return None
    try:
        recording_id = int(context["recording_id"])
        ego_id = int(context["ego_id"])
        target_id = int(context["target_id"])
    except (KeyError, TypeError, ValueError):
        return None
    anchor_frame = _anchor_frame_from_context(context)
    if anchor_frame is None:
        return None
    rec = _load_background_recording(
        str(SCRIPT_DEFAULTS.get("background_config_path") or ""),
        recording_id,
    )
    if rec is None:
        return None
    try:
        ego_track = rec.get_vehicle_track(ego_id)
        if anchor_frame not in ego_track.index:
            available = np.asarray(ego_track.index, dtype=np.int64)
            anchor_frame = int(available[np.argmin(np.abs(available - anchor_frame))])
        ego_row = ego_track.loc[anchor_frame]
    except Exception as exc:
        logger.warning("Could not align background ego at anchor frame: %s", exc)
        return None
    first = trace[0]
    lane_width = float(SCRIPT_DEFAULTS["background_lane_width"])
    initial_lateral_offset = float(first["lead_y"]) - float(first["ego_y"])
    source_y = None
    if abs(initial_lateral_offset) > 0.5 * lane_width:
        source_y = float(first["lead_y"])
    exclude_ids = {ego_id, target_id}
    exclude_ids.update(
        _lane_changing_background_ids(
            rec,
            start_frame=int(anchor_frame),
            end_frame=int(anchor_frame) + len(trace) - 1,
            exclude_ids=exclude_ids,
        )
    )
    return {
        "recording": rec,
        "anchor_frame": int(anchor_frame),
        "anchor_ego_x": float(ego_row["x"]),
        "anchor_ego_y": float(ego_row["y"]),
        "display_ego_x0": float(first["ego_position"]),
        "display_ego_y0": float(first["ego_y"]),
        "exclude_ids": exclude_ids,
        "target_source_display_y": source_y,
    }


def _vehicle_heading_from_row(row: Any) -> float:
    vx = float(row.get("xVelocity", row.get("vx", 0.0)))
    vy = -float(row.get("yVelocity", row.get("vy", 0.0)))
    if abs(vx) + abs(vy) < 1.0e-9:
        return 0.0
    return float(np.arctan2(vy, vx))


def _draw_background_traffic(
    ax: plt.Axes,
    *,
    background: dict[str, Any] | None,
    frame_offset: int,
    ego_y: float,
    lead_y: float,
    xlim: tuple[float, float],
    vehicle_width: float,
) -> None:
    if background is None:
        return
    rec: HighDRecording = background["recording"]
    frame = int(background["anchor_frame"]) + int(frame_offset)
    frame_df = rec.get_frame(frame)
    if frame_df.empty:
        return
    lane_width = float(SCRIPT_DEFAULTS["background_lane_width"])
    margin = float(SCRIPT_DEFAULTS["background_neighbor_margin"])
    blocked_lane_centers = [float(ego_y), float(lead_y)]
    source_y = background.get("target_source_display_y")
    if source_y is not None:
        blocked_lane_centers.append(float(source_y))
    for idx, row in frame_df.iterrows():
        vehicle_id = int(idx[0]) if isinstance(idx, tuple) else int(idx)
        if vehicle_id in background["exclude_ids"]:
            continue
        x = (
            float(row["x"])
            - float(background["anchor_ego_x"])
            + float(background["display_ego_x0"])
        )
        y = -(
            float(row["y"]) - float(background["anchor_ego_y"])
        ) + float(background["display_ego_y0"])
        if x < xlim[0] - margin or x > xlim[1] + margin:
            continue
        if y < -7.5 or y > 7.5:
            continue
        if any(abs(y - center_y) <= 0.5 * lane_width for center_y in blocked_lane_centers):
            continue
        _add_vehicle(
            ax,
            x=x,
            y=y,
            heading=_vehicle_heading_from_row(row),
            length=float(row.get("width", 4.5)),
            width=float(row.get("height", vehicle_width)),
            color="#bdbdbd",
            label=None,
            zorder=1,
            alpha=0.50,
            linewidth=0.45,
        )


def _add_vehicle(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    heading: float,
    length: float,
    width: float,
    color: str,
    label: str | None,
    zorder: int,
    alpha: float = 0.92,
    linewidth: float = 0.8,
) -> None:
    rect = Rectangle(
        (-0.5 * length, -0.5 * width),
        length,
        width,
        facecolor=color,
        edgecolor="black",
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    rect.set_transform(Affine2D().rotate(heading).translate(x, y) + ax.transData)
    ax.add_patch(rect)
    if not label:
        return
    ax.text(
        x,
        y + 0.8 * width,
        label,
        ha="center",
        va="bottom",
        fontsize=8,
        color="black",
        zorder=zorder + 1,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
    )


def _write_overview_png(
    trace: list[dict[str, float]],
    row: dict[str, Any],
    context: dict[str, Any],
    metrics: dict[str, float],
    output_path: Path,
) -> None:
    if not trace:
        raise RuntimeError("Cannot write overview for an empty rollout trace")
    steps = _trace_array(trace, "step")
    ego_x = _trace_array(trace, "ego_position")
    lead_x = _trace_array(trace, "lead_position")
    gap = _trace_array(trace, "gap")
    raw_ttc = _trace_array(trace, "ttc")
    ttc = np.where(raw_ttc >= 999.0, np.nan, np.clip(raw_ttc, 0.0, 60.0))
    ego_accel = _trace_array(trace, "ego_accel")
    lead_accel = _trace_array(trace, "lead_accel")
    collisions = _trace_array(trace, "collision") > 0.0

    fig, axes = plt.subplots(3, 1, figsize=(12.0, 8.0), sharex=False)
    axes[0].plot(steps, ego_x, label="ego x", color="tab:red")
    axes[0].plot(steps, lead_x, label="lead x", color="tab:blue")
    axes[0].set_ylabel("x [m]")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(steps, gap, label="gap", color="tab:purple")
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    axes[1].axhline(2.0, color="tab:orange", linewidth=0.9, alpha=0.7, linestyle="--")
    axes[1].set_ylabel("gap [m]")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(steps, ttc, label="TTC", color="tab:green")
    axes[2].plot(steps, ego_accel, label="ego accel", color="tab:red", alpha=0.75)
    axes[2].plot(steps, lead_accel, label="lead accel", color="tab:blue", alpha=0.75)
    if np.any(collisions):
        axes[2].scatter(
            steps[collisions],
            np.zeros(int(np.sum(collisions)), dtype=np.float32),
            label="collision",
            color="black",
            marker="x",
            zorder=5,
        )
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("TTC [s] / accel [m/s^2]")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="best")

    event_id = context.get("event_id", "")
    fig.suptitle(
        "subset final-level case "
        f"rank={row['rank']} level={row['level']} sample={row['sample_index']} "
        f"context={row['context_index']} event={event_id}\n"
        f"score={row['score']:.4f} replay_risk={metrics.get('risk_score', np.nan):.4f} "
        f"min_gap={metrics.get('min_gap', np.nan):.3f} "
        f"min_ttc={metrics.get('min_ttc', np.nan):.3f} "
        f"collision={metrics.get('collision', np.nan):.0f}",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _write_gif(
    trace: list[dict[str, float]],
    row: dict[str, Any],
    context: dict[str, Any],
    metrics: dict[str, float],
    output_path: Path,
    *,
    view_width: float,
    vehicle_width: float,
    tail_steps: int,
    fps: float,
) -> Path:
    if not trace:
        raise RuntimeError("Cannot write GIF for an empty rollout trace")
    actual_path = output_path.with_suffix(".gif")
    writer = PillowWriter(fps=max(min(float(fps), 15.0), 1.0))

    ego_length = float(context.get("ego_length", 4.8))
    lead_length = float(context.get("adv_length", 4.8))
    ego_y = _trace_array(trace, "ego_y")
    lead_y = _trace_array(trace, "lead_y")
    lane_width = float(SCRIPT_DEFAULTS["background_lane_width"])
    road_half_width = 1.76 * lane_width
    vehicle_y = np.concatenate([ego_y, lead_y])
    ymin = min(float(np.nanmin(vehicle_y)) - 2.5, -road_half_width)
    ymax = max(float(np.nanmax(vehicle_y)) + 2.5, road_half_width)
    half_width = 0.5 * float(view_width)
    background = _background_reference(context, trace)

    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#6f7378")
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    title = ax.set_title("")
    ego_line, = ax.plot([], [], color="#e31a1c", linewidth=1.6, alpha=0.8, zorder=3)
    lead_line, = ax.plot([], [], color="#1f78b4", linewidth=1.6, alpha=0.8, zorder=3)
    lane_boundaries = np.array([-1.5, -0.5, 0.5, 1.5], dtype=float) * lane_width
    for lane_idx, y_lane in enumerate(lane_boundaries):
        is_outer = lane_idx == 0 or lane_idx == len(lane_boundaries) - 1
        ax.axhline(
            y_lane,
            color="#ffffff",
            linewidth=0.8,
            linestyle="-" if is_outer else "--",
            alpha=0.45 if is_outer else 0.28,
        )
    frame_artists: list[Any] = []

    with writer.saving(fig, str(actual_path), dpi=110):
        for frame_idx, item in enumerate(trace):
            for artist in frame_artists:
                artist.remove()
            frame_artists.clear()

            ego_x = float(item["ego_position"])
            lead_x = float(item["lead_position"])
            center_x = 0.5 * (ego_x + lead_x)
            ax.set_xlim(center_x - half_width, center_x + half_width)
            xlim = (center_x - half_width, center_x + half_width)

            trail_start = max(0, frame_idx - int(tail_steps))
            trail = trace[trail_start:frame_idx + 1]
            ego_line.set_data(
                [float(t["ego_position"]) for t in trail],
                [float(t["ego_y"]) for t in trail],
            )
            lead_line.set_data(
                [float(t["lead_position"]) for t in trail],
                [float(t["lead_y"]) for t in trail],
            )
            before = len(ax.patches), len(ax.texts)
            _draw_background_traffic(
                ax,
                background=background,
                frame_offset=frame_idx,
                ego_y=float(item["ego_y"]),
                lead_y=float(item["lead_y"]),
                xlim=xlim,
                vehicle_width=float(vehicle_width),
            )
            frame_artists.extend(list(ax.patches)[before[0]:])
            frame_artists.extend(list(ax.texts)[before[1]:])
            before = len(ax.patches), len(ax.texts)
            _add_vehicle(
                ax,
                x=ego_x,
                y=float(item["ego_y"]),
                heading=float(item["ego_yaw"]),
                length=ego_length,
                width=float(vehicle_width),
                color="#e31a1c",
                label="ego",
                zorder=5,
            )
            frame_artists.extend(list(ax.patches)[before[0]:])
            frame_artists.extend(list(ax.texts)[before[1]:])
            before = len(ax.patches), len(ax.texts)
            target_label = (
                "target"
                if context.get("cross_frame") is not None
                or context.get("cutin_start_frame") is not None
                else "lead"
            )
            _add_vehicle(
                ax,
                x=lead_x,
                y=float(item["lead_y"]),
                heading=float(item["lead_yaw"]),
                length=lead_length,
                width=float(vehicle_width),
                color="#1f78b4",
                label=target_label,
                zorder=4,
            )
            frame_artists.extend(list(ax.patches)[before[0]:])
            frame_artists.extend(list(ax.texts)[before[1]:])
            title.set_text(
                f"rank={row['rank']} sample={row['sample_index']} "
                f"score={row['score']:.3f} replay_risk={metrics.get('risk_score', np.nan):.3f} | "
                f"step={int(item['step'])} gap={float(item['gap']):.2f}m "
                f"TTC={_display_ttc_label(item)}"
            )
            writer.grab_frame()
    plt.close(fig)
    return actual_path


def _manifest_row(
    row: dict[str, Any],
    context: dict[str, Any],
    metrics: dict[str, float],
    *,
    png_path: Path,
    animation_path: Path | None,
) -> dict[str, Any]:
    context_kin = _context_kinematics(context)
    return {
        "rank": int(row["rank"]),
        "score_rank": int(row.get("score_rank", row["rank"])),
        "level": int(row["level"]),
        "sample_index": int(row["sample_index"]),
        "context_index": int(row["context_index"]),
        "test_scenario_id": str(row.get("test_scenario_id", "")),
        "recording_id": context.get("recording_id"),
        "event_id": context.get("event_id"),
        "source_type": context.get("source_type"),
        "tail_threshold": float(context.get("tail_threshold", np.nan)),
        "context_y_long": float(context.get("y_long", np.nan)),
        "context_risk_score": float(context.get("risk_score", np.nan)),
        "recorded_min_gap": float(context.get("recorded_min_gap", np.nan)),
        "recorded_min_ttc": float(context.get("recorded_min_ttc", np.nan)),
        **context_kin,
        "subset_score": float(row["score"]),
        "failure_threshold": float(row.get("failure_threshold", np.nan)),
        "subset_level_threshold": float(row.get("threshold", np.nan)),
        "replay_risk": float(metrics.get("risk_score", np.nan)),
        "risk_score": float(metrics.get("risk_score", np.nan)),
        "y_long": float(metrics.get("y_long", np.nan)),
        "y_cutin": float(metrics.get("y_cutin", np.nan)),
        "evt_tail_probability": float(metrics.get("evt_tail_probability", np.nan)),
        "collision": float(metrics.get("collision", np.nan)),
        "near_collision": float(metrics.get("near_collision", np.nan)),
        "min_gap": float(metrics.get("min_gap", np.nan)),
        "min_ttc": float(metrics.get("min_ttc", np.nan)),
        "hard_brake": float(metrics.get("hard_brake", np.nan)),
        "cutin_safety_risk_score": float(
            metrics.get("cutin_safety_risk_score", np.nan)
        ),
        "cutin_time_headway": float(metrics.get("cutin_time_headway", np.nan)),
        "cutin_lateral_time_gap": float(
            metrics.get("cutin_lateral_time_gap", np.nan)
        ),
        "max_post_cutin_drac": float(
            metrics.get("max_post_cutin_drac", np.nan)
        ),
        "min_abs_lateral_offset": float(
            metrics.get("min_abs_lateral_offset", np.nan)
        ),
        "final_abs_lateral_offset": float(
            metrics.get("final_abs_lateral_offset", np.nan)
        ),
        "max_lateral_approach_speed": float(
            metrics.get("max_lateral_approach_speed", np.nan)
        ),
        "lateral_overlap_fraction": float(
            metrics.get("lateral_overlap_fraction", np.nan)
        ),
        "is_cutin": float(metrics.get("is_cutin", np.nan)),
        "planned_steps": int(row["steps"]),
        "steps": int(metrics.get("steps", row["steps"])),
        "png": str(png_path),
        "animation": str(animation_path) if animation_path is not None else None,
    }


def replay_final_level(
    config: dict[str, Any],
    config_dir: Path,
    *,
    expected_event_type: str | None = None,
) -> Path:
    paths = _paths(
        config,
        config_dir,
        samples_path=SCRIPT_DEFAULTS["samples_path"],
        output_dir=SCRIPT_DEFAULTS["output_dir"],
    )
    if not paths["samples"].exists():
        raise FileNotFoundError(f"Subset samples not found: {paths['samples']}")
    evt_cfg = config.setdefault("evt", {})
    evt_cfg["model_path"] = str(paths["evt_model"])
    evt_cfg["score_space"] = str(evt_cfg.get("score_space", "evt"))
    samples = load_npz(paths["samples"])
    level_idx = _level_index(samples, int(SCRIPT_DEFAULTS["level"]))
    failure_threshold = _failure_threshold(samples, paths, config, config_dir)
    cases = _case_rows(
        samples,
        level_idx,
        num_cases=int(SCRIPT_DEFAULTS["num_cases"]),
        unique_test_scenarios=bool(SCRIPT_DEFAULTS["unique_test_scenarios"]),
        failure_threshold=failure_threshold,
        random_seed=int(SCRIPT_DEFAULTS["random_seed"]),
    )
    if not cases:
        raise RuntimeError(
            "No final-level subset cases meet the failure threshold "
            f"{failure_threshold:.6g}"
        )
    event_type = _event_type_from_config(config)
    if expected_event_type is not None and event_type != expected_event_type:
        raise ValueError(f"Expected {expected_event_type} config, got {event_type}")
    _apply_shared_idm_ego_config(config, config_dir, event_type=event_type)

    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_generated_playbacks(output_dir)
    runner = _make_runner(config, config_dir)
    target_fps = float(config.get("sampling", {}).get("target_fps", 25.0))
    fps = target_fps * SCRIPT_DEFAULTS["speed"]
    manifest: list[dict[str, Any]] = []

    for row in cases:
        context_idx = int(row["context_index"])
        context = _context_from_saved_sample(row)
        context["context_index"] = int(context_idx)
        actions = np.asarray(row["actions"], dtype=np.float32)
        result = runner.rollout_pre_sampled_plan(
            context,
            actions,
            episode_steps=int(row["steps"]),
        )
        safe_event_id = str(context.get("event_id", "event")).replace("/", "_")
        stem = (
            f"level{level_idx:02d}_rank{int(row['rank']):02d}_"
            f"sample{int(row['sample_index']):04d}_ctx{context_idx:03d}_{safe_event_id}"
        )
        png_path = output_dir / f"{stem}.png"
        animation_path = None
        gif_path = output_dir / f"{stem}.gif"
        _write_overview_png(
            result.trace,
            row,
            context,
            result.metrics,
            png_path,
        )
        if bool(SCRIPT_DEFAULTS["render_gif"]):
            animation_path = _write_gif(
                result.trace,
                row,
                context,
                result.metrics,
                gif_path,
                view_width=SCRIPT_DEFAULTS["view_width"],
                vehicle_width=SCRIPT_DEFAULTS["vehicle_width"],
                tail_steps=int(SCRIPT_DEFAULTS["tail_steps"]),
                fps=fps,
            )
        manifest.append(
            _manifest_row(
                row,
                context,
                result.metrics,
                png_path=png_path,
                animation_path=animation_path
                if animation_path is not None and animation_path.exists()
                else None,
            )
        )
        logger.info(
            "Rendered rank %d sample %d score %.4f -> %s",
            row["rank"],
            row["sample_index"],
            row["score"],
            png_path,
        )

    manifest_path = output_dir / "final_level_playback_manifest.json"
    save_json(
        {
            "samples": str(paths["samples"]),
            "level": int(level_idx),
            "failure_threshold": float(failure_threshold),
            "num_cases": int(len(manifest)),
            "deduplication": "unique_test_scenario"
            if bool(SCRIPT_DEFAULTS["unique_test_scenarios"])
            else "none",
            "unique_test_scenarios": bool(
                SCRIPT_DEFAULTS["unique_test_scenarios"]
            ),
            "selection": "random_without_replacement",
            "random_seed": int(SCRIPT_DEFAULTS["random_seed"]),
            "cases": manifest,
        },
        manifest_path,
    )
    return manifest_path
