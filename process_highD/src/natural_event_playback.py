"""GIF playback for selected highD natural tail segments."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np
import pandas as pd

from process_highD.src.io_utils import ensure_dir, load_config, resolve_data_path
from process_highD.src.loader import HighDRecording, load_recording
from process_highD.src.natural_segments import SLOT_NAMES
from process_highD.src.preprocess import (
    filter_abnormal_tracks,
    normalize_driving_direction,
    resample_recording,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NaturalPlaybackOptions:
    fps: float = 12.5
    frame_stride: int = 2
    view_width_m: float = 160.0
    max_background_vehicles: int = 100
    dpi: int = 120


def _vehicle_row(frame: pd.DataFrame, vehicle_id: int) -> pd.Series | None:
    if frame.empty:
        return None
    ids = frame.index.get_level_values("id")
    mask = ids == int(vehicle_id)
    if not np.any(mask):
        return None
    return frame.loc[mask].iloc[0]


def _lane_limits(recording: HighDRecording, frame_slice: pd.DataFrame) -> tuple[float, float]:
    markings: list[float] = []
    for key in ("upperLaneMarkings", "lowerLaneMarkings"):
        values = np.asarray(recording.recording_meta.get(key, []), dtype=float)
        values = values[np.isfinite(values)]
        markings.extend(float(value) for value in values)
    if markings:
        return min(markings) - 2.5, max(markings) + 2.5
    if "y" not in frame_slice:
        return -2.0, 14.0
    y = pd.to_numeric(frame_slice["y"], errors="coerce")
    y = y[np.isfinite(y)]
    if y.empty:
        return -2.0, 14.0
    return float(y.min() - 4.0), float(y.max() + 4.0)


def _lane_markings(recording: HighDRecording) -> list[float]:
    lines: list[float] = []
    for key in ("upperLaneMarkings", "lowerLaneMarkings"):
        values = np.asarray(recording.recording_meta.get(key, []), dtype=float)
        values = values[np.isfinite(values)]
        lines.extend(float(value) for value in values)
    return sorted(set(round(value, 6) for value in lines))


def _slot_vehicle_ids(row: pd.Series) -> dict[int, str]:
    out: dict[int, str] = {}
    for slot in SLOT_NAMES:
        column = f"{slot}_id"
        if column not in row:
            continue
        try:
            vehicle_id = int(row[column])
        except (TypeError, ValueError):
            continue
        if vehicle_id > 0:
            out[vehicle_id] = slot
    return out


def _window_track_slice(
    recording: HighDRecording,
    *,
    start_frame: int,
    end_frame: int,
) -> pd.DataFrame:
    frames = recording.tracks.index.get_level_values("frame")
    return recording.tracks.loc[(frames >= int(start_frame)) & (frames <= int(end_frame))]


def _draw_vehicle(
    ax: Any,
    *,
    recording: HighDRecording,
    row: pd.Series,
    vehicle_id: int,
    color: str,
    edgecolor: str,
    alpha: float,
    label: str | None = None,
) -> None:
    if int(vehicle_id) in recording.tracks_meta.index:
        meta = recording.tracks_meta.loc[int(vehicle_id)]
        length = max(float(meta.get("width", 4.5)), 0.5)
        width = max(float(meta.get("height", 1.8)), 0.4)
    else:
        length = max(float(row.get("width", 4.5)), 0.5)
        width = max(float(row.get("height", 1.8)), 0.4)
    x = float(row["x"])
    y = float(row["y"])
    rect = Rectangle(
        (x - 0.5 * length, y - 0.5 * width),
        length,
        width,
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=1.0,
        alpha=alpha,
        zorder=3 if label else 2,
    )
    ax.add_patch(rect)
    if label:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=6,
            color="white",
            zorder=4,
        )
    elif vehicle_id > 0:
        ax.text(
            x,
            y,
            str(vehicle_id),
            ha="center",
            va="center",
            fontsize=4.5,
            color="#374151",
            alpha=0.72,
            zorder=3,
        )


def _draw_frame(
    *,
    axes: np.ndarray,
    recording: HighDRecording,
    row: pd.Series,
    frame_id: int,
    risk_trace: np.ndarray | None,
    options: NaturalPlaybackOptions,
    y_limits: tuple[float, float],
    lane_lines: list[float],
) -> None:
    ax_scene, ax_risk = axes
    ax_scene.clear()
    ax_risk.clear()

    ego_id = int(row["ego_id"])
    slot_ids = _slot_vehicle_ids(row)
    frame = recording.get_frame(int(frame_id))
    ego = _vehicle_row(frame, ego_id)
    if ego is None:
        return
    ego_x = float(ego["x"])
    half_width = 0.5 * float(options.view_width_m)
    x_limits = (ego_x - half_width, ego_x + half_width)

    in_view = (
        (pd.to_numeric(frame["x"], errors="coerce") >= x_limits[0])
        & (pd.to_numeric(frame["x"], errors="coerce") <= x_limits[1])
        & (pd.to_numeric(frame["y"], errors="coerce") >= y_limits[0])
        & (pd.to_numeric(frame["y"], errors="coerce") <= y_limits[1])
    )
    visible = frame.loc[in_view].copy()
    if len(visible) > int(options.max_background_vehicles):
        distance = np.abs(pd.to_numeric(visible["x"], errors="coerce") - ego_x)
        keep = np.argsort(distance.to_numpy())[: int(options.max_background_vehicles)]
        visible = visible.iloc[np.sort(keep)]

    for y in lane_lines:
        if y_limits[0] <= y <= y_limits[1]:
            ax_scene.axhline(y, color="#9ca3af", linewidth=0.8, alpha=0.65, zorder=1)
    ax_scene.axvline(ego_x, color="#2563eb", linewidth=0.8, alpha=0.18, zorder=1)

    for (vehicle_id, _frame), vehicle in visible.iterrows():
        vid = int(vehicle_id)
        if vid == ego_id or vid in slot_ids:
            continue
        _draw_vehicle(
            ax_scene,
            recording=recording,
            row=vehicle,
            vehicle_id=vid,
            color="#d1d5db",
            edgecolor="#9ca3af",
            alpha=0.46,
        )

    for vid, slot in slot_ids.items():
        vehicle = _vehicle_row(frame, vid)
        if vehicle is None:
            continue
        _draw_vehicle(
            ax_scene,
            recording=recording,
            row=vehicle,
            vehicle_id=vid,
            color="#f97316",
            edgecolor="#9a3412",
            alpha=0.88,
            label=slot.replace("_", "\n"),
        )

    _draw_vehicle(
        ax_scene,
        recording=recording,
        row=ego,
        vehicle_id=ego_id,
        color="#2563eb",
        edgecolor="#1e3a8a",
        alpha=0.95,
        label=f"ego\n{ego_id}",
    )

    peak_frame = int(row.get("peak_risk_frame", row.get("window_end_frame", frame_id)))
    ax_scene.set_xlim(*x_limits)
    ax_scene.set_ylim(*y_limits)
    ax_scene.set_aspect("equal", adjustable="box")
    ax_scene.set_title(
        f"{row['segment_id']} | rec {int(row['recording_id']):02d} | "
        f"frame {int(frame_id)} | R={float(row['event_risk']):.3f}",
        fontsize=9,
    )
    ax_scene.set_xlabel("x [m], normalized driving direction")
    ax_scene.set_ylabel("y [m]")
    if int(frame_id) == peak_frame:
        ax_scene.text(
            0.01,
            0.96,
            "peak risk frame",
            transform=ax_scene.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#dc2626",
            fontweight="bold",
        )

    if risk_trace is not None and risk_trace.size:
        start = int(row["window_start_frame"])
        offset = int(np.clip(int(frame_id) - start, 0, risk_trace.size - 1))
        xs = np.arange(risk_trace.size, dtype=int)
        ax_risk.plot(xs, risk_trace, color="#111827", linewidth=1.2)
        ax_risk.plot(xs[: offset + 1], risk_trace[: offset + 1], color="#2563eb", linewidth=2.0)
        ax_risk.axvline(offset, color="#dc2626", linewidth=1.0, alpha=0.8)
        ax_risk.scatter([offset], [risk_trace[offset]], color="#dc2626", s=18, zorder=5)
        ax_risk.set_xlim(0, risk_trace.size - 1)
        ymax = max(float(np.nanmax(risk_trace)) * 1.08, 1.0e-6)
        ax_risk.set_ylim(0.0, ymax)
        ax_risk.set_ylabel("R_SEI prefix")
        ax_risk.set_xlabel("frame offset in 6 s window")
        ax_risk.grid(True, linewidth=0.4, alpha=0.35)
    else:
        ax_risk.text(0.5, 0.5, "risk trace unavailable", ha="center", va="center")
        ax_risk.set_axis_off()


def _risk_trace_for_row(
    row: pd.Series,
    *,
    risk_trace_npz: Path,
) -> np.ndarray | None:
    if not risk_trace_npz.exists() or "risk_trace_row" not in row:
        return None
    trace_index = int(row["risk_trace_row"])
    with np.load(risk_trace_npz) as data:
        traces = data["risk_trace"]
        if trace_index < 0 or trace_index >= traces.shape[0]:
            return None
        return np.asarray(traces[trace_index], dtype=np.float64)


def render_natural_tail_event_gif(
    *,
    config_path: Path,
    segment_row: pd.Series,
    output_path: Path,
    risk_trace_npz: Path,
    options: NaturalPlaybackOptions | None = None,
) -> Path:
    options = options or NaturalPlaybackOptions()
    cfg = load_config(str(config_path))
    raw_dir = resolve_data_path(cfg["paths"]["raw_dir"], config_path)
    recording_id = int(segment_row["recording_id"])
    target_fps = int(cfg.get("sampling", {}).get("target_fps", 25))
    recording = load_recording(str(raw_dir), recording_id)
    recording = normalize_driving_direction(recording)
    recording = filter_abnormal_tracks(recording, cfg)
    recording = resample_recording(recording, target_fps)

    start_frame = int(segment_row["window_start_frame"])
    end_frame = int(segment_row["window_end_frame"])
    frames = list(range(start_frame, end_frame + 1, max(int(options.frame_stride), 1)))
    if frames[-1] != end_frame:
        frames.append(end_frame)
    window_slice = _window_track_slice(
        recording,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    y_limits = _lane_limits(recording, window_slice)
    lane_lines = _lane_markings(recording)
    risk_trace = _risk_trace_for_row(segment_row, risk_trace_npz=risk_trace_npz)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.2, 5.2),
        dpi=int(options.dpi),
        gridspec_kw={"height_ratios": [3.2, 1.0]},
        constrained_layout=True,
    )
    duration = 1.0 / max(float(options.fps), 1.0e-6)
    with imageio.get_writer(output_path, mode="I", duration=duration) as writer:
        for frame_id in frames:
            _draw_frame(
                axes=axes,
                recording=recording,
                row=segment_row,
                frame_id=int(frame_id),
                risk_trace=risk_trace,
                options=options,
                y_limits=y_limits,
                lane_lines=lane_lines,
            )
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(np.asarray(rgba[:, :, :3], dtype=np.uint8))
    plt.close(fig)
    LOGGER.info("Wrote natural tail playback: %s", output_path)
    return output_path


def render_natural_tail_events(
    *,
    config_path: Path,
    tail_contexts_csv: Path,
    output_dir: Path,
    risk_trace_npz: Path,
    top_k: int,
    options: NaturalPlaybackOptions | None = None,
) -> list[Path]:
    if not tail_contexts_csv.exists():
        raise FileNotFoundError(f"Natural tail contexts CSV not found: {tail_contexts_csv}")
    contexts = pd.read_csv(tail_contexts_csv)
    if contexts.empty:
        raise RuntimeError(f"Natural tail contexts CSV is empty: {tail_contexts_csv}")
    contexts = contexts.sort_values(
        ["event_risk", "recording_id", "window_start_frame"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    if int(top_k) > 0:
        contexts = contexts.head(int(top_k))

    output_dir = ensure_dir(output_dir)
    outputs: list[Path] = []
    for rank, (_, row) in enumerate(contexts.iterrows(), start=1):
        segment_id = str(row["segment_id"]).replace("/", "_")
        output_path = output_dir / f"natural_tail_{rank:03d}_{segment_id}.gif"
        outputs.append(
            render_natural_tail_event_gif(
                config_path=config_path,
                segment_row=row,
                output_path=output_path,
                risk_trace_npz=risk_trace_npz,
                options=options,
            )
        )
    return outputs
