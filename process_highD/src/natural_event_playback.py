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
from matplotlib.transforms import Affine2D  # noqa: E402
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
    fps: float = 25.0
    frame_stride: int = 1
    view_width_m: float = 160.0
    trail_frames: int = 50
    max_background_vehicles: int = 120
    dpi: int = 100
    figsize: tuple[float, float] = (12.0, 4.8)
    lane_width_m: float = 3.75


EGO_COLOR = "#e31a1c"
TARGET_COLOR = "#1f78b4"
SLOT_COLOR = "#ff7f0e"
BACKGROUND_COLOR = "#bdbdbd"
ROAD_COLOR = "#6f7378"

SLOT_LABELS = {
    "same_front": "front",
    "same_rear": "rear",
    "left_front": "left front",
    "left_rear": "left rear",
    "right_front": "right front",
    "right_rear": "right rear",
}


def _vehicle_row(frame: pd.DataFrame, vehicle_id: int) -> pd.Series | None:
    if frame.empty:
        return None
    ids = frame.index.get_level_values("id")
    mask = ids == int(vehicle_id)
    if not np.any(mask):
        return None
    return frame.loc[mask].iloc[0]


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


def _lateral_sign(driving_direction: int) -> float:
    # highD image y increases downward.  Keep playback in an ego-left-positive
    # coordinate system after x has been normalized to forward-positive.
    return 1.0 if int(driving_direction) == 1 else -1.0


def _focused_y_limits(
    *,
    frame_slice: pd.DataFrame,
    focus_ids: set[int],
    lane_width_m: float,
) -> tuple[float, float]:
    road_half_width = 1.76 * float(lane_width_m)
    if "y" not in frame_slice or frame_slice.empty:
        return -road_half_width, road_half_width
    ids = frame_slice.index.get_level_values("id")
    focus_slice = frame_slice.loc[ids.isin(focus_ids)] if focus_ids else frame_slice
    if focus_slice.empty:
        focus_slice = frame_slice
    y = pd.to_numeric(focus_slice["y"], errors="coerce")
    y = y[np.isfinite(y)]
    if y.empty:
        y = pd.to_numeric(frame_slice["y"], errors="coerce")
        y = y[np.isfinite(y)]
    if y.empty:
        return -road_half_width, road_half_width
    center_y = float(np.nanmedian(y.to_numpy(dtype=np.float64)))
    ymin = min(float(y.min()) - 2.5, center_y - road_half_width)
    ymax = max(float(y.max()) + 2.5, center_y + road_half_width)
    return ymin, ymax


def _window_track_slice(
    recording: HighDRecording,
    *,
    start_frame: int,
    end_frame: int,
) -> pd.DataFrame:
    frames = recording.tracks.index.get_level_values("frame")
    return recording.tracks.loc[(frames >= int(start_frame)) & (frames <= int(end_frame))]


def _vehicle_dimensions(
    *,
    recording: HighDRecording,
    row: pd.Series,
    vehicle_id: int,
) -> tuple[float, float]:
    if int(vehicle_id) in recording.tracks_meta.index:
        meta = recording.tracks_meta.loc[int(vehicle_id)]
        length = max(float(meta.get("width", 4.5)), 0.5)
        width = max(float(meta.get("height", 1.8)), 0.4)
    else:
        length = max(float(row.get("width", 4.5)), 0.5)
        width = max(float(row.get("height", 1.8)), 0.4)
    return length, width


def _vehicle_heading(row: pd.Series, *, lateral_sign: float) -> float:
    vx = float(row.get("xVelocity", row.get("vx", 0.0)))
    vy = float(lateral_sign) * float(row.get("yVelocity", row.get("vy", 0.0)))
    if abs(vx) + abs(vy) < 1.0e-9:
        return 0.0
    return float(np.arctan2(vy, vx))


def _draw_vehicle(
    ax: Any,
    *,
    recording: HighDRecording,
    row: pd.Series,
    vehicle_id: int,
    origin_x: float,
    origin_y: float,
    lateral_sign: float,
    color: str,
    alpha: float,
    zorder: int,
    label: str | None = None,
) -> None:
    length, width = _vehicle_dimensions(
        recording=recording,
        row=row,
        vehicle_id=vehicle_id,
    )
    x = float(row["x"]) - float(origin_x)
    y = float(lateral_sign) * (float(row["y"]) - float(origin_y))
    rect = Rectangle(
        (-0.5 * length, -0.5 * width),
        length,
        width,
        facecolor=color,
        edgecolor="black",
        linewidth=0.8,
        alpha=alpha,
        zorder=zorder,
    )
    rect.set_transform(
        Affine2D().rotate(_vehicle_heading(row, lateral_sign=lateral_sign)).translate(
            x,
            y,
        )
        + ax.transData
    )
    ax.add_patch(rect)
    if label:
        ax.text(
            x,
            y + 0.8 * width,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            color="black",
            zorder=zorder + 1,
            bbox={
                "facecolor": "white",
                "alpha": 0.78,
                "edgecolor": "none",
                "pad": 1.4,
            },
        )


def _draw_vehicle_trail(
    ax: Any,
    *,
    recording: HighDRecording,
    vehicle_id: int,
    frame_id: int,
    start_frame: int,
    origin_x: float,
    origin_y: float,
    lateral_sign: float,
    trail_frames: int,
    color: str,
    zorder: int,
) -> None:
    try:
        track = recording.get_vehicle_track(int(vehicle_id))
    except KeyError:
        return
    trail_start = max(int(start_frame), int(frame_id) - int(trail_frames))
    window = track.loc[
        (track.index >= trail_start) & (track.index <= int(frame_id))
    ]
    if len(window) < 2 or "x" not in window or "y" not in window:
        return
    xs = pd.to_numeric(window["x"], errors="coerce").to_numpy(dtype=np.float64)
    ys = pd.to_numeric(window["y"], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if int(np.sum(mask)) < 2:
        return
    ax.plot(
        xs[mask] - float(origin_x),
        float(lateral_sign) * (ys[mask] - float(origin_y)),
        color=color,
        linewidth=1.6,
        alpha=0.78,
        zorder=zorder,
    )


def _draw_lane_markings(
    ax: Any,
    *,
    lane_lines: list[float],
    y_limits: tuple[float, float],
    origin_y: float,
    lateral_sign: float,
) -> None:
    visible_lines = [
        y for y in lane_lines if y_limits[0] - 0.2 <= float(y) <= y_limits[1] + 0.2
    ]
    for idx, y in enumerate(visible_lines):
        is_outer = idx == 0 or idx == len(visible_lines) - 1
        ax.axhline(
            float(lateral_sign) * (float(y) - float(origin_y)),
            color="#ffffff",
            linewidth=0.8,
            linestyle="-" if is_outer else "--",
            alpha=0.45 if is_outer else 0.28,
            zorder=0,
        )


def _draw_frame(
    *,
    ax: Any,
    recording: HighDRecording,
    row: pd.Series,
    frame_id: int,
    risk_trace: np.ndarray | None,
    options: NaturalPlaybackOptions,
    origin_x: float,
    origin_y: float,
    lateral_sign: float,
    y_limits: tuple[float, float],
    lane_lines: list[float],
    target_fps: int,
) -> None:
    ax.clear()
    ax.set_facecolor(ROAD_COLOR)

    ego_id = int(row["ego_id"])
    slot_ids = _slot_vehicle_ids(row)
    peak_neighbor_id = int(row.get("peak_neighbor_id", -1))
    peak_slot_name = str(row.get("peak_slot_name", "") or "")
    frame = recording.get_frame(int(frame_id))
    ego = _vehicle_row(frame, ego_id)
    if ego is None:
        return
    target = _vehicle_row(frame, peak_neighbor_id) if peak_neighbor_id > 0 else None
    if target is not None:
        center_x = 0.5 * (float(ego["x"]) + float(target["x"])) - float(origin_x)
    else:
        center_x = float(ego["x"]) - float(origin_x)
    half_width = 0.5 * float(options.view_width_m)
    x_limits = (center_x - half_width, center_x + half_width)

    local_x = pd.to_numeric(frame["x"], errors="coerce") - float(origin_x)
    in_view = (
        (local_x >= x_limits[0])
        & (local_x <= x_limits[1])
        & (pd.to_numeric(frame["y"], errors="coerce") >= y_limits[0])
        & (pd.to_numeric(frame["y"], errors="coerce") <= y_limits[1])
    )
    visible = frame.loc[in_view].copy()
    if len(visible) > int(options.max_background_vehicles):
        distance = np.abs(pd.to_numeric(visible["x"], errors="coerce") - float(ego["x"]))
        keep = np.argsort(distance.to_numpy())[: int(options.max_background_vehicles)]
        visible = visible.iloc[np.sort(keep)]

    _draw_lane_markings(
        ax,
        lane_lines=lane_lines,
        y_limits=y_limits,
        origin_y=origin_y,
        lateral_sign=lateral_sign,
    )

    for (vehicle_id, _frame), vehicle in visible.iterrows():
        vid = int(vehicle_id)
        if vid == ego_id or vid in slot_ids:
            continue
        _draw_vehicle(
            ax,
            recording=recording,
            row=vehicle,
            vehicle_id=vid,
            origin_x=origin_x,
            origin_y=origin_y,
            lateral_sign=lateral_sign,
            color=BACKGROUND_COLOR,
            alpha=0.48,
            zorder=1,
        )

    for vid, slot in slot_ids.items():
        if vid == peak_neighbor_id:
            continue
        vehicle = _vehicle_row(frame, vid)
        if vehicle is None:
            continue
        _draw_vehicle_trail(
            ax,
            recording=recording,
            vehicle_id=vid,
            frame_id=frame_id,
            start_frame=int(row["window_start_frame"]),
            origin_x=origin_x,
            origin_y=origin_y,
            lateral_sign=lateral_sign,
            trail_frames=options.trail_frames,
            color=SLOT_COLOR,
            zorder=2,
        )
        _draw_vehicle(
            ax,
            recording=recording,
            row=vehicle,
            vehicle_id=vid,
            origin_x=origin_x,
            origin_y=origin_y,
            lateral_sign=lateral_sign,
            color=SLOT_COLOR,
            alpha=0.92,
            zorder=4,
            label=SLOT_LABELS.get(slot, slot.replace("_", " ")),
        )

    if peak_neighbor_id > 0 and target is not None:
        _draw_vehicle_trail(
            ax,
            recording=recording,
            vehicle_id=peak_neighbor_id,
            frame_id=frame_id,
            start_frame=int(row["window_start_frame"]),
            origin_x=origin_x,
            origin_y=origin_y,
            lateral_sign=lateral_sign,
            trail_frames=options.trail_frames,
            color=TARGET_COLOR,
            zorder=3,
        )
        _draw_vehicle(
            ax,
            recording=recording,
            row=target,
            vehicle_id=peak_neighbor_id,
            origin_x=origin_x,
            origin_y=origin_y,
            lateral_sign=lateral_sign,
            color=TARGET_COLOR,
            alpha=0.95,
            zorder=5,
            label="target",
        )

    _draw_vehicle_trail(
        ax,
        recording=recording,
        vehicle_id=ego_id,
        frame_id=frame_id,
        start_frame=int(row["window_start_frame"]),
        origin_x=origin_x,
        origin_y=origin_y,
        lateral_sign=lateral_sign,
        trail_frames=options.trail_frames,
        color=EGO_COLOR,
        zorder=4,
    )
    _draw_vehicle(
        ax,
        recording=recording,
        row=ego,
        vehicle_id=ego_id,
        origin_x=origin_x,
        origin_y=origin_y,
        lateral_sign=lateral_sign,
        color=EGO_COLOR,
        alpha=0.96,
        zorder=6,
        label="ego",
    )

    start = int(row["window_start_frame"])
    offset = int(frame_id) - start
    elapsed = float(offset) / max(float(target_fps), 1.0e-6)
    event_risk = float(row["event_risk"])
    current_risk = float("nan")
    if risk_trace is not None and risk_trace.size:
        trace_offset = int(np.clip(offset, 0, risk_trace.size - 1))
        current_risk = float(risk_trace[trace_offset])
    risk_text = (
        f"R={current_risk:.3f}/{event_risk:.3f}"
        if np.isfinite(current_risk)
        else f"R={event_risk:.3f}"
    )
    peak_frame = int(row.get("peak_risk_frame", row.get("window_end_frame", frame_id)))
    peak_text = (
        f"peak={peak_slot_name}#{peak_neighbor_id}"
        if peak_neighbor_id > 0 and peak_slot_name
        else "peak=n/a"
    )
    ax.set_xlim(*x_limits)
    transformed_y_limits = [
        float(lateral_sign) * (float(value) - float(origin_y))
        for value in y_limits
    ]
    ax.set_ylim(min(transformed_y_limits), max(transformed_y_limits))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"{row['segment_id']} | rec {int(row['recording_id']):02d} | "
        f"t={elapsed:.2f}s | {risk_text} | {peak_text}",
        fontsize=9,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.tick_params(labelsize=8)
    if abs(int(frame_id) - peak_frame) <= max(int(options.frame_stride), 1) // 2:
        ax.text(
            0.01,
            0.93,
            "peak risk",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#b91c1c",
            fontweight="bold",
            bbox={
                "facecolor": "white",
                "alpha": 0.78,
                "edgecolor": "none",
                "pad": 2.0,
            },
        )


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
    slot_ids = _slot_vehicle_ids(segment_row)
    focus_ids = {int(segment_row["ego_id"]), *slot_ids.keys()}
    peak_neighbor_id = int(segment_row.get("peak_neighbor_id", -1))
    if peak_neighbor_id > 0:
        focus_ids.add(peak_neighbor_id)
    y_limits = _focused_y_limits(
        frame_slice=window_slice,
        focus_ids=focus_ids,
        lane_width_m=options.lane_width_m,
    )
    lane_lines = _lane_markings(recording)
    risk_trace = _risk_trace_for_row(segment_row, risk_trace_npz=risk_trace_npz)
    first_frame = recording.get_frame(start_frame)
    ego_first = _vehicle_row(first_frame, int(segment_row["ego_id"]))
    origin_x = 0.0 if ego_first is None else float(ego_first["x"])
    origin_y = 0.0 if ego_first is None else float(ego_first["y"])
    if "ego_driving_direction" in segment_row:
        ego_direction = int(segment_row["ego_driving_direction"])
    elif int(segment_row["ego_id"]) in recording.tracks_meta.index:
        ego_direction = int(
            recording.tracks_meta.loc[int(segment_row["ego_id"])].get(
                "drivingDirection",
                1,
            )
        )
    else:
        ego_direction = 1
    display_lateral_sign = _lateral_sign(ego_direction)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=options.figsize, dpi=int(options.dpi))
    fig.subplots_adjust(left=0.065, right=0.965, bottom=0.18, top=0.83)
    duration_ms = max(int(round(1000.0 / max(float(options.fps), 1.0e-6))), 1)
    with imageio.get_writer(output_path, mode="I", duration=duration_ms) as writer:
        for frame_id in frames:
            _draw_frame(
                ax=ax,
                recording=recording,
                row=segment_row,
                frame_id=int(frame_id),
                risk_trace=risk_trace,
                options=options,
                origin_x=origin_x,
                origin_y=origin_y,
                lateral_sign=display_lateral_sign,
                y_limits=y_limits,
                lane_lines=lane_lines,
                target_fps=target_fps,
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
