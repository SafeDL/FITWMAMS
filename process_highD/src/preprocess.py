"""将 highD 原始轨迹转为车辆中心坐标、统一行驶方向并标记异常。

方向统一原则 (参考 Matlab longfilter_onlycar.m):
  - highD 中 drivingDirection == 1 的车辆沿 x 负方向行驶
  - 统一为 ego forward = positive x
  - 对 drivingDirection == 1: x → −x, xVelocity → −xVelocity, xAcceleration → −xAcceleration

参考:
  - highD-dataset/Matlab/tools/longfilter_onlycar.m (行 264, 285: sign(xVelocity(1)))
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .loader import HighDRecording, load_recording

logger = logging.getLogger(__name__)


def check_frame_continuity(track: pd.DataFrame) -> bool:
    """检查轨迹帧是否连续 (步长为 1)。"""
    frames = track.index.values
    if len(frames) <= 1:
        return True
    return bool(np.all(np.diff(frames) == 1))


def normalize_driving_direction(recording: HighDRecording) -> HighDRecording:
    """就地转换为车辆中心坐标，并将纵向统一为正向行驶。"""
    tracks = recording.tracks
    meta = recording.tracks_meta

    flip_ids = set(meta[meta["drivingDirection"] == 1].index)

    # 统一将 x 和 y 转换为车辆几何中心 (highD 原始 x,y 为 top-left 边界框)
    vehicle_ids = tracks.index.get_level_values("id")
    widths = meta.loc[vehicle_ids, "width"].values
    heights = meta.loc[vehicle_ids, "height"].values
    tracks["x"] = tracks["x"] + widths / 2.0
    tracks["y"] = tracks["y"] + heights / 2.0

    if not flip_ids:
        logger.debug("Recording %02d: 无需方向翻转。", recording.recording_id)
        return recording

    logger.debug(
        "Recording %02d: 翻转 %d 辆车的 x 方向。",
        recording.recording_id,
        len(flip_ids),
    )

    mask = vehicle_ids.isin(flip_ids)

    tracks.loc[mask, "x"] = -tracks.loc[mask, "x"]
    tracks.loc[mask, "xVelocity"] = -tracks.loc[mask, "xVelocity"]
    tracks.loc[mask, "xAcceleration"] = -tracks.loc[mask, "xAcceleration"]

    # precedingXVelocity 也要翻转（它是前车的速度，同方向行驶）
    if "precedingXVelocity" in tracks.columns:
        tracks.loc[mask, "precedingXVelocity"] = -tracks.loc[mask, "precedingXVelocity"]

    recording._vehicle_cache.clear()
    recording._frame_cache.clear()

    return recording


def filter_abnormal_tracks(recording: HighDRecording, config: dict) -> HighDRecording:
    """就地添加 '_abnormal' 标记，不删除轨迹。

    检查帧连续性、加速度、jerk、最低速度、位置跳变和车辆尺寸。
    """
    filters = config["filters"]
    max_accel = float(filters["max_abs_accel"])
    max_jerk = float(filters["max_abs_jerk"])
    max_jump = float(filters["max_position_jump"])
    min_speed = float(filters["min_vehicle_speed"])
    fps = float(recording.recording_meta["frameRate"])

    tracks = recording.tracks

    tracks["_abnormal"] = False

    accel_mask = tracks["xAcceleration"].abs() > max_accel
    tracks.loc[accel_mask, "_abnormal"] = True
    n_accel = accel_mask.sum()

    jerk_x = tracks.groupby(level="id")["xAcceleration"].diff().abs() * fps
    jerk_y = tracks.groupby(level="id")["yAcceleration"].diff().abs() * fps
    jerk_mask = (jerk_x > max_jerk) | (jerk_y > max_jerk)
    tracks.loc[jerk_mask.fillna(False), "_abnormal"] = True
    n_jerk = int(jerk_mask.sum())

    speed = np.hypot(tracks["xVelocity"], tracks["yVelocity"])
    speed_mask = speed < min_speed
    tracks.loc[speed_mask, "_abnormal"] = True
    n_speed = int(speed_mask.sum())

    x_diff = tracks.groupby(level="id")["x"].diff()
    y_diff = tracks.groupby(level="id")["y"].diff()
    jump_mask = np.hypot(x_diff, y_diff) > max_jump
    tracks.loc[jump_mask, "_abnormal"] = True
    n_jump = jump_mask.sum()

    meta = recording.tracks_meta
    bad_size_ids = set()
    for vid in meta.index:
        w = meta.loc[vid, "width"]
        h = meta.loc[vid, "height"]
        if pd.isna(w) or pd.isna(h) or w <= 0 or h <= 0:
            bad_size_ids.add(vid)
    if bad_size_ids:
        size_mask = tracks.index.get_level_values("id").isin(bad_size_ids)
        tracks.loc[size_mask, "_abnormal"] = True

    discontinuous_ids = set()
    for vid in meta.index:
        if not check_frame_continuity(recording.get_vehicle_track(vid)):
            discontinuous_ids.add(vid)
    if discontinuous_ids:
        discontinuous_mask = tracks.index.get_level_values("id").isin(discontinuous_ids)
        tracks.loc[discontinuous_mask, "_abnormal"] = True

    n_total_abnormal = tracks["_abnormal"].sum()
    logger.info(
        "Recording %02d: 异常帧标记 — 加速度过大=%d, jerk过大=%d, "
        "速度过低=%d, 位置跳变=%d, 尺寸异常=%d辆, 帧不连续=%d辆, 总标记帧=%d",
        recording.recording_id,
        n_accel,
        n_jerk,
        n_speed,
        n_jump,
        len(bad_size_ids),
        len(discontinuous_ids),
        n_total_abnormal,
    )

    recording._vehicle_cache.clear()
    recording._frame_cache.clear()

    return recording


def resample_recording(recording: HighDRecording, target_fps: int) -> HighDRecording:
    """确认处理帧率与 highD 原生帧率相同。

    窗口连续性依赖原始 frame 编号。改变帧率需要同时重建时间轴、
    frame 编号和运动量，当前预处理不做不完整的抽帧。
    """
    source_fps = int(recording.recording_meta["frameRate"])

    if source_fps == target_fps:
        logger.debug(
            "Recording %02d: 帧率已为 %d, 无需重采样。",
            recording.recording_id,
            target_fps,
        )
        return recording

    raise ValueError(
        "process_highD currently requires native highD frame rate: "
        f"source_fps={source_fps}, target_fps={target_fps}"
    )


def prepare_recording(
    raw_dir: str | Path,
    recording_id: int,
    config: dict,
) -> HighDRecording:
    """加载并执行项目统一的 highD 预处理步骤。"""
    recording = load_recording(str(raw_dir), int(recording_id))
    recording = normalize_driving_direction(recording)
    recording = filter_abnormal_tracks(recording, config)
    target_fps = int(config["sampling"]["target_fps"])
    return resample_recording(recording, target_fps)
