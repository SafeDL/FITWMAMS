"""
preprocess.py — 轨迹清洗、方向统一与重采样
===========================================
将 highD 原始轨迹转换为统一方向、统一帧率、异常已过滤的数据。

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


# 帧连续性检查


def check_frame_continuity(track: pd.DataFrame) -> bool:
    """检查轨迹帧是否连续 (步长为 1)。"""
    frames = track.index.values
    if len(frames) <= 1:
        return True
    return bool(np.all(np.diff(frames) == 1))


# 方向统一


def normalize_driving_direction(recording: HighDRecording) -> HighDRecording:
    """将所有车辆统一为 ego forward = positive x。

    对 drivingDirection == 1 的车辆翻转 x, xVelocity, xAcceleration。
    precedingXVelocity 也需要根据该车的行驶方向翻转。

    Parameters
    ----------
    recording : HighDRecording
        原始 recording (会被就地修改)。

    Returns
    -------
    HighDRecording
        修改后的 recording。
    """
    tracks = recording.tracks
    meta = recording.tracks_meta

    # 找到需要翻转的车辆 (drivingDirection == 1)
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

    # 翻转坐标和速度/加速度
    tracks.loc[mask, "x"] = -tracks.loc[mask, "x"]
    tracks.loc[mask, "xVelocity"] = -tracks.loc[mask, "xVelocity"]
    tracks.loc[mask, "xAcceleration"] = -tracks.loc[mask, "xAcceleration"]

    # precedingXVelocity 也要翻转（它是前车的速度，同方向行驶）
    if "precedingXVelocity" in tracks.columns:
        tracks.loc[mask, "precedingXVelocity"] = -tracks.loc[mask, "precedingXVelocity"]

    # 清空缓存（数据已修改）
    recording._vehicle_cache.clear()
    recording._frame_cache.clear()

    return recording


# 异常轨迹过滤


def filter_abnormal_tracks(recording: HighDRecording, config: dict) -> HighDRecording:
    """标记异常轨迹段 (不立即删除，添加 '_abnormal' 标记列)。

    过滤规则:
    1. 帧不连续的轨迹标记
    2. |xAcceleration| 超过 max_abs_accel
    3. 相邻帧位置跳变超过 max_position_jump
    4. 车辆宽度/高度缺失或为 0

    Parameters
    ----------
    recording : HighDRecording
    config : dict
        filters 子字典。

    Returns
    -------
    HighDRecording
    """
    filters = config["filters"]
    max_accel = float(filters["max_abs_accel"])
    max_jerk = float(filters["max_abs_jerk"])
    max_jump = float(filters["max_position_jump"])
    min_speed = float(filters["min_vehicle_speed"])
    fps = float(recording.recording_meta["frameRate"])

    tracks = recording.tracks

    # 初始化异常标记列
    tracks["_abnormal"] = False

    # ── 规则 2: 加速度过大 ──
    accel_mask = tracks["xAcceleration"].abs() > max_accel
    tracks.loc[accel_mask, "_abnormal"] = True
    n_accel = accel_mask.sum()

    # ── 规则 2b: jerk 过大 ──
    jerk_x = tracks.groupby(level="id")["xAcceleration"].diff().abs() * fps
    jerk_y = tracks.groupby(level="id")["yAcceleration"].diff().abs() * fps
    jerk_mask = (jerk_x > max_jerk) | (jerk_y > max_jerk)
    tracks.loc[jerk_mask.fillna(False), "_abnormal"] = True
    n_jerk = int(jerk_mask.sum())

    # ── 规则 2c: 速度过低 ──
    speed = np.hypot(tracks["xVelocity"], tracks["yVelocity"])
    speed_mask = speed < min_speed
    tracks.loc[speed_mask, "_abnormal"] = True
    n_speed = int(speed_mask.sum())

    # ── 规则 3: 二维位置跳变 ──
    x_diff = tracks.groupby(level="id")["x"].diff()
    y_diff = tracks.groupby(level="id")["y"].diff()
    jump_mask = np.hypot(x_diff, y_diff) > max_jump
    tracks.loc[jump_mask, "_abnormal"] = True
    n_jump = jump_mask.sum()

    # ── 规则 4: 尺寸异常 ──
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

    # ── 规则 1: 帧连续性标记 ──
    discontinuous_ids = set()
    for vid in meta.index:
        if not check_frame_continuity(recording.get_vehicle_track(vid)):
            discontinuous_ids.add(vid)
    if discontinuous_ids:
        discontinuous_mask = tracks.index.get_level_values("id").isin(
            discontinuous_ids
        )
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

    # 清空缓存
    recording._vehicle_cache.clear()
    recording._frame_cache.clear()

    return recording


# 重采样


def resample_recording(recording: HighDRecording, target_fps: int) -> HighDRecording:
    """确认处理帧率与 highD 原生帧率相同。

    窗口连续性依赖原始 frame 编号。改变帧率需要同时重建时间轴、
    frame 编号和运动量，当前预处理不做不完整的抽帧。

    Parameters
    ----------
    recording : HighDRecording
    target_fps : int

    Returns
    -------
    HighDRecording
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
