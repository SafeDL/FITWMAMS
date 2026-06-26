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

import numpy as np
import pandas as pd

from .loader import HighDRecording

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
        recording.recording_id, len(flip_ids),
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

def filter_abnormal_tracks(
    recording: HighDRecording, config: dict
) -> HighDRecording:
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
    filters = config.get("filters", {})
    max_accel = filters.get("max_abs_accel", 8.0)
    max_jerk = filters.get("max_abs_jerk")
    max_jump = filters.get("max_position_jump", 5.0)
    min_speed = filters.get("min_vehicle_speed", 0.0)
    fps = float(recording.recording_meta.get("frameRate", 25))

    tracks = recording.tracks

    # 初始化异常标记列
    tracks["_abnormal"] = False

    # ── 规则 2: 加速度过大 ──
    accel_mask = tracks["xAcceleration"].abs() > max_accel
    tracks.loc[accel_mask, "_abnormal"] = True
    n_accel = accel_mask.sum()

    # ── 规则 2b: jerk 过大 ──
    n_jerk = 0
    if max_jerk is not None:
        jerk_x = tracks.groupby(level="id")["xAcceleration"].diff().abs() * fps
        jerk_mask = jerk_x > float(max_jerk)
        if "yAcceleration" in tracks.columns:
            jerk_y = tracks.groupby(level="id")["yAcceleration"].diff().abs() * fps
            jerk_mask |= jerk_y > float(max_jerk)
        tracks.loc[jerk_mask.fillna(False), "_abnormal"] = True
        n_jerk = int(jerk_mask.sum())

    # ── 规则 2c: 速度过低 ──
    n_speed = 0
    if min_speed is not None and float(min_speed) > 0:
        if "yVelocity" in tracks.columns:
            speed = np.hypot(tracks["xVelocity"], tracks["yVelocity"])
        else:
            speed = tracks["xVelocity"].abs()
        speed_mask = speed < float(min_speed)
        tracks.loc[speed_mask, "_abnormal"] = True
        n_speed = int(speed_mask.sum())

    # ── 规则 3: 位置跳变 ──
    # 按车辆分组计算相邻帧 x 差值
    x_diff = tracks.groupby(level="id")["x"].diff().abs()
    jump_mask = x_diff > max_jump
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
        try:
            vtrack = recording.get_vehicle_track(vid)
            if not check_frame_continuity(vtrack):
                discontinuous_ids.add(vid)
        except KeyError:
            continue

    n_total_abnormal = tracks["_abnormal"].sum()
    logger.info(
        "Recording %02d: 异常帧标记 — 加速度过大=%d, jerk过大=%d, "
        "速度过低=%d, 位置跳变=%d, 尺寸异常=%d辆, 帧不连续=%d辆, 总标记帧=%d",
        recording.recording_id, n_accel, n_jerk, n_speed, n_jump,
        len(bad_size_ids), len(discontinuous_ids), n_total_abnormal,
    )

    # 清空缓存
    recording._vehicle_cache.clear()
    recording._frame_cache.clear()

    return recording


# 重采样

def resample_recording(
    recording: HighDRecording, target_fps: int
) -> HighDRecording:
    """将 recording 从 source_fps 重采样到 target_fps。

    使用每隔 step 帧取样的方式 (step = source_fps / target_fps)。
    仅当 source_fps 是 target_fps 的整数倍时支持。

    Parameters
    ----------
    recording : HighDRecording
    target_fps : int

    Returns
    -------
    HighDRecording
    """
    source_fps = int(recording.recording_meta.get("frameRate", 25))

    if source_fps == target_fps:
        logger.debug("Recording %02d: 帧率已为 %d, 无需重采样。",
                      recording.recording_id, target_fps)
        return recording

    step = source_fps / target_fps  # 允许浮点步长
    logger.info(
        "Recording %02d: 重采样 %d → %d fps (step=%.2f)",
        recording.recording_id, source_fps, target_fps, step,
    )

    tracks = recording.tracks

    # 按车辆分组，每 step 帧取一行
    resampled_parts = []
    for vid, vtrack in tracks.groupby(level="id"):
        frames = vtrack.index.get_level_values("frame").values
        n = len(frames)
        # 使用浮点步长选取帧索引
        keep_indices = np.round(np.arange(0, n, step)).astype(int)
        keep_indices = keep_indices[keep_indices < n]
        keep_frames = frames[keep_indices]
        idx = pd.MultiIndex.from_arrays(
            [[vid] * len(keep_frames), keep_frames],
            names=["id", "frame"],
        )
        resampled_parts.append(vtrack.loc[idx])

    recording.tracks = pd.concat(resampled_parts)
    recording.tracks.sort_index(inplace=True)

    # 更新元数据中的帧率
    recording.recording_meta["frameRate"] = target_fps

    # 清空缓存
    recording._vehicle_cache.clear()
    recording._frame_cache.clear()

    return recording
