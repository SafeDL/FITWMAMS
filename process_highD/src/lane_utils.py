"""
lane_utils.py — 车道几何工具
=============================
解析 recordingMeta 中车道线信息，支持车道中心线、车道宽度、
相邻车道检测和车道变化检测。

参考:
  - highD-dataset/Matlab/tools/readInVideoCsv.m (upperLanes / lowerLanes 解析)
  - highD-dataset/Matlab/tools/CutInFilter.m  (换道检测逻辑)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# 车道线解析

def parse_lane_markings(recording_meta: dict) -> dict:
    """解析 recordingMeta 中的车道线信息。

    highD 将公路分为 upper (方向1) 和 lower (方向2) 两组车道，
    每组由分号分隔的 y 坐标表示车道边界线。

    Parameters
    ----------
    recording_meta : dict
        recordingMeta 字典。

    Returns
    -------
    dict
        包含以下键:
        - 'upper_markings': ndarray, 上方车道边界 y 坐标
        - 'lower_markings': ndarray, 下方车道边界 y 坐标
        - 'lanes': dict, lane_id -> {'center': float, 'width': float,
                                      'upper': float, 'lower': float,
                                      'direction': int}
        - 'direction_1_lanes': list[int], 方向1的 lane_id 列表
        - 'direction_2_lanes': list[int], 方向2的 lane_id 列表

    Raises
    ------
    ValueError
        如果无法解析车道线。
    """
    upper = recording_meta.get("upperLaneMarkings", np.array([]))
    lower = recording_meta.get("lowerLaneMarkings", np.array([]))

    # 如果是字符串则解析
    if isinstance(upper, str):
        upper = np.fromstring(upper, sep=";")
    if isinstance(lower, str):
        lower = np.fromstring(lower, sep=";")

    upper = np.array(upper, dtype=float)
    lower = np.array(lower, dtype=float)

    if len(upper) < 2 and len(lower) < 2:
        raise ValueError(
            f"无法解析车道线: upper={upper}, lower={lower}"
        )

    lanes = {}

    # highD laneId 编号规则:
    # - 上方车道 (direction 1): laneId = 2, 3, ..., n_upper+1
    # - 下方车道 (direction 2): laneId = n_upper+3, n_upper+4, ...
    # (中间跳过一个编号用于分隔)
    n_upper = len(upper) - 1  # 上方车道数

    # 方向1 (upper) 车道
    direction_1_lanes = []
    for i in range(n_upper):
        lane_id = i + 2  # highD 从 2 开始
        y_top = upper[i]
        y_bottom = upper[i + 1]
        center = (y_top + y_bottom) / 2.0
        width = abs(y_bottom - y_top)
        lanes[lane_id] = {
            "center": center,
            "width": width,
            "upper": min(y_top, y_bottom),
            "lower": max(y_top, y_bottom),
            "direction": 1,
        }
        direction_1_lanes.append(lane_id)

    # 方向2 (lower) 车道
    n_lower = len(lower) - 1
    direction_2_lanes = []
    for i in range(n_lower):
        lane_id = n_upper + 3 + i  # 跳过分隔
        y_top = lower[i]
        y_bottom = lower[i + 1]
        center = (y_top + y_bottom) / 2.0
        width = abs(y_bottom - y_top)
        lanes[lane_id] = {
            "center": center,
            "width": width,
            "upper": min(y_top, y_bottom),
            "lower": max(y_top, y_bottom),
            "direction": 2,
        }
        direction_2_lanes.append(lane_id)

    result = {
        "upper_markings": upper,
        "lower_markings": lower,
        "lanes": lanes,
        "direction_1_lanes": direction_1_lanes,
        "direction_2_lanes": direction_2_lanes,
    }

    logger.debug(
        "解析车道: 方向1=%d条, 方向2=%d条",
        len(direction_1_lanes), len(direction_2_lanes),
    )
    return result


def are_adjacent_lanes(lane_a: int, lane_b: int, lane_info: dict) -> bool:
    """判断两个 highD laneId 是否相邻。

    highD laneId 编号规则:
    - 上方车道 (drivingDirection=1): laneId 从 2 开始
    - 下方车道 (drivingDirection=2): laneId 接续上方之后
    - 中间可能跳号 (因为边界线不算车道)

    相邻定义: laneId 差为 1 且属于同一方向。
    """
    if lane_a == lane_b:
        return False
    if abs(lane_a - lane_b) != 1:
        return False

    # 确保属于同一方向组
    lanes = lane_info.get("lanes", {})
    if lanes:
        # 如果我们有方向信息
        dir_a = lanes.get(lane_a, {}).get("direction")
        dir_b = lanes.get(lane_b, {}).get("direction")
        if dir_a is not None and dir_b is not None:
            return dir_a == dir_b

    # 如果没有解析信息，仅根据方向组判断
    d1 = lane_info.get("direction_1_lanes", [])
    d2 = lane_info.get("direction_2_lanes", [])
    if (lane_a in d1 and lane_b in d1) or (lane_a in d2 and lane_b in d2):
        return True

    # 默认: 差为1就认为相邻
    return True


# 车道变化检测

def detect_lane_changes(
    track: pd.DataFrame,
    vehicle_id: int,
    min_stable_steps: int = 5,
) -> list[dict]:
    """检测轨迹中的车道变化事件。

    策略 (参考 CutInFilter.m):
    1. 遍历 laneId 序列，找到 laneId 变化点 (cross_frame)
    2. 向前搜索变化前 laneId 连续稳定的起始帧
    3. 向后搜索变化后 laneId 连续稳定的结束帧

    Parameters
    ----------
    track : pd.DataFrame
        某车辆的轨迹 (index=frame)。
    vehicle_id : int
        车辆 ID。
    min_stable_steps : int
        变化前后 laneId 稳定的最少帧数。

    Returns
    -------
    list[dict]
        每个元素为一次车道变化事件:
        {
            "vehicle_id": int,
            "from_lane": int,
            "to_lane": int,
            "cross_frame": int,
            "stable_before_start": int,
            "stable_after_end": int,
        }
    """
    if "laneId" not in track.columns:
        return []

    lane_ids = track["laneId"].values
    frames = track.index.get_level_values("frame").values if "frame" in track.index.names else track.index.values

    if len(lane_ids) < 2:
        return []

    changes = []

    # 找到 laneId 变化点
    diff = np.diff(lane_ids)
    change_indices = np.where(diff != 0)[0]

    for idx in change_indices:
        from_lane = int(lane_ids[idx])
        to_lane = int(lane_ids[idx + 1])
        cross_frame = int(frames[idx + 1])

        # 向前搜索稳定起始帧
        stable_before_start = int(frames[0])
        for i in range(idx, -1, -1):
            if lane_ids[i] != from_lane:
                stable_before_start = int(frames[i + 1])
                break
            if idx - i >= min_stable_steps:
                stable_before_start = int(frames[i])
                break

        # 向后搜索稳定结束帧
        stable_after_end = int(frames[-1])
        for i in range(idx + 1, len(lane_ids)):
            if lane_ids[i] != to_lane:
                stable_after_end = int(frames[i - 1])
                break
            if i - (idx + 1) >= min_stable_steps:
                stable_after_end = int(frames[i])
                break

        changes.append({
            "vehicle_id": vehicle_id,
            "from_lane": from_lane,
            "to_lane": to_lane,
            "cross_frame": cross_frame,
            "stable_before_start": stable_before_start,
            "stable_after_end": stable_after_end,
        })

    return changes
