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

import numpy as np


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
    upper = np.asarray(recording_meta["upperLaneMarkings"], dtype=float)
    lower = np.asarray(recording_meta["lowerLaneMarkings"], dtype=float)

    if len(upper) < 2 and len(lower) < 2:
        raise ValueError(f"无法解析车道线: upper={upper}, lower={lower}")

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

    return {
        "upper_markings": upper,
        "lower_markings": lower,
        "lanes": lanes,
        "direction_1_lanes": direction_1_lanes,
        "direction_2_lanes": direction_2_lanes,
    }



def are_adjacent_lanes(lane_a: int, lane_b: int, lane_info: dict) -> bool:
    """判断两个 highD laneId 是否相邻。

    highD laneId 编号规则:
    - 上方车道 (drivingDirection=1): laneId 从 2 开始
    - 下方车道 (drivingDirection=2): laneId 接续上方之后
    - 中间可能跳号 (因为边界线不算车道)

    相邻定义: laneId 差为 1 且属于同一方向。
    """
    lanes = lane_info["lanes"]
    return (
        abs(lane_a - lane_b) == 1
        and lanes[lane_a]["direction"] == lanes[lane_b]["direction"]
    )
