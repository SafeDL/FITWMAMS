"""
loader.py — highD 数据读取器
=============================
读取 highD 三类 CSV 文件 (tracks / tracksMeta / recordingMeta)，
构建便于按车辆和按帧查询的 HighDRecording 数据结构。

参考实现:
  - highD-dataset/Python/src/data_management/read_csv.py (官方 Python 示例)
  - highD-dataset/Matlab/tools/readInTracksCsv.m  (Matlab 读取逻辑)
  - highD-dataset/Matlab/tools/readInVideoCsv.m   (Matlab 录像元数据)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ID_COLUMNS = [
    "precedingId", "followingId",
    "leftPrecedingId", "leftAlongsideId", "leftFollowingId",
    "rightPrecedingId", "rightAlongsideId", "rightFollowingId",
]


# HighDRecording 核心类

class HighDRecording:
    """封装单个 highD recording 的全部数据。

    Parameters
    ----------
    recording_id : int
        录像编号 (1-60)。
    tracks : pd.DataFrame
        tracks CSV 数据，按 (id, frame) 建立 MultiIndex。
    tracks_meta : pd.DataFrame
        tracksMeta CSV 数据，index = vehicle_id。
    recording_meta : dict
        recordingMeta 解析后的字典。
    """

    def __init__(
        self,
        recording_id: int,
        tracks: pd.DataFrame,
        tracks_meta: pd.DataFrame,
        recording_meta: dict,
    ):
        self.recording_id = recording_id
        self.tracks = tracks
        self.tracks_meta = tracks_meta
        self.recording_meta = recording_meta

        # 缓存字典: vehicle_id -> track_df
        self._vehicle_cache: dict[int, pd.DataFrame] = {}
        # 缓存字典: frame_id -> frame_df
        self._frame_cache: dict[int, pd.DataFrame] = {}

    def get_vehicle_track(self, vehicle_id: int) -> pd.DataFrame:
        """返回指定车辆的完整轨迹 DataFrame。"""
        if vehicle_id not in self._vehicle_cache:
            try:
                self._vehicle_cache[vehicle_id] = (
                    self.tracks.loc[vehicle_id].copy()
                )
            except KeyError:
                raise KeyError(
                    f"Recording {self.recording_id}: 车辆 {vehicle_id} 不存在。"
                )
        return self._vehicle_cache[vehicle_id]

    def get_frame(self, frame_id: int) -> pd.DataFrame:
        """返回指定帧中所有车辆的 DataFrame。"""
        if frame_id not in self._frame_cache:
            idx = self.tracks.index.get_level_values("frame") == frame_id
            self._frame_cache[frame_id] = self.tracks.loc[idx].copy()
        return self._frame_cache[frame_id]

    def vehicle_ids(self) -> list[int]:
        """返回所有车辆 ID 列表。"""
        return list(self.tracks_meta.index)

    def frame_ids(self) -> list[int]:
        """返回所有帧 ID 列表 (已排序)。"""
        return sorted(self.tracks.index.get_level_values("frame").unique())

    def __repr__(self) -> str:
        n_veh = len(self.tracks_meta)
        n_frames = len(self.frame_ids())
        return (
            f"HighDRecording(id={self.recording_id}, "
            f"vehicles={n_veh}, frames={n_frames})"
        )


# 公开加载函数

def load_recording(raw_dir: str, recording_id: int) -> HighDRecording:
    """读取指定 recording_id 的 highD 三类 CSV 并构建 HighDRecording

    Parameters
    ----------
    raw_dir : str
        原始数据目录，包含 XX_tracks.csv 等文件。
    recording_id : int
        录像编号 (1-60)。

    Returns
    -------
    HighDRecording
    """
    raw_dir = Path(raw_dir)
    prefix = f"{recording_id:02d}"

    tracks_path = raw_dir / f"{prefix}_tracks.csv"
    meta_path = raw_dir / f"{prefix}_tracksMeta.csv"
    rec_meta_path = raw_dir / f"{prefix}_recordingMeta.csv"

    for p in [tracks_path, meta_path, rec_meta_path]:
        if not p.exists():
            raise FileNotFoundError(f"找不到文件: {p}")

    logger.info("加载 recording %02d ...", recording_id)

    # ── 读取 tracks ──
    tracks_df = pd.read_csv(tracks_path)

    # 统一无效 ID 字段为 -1（原始数据中用 0 表示无效）
    for col in _ID_COLUMNS:
        if col in tracks_df.columns:
            tracks_df[col] = tracks_df[col].replace(0, -1).astype(int)

    # 建立 MultiIndex: (id, frame)
    tracks_df = tracks_df.set_index(["id", "frame"])
    tracks_df.sort_index(inplace=True)

    # ── 读取 tracksMeta ──
    tracks_meta_df = pd.read_csv(meta_path)
    tracks_meta_df = tracks_meta_df.set_index("id")

    # ── 读取 recordingMeta ──
    rec_meta_df = pd.read_csv(rec_meta_path)
    rec_meta = rec_meta_df.iloc[0].to_dict()

    # 解析 lane markings
    rec_meta["upperLaneMarkings"] = np.fromstring(
        str(rec_meta.get("upperLaneMarkings", "")), sep=";"
    )
    rec_meta["lowerLaneMarkings"] = np.fromstring(
        str(rec_meta.get("lowerLaneMarkings", "")), sep=";"
    )

    # ── 验证 ──
    n_meta = len(tracks_meta_df)
    n_tracks = tracks_df.index.get_level_values(0).nunique()
    if n_meta != n_tracks:
        logger.warning(
            "Recording %02d: tracksMeta 车辆数 (%d) != tracks 车辆数 (%d)",
            recording_id, n_meta, n_tracks,
        )

    recording = HighDRecording(
        recording_id=recording_id,
        tracks=tracks_df,
        tracks_meta=tracks_meta_df,
        recording_meta=rec_meta,
    )
    logger.info("加载完成: %s", recording)
    return recording
