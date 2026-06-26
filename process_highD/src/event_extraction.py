"""
event_extraction.py — 事件抽取
===============================
从 highD recording 中提取 following 和 cut-in 交互事件。
参考: Matlab longfilter_onlycar.m, CutInFilter.m

事件筛选原则:
  所有过滤条件均为语义/运动学规则；本模块只输出事件元数据，
  不计算或输出基础交互指标或综合危险得分。
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

import numpy as np
import pandas as pd

from .lane_utils import are_adjacent_lanes, detect_lane_changes, parse_lane_markings

logger = logging.getLogger(__name__)
PASSENGER_CAR_CLASS = "car"


@dataclass
class EventRecord:
    """Single extracted highD interaction event."""

    event_id: str = ""
    event_type: str = ""
    recording_id: int = -1
    ego_id: int = -1
    target_id: int = -1
    ego_class: str = ""
    target_class: str = ""

    start_frame: int = -1
    end_frame: int = -1
    anchor_frame: int = -1

    cross_frame: Optional[int] = None
    cutin_start_frame: Optional[int] = None
    cutin_end_frame: Optional[int] = None
    source_lane: Optional[int] = None
    target_lane: Optional[int] = None

    def to_row(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "recording_id": self.recording_id,
            "ego_id": self.ego_id,
            "target_id": self.target_id,
            "ego_class": self.ego_class,
            "target_class": self.target_class,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "anchor_frame": self.anchor_frame,
            "cross_frame": self.cross_frame,
            "cutin_start_frame": self.cutin_start_frame,
            "cutin_end_frame": self.cutin_end_frame,
            "source_lane": self.source_lane,
            "target_lane": self.target_lane,
        }


def _cutin_min_post_steps(recording, config) -> int:
    cutin_cfg = config.get("cutin", {})
    fps = int(recording.recording_meta.get(
        "frameRate",
        config.get("sampling", {}).get("target_fps", 25),
    ))
    if "min_post_cutin_duration_seconds" in cutin_cfg:
        return max(
            int(np.ceil(float(cutin_cfg["min_post_cutin_duration_seconds"]) * fps)),
            1,
        )
    return max(int(cutin_cfg.get("min_post_cutin_duration_steps", 10)), 1)


def _align_frames(rec, ego_id, target_id, frame_range=None):
    """对齐 ego 与 target 的公共帧范围"""
    ego_t = rec.get_vehicle_track(ego_id)
    tgt_t = rec.get_vehicle_track(target_id)
    common = sorted(set(ego_t.index) & set(tgt_t.index))
    if frame_range is not None:
        common = [f for f in common if frame_range[0] <= f <= frame_range[1]]
    if not common:
        return np.array([]), pd.DataFrame(), pd.DataFrame()
    common = np.array(common)
    return common, ego_t.loc[common], tgt_t.loc[common]


def _net_gap(ego_x, target_x, ego_length, target_length):
    return target_x - ego_x - 0.5 * (target_length + ego_length)


def _net_gap_series(ego_df, tgt_df, ego_length, target_length):
    return _net_gap(
        ego_df["x"].values,
        tgt_df["x"].values,
        ego_length,
        target_length,
    )


# Following 事件抽取

def _is_passenger_car(meta_row) -> bool:
    return str(meta_row.get("class", "")).strip().lower() == PASSENGER_CAR_CLASS


def extract_following_segments(recording, config):
    """提取所有跟驰事件段

    筛选规则 (语义/运动学):
    1. ego 和 lead 均为小汽车
    2. segment 内 ego 无换道
    3. precedingId 连续 >= min_same_preceding_steps
    4. 两车同车道 (>= 80%)
    5. median gap > min_positive_gap
    """
    fol_cfg = config.get("following", {})
    filt_cfg = config.get("filters", {})
    min_steps = int(fol_cfg.get("min_same_preceding_steps", 40))
    min_gap = filt_cfg.get("min_positive_gap", 0.5)

    meta = recording.tracks_meta
    events = []
    event_counter = 0

    for ego_id in meta.index:
        ego_meta = meta.loc[ego_id]
        if not _is_passenger_car(ego_meta):
            continue
        ego_track = recording.get_vehicle_track(ego_id)
        prec_ids = ego_track["precedingId"].values
        frames = ego_track.index.values

        # 找连续相同 precedingId 的段
        segments = []
        seg_start = 0
        for i in range(1, len(prec_ids)):
            if prec_ids[i] != prec_ids[seg_start] or prec_ids[i] == -1:
                if prec_ids[seg_start] != -1 and (i - seg_start) >= min_steps:
                    segments.append((seg_start, i - 1, int(prec_ids[seg_start])))
                seg_start = i
        if prec_ids[seg_start] != -1 and (len(prec_ids) - seg_start) >= min_steps:
            segments.append((seg_start, len(prec_ids) - 1, int(prec_ids[seg_start])))

        for s_start, s_end, lead_id in segments:
            if (
                lead_id not in meta.index
                or not _is_passenger_car(meta.loc[lead_id])
            ):
                continue

            seg_frames = frames[s_start:s_end + 1]
            fr_range = (int(seg_frames[0]), int(seg_frames[-1]))
            common_f, ego_df, tgt_df = _align_frames(
                recording,
                ego_id,
                lead_id,
                fr_range,
            )
            if len(common_f) < min_steps:
                continue

            if "_abnormal" in ego_df.columns and ego_df["_abnormal"].any():
                continue
            if "_abnormal" in tgt_df.columns and tgt_df["_abnormal"].any():
                continue

            # 同车道检查
            ego_lanes = ego_df["laneId"].values
            tgt_lanes = tgt_df["laneId"].values
            if len(np.unique(ego_lanes)) > 1:
                continue
            if np.mean(ego_lanes == tgt_lanes) < 0.8:
                continue

            ego_len = float(meta.loc[ego_id]["width"])
            tgt_len = float(meta.loc[lead_id]["width"])
            gap = _net_gap_series(ego_df, tgt_df, ego_len, tgt_len)
            if np.median(gap) < min_gap:
                continue
            anchor_idx = len(common_f) // 2

            event_counter += 1
            events.append(
                EventRecord(
                    event_id=(
                        f"fol_{recording.recording_id:02d}_{event_counter:05d}"
                    ),
                    event_type="following",
                    recording_id=recording.recording_id,
                    ego_id=ego_id,
                    target_id=lead_id,
                    ego_class=str(ego_meta.get("class", "")),
                    target_class=str(meta.loc[lead_id].get("class", "")),
                    start_frame=int(common_f[0]),
                    end_frame=int(common_f[-1]),
                    anchor_frame=int(common_f[anchor_idx]),
                )
            )

    logger.info("Recording %02d: 提取 %d 个 following 事件",
                recording.recording_id, len(events))
    return events


# Cut-in 事件抽取

def _has_vehicle_between(recording, ego_id, target_id, frame, lane_id):
    """Return True when another vehicle sits between ego and target in lane."""
    try:
        ego_row = recording.get_vehicle_track(ego_id).loc[frame]
        target_row = recording.get_vehicle_track(target_id).loc[frame]
    except KeyError:
        return True

    ego_x = float(ego_row["x"])
    target_x = float(target_row["x"])
    if target_x <= ego_x:
        return True

    frame_df = recording.get_frame(frame)
    vids = frame_df.index.get_level_values("id").unique()
    for vid in vids:
        if vid in {ego_id, target_id}:
            continue
        row = frame_df.loc[(vid, frame)]
        if int(row["laneId"]) != lane_id:
            continue
        other_x = float(row["x"])
        if ego_x < other_x < target_x:
            return True
    return False


def _is_valid_cutin_ego(recording, cutin_id, ego_id, frame, target_lane, min_gap):
    try:
        cutin_row = recording.get_vehicle_track(cutin_id).loc[frame]
        ego_row = recording.get_vehicle_track(ego_id).loc[frame]
    except KeyError:
        return False

    if int(ego_row["laneId"]) != target_lane:
        return False

    meta = recording.tracks_meta
    cutin_len = float(meta.loc[cutin_id]["width"])
    ego_len = float(meta.loc[ego_id]["width"])
    gap = float(cutin_row["x"] - ego_row["x"]) - 0.5 * (cutin_len + ego_len)
    if gap <= min_gap:
        return False
    if _has_vehicle_between(recording, ego_id, cutin_id, frame, target_lane):
        return False

    return True


def match_cutin_ego(recording, lane_change, config):
    """为换道事件匹配被切入的 ego 车辆

    优先使用 followingId,否则在目标车道后方找最近小汽车
    """
    cutin_id = lane_change["vehicle_id"]
    cross_frame = lane_change["cross_frame"]
    target_lane = lane_change["to_lane"]
    meta = recording.tracks_meta
    cutin_track = recording.get_vehicle_track(cutin_id)
    min_gap = config.get("filters", {}).get("min_positive_gap", 0.0)
    end_frame = lane_change.get("stable_after_end", cross_frame)
    check_frames = []
    for frame in [end_frame, cross_frame]:
        if frame not in check_frames:
            check_frames.append(frame)

    # 优先: followingId
    for check_frame in check_frames:
        if check_frame in cutin_track.index:
            fid = int(cutin_track.loc[check_frame, "followingId"])
            if fid != -1 and fid in meta.index:
                if str(meta.loc[fid].get("class", "")).lower() != "truck":
                    if _is_valid_cutin_ego(recording, cutin_id, fid, check_frame, target_lane, min_gap):
                        return fid

    # 候选匹配：在目标车道后方找最近小汽车。优先使用稳定进入目标车道后的帧，
    # 再检查 cross frame；仍要求 target 在 ego 前方且两者之间没有其他车。
    for check_frame in check_frames:
        if check_frame not in cutin_track.index:
            continue
        cutin_x = float(cutin_track.loc[check_frame, "x"])
        frame_df = recording.get_frame(check_frame)
        vids = frame_df.index.get_level_values("id").unique()
        candidates = []
        for vid in vids:
            if vid == cutin_id:
                continue
            row = frame_df.loc[(vid, check_frame)]
            if int(row["laneId"]) != target_lane:
                continue
            if vid in meta.index and str(meta.loc[vid].get("class", "")).lower() == "truck":
                continue
            vx = float(row["x"])
            if vx < cutin_x:
                candidates.append((vid, cutin_x - vx))

        candidates.sort(key=lambda x: x[1])
        for ego_id, _ in candidates:
            if _is_valid_cutin_ego(recording, cutin_id, ego_id, check_frame, target_lane, min_gap):
                return ego_id
    return None


def _extend_cutin_end_to_motion_settled(track, cross_frame, lane_stable_end, config):
    """Extend cut-in end until lateral velocity settles when available."""
    if "yVelocity" not in track.columns:
        return int(lane_stable_end)
    lat_thresh = config.get("cutin", {}).get("lateral_velocity_threshold", 0.15)
    settle_threshold = abs(float(lat_thresh)) * 0.3
    frames = track.index.values
    if len(frames) == 0:
        return int(lane_stable_end)
    cross_idx = min(np.searchsorted(frames, int(cross_frame)), len(frames) - 1)
    yvel = np.abs(track["yVelocity"].astype(float).values)
    settled = int(lane_stable_end)
    for idx in range(cross_idx + 1, len(frames)):
        settled = int(frames[idx])
        if yvel[idx] < settle_threshold:
            break
    return max(int(lane_stable_end), settled)


def extract_cutin_events(recording, config):
    """提取所有 cut-in 事件

    筛选规则 (语义/运动学):
    1. 目标车: 至少 1 次换道的小汽车，遍历所有相邻车道变化
    2. 切入后 target 在 ego 前方 (median post_gap > 0)
    3. 切入后两车同车道 (>= 70%)
    4. cutin 持续时间 >= min_cutin_duration_steps
    5. 间距不全为负 (排除数据对齐错误)
    6. cross_frame 后进入 ego 车道的帧数 >= min_post_cutin_duration_seconds
    7. cross_frame 在 ego-target 公共帧内，且 post window 内 target 是 ego 最近前车
    """
    cutin_cfg = config.get("cutin", {})
    min_post_steps = _cutin_min_post_steps(recording, config)
    min_cutin_duration_steps = cutin_cfg.get("min_cutin_duration_steps", 5)
    min_stable = cutin_cfg.get("min_lane_stable_steps", 5)
    require_immediate = cutin_cfg.get("require_immediate_leader", True)

    meta = recording.tracks_meta
    lane_info = parse_lane_markings(recording.recording_meta)

    events = []
    event_counter = 0

    for vid in meta.index:
        vm = meta.loc[vid]
        if vm.get("numLaneChanges", 0) < 1 or str(vm.get("class", "")).lower() != "car":
            continue

        track = recording.get_vehicle_track(vid)
        lc_list = detect_lane_changes(track, vid, min_stable)
        if not lc_list:
            continue

        for lc in lc_list:
            if lane_info and not are_adjacent_lanes(lc["from_lane"], lc["to_lane"], lane_info):
                continue

            ego_id = match_cutin_ego(recording, lc, config)
            if ego_id is None:
                continue
            if str(meta.loc[ego_id].get("class", "")).lower() == "truck":
                continue

            # Use the lane-id stable envelope as the complete cut-in event time.
            # The lateral-velocity tail in highD can extend for many seconds and
            # is too long for fixed-horizon diffusion windows; stable source lane
            # -> cross -> stable target lane is the semantic completed maneuver.
            cutin_start = int(lc.get("stable_before_start", lc["cross_frame"]))
            cutin_end = _extend_cutin_end_to_motion_settled(
                track,
                lc["cross_frame"],
                int(lc.get("stable_after_end", lc["cross_frame"])),
                config,
            )
            if (cutin_end - cutin_start) < min_cutin_duration_steps:
                continue

            common_f, ego_df, tgt_df = _align_frames(recording, ego_id, vid)
            if lc["cross_frame"] not in common_f:
                continue

            ego_len = float(meta.loc[ego_id]["width"])
            tgt_len = float(meta.loc[vid]["width"])

            # 只在换道和刚完成后的语义窗口内检查 cut-in 关系。后续很久之后的
            # 跟驰变化不应反过来否定一个已经完成的切入事件。
            cross_idx = int(np.flatnonzero(common_f == lc["cross_frame"])[0])
            post_end_frame = max(
                int(cutin_end),
                int(lc["cross_frame"]) + int(min_post_steps) - 1,
            )
            semantic_mask = (
                (common_f >= int(cutin_start))
                & (common_f <= int(post_end_frame))
            )
            if not np.any(semantic_mask):
                continue
            ego_semantic = ego_df.loc[common_f[semantic_mask]]
            tgt_semantic = tgt_df.loc[common_f[semantic_mask]]
            if "_abnormal" in ego_semantic.columns and ego_semantic["_abnormal"].any():
                continue
            if "_abnormal" in tgt_semantic.columns and tgt_semantic["_abnormal"].any():
                continue

            post_mask = (
                (common_f >= int(lc["cross_frame"]))
                & (common_f <= int(post_end_frame))
            )
            if not np.any(post_mask):
                continue
            post_ego = ego_df.loc[common_f[post_mask]]
            post_tgt = tgt_df.loc[common_f[post_mask]]
            post_gap = post_tgt["x"].values - post_ego["x"].values - 0.5 * (tgt_len + ego_len)
            if len(post_gap) < min_post_steps:
                continue

            max_post_gap = cutin_cfg.get("max_post_cutin_gap", 120.0)
            post_gap_median = float(np.median(post_gap))
            if post_gap_median < 0 or post_gap_median > max_post_gap:
                continue

            post_ego_lanes = post_ego["laneId"].values
            post_tgt_lanes = post_tgt["laneId"].values
            if len(post_ego_lanes) > 0 and np.mean(post_ego_lanes == post_tgt_lanes) < 0.7:
                continue
            if require_immediate:
                post_frames = common_f[cross_idx:cross_idx + min_post_steps]
                blocked = any(
                    _has_vehicle_between(recording, ego_id, vid, int(frame), lc["to_lane"])
                    for frame in post_frames
                )
                if blocked:
                    continue

            anchor_phase = str(cutin_cfg.get("anchor_phase", "cutin_start"))
            if anchor_phase == "cross":
                anchor_frame = int(lc["cross_frame"])
            elif anchor_phase == "cutin_start":
                anchor_frame = int(cutin_start)
            else:
                raise ValueError(
                    "cutin.anchor_phase must be 'cutin_start' or 'cross', "
                    f"got {anchor_phase!r}"
                )

            event_counter += 1
            events.append(EventRecord(
                event_id=f"cin_{recording.recording_id:02d}_{event_counter:05d}",
                event_type="cut_in",
                recording_id=recording.recording_id,
                ego_id=ego_id, target_id=vid,
                start_frame=int(common_f[0]), end_frame=int(common_f[-1]),
                anchor_frame=anchor_frame,
                cross_frame=lc["cross_frame"],
                cutin_start_frame=cutin_start, cutin_end_frame=cutin_end,
                source_lane=lc["from_lane"], target_lane=lc["to_lane"],
            ))

    logger.info("Recording %02d: 提取 %d 个 cut-in 事件",
                recording.recording_id, len(events))
    return events
