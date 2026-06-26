#!/usr/bin/env python3
"""
extract_highd_events.py — 从 highD 中抽取驾驶事件
=====================================================
输出:
  results/highd_events/events.csv

用法:
  conda activate tread
  python process_highD/scripts/extract_highd_events.py
"""
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from process_highD.src.event_extraction import extract_cutin_events, extract_following_segments
from process_highD.src.io_utils import ensure_dir, load_config, resolve_data_path, resolve_recording_ids
from process_highD.src.loader import load_recording
from process_highD.src.preprocess import filter_abnormal_tracks, normalize_driving_direction, resample_recording
from tools.highd_exposure import (
    all_vehicle_exposure_for_recording,
    following_exposure_for_recording,
)
from tools.highd_cutin import (
    build_highd_cutin_event_rows_from_recording,
    filter_semantic_cutin_rows,
    highd_cutin_options_from_config,
    highd_cutin_score_table,
    save_highd_cutin_event_context_cache,
    score_highd_cutin_event_rows,
)
from tools.highd_longitudinal import (
    build_highd_event_rows_from_recording,
    build_highd_following_segment_rows_from_recording,
    highd_options_from_config,
    highd_score_table,
    save_highd_event_context_cache,
    save_highd_following_segment_cache,
    score_highd_event_rows,
)
from tools.io import write_csv, write_json

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "highd_default.yaml"
FOLLOWING_SCORE_CACHE = "following_event_scores.csv"
FOLLOWING_CONTEXT_CACHE = "following_event_contexts.npz"
FOLLOWING_SEGMENT_CACHE = "following_event_segments.npz"
FOLLOWING_CACHE_SUMMARY = "following_event_cache_summary.json"
CUTIN_SCORE_CACHE = "cutin_event_scores.csv"
CUTIN_CONTEXT_CACHE = "cutin_event_contexts.npz"
CUTIN_CACHE_SUMMARY = "cutin_event_cache_summary.json"
EXPOSURE_PER_RECORDING_CSV = "exposure_per_recording.csv"


def events_to_dataframe(events):
    if not events:
        return pd.DataFrame()
    return pd.DataFrame([event.to_row() for event in events])


def validate_raw_dir(raw_dir: Path) -> None:
    """Fail early when the configured highD raw data directory is missing or empty."""
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"highD 原始数据目录不存在: {raw_dir}\n"
            "请把 highD CSV 文件放到该目录，或修改配置文件中的 paths.raw_dir。\n"
            "期望文件名示例: 01_tracks.csv, 01_tracksMeta.csv, 01_recordingMeta.csv"
        )
    tracks_files = sorted(raw_dir.glob("*_tracks.csv"))
    if not tracks_files:
        raise FileNotFoundError(
            f"highD 原始数据目录中没有找到 *_tracks.csv: {raw_dir}\n"
            "请确认 raw_dir 指向包含 highD 原始 CSV 的目录。"
        )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("extract")

    config_path = DEFAULT_CONFIG_PATH
    cfg = load_config(config_path)
    raw_dir_path = resolve_data_path(cfg["paths"]["raw_dir"], config_path)
    validate_raw_dir(raw_dir_path)
    raw_dir = str(raw_dir_path)
    out_dir = Path(str(resolve_data_path(cfg["paths"]["output_dir"], config_path)))
    ensure_dir(out_dir)

    ids = resolve_recording_ids(raw_dir, cfg.get("recordings", {}))
    logger.info("将处理 recording IDs: %s", ids)
    risk_options = highd_options_from_config(cfg)
    cutin_risk_options = highd_cutin_options_from_config(cfg)

    target_fps = cfg.get("sampling", {}).get("target_fps", 10)
    all_events = []
    following_rows = []
    following_segment_rows = []
    following_skipped = 0
    following_segment_skipped = 0
    cutin_rows = []
    cutin_skipped = 0
    cutin_candidates_scored = 0
    cutin_semantic_rejected = 0
    exposure_rows = []

    for rid in tqdm(ids, desc="Extracting events"):
        rec = load_recording(raw_dir, rid)
        rec = normalize_driving_direction(rec)
        rec = filter_abnormal_tracks(rec, cfg)
        rec = resample_recording(rec, target_fps)
        recording_events = []
        recording_events.extend(extract_following_segments(rec, cfg))
        recording_events.extend(extract_cutin_events(rec, cfg))
        all_events.extend(recording_events)

        recording_df = events_to_dataframe(recording_events)
        if len(recording_df) == 0:
            continue
        following = recording_df[
            recording_df["event_type"] == "following"
        ]
        cutin = recording_df[
            recording_df["event_type"] == "cut_in"
        ]

        # 曝光计算：与事件提取在同一遍历中完成，避免二次加载原始数据
        exposure_rows.append({
            **following_exposure_for_recording(
                following.copy() if not following.empty else following,
                recording_id=rid,
                get_track=rec.get_vehicle_track,
                fps=float(rec.recording_meta.get("frameRate", target_fps)),
            ),
            **all_vehicle_exposure_for_recording(
                recording_id=rid,
                vehicle_ids=rec.vehicle_ids(),
                get_track=rec.get_vehicle_track,
                fps=float(rec.recording_meta.get("frameRate", target_fps)),
            ),
        })

        if not following.empty:
            rows, skipped = build_highd_event_rows_from_recording(
                rec,
                following.reset_index(drop=True),
                options=risk_options,
            )
            score_highd_event_rows(rows, options=risk_options)
            following_rows.extend(rows)
            following_skipped += int(skipped)
            (
                segment_rows,
                segment_skipped,
            ) = build_highd_following_segment_rows_from_recording(
                rec,
                following.reset_index(drop=True),
                options=risk_options,
            )
            following_segment_rows.extend(segment_rows)
            following_segment_skipped += int(segment_skipped)

        if not cutin.empty:
            rows, skipped = build_highd_cutin_event_rows_from_recording(
                rec,
                cutin.reset_index(drop=True),
                options=cutin_risk_options,
            )
            score_highd_cutin_event_rows(rows, options=cutin_risk_options)
            semantic_rows = filter_semantic_cutin_rows(rows)
            cutin_candidates_scored += int(len(rows))
            cutin_semantic_rejected += int(len(rows) - len(semantic_rows))
            cutin_rows.extend(semantic_rows)
            cutin_skipped += int(skipped)

    df = events_to_dataframe(all_events)
    if len(df) > 0:
        df.to_csv(out_dir / "events.csv", index=False)
        if following_rows:
            score_path = out_dir / FOLLOWING_SCORE_CACHE
            context_path = out_dir / FOLLOWING_CONTEXT_CACHE
            segment_path = out_dir / FOLLOWING_SEGMENT_CACHE
            following_horizon = int(risk_options.get("min_future_steps", 125))
            write_csv(score_path, highd_score_table(following_rows))
            save_highd_event_context_cache(context_path, following_rows)
            save_highd_following_segment_cache(
                segment_path,
                following_segment_rows,
                target_fps=float(target_fps),
            )
            write_json(
                out_dir / FOLLOWING_CACHE_SUMMARY,
                {
                    "score_cache": str(score_path),
                    "context_cache": str(context_path),
                    "segment_cache": str(segment_path),
                    "num_following_contexts": int(len(following_rows)),
                    "num_following_segments": int(len(following_segment_rows)),
                    "skipped_following_contexts": int(following_skipped),
                    "skipped_following_segments": int(following_segment_skipped),
                    "risk_window": "fixed_horizon_context",
                    "score_start_frame": "context_anchor_frame",
                    "score_horizon_steps": following_horizon,
                    "context_anchor_frame": (
                        "clamped_event_anchor_with_125_step_future"
                    ),
                    "context_horizon_steps": following_horizon,
                    "risk_variable": "y_long",
                },
            )
            logger.info(
                "following 风险缓存: %d 条, segments=%d, skipped=%d/%d, 输出: %s, %s 和 %s",
                len(following_rows),
                len(following_segment_rows),
                following_skipped,
                following_segment_skipped,
                score_path,
                context_path,
                segment_path,
            )
        else:
            logger.warning("没有生成 following 风险/context 缓存")
        if cutin_rows:
            score_path = out_dir / CUTIN_SCORE_CACHE
            context_path = out_dir / CUTIN_CONTEXT_CACHE
            cutin_horizon = int(cutin_risk_options.get("context_horizon_steps", 100))
            min_post_cutin_seconds = float(
                cfg.get("cutin", {}).get(
                    "min_post_cutin_duration_seconds",
                    float(cfg.get("cutin", {}).get("min_post_cutin_duration_steps", 0))
                    / max(float(target_fps), 1.0),
                )
            )
            write_csv(score_path, highd_cutin_score_table(cutin_rows))
            save_highd_cutin_event_context_cache(context_path, cutin_rows)
            write_json(
                out_dir / CUTIN_CACHE_SUMMARY,
                {
                    "score_cache": str(score_path),
                    "context_cache": str(context_path),
                    "num_cutin_contexts": int(len(cutin_rows)),
                    "num_scored_cutin_candidates": int(cutin_candidates_scored),
                    "semantic_rejected_cutin_candidates": int(cutin_semantic_rejected),
                    "skipped_cutin_contexts": int(cutin_skipped),
                    "anchor_phase": str(
                        cfg.get("cutin", {}).get("anchor_phase", "cross")
                    ),
                    "context_pre_cross_steps": cfg.get("cutin", {}).get(
                        "context_pre_cross_steps",
                        25,
                    ),
                    "min_post_cutin_duration_seconds": min_post_cutin_seconds,
                    "context_horizon_steps": cutin_horizon,
                    "score_horizon_steps": cutin_horizon,
                    "context_anchor": "cross_frame_minus_context_pre_cross_steps",
                    "context_end": "anchor_frame_plus_context_horizon_steps",
                    "risk_start_frame": "cross_frame",
                    "risk_window": "fixed_horizon_context_from_cross_to_window_end",
                    "context_cache_format": "fixed_horizon_cutin_pre_cross_window",
                    "risk_variable": "y_cutin",
                },
            )
            logger.info(
                "cut-in 风险缓存: %d 条真实 cut-in, semantic_rejected=%d, skipped=%d, 输出: %s 和 %s",
                len(cutin_rows),
                cutin_semantic_rejected,
                cutin_skipped,
                score_path,
                context_path,
            )
        else:
            logger.warning("没有生成 cut-in 风险/context 缓存")
        if exposure_rows:
            write_csv(out_dir / EXPOSURE_PER_RECORDING_CSV, exposure_rows)
            logger.info(
                "曝光 per-recording: %d 条记录, 输出: %s",
                len(exposure_rows),
                out_dir / EXPOSURE_PER_RECORDING_CSV,
            )
        logger.info(
            "筛选后事件总数: %d",
            len(df),
        )
    else:
        logger.warning("未提取到任何事件!")

    logger.info("完成! 输出: %s", out_dir)


if __name__ == "__main__":
    main()
