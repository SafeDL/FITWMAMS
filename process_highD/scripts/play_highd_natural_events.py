#!/usr/bin/env python3
"""Render selected highD natural driving segments to GIF."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from process_highD.src.natural_event_playback import (  # noqa: E402
    NaturalPlaybackOptions,
    render_natural_tail_event_gif,
    render_natural_tail_events,
)
from process_highD.src.natural_evt_pipeline import (  # noqa: E402
    natural_output_paths,
    select_natural_tail_contexts,
)
from process_highD.src.io_utils import load_config  # noqa: E402
from process_highD.src.natural_segments import validate_lateral_integrity  # noqa: E402

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "highd_natural_evt.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--tail-contexts-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Render the top-k highest-risk tail contexts; use <=0 for all rows.",
    )
    parser.add_argument(
        "--random-count",
        type=int,
        default=0,
        help="Randomly render this many rows from the complete natural-segment cohort.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--cohort",
        choices=("all", "lane-change", "strict-cutin", "no-lane-change"),
        default="all",
        help="Semantic cohort used with --random-count.",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config_path = Path(args.config).resolve()
    cfg = load_config(str(config_path))
    paths = natural_output_paths(cfg, config_path)
    tail_csv = (
        Path(args.tail_contexts_csv).resolve()
        if args.tail_contexts_csv
        else paths["tail_contexts"]
    )
    if not tail_csv.exists():
        select_natural_tail_contexts(
            config_path=config_path,
            top_k=0,
        )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else paths["out_dir"] / "playbacks"
    )
    options = NaturalPlaybackOptions(frame_stride=max(int(args.frame_stride), 1))
    if int(args.random_count) <= 0:
        render_natural_tail_events(
            config_path=config_path,
            tail_contexts_csv=tail_csv,
            output_dir=output_dir,
            risk_trace_npz=paths["risk_trace_npz"],
            top_k=int(args.top_k),
            options=options,
        )
        return

    segments = pd.read_csv(paths["segment_csv"])
    validate_lateral_integrity(
        segments,
        required=True,
        source=str(paths["segment_csv"]),
    )
    if args.cohort == "lane-change":
        segments = segments.loc[segments["num_lane_changes"] > 0]
    elif args.cohort == "strict-cutin":
        segments = segments.loc[segments["num_strict_cutins"] > 0]
    elif args.cohort == "no-lane-change":
        segments = segments.loc[segments["num_lane_changes"] == 0]
    if segments.empty:
        raise RuntimeError(f"No rows available for cohort={args.cohort!r}")
    count = min(int(args.random_count), len(segments))
    selected = segments.sample(n=count, random_state=int(args.random_seed))
    selected = selected.sort_values("segment_id", kind="mergesort")
    output_dir.mkdir(parents=True, exist_ok=True)
    for rank, (_, row) in enumerate(selected.iterrows(), start=1):
        segment_id = str(row["segment_id"]).replace("/", "_")
        render_natural_tail_event_gif(
            config_path=config_path,
            segment_row=row,
            output_path=(
                output_dir / f"natural_random_{args.cohort}_{rank:03d}_{segment_id}.gif"
            ),
            risk_trace_npz=paths["risk_trace_npz"],
            options=options,
        )


if __name__ == "__main__":
    main()
