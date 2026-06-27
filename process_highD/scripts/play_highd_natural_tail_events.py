#!/usr/bin/env python3
"""Render selected highD natural tail segments to GIF."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from process_highD.src.natural_event_playback import (  # noqa: E402
    NaturalPlaybackOptions,
    render_natural_tail_events,
)
from process_highD.src.natural_evt_pipeline import (  # noqa: E402
    natural_output_paths,
    select_natural_tail_contexts,
)
from process_highD.src.io_utils import load_config  # noqa: E402


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "highd_natural_evt.yaml"
DEFAULT_PLAYBACK_OPTIONS = NaturalPlaybackOptions()


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
    render_natural_tail_events(
        config_path=config_path,
        tail_contexts_csv=tail_csv,
        output_dir=output_dir,
        risk_trace_npz=paths["risk_trace_npz"],
        top_k=int(args.top_k),
        options=DEFAULT_PLAYBACK_OPTIONS,
    )


if __name__ == "__main__":
    main()
