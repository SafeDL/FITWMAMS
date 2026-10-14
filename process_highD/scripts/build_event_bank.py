#!/usr/bin/env python3
"""Build canonical normal/empirical/EVT event manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from process_highD.src.event_bank import (  # noqa: E402
    _sequence_splits,
    build_testing_event_bank,
    write_event_bank,
)
from process_highD.src.evt_pipeline import (  # noqa: E402
    evt_output_paths,
    load_evt_context,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config.yaml",
    )
    args = parser.parse_args()
    context = load_evt_context(args.config.resolve())
    evt_paths = evt_output_paths(context)
    model_path = evt_paths["model"]
    model = json.loads(model_path.read_text(encoding="utf-8"))
    segment_path = Path(context["preprocessing_paths"]["segment_csv"]).resolve()
    grouping = context["config"]["event_grouping"]
    bank_cfg = context["config"]["event_bank"]
    result = build_testing_event_bank(
        pd.read_csv(segment_path),
        evt_threshold=float(model["u"]),
        exceedance_rate=float(model["exceedance_rate"]),
        split_by_event=_sequence_splits(context["sequence_cache"]),
        seed=int(bank_cfg["seed"]),
        min_shared_actors=int(grouping["min_shared_actors"]),
        min_actor_overlap=float(grouping["min_actor_overlap"]),
        max_group_span_windows=float(grouping["max_group_span_windows"]),
    )
    result.summary["evt_model"] = str(model_path)
    result.summary["source_segment_csv"] = str(segment_path)
    write_event_bank(result, context["event_bank_dir"])
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
