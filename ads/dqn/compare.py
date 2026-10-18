#!/usr/bin/env python3
"""Produce the paired four-cell repair/regression table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .paired import paired_repair_summary
from .provenance import artifact_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = pd.read_csv(args.before)
    after = pd.read_csv(args.after)
    cluster_column = None
    if args.manifest is not None:
        manifest = pd.read_csv(args.manifest)
        columns = ["event_id"]
        if "recording_id" in manifest:
            columns.append("recording_id")
            cluster_column = "recording_id"
        elif "source_event_id" in manifest:
            columns.append("source_event_id")
            cluster_column = "source_event_id"
        else:
            raise KeyError("manifest requires recording_id or source_event_id")
        before = before.merge(manifest[columns], on="event_id", validate="one_to_one")
        after = after.merge(manifest[columns], on="event_id", validate="one_to_one")
    summary = paired_repair_summary(
        before, after, cluster_column=cluster_column
    )
    summary["input_artifacts"] = {
        "before": artifact_record(args.before),
        "after": artifact_record(args.after),
        "manifest": artifact_record(args.manifest),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
