#!/usr/bin/env python3
"""Produce the paired four-cell repair/regression table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .paired import paired_repair_summary
from .provenance import artifact_record
from .verify_protocol import verify_protocol_lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path)
    args = parser.parse_args()
    protocol = (
        None
        if args.protocol_lock is None
        else verify_protocol_lock(
            args.protocol_lock, Path(__file__).resolve().parents[1]
        )
    )
    if protocol is not None and not protocol["verified"]:
        raise RuntimeError(f"protocol lock mismatch: {protocol['mismatches'][:3]}")
    if protocol is not None:
        for evaluation in (args.before, args.after):
            metadata_path = evaluation.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                (metadata.get("protocol") or {}).get("protocol_lock_sha256")
                != protocol["protocol_lock_sha256"]
            ):
                raise ValueError(f"{evaluation} uses a different protocol lock")
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
    summary["protocol"] = protocol
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
