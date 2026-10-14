#!/usr/bin/env python3
"""Create deterministic, interaction-deduplicated comparison manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

COHORTS = ("normal", "empirical", "evt")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from process_highD.src.evt_pipeline import load_evt_context  # noqa: E402


def select_manifest(
    frame: pd.DataFrame,
    *,
    split: str,
    size: int,
    seed: int,
) -> pd.DataFrame:
    selected = frame.loc[frame["split"].eq(split)].sort_values("event_id")
    selected = selected.sample(frac=1.0, random_state=seed).drop_duplicates(
        "interaction_group", keep="first"
    )
    if len(selected) < int(size):
        raise ValueError(f"{split} has only {len(selected)} unique interaction groups")
    return selected.iloc[: int(size)].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config.yaml",
    )
    args = parser.parse_args()
    context = load_evt_context(args.config.resolve())
    manifest_cfg = context["config"]["manifests"]
    validation_size = int(manifest_cfg["validation_size_per_cohort"])
    test_size = int(manifest_cfg["test_size_per_cohort"])
    train_size = int(manifest_cfg["train_size_per_cohort"])
    seed = int(manifest_cfg["seed"])
    event_bank_dir = context["event_bank_dir"]
    output_dir = context["manifest_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    event_table = pd.read_csv(event_bank_dir / "events.csv")
    cohorts = {
        name: event_table.loc[event_table[f"{name}_selected"]].copy()
        for name in COHORTS
    }
    evt_tail_all = (
        event_table.loc[event_table["evt_tail"]]
        .sort_values("event_id")
        .reset_index(drop=True)
    )
    evt_tail_all.to_csv(output_dir / "evt_tail_all.csv", index=False)
    manifests: dict[str, pd.DataFrame] = {}
    selection_index = 0
    for name, frame in cohorts.items():
        for split, size in (
            ("val", validation_size),
            ("test", test_size),
            ("train", train_size),
        ):
            key = f"{name}_{split}_{size}"
            manifests[key] = select_manifest(
                frame,
                split=split,
                size=size,
                seed=seed + selection_index,
            )
            selection_index += 1
            manifests[key].to_csv(output_dir / f"{key}.csv", index=False)
    train_parts = []
    for name in ("normal", "evt"):
        key = f"{name}_train_{train_size}"
        part = manifests[key].copy()
        part["training_cohort"] = name
        train_parts.append(part)
    mixed = (
        pd.concat(train_parts, ignore_index=True)
        .drop_duplicates("source_event_id", keep="first")
        .drop_duplicates("interaction_group", keep="first")
    )
    mixed = mixed.sort_values("event_id").sample(frac=1.0, random_state=seed + 100)
    mixed.to_csv(output_dir / "mixed_normal_evt_train.csv", index=False)
    split_sources = {
        split: set(
            pd.concat(
                [value[value["split"].eq(split)] for value in cohorts.values()],
                ignore_index=True,
            )["source_event_id"].astype(str)
        )
        for split in ("train", "val", "test")
    }
    leakage = {
        f"{left}_{right}": len(split_sources[left] & split_sources[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    overlap: dict[str, int] = {}
    for split, size in (("val", validation_size), ("test", test_size)):
        pairs = (("normal", "empirical"), ("normal", "evt"), ("empirical", "evt"))
        for left, right in pairs:
            left_ids = set(
                manifests[f"{left}_{split}_{size}"]["source_event_id"].astype(str)
            )
            right_ids = set(
                manifests[f"{right}_{split}_{size}"]["source_event_id"].astype(str)
            )
            overlap[f"{split}:{left}:{right}"] = len(left_ids & right_ids)
    summary = {
        "seed": seed,
        "interaction_group_deduplicated": True,
        "evt_tail_all": int(len(evt_tail_all)),
        "evt_tail_all_split_for_use": False,
        "validation_size_per_cohort": validation_size,
        "test_size_per_cohort": test_size,
        "train_size_per_cohort": train_size,
        "mixed_train_events": int(len(mixed)),
        "split_source_leakage": leakage,
        "selected_cohort_overlap": overlap,
        "manifests": {
            name: {
                "events": int(len(frame)),
                "recordings": int(frame["recording_id"].nunique()),
                "risk_mean": float(frame["event_risk"].mean()),
                "risk_median": float(frame["event_risk"].median()),
            }
            for name, frame in manifests.items()
        },
    }
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
