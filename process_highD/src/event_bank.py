"""Canonical event identities and EVT comparison cohorts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

ACTOR_COLUMNS = (
    "same_front_id",
    "same_rear_id",
    "left_front_id",
    "left_rear_id",
    "right_front_id",
    "right_rear_id",
)
SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}


@dataclass(frozen=True)
class EventBankResult:
    events: pd.DataFrame
    thresholds: pd.DataFrame
    summary: dict[str, object]


def _actors(row: pd.Series) -> tuple[int, ...]:
    values = {int(row["ego_id"])}
    values.update(int(row[name]) for name in ACTOR_COLUMNS if int(row[name]) >= 0)
    return tuple(sorted(values))


def _interaction_groups(
    frame: pd.DataFrame,
    *,
    min_shared_actors: int = 2,
    min_actor_overlap: float = 0.5,
    max_group_span_windows: float = 2.0,
) -> np.ndarray:
    """Group nearby repeated views without transitive chain merging.

    A row is compared with the fixed seed of a group.  It must overlap that
    seed in time, share enough actors, and keep the whole group within a short
    time span.  Comparing with a fixed seed prevents A-B-C chains from turning
    a long part of one recording into one event.
    """
    groups = np.full(len(frame), -1, dtype=np.int64)
    next_group = 0
    for _, indices in frame.groupby("recording_id", sort=True).groups.items():
        ordered = sorted(
            (int(index) for index in indices),
            key=lambda index: (
                int(frame.iloc[index]["window_start_frame"]),
                int(frame.iloc[index]["window_end_frame"]),
            ),
        )
        active: list[dict[str, object]] = []
        for index in ordered:
            row = frame.iloc[index]
            start, stop = int(row["window_start_frame"]), int(row["window_end_frame"])
            actor_set = set(_actors(row))
            active = [item for item in active if int(item["seed_stop"]) >= start]
            matches: list[tuple[float, int, int, dict[str, object]]] = []
            for item in active:
                seed_actors = item["seed_actors"]
                shared = len(actor_set & seed_actors)
                overlap = shared / max(min(len(actor_set), len(seed_actors)), 1)
                seed_steps = int(item["seed_stop"]) - int(item["seed_start"]) + 1
                span = (
                    max(stop, int(item["max_stop"]))
                    - min(start, int(item["min_start"]))
                    + 1
                )
                if (
                    stop >= int(item["seed_start"])
                    and shared >= int(min_shared_actors)
                    and overlap >= float(min_actor_overlap)
                    and span <= float(max_group_span_windows) * seed_steps
                ):
                    matches.append((overlap, shared, -int(item["group"]), item))
            if matches:
                chosen = max(matches, key=lambda value: value[:3])[3]
                group = int(chosen["group"])
                chosen["min_start"] = min(start, int(chosen["min_start"]))
                chosen["max_stop"] = max(stop, int(chosen["max_stop"]))
            else:
                group = next_group
                next_group += 1
                active.append(
                    {
                        "group": group,
                        "seed_start": start,
                        "seed_stop": stop,
                        "min_start": start,
                        "max_stop": stop,
                        "seed_actors": actor_set,
                    }
                )
            groups[index] = group
    return groups


def _sequence_splits(sequence_cache: Path) -> dict[str, str]:
    ids = np.load(sequence_cache / "sequence_id.npy", mmap_mode="r", allow_pickle=False)
    split = np.load(
        sequence_cache / "split_index.npy", mmap_mode="r", allow_pickle=False
    )
    return {
        str(event_id): SPLIT_NAMES[int(value)] for event_id, value in zip(ids, split)
    }


def build_testing_event_bank(
    segments: pd.DataFrame,
    *,
    evt_threshold: float,
    exceedance_rate: float,
    split_by_event: dict[str, str],
    seed: int = 20260926,
    max_per_interaction_group: int = 1,
    min_shared_actors: int = 2,
    min_actor_overlap: float = 0.5,
    max_group_span_windows: float = 2.0,
) -> EventBankResult:
    """Build normal, empirical-tail and recording-adaptive EVT cohorts.

    The three cohort flags live on one canonical event table, so later
    mutation/search code cannot silently change event identity or split.
    """
    required = {
        "segment_id",
        "recording_id",
        "ego_id",
        "window_start_frame",
        "window_end_frame",
        "event_risk",
        "num_untracked_candidates",
        "max_untracked_pair_risk",
        "untracked_risk_exceeds_tracked_peak",
        *ACTOR_COLUMNS,
    }
    missing = required - set(segments.columns)
    if missing:
        raise KeyError(f"segment table is missing columns: {sorted(missing)}")
    frame = segments.reset_index(drop=True).copy()
    frame["event_id"] = frame["segment_id"].astype(str)
    frame["source_event_id"] = frame["event_id"]
    frame["split"] = frame["event_id"].map(split_by_event)
    if frame["split"].isna().any():
        missing_ids = frame.loc[frame["split"].isna(), "event_id"].head().tolist()
        raise ValueError(
            f"events are missing canonical split assignments: {missing_ids}"
        )
    frame["actor_ids"] = [json.dumps(_actors(row)) for _, row in frame.iterrows()]
    frame["interaction_group"] = _interaction_groups(
        frame,
        min_shared_actors=min_shared_actors,
        min_actor_overlap=min_actor_overlap,
        max_group_span_windows=max_group_span_windows,
    )
    frame["evt_tail"] = frame["event_risk"].to_numpy(float) > float(evt_threshold)
    frame["empirical_selected"] = False
    frame["evt_selected"] = False
    frame["normal_selected"] = False
    threshold_rows: list[dict[str, object]] = []
    for split_name, split_indices in frame.groupby("split", sort=True).groups.items():
        split_indices = np.asarray(list(split_indices), dtype=np.int64)
        count = max(1, int(round(len(split_indices) * float(exceedance_rate))))
        ranked = split_indices[
            np.argsort(
                frame.loc[split_indices, "event_risk"].to_numpy(float), kind="stable"
            )
        ]
        frame.loc[ranked[-count:], "empirical_selected"] = True

        selected_evt: list[int] = []
        for recording_id, recording_indices in (
            frame.loc[split_indices].groupby("recording_id", sort=True).groups.items()
        ):
            recording_indices = np.asarray(list(recording_indices), dtype=np.int64)
            recording_risk = frame.loc[recording_indices, "event_risk"].to_numpy(float)
            local_count = max(
                1, int(round(len(recording_indices) * float(exceedance_rate)))
            )
            local_threshold = max(
                float(evt_threshold),
                float(np.partition(recording_risk, -local_count)[-local_count]),
            )
            local = recording_indices[recording_risk >= local_threshold]
            selected_evt.extend(int(value) for value in local)
            threshold_rows.append(
                {
                    "split": split_name,
                    "recording_id": int(recording_id),
                    "events": int(len(recording_indices)),
                    "r_select": local_threshold,
                    "selected": int(len(local)),
                }
            )
        frame.loc[selected_evt, "evt_selected"] = True

        # Sort before shuffling so a fixed seed gives the same sample even if
        # the input CSV row order changes.
        split_seed = int(seed) + {"train": 0, "val": 1, "test": 2}[split_name]
        normal_candidates = (
            frame.loc[split_indices]
            .sort_values("event_id")
            .sample(frac=1.0, random_state=split_seed)
        )
        representatives = normal_candidates.groupby(
            "interaction_group", sort=False
        ).head(int(max_per_interaction_group))
        choose = (
            representatives.sort_values("event_id")
            .sample(frac=1.0, random_state=split_seed + 10)
            .head(min(count, len(representatives)))
            .index.to_numpy(dtype=np.int64)
        )
        frame.loc[choose, "normal_selected"] = True

    output_columns = [
        "event_id",
        "recording_id",
        "window_start_frame",
        "window_end_frame",
        "ego_id",
        "actor_ids",
        "event_risk",
        "evt_tail",
        "split",
        "source_event_id",
        "interaction_group",
        "normal_selected",
        "empirical_selected",
        "evt_selected",
        "num_untracked_candidates",
        "max_untracked_pair_risk",
        "untracked_risk_exceeds_tracked_peak",
    ]
    events = frame[output_columns].copy()
    summary: dict[str, object] = {
        "schema": "fitwmams_testing_event_bank_v2",
        "events": int(len(events)),
        "evt_fit_threshold_u": float(evt_threshold),
        "target_exceedance_rate": float(exceedance_rate),
        "split_counts": events["split"].value_counts().sort_index().to_dict(),
        "cohort_counts": {
            name: int(events[f"{name}_selected"].sum())
            for name in ("normal", "empirical", "evt")
        },
        "interaction_groups": int(events["interaction_group"].nunique()),
        "interaction_grouping": {
            "min_shared_actors": int(min_shared_actors),
            "min_actor_overlap": float(min_actor_overlap),
            "max_group_span_windows": float(max_group_span_windows),
            "transitive_merge": False,
        },
        "untracked_higher_risk_events": int(
            events["untracked_risk_exceeds_tracked_peak"].sum()
        ),
    }
    return EventBankResult(events, pd.DataFrame(threshold_rows), summary)


def write_event_bank(result: EventBankResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.events.to_csv(output_dir / "events.csv", index=False)
    result.thresholds.to_csv(output_dir / "recording_thresholds.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
