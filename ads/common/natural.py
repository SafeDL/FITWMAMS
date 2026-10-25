"""Empirical natural Test sampling; no EVT selection or generated initialization."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ads.common.instances import GeneratedInstance, file_hash, json_hash
from ads.common.world import HighDEventStore
from hierarchical_world_model.src.core.sequential_dataset import (
    _segment_slot_ids,
    _state_window,
)
from process_highD.src.io_utils import load_config, resolve_data_path
from process_highD.src.natural_segments import _build_vehicle_cache
from process_highD.src.preprocess import prepare_recording

HISTORY_FRAMES = 51  # Two recorded seconds at 25 Hz, including the current state.


def recorded_history_sources(recordings, config_path):
    config_path = Path(config_path)
    config = load_config(config_path)
    raw = resolve_data_path(config["paths"]["raw_dir"], config_path)
    files = {"history_preprocessing": config_path}
    for recording in sorted(map(int, recordings)):
        for name in ("tracks", "tracksMeta", "recordingMeta"):
            files[f"history_{recording:02d}_{name}"] = (
                raw / f"{recording:02d}_{name}.csv"
            )
    return files


def recorded_history(vehicles, row, event):
    anchor = int(row["window_start_frame"])
    history, valid = _state_window(
        vehicles,
        ego_id=int(row["ego_id"]),
        slot_ids=_segment_slot_ids(row),
        start_frame=anchor - HISTORY_FRAMES + 1,
        steps=HISTORY_FRAMES,
        origin_frame=anchor,
    )
    np.testing.assert_allclose(history[-1], event.initial_states, rtol=1e-6, atol=1e-5)
    np.testing.assert_array_equal(valid[-1], event.valid)
    return replace(event, initial_history=history, initial_history_valid=valid)


def empirical_manifest(
    natural_frame, arrays, *, count=2000, seed=20260926, split_indices=(2,)
):
    test_ids = np.asarray(arrays["segment_id"])[
        np.isin(np.asarray(arrays["split_index"]), split_indices)
    ]
    frame = natural_frame[natural_frame.segment_id.isin(test_ids)].copy()
    if len(frame) != len(test_ids) or frame.segment_id.duplicated().any():
        raise ValueError(
            "natural Test metadata does not match the frozen source population"
        )
    if count > len(frame):
        raise ValueError("requested empirical cohort exceeds available Test segments")
    columns = ["recording_id", "ego_driving_direction"]
    sizes = frame.groupby(columns, sort=True).size()
    exact = sizes.to_numpy() * count / len(frame)
    quotas = np.floor(exact).astype(int)
    for index in np.argsort(-(exact - quotas), kind="stable")[: count - quotas.sum()]:
        quotas[index] += 1
    rng = np.random.default_rng(seed)
    selected = []
    for (_, group), quota in zip(frame.groupby(columns, sort=True), quotas):
        indices = rng.choice(len(group), int(quota), replace=False)
        selected.append(group.iloc[indices])
    result = pd.concat(selected).sort_values("segment_id").reset_index(drop=True)
    result["event_id"] = result.segment_id
    result["base_event_id"] = result.segment_id
    result["source_event_id"] = result.segment_id
    splits = dict(zip(arrays["segment_id"], arrays["split_index"]))
    result["split"] = [
        ("train", "val", "test")[int(splits[identity])]
        for identity in result.segment_id
    ]
    # Association components retain shared vehicles in overlapping recorded windows.
    parents = np.arange(len(result))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    vehicle_columns = ["ego_id"] + [
        f"{side}_{role}_id"
        for side in ("same", "left", "right")
        for role in ("front", "rear")
    ]
    vehicles = [
        {
            int(row[key])
            for key in vehicle_columns
            if key in row and pd.notna(row[key]) and int(row[key]) > 0
        }
        for _, row in result.iterrows()
    ]
    for _, group in result.groupby("recording_id"):
        ordered = group.sort_values("window_start_frame").index.tolist()
        for position, left in enumerate(ordered):
            for right in ordered[position + 1 :]:
                if (
                    result.loc[right, "window_start_frame"]
                    > result.loc[left, "window_end_frame"]
                ):
                    break
                if vehicles[left] & vehicles[right]:
                    parents[find(right)] = find(left)
    result["association_group"] = [
        f"recording_{int(result.loc[i, 'recording_id']):02d}_group_{find(i)}"
        for i in range(len(result))
    ]
    tail_ids = set(
        np.asarray(arrays["segment_id"])[np.asarray(arrays["is_evt_tail"], bool)]
    )
    result["source_used_by_tail_flow"] = result.segment_id.isin(tail_ids)
    return result


def empirical_instances(
    manifest_path,
    sequence_cache,
    assets,
    *,
    history_config,
    seed=20260926,
    purpose="final_test",
    namespace="sealed_empirical_natural",
):
    store = HighDEventStore(manifest_path, sequence_cache, world_seed=seed)
    if purpose != "final_test" and (store.frame["split"] == "test").any():
        raise ValueError("empirical learning/development must use Train/Val sources")
    config = load_config(history_config)
    raw = resolve_data_path(config["paths"]["raw_dir"], history_config)
    current_recording, vehicles = None, None
    for index, row in store.frame.iterrows():
        recording_id = int(row["recording_id"])
        if recording_id != current_recording:
            recording = prepare_recording(raw, recording_id, config)
            vehicles = _build_vehicle_cache(recording)
            current_recording = recording_id
        event = recorded_history(vehicles, row, store[index])
        metadata = {
            key: row[key].item() if isinstance(row[key], np.generic) else row[key]
            for key in (
                "recording_id",
                "ego_driving_direction",
                "window_start_frame",
                "window_end_frame",
                "association_group",
                "source_used_by_tail_flow",
            )
        }
        metadata["history_source"] = "recorded_highd_tracks"
        metadata["source_split"] = row["split"]
        metadata["history_start_frame"] = (
            int(row["window_start_frame"]) - HISTORY_FRAMES + 1
        )
        metadata["history_end_frame"] = int(row["window_start_frame"])
        source_assets = {**assets, "empirical_manifest": file_hash(manifest_path)}
        root = json_hash(
            {
                "namespace": namespace,
                "seed": seed,
                "source_event_id": event.source_event_id,
            }
        )
        instance = GeneratedInstance(
            event,
            root,
            None,
            purpose,
            source_assets,
            source_kind="empirical_natural",
            source_metadata=metadata,
        )
        instance = replace(
            instance, event=replace(event, event_id=instance.instance_hash)
        )
        instance.validate()
        yield instance
