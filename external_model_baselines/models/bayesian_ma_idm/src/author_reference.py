"""Exact-data protocol used by the public Zhang--Sun MA-IDM notebook.

This module reads only the author's *preprocessed highD cache* supplied to an
explicit command-line path.  It does not import the author's code or pickle a
PyMC model.  The resulting NPZ is a transparent numerical dataset containing
the twenty published pair indices and their already-downsampled 5 Hz states.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
import numpy as np


AUTHOR_COMMIT = "7520b15a8b163f52532cd3e5c7f7d4db05fc43f0"
CAR_PAIRS = (14, 35, 23, 25, 36, 38, 60, 90, 228, 232)
TRUCK_PAIRS = (3, 4, 18, 52, 81, 144, 153, 162, 5, 241)
PAIR_ORDER = CAR_PAIRS + TRUCK_PAIRS


def _cache_file(cache_dir: Path, recording: int) -> Path:
    return cache_dir / f"track_pair_list_temp{recording:02d}_MinLength50_freq_5.pkl"


def prepare_author_reference(cache_dir: str | Path, output_dir: str | Path) -> Path:
    """Export the exact 20 cached 5 Hz pair records selected by Config.py.

    Pickles are deserialized only after the caller explicitly designates the
    checkout/cache directory; the cache is a numerical highD preprocessing
    artifact from the pinned public author repository.
    """
    cache_dir, output_dir = Path(cache_dir), Path(output_dir)
    source: dict[int, dict] = {}
    for recording in range(1, 61):
        file = _cache_file(cache_dir, recording)
        if not file.exists() or file.stat().st_size <= 16:
            continue
        with file.open("rb") as handle:
            entries = pickle.load(handle)
        for entry in entries:
            source[int(entry["pair_No"])] = entry
    missing = sorted(set(PAIR_ORDER) - source.keys())
    if missing:
        raise RuntimeError(f"author reference cache is incomplete; missing pair_No values {missing}")
    chosen = [source[number] for number in PAIR_ORDER]
    offsets = np.r_[0, np.cumsum([len(item["sReal"]) for item in chosen])].astype(np.int64)
    arrays = {
        "offsets": offsets,
        "follower_v": np.concatenate([np.asarray(item["vFollReal"], float) for item in chosen]),
        "leader_v": np.concatenate([np.asarray(item["vLeaderReal"], float) for item in chosen]),
        "follower_x": np.concatenate([np.asarray(item["xFollReal"], float) for item in chosen]),
        "leader_x": np.concatenate([np.asarray(item["xLeaderReal"], float) for item in chosen]),
        "follower_a": np.concatenate([np.asarray(item["aFollReal"], float) for item in chosen]),
        "gap": np.concatenate([np.asarray(item["sReal"], float) for item in chosen]),
        "next_speed": np.concatenate([np.asarray(item["vFollReal_next"], float) for item in chosen]),
        "pair_no": np.asarray(PAIR_ORDER, np.int16),
        "follower_id": np.asarray([int(item["id_Foll"]) for item in chosen], np.int32),
        "leader_id": np.asarray([int(np.asarray(item["precedingId_Foll"])[0]) for item in chosen], np.int32),
        "frame_start": np.asarray([int(np.asarray(item["frame_Foll"])[0]) for item in chosen], np.int32),
        "follower_length": np.asarray([float(np.asarray(item["vehicle_length"])[0]) for item in chosen]),
        "vehicle_class": np.asarray(["Car"] * len(CAR_PAIRS) + ["Truck"] * len(TRUCK_PAIRS), dtype="U8"),
        "dt_s": np.asarray(.2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "zhang_sun_author_20pairs_5hz.npz"
    np.savez_compressed(path, **arrays)
    audit = {
        "protocol": "exact Config.car_interactive_pair_list + truck_interactive_pair_list",
        "author_repository_commit": AUTHOR_COMMIT,
        "source_cache_directory": str(cache_dir.resolve()),
        "sampling_fps": 5,
        "dt_s": .2,
        "pair_numbers": list(PAIR_ORDER),
        "cars": list(CAR_PAIRS),
        "trucks": list(TRUCK_PAIRS),
        "selected": [{"pair_no": int(item["pair_No"]), "follower_id": int(item["id_Foll"]),
                      "leader_id": int(np.asarray(item["precedingId_Foll"])[0]),
                      "frame_start": int(np.asarray(item["frame_Foll"])[0]),
                      "cache_class": str(item["class_foll"]), "samples_5hz": int(len(item["sReal"]))}
                     for item in chosen],
        "note": "cache_class is retained for audit; paper cohort strata follow Config.py lists",
    }
    path.with_suffix(".json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return path


def load_author_reference(path: str | Path) -> tuple[dict[str, np.ndarray], list[dict[str, np.ndarray]]]:
    with np.load(path, allow_pickle=False) as raw:
        data = {key: raw[key] for key in raw.files}
    pairs: list[dict[str, np.ndarray]] = []
    for index, (begin, end) in enumerate(zip(data["offsets"][:-1], data["offsets"][1:])):
        pair = {key: np.asarray(data[key][begin:end], float) for key in ("follower_x", "leader_x", "follower_v", "leader_v", "follower_a", "gap", "next_speed")}
        pair.update({key: data[key][index] for key in ("pair_no", "follower_id", "leader_id", "frame_start", "follower_length", "vehicle_class")})
        pairs.append(pair)
    return data, pairs
