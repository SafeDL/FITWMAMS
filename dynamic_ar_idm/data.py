"""Paper-cohort highD preparation with native 25 Hz trajectories retained."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

# The published Config.py lists point to these cache records.  The fingerprint
# below (from its 7520b15 cache) makes the selection reproducible from raw highD
# files without shipping or unpickling opaque cache files.
PAPER_PAIRS = (
    (14, 273, 268, 2206, "Car", 290), (35, 352, 342, 3011, "Car", 319),
    (23, 309, 301, 2556, "Car", 316), (25, 314, 309, 2601, "Car", 316),
    (36, 355, 352, 2996, "Car", 337), (38, 361, 355, 3049, "Car", 335),
    (60, 412, 406, 3705, "Car", 314), (90, 1005, 1001, 10183, "Car", 415),
    (228, 343, 330, 2829, "Car", 283), (232, 404, 398, 3545, "Car", 324),
    (3, 211, 203, 1518, "Truck", 319), (4, 216, 211, 1564, "Truck", 363),
    (18, 287, 281, 2348, "Truck", 311), (52, 398, 390, 3519, "Truck", 287),
    (81, 747, 743, 7436, "Truck", 258), (144, 2486, 2485, 25350, "Truck", 306),
    (153, 2502, 2497, 25508, "Truck", 325), (162, 2524, 2517, 25712, "Truck", 325),
    (5, 221, 216, 1619, "Truck", 362), (241, 816, 806, 7906, "Truck", 277),
)
TRACK_COLUMNS = ["frame", "id", "x", "width", "xVelocity", "precedingId"]
FULL_TRACK_COLUMNS = TRACK_COLUMNS + ["followingId"]


def _find_pair(raw_dir: Path, follower_id: int, leader_id: int, frame_start: int,
               recording_cache: dict[int, tuple[pd.DataFrame, pd.DataFrame]], expected_5hz: int) -> tuple[int, pd.DataFrame, pd.DataFrame, float]:
    """Locate one cache-fingerprinted pair and retain its contiguous raw run."""
    candidates = []
    for recording in range(1, 61):
        stem = raw_dir / f"{recording:02d}"
        track_path, meta_path = Path(f"{stem}_tracks.csv"), Path(f"{stem}_tracksMeta.csv")
        if not track_path.exists():
            continue
        if recording not in recording_cache:
            recording_cache[recording] = (pd.read_csv(track_path, usecols=TRACK_COLUMNS),
                                          pd.read_csv(meta_path).set_index("id"))
        tracks, meta = recording_cache[recording]
        follower = tracks[(tracks.id == follower_id) & (tracks.precedingId == leader_id)].copy()
        if not len(follower) or frame_start not in set(follower.frame):
            continue
        leader = tracks[tracks.id == leader_id].set_index("frame")
        follower = follower.sort_values("frame").set_index("frame")
        # The source cache's long dominant-leader section is continuous.  Use
        # the one containing its published starting frame, not a later match.
        frames = follower.index.to_numpy()
        where = int(np.flatnonzero(frames == frame_start)[0])
        left = where
        while left and frames[left] - frames[left - 1] == 1:
            left -= 1
        right = where + 1
        while right < len(frames) and frames[right] - frames[right - 1] == 1:
            right += 1
        follower = follower.iloc[left:right]
        leader = leader.reindex(follower.index)
        if leader.isna().any().any():
            continue
        direction = int(meta.loc[follower_id, "drivingDirection"])
        sign = -1. if direction == 1 else 1.
        # Public source code uses bounding-box leading x coordinates and
        # subtracts the follower's longitudinal length (not two centres).
        # Retain this source-compatible effective length in the artifact.
        length_sum = float(meta.loc[follower_id, "width"])
        candidates.append((recording, follower, leader, sign * length_sum))
    # Vehicle IDs repeat across highD recordings.  The source-cache trajectory
    # length is therefore part of the published pair fingerprint.
    decision_count = lambda candidate: (len(candidate[1]) - 2) // 5  # len(arange(0, n-6, 5))
    candidates = [candidate for candidate in candidates if decision_count(candidate) >= expected_5hz]
    if candidates:
        return min(candidates, key=lambda candidate: abs(decision_count(candidate) - expected_5hz))
    raise RuntimeError(f"Could not locate published pair follower={follower_id}, leader={leader_id}, frame={frame_start}")


def prepare_paper_highd(raw_dir: str | Path, output_dir: str | Path) -> Path:
    """Build the exact 20-pair paper cohort while preserving 25 Hz states."""
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    records = []
    recording_cache: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for pair_no, follower_id, leader_id, frame_start, vehicle_class, expected_5hz in PAPER_PAIRS:
        recording, follower, leader, signed_length_sum = _find_pair(raw_dir, follower_id, leader_id, frame_start, recording_cache, expected_5hz)
        sign = 1. if signed_length_sum >= 0 else -1.
        # This is exactly `read_track_csv`: reverse negative-direction
        # bounding boxes around their leading edge, rather than merely
        # negating centres/left edges.  It preserves the paper's gap series.
        if sign < 0:
            fx = -(follower.x.to_numpy(float) + follower.width.to_numpy(float))
            lx = -(leader.x.to_numpy(float) + leader.width.to_numpy(float))
        else:
            fx, lx = follower.x.to_numpy(float), leader.x.to_numpy(float)
        fv, lv = sign * follower.xVelocity.to_numpy(float), sign * leader.xVelocity.to_numpy(float)
        length_sum = abs(signed_length_sum)
        # Match author current/next slicing: current raw indices 0:-6:5.
        decision = np.arange(0, max(0, len(fx) - 6), 5, dtype=np.int64)
        if len(decision) < expected_5hz:
            raise RuntimeError(f"pair {pair_no}: raw run yields {len(decision)} 5 Hz samples, expected at least {expected_5hz}")
        decision = decision[:expected_5hz]
        end = int(decision[-1] + 6)
        records.append(dict(pair_no=pair_no, recording_id=recording, follower_id=follower_id, leader_id=leader_id,
                            frame_start=frame_start, vehicle_class=vehicle_class, length_sum=length_sum,
                            follower_x=fx[:end], follower_v=fv[:end], leader_x=lx[:end], leader_v=lv[:end], decision=decision))
    offsets = np.r_[0, np.cumsum([len(row["follower_x"]) for row in records])].astype(np.int64)
    arrays = {key: np.concatenate([row[key] for row in records]) for key in ("follower_x", "follower_v", "leader_x", "leader_v")}
    arrays.update(offsets=offsets,
                  decision_offsets=np.r_[0, np.cumsum([len(row["decision"]) for row in records])].astype(np.int64),
                  decision_indices=np.concatenate([row["decision"] for row in records]),
                  pair_no=np.asarray([row["pair_no"] for row in records], np.int16),
                  recording_id=np.asarray([row["recording_id"] for row in records], np.int16),
                  follower_id=np.asarray([row["follower_id"] for row in records], np.int32),
                  leader_id=np.asarray([row["leader_id"] for row in records], np.int32),
                  frame_start=np.asarray([row["frame_start"] for row in records], np.int32),
                  length_sum=np.asarray([row["length_sum"] for row in records]),
                  vehicle_class=np.asarray([row["vehicle_class"] for row in records], dtype="U8"), native_dt_s=np.asarray(.04))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "paper_highd_20pairs_25hz.npz"
    np.savez_compressed(path, **arrays)
    audit = {"paper": "Zhang, Wang & Sun (2024)", "author_commit": "7520b15a8b163f52532cd3e5c7f7d4db05fc43f0",
             "native_fps": 25, "model_decision_fps": 5, "selection": "published Config.py pair lists, re-located by source-cache fingerprint",
             "pairs": [{key: (int(row[key]) if key not in {"vehicle_class"} else row[key]) for key in ("pair_no", "recording_id", "follower_id", "leader_id", "frame_start", "vehicle_class")} for row in records]}
    path.with_suffix(".json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (output_dir / "clock_contract.json").write_text(json.dumps({
        "native_plant_dt_s": .04, "native_plant_fps": 25, "driver_decision_dt_s": .2,
        "driver_decision_fps": 5, "native_ticks_per_decision": 5,
        "rule": "one innovation is consumed at each native tick divisible by five; held acceleration is used on intervening ticks",
        "snapshot_requirement": ["residual_history_newest_first", "last_native_decision_time", "held_acceleration", "random_stream_index"]
    }, indent=2), encoding="utf-8")
    return path


def load_pairs(path: str | Path) -> list[dict[str, np.ndarray]]:
    """Load ragged paper cohort records, including native and decision grids."""
    with np.load(path, allow_pickle=False) as raw:
        value = {key: raw[key] for key in raw.files}
    pairs = []
    for i, (start, stop) in enumerate(zip(value["offsets"][:-1], value["offsets"][1:])):
        pair = {key: np.asarray(value[key][start:stop], float) for key in ("follower_x", "follower_v", "leader_x", "leader_v")}
        if "decision_offsets" in value:
            dstart, dstop = value["decision_offsets"][i:i + 2]
            pair["decision"] = np.asarray(value["decision_indices"][dstart:dstop], int)
        else:
            # Shared ragged highD cohorts need not duplicate a decision index:
            # derive the same completed 5 Hz action grid from native 25 Hz rows.
            pair["decision"] = np.arange(0, max(0, len(pair["follower_v"]) - 6), 5, dtype=int)
        pair["pair_no"] = value["pair_no"][i] if "pair_no" in value else i
        pair.update({key: value[key][i] for key in ("recording_id", "follower_id", "leader_id", "length_sum", "vehicle_class")})
        pairs.append(pair)
    return pairs


def _full_pairs_in_recording(recording_id: int, raw_dir: Path, minimum_frames: int) -> list[dict[str, object]]:
    """Extract all source-loader-compatible long following runs in one recording.

    This mirrors the project's retained full-highD cohort policy, including its
    source-loader lane-change indexing convention, while retaining native-rate
    states needed by the Dynamic-IDM plant.
    """
    stem = raw_dir / f"{recording_id:02d}"
    tracks_path, meta_path = Path(f"{stem}_tracks.csv"), Path(f"{stem}_tracksMeta.csv")
    if not tracks_path.exists() or not meta_path.exists():
        return []
    meta_raw = pd.read_csv(meta_path)
    meta = meta_raw.set_index("id")
    # Deliberately match the public loader's category-index expression rather
    # than silently correcting it.  It is part of the existing 251-pair
    # project protocol against which this full-cohort artifact is checked.
    lane_changed = set(meta_raw.groupby("numLaneChanges", sort=False).groups.get(1, []).values)
    tracks = pd.read_csv(tracks_path, usecols=FULL_TRACK_COLUMNS)
    by_id = {int(key): value.set_index("frame").sort_index() for key, value in tracks.groupby("id", sort=False)}
    records: list[dict[str, object]] = []
    for follower_id, follower_all in by_id.items():
        if follower_id not in meta.index or follower_id in lane_changed:
            continue
        preceding = follower_all["precedingId"].to_numpy(np.int64)
        valid = np.flatnonzero(preceding)
        if not len(valid):
            continue
        leaders, counts = np.unique(preceding, return_counts=True)
        # The public loader's check was ``'0' in integer_array`` and therefore
        # falls through to this branch whenever multiple IDs occur.
        if len(leaders) > 1:
            leader_id = int(leaders[np.argmax(counts[1:]) + 1])
            valid = np.flatnonzero(preceding == leader_id)
        else:
            leader_id = int(preceding[valid[0]])
        if len(valid) < minimum_frames or leader_id not in by_id or leader_id not in meta.index:
            continue
        follower = follower_all.iloc[valid]
        leader = by_id[leader_id].reindex(follower.index)
        frames = follower.index.to_numpy(np.int64)
        if (leader.isna().any().any() or not np.all(np.diff(frames) == 1)
                or np.unique(leader["followingId"].to_numpy(np.int64))[0] != follower_id):
            continue
        direction = int(meta.loc[follower_id, "drivingDirection"])
        sign = -1. if direction == 1 else 1.
        follower_length, leader_length = float(meta.loc[follower_id, "width"]), float(meta.loc[leader_id, "width"])
        # Centre-coordinate representation avoids a direction-dependent bbox
        # edge convention in population rollouts; `length_sum` restores its
        # physical bumper-to-bumper gap exactly.
        fx = sign * (follower["x"].to_numpy(float) + .5 * follower_length)
        lx = sign * (leader["x"].to_numpy(float) + .5 * leader_length)
        fv = np.maximum(sign * follower["xVelocity"].to_numpy(float), 0.)
        lv = np.maximum(sign * leader["xVelocity"].to_numpy(float), 0.)
        if np.min(lx - fx - .5 * (follower_length + leader_length)) <= .1:
            continue
        records.append({"recording_id": recording_id, "follower_id": follower_id, "leader_id": leader_id,
                        "frame_start": int(frames[0]), "vehicle_class": str(meta.loc[follower_id, "class"]),
                        "length_sum": .5 * (follower_length + leader_length), "follower_x": fx, "follower_v": fv,
                        "leader_x": lx, "leader_v": lv})
    return records


def prepare_full_highd(raw_dir: str | Path, output_dir: str | Path, *, minimum_seconds: float = 50.) -> Path:
    """Build the all-recording 25 Hz highD cohort used for project-scale calibration."""
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    minimum_frames = int(round(minimum_seconds * 25))
    records = [record for recording in range(1, 61) for record in _full_pairs_in_recording(recording, raw_dir, minimum_frames)]
    if not records:
        raise RuntimeError("No qualifying highD following pairs found")
    prepared = []
    for pair_no, record in enumerate(records):
        decision = np.arange(0, len(record["follower_x"]) - 6, 5, dtype=np.int64)
        # A 1250-frame source run is eligible at the 50 s threshold even
        # though author current/next slicing leaves 249 completed decisions.
        if not len(decision):
            continue
        end = int(decision[-1] + 6)
        prepared.append({**record, "pair_no": pair_no, "decision": decision,
                         **{key: np.asarray(record[key])[:end] for key in ("follower_x", "follower_v", "leader_x", "leader_v")}})
    offsets = np.r_[0, np.cumsum([len(record["follower_x"]) for record in prepared])].astype(np.int64)
    arrays = {key: np.concatenate([record[key] for record in prepared]).astype(float) for key in ("follower_x", "follower_v", "leader_x", "leader_v")}
    arrays.update(offsets=offsets, decision_offsets=np.r_[0, np.cumsum([len(record["decision"]) for record in prepared])].astype(np.int64),
                  decision_indices=np.concatenate([record["decision"] for record in prepared]),
                  pair_no=np.asarray([record["pair_no"] for record in prepared], np.int32),
                  recording_id=np.asarray([record["recording_id"] for record in prepared], np.int16),
                  follower_id=np.asarray([record["follower_id"] for record in prepared], np.int32),
                  leader_id=np.asarray([record["leader_id"] for record in prepared], np.int32),
                  frame_start=np.asarray([record["frame_start"] for record in prepared], np.int32),
                  length_sum=np.asarray([record["length_sum"] for record in prepared], float),
                  vehicle_class=np.asarray([record["vehicle_class"] for record in prepared], dtype="U8"), native_dt_s=np.asarray(.04))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "full_highd_25hz.npz"
    np.savez_compressed(path, **arrays)
    per_recording = {str(recording): int(np.sum(arrays["recording_id"] == recording)) for recording in np.unique(arrays["recording_id"])}
    manifest = {"source": "all 60 local highD recordings", "native_fps": 25, "decision_fps": 5,
                "minimum_following_seconds": minimum_seconds, "selection": "source-loader lane policy; dominant preceding vehicle; continuous reciprocal relation; project nonnegative velocity; all qualifying pairs",
                "pair_count": len(prepared), "per_recording_pair_count": per_recording,
                "expected_project_protocol_pair_count": 251,
                "paper_relationship": "full project cohort; paper's published 20-pair selection remains a separately auditable exact subset"}
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if len(prepared) != 251:
        raise RuntimeError(f"full highD cohort count {len(prepared)} differs from project protocol expectation 251")
    return path
