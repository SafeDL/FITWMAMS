"""Causal car-following extraction from the native 25 Hz highD recordings."""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


TRACK_COLUMNS = ["frame", "id", "x", "xVelocity", "xAcceleration", "precedingId"]
AUTHOR_TRACK_COLUMNS = TRACK_COLUMNS + ["followingId"]


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _author_pairs(recording_id: int, raw_dir: Path, minimum_frames: int, *, velocity_policy: str = "strict",
                  lane_change_policy: str = "correct_meta") -> list[dict]:
    """Extract every source-compatible long following pair from one recording.

    This follows the author's public loader semantics where a pair has one
    dominant preceding vehicle and at least 50 s worth of following samples.
    No class quota, recording quota, or early stopping is applied. We additionally
    require a continuous, reciprocal
    leader--follower relation so that a 25 Hz GP window never spans a missing
    physical observation.
    """
    stem = raw_dir / f"{recording_id:02d}"
    meta_path, tracks_path = Path(f"{stem}_tracksMeta.csv"), Path(f"{stem}_tracks.csv")
    if not (meta_path.exists() and tracks_path.exists()):
        return []
    meta_raw = pd.read_csv(meta_path)
    meta = meta_raw.set_index("id")
    source_loader_lc_ids = set(meta_raw.groupby("numLaneChanges", sort=False).groups.get(1, []).values)
    tracks = pd.read_csv(tracks_path, usecols=AUTHOR_TRACK_COLUMNS)
    by_id = {int(key): value.set_index("frame").sort_index() for key, value in tracks.groupby("id", sort=False)}
    output: list[dict] = []
    for follower_id, follower in by_id.items():
        if follower_id not in meta.index:
            continue
        lane_changed = (follower_id in source_loader_lc_ids if lane_change_policy == "source_loader_index"
                        else int(meta.loc[follower_id, "numLaneChanges"]) != 0)
        if lane_changed:
            continue
        # Mirror public ``load_CF_pairs`` before adding causal physical
        # projection.  In particular, its dominant-leader selection is made
        # from the full precedingId sequence (including zero), not merely by
        # taking the most frequent positive ID.  This seemingly small branch
        # changes eleven local 50 s candidates.
        preceding = follower["precedingId"].to_numpy(np.int64)
        valid_position = np.flatnonzero(preceding)
        if len(valid_position) == 0:
            continue
        leaders, counts = np.unique(preceding, return_counts=True)
        # Equivalent to the author's source branch (where the zero ID occupies
        # sorted position zero): with more than one possible preceding ID,
        # choose the most common non-zero candidate.
        if len(leaders) > 1:
            leader_id = int(leaders[np.argmax(counts[1:]) + 1])
            valid_position = np.flatnonzero(preceding == leader_id)
        else:
            leader_id = int(preceding[valid_position[0]])
        follower = follower.iloc[valid_position]
        if len(follower) < minimum_frames or leader_id not in by_id or leader_id not in meta.index:
            continue
        leader = by_id[leader_id].reindex(follower.index)
        frames = follower.index.to_numpy()
        # Public code rejects if the first unique reciprocal ID differs.  The
        # source cache is continuous, so retain continuity as a causal
        # invariant while matching that reciprocal comparison exactly.
        if (leader.isna().any().any() or not np.all(np.diff(frames) == 1)
                or np.unique(leader["followingId"].to_numpy(np.int64))[0] != follower_id):
            continue
        direction = int(meta.loc[follower_id, "drivingDirection"])
        sign = -1.0 if direction == 1 else 1.0
        follower_length, leader_length = float(meta.loc[follower_id, "width"]), float(meta.loc[leader_id, "width"])
        follower_x = sign * (follower["x"].to_numpy(float) + .5 * follower_length)
        leader_x = sign * (leader["x"].to_numpy(float) + .5 * leader_length)
        follower_v = sign * follower["xVelocity"].to_numpy(float)
        leader_v = sign * leader["xVelocity"].to_numpy(float)
        gap = leader_x - follower_x - .5 * (follower_length + leader_length)
        if np.min(gap) <= .1:
            continue
        if velocity_policy == "strict" and (np.min(follower_v) < 0 or np.min(leader_v) < 0):
            continue
        if velocity_policy == "project_nonnegative":
            # highD reports small negative longitudinal velocities around
            # standstill (minimum -0.20 m/s in the author cache).  The causal
            # plant already enforces v>=0, so project measurement jitter
            # rather than deleting an otherwise source-selected 50 s pair.
            follower_v, leader_v = np.maximum(follower_v, 0.), np.maximum(leader_v, 0.)
        output.append({"recording_id": recording_id, "follower_id": follower_id, "leader_id": leader_id,
                       "vehicle_class": str(meta.loc[follower_id, "class"]), "frame_start": int(frames[0]),
                       "follower_x": follower_x, "follower_v": follower_v, "leader_x": leader_x, "leader_v": leader_v,
                       "gap": gap, "length_sum": .5 * (follower_length + leader_length)})
    return output


def build_author_full_dataset(raw_dir: str | Path, output_dir: str | Path, recording_ids: list[int] | None = None, *,
                              minimum_seconds: float = 50.0, velocity_policy: str = "strict",
                              lane_change_policy: str = "correct_meta") -> Path:
    """Build the full 60-recording highD long-following cohort as ragged 25 Hz arrays."""
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recordings = list(range(1, 61)) if recording_ids is None else [int(value) for value in recording_ids]
    if velocity_policy not in {"strict", "project_nonnegative"}:
        raise ValueError("velocity_policy must be strict or project_nonnegative")
    if lane_change_policy not in {"correct_meta", "source_loader_index"}:
        raise ValueError("lane_change_policy must be correct_meta or source_loader_index")
    minimum_frames = int(round(minimum_seconds * 25.0))
    pairs: list[dict] = []
    for recording_id in recordings:
        pairs.extend(_author_pairs(recording_id, raw_dir, minimum_frames, velocity_policy=velocity_policy,
                                   lane_change_policy=lane_change_policy))
    if not pairs:
        raise RuntimeError("No source-compatible long highD following pairs were found")
    offsets = np.r_[0, np.cumsum([len(pair["gap"]) for pair in pairs])].astype(np.int64)
    arrays = {key: np.concatenate([pair[key] for pair in pairs]).astype(np.float64)
              for key in ("follower_x", "follower_v", "leader_x", "leader_v", "gap")}
    arrays.update({"offsets": offsets,
                   "recording_id": np.asarray([pair["recording_id"] for pair in pairs], np.int16),
                   "follower_id": np.asarray([pair["follower_id"] for pair in pairs], np.int32),
                   "leader_id": np.asarray([pair["leader_id"] for pair in pairs], np.int32),
                   "frame_start": np.asarray([pair["frame_start"] for pair in pairs], np.int32),
                   "length_sum": np.asarray([pair["length_sum"] for pair in pairs], np.float64),
                   "vehicle_class": np.asarray([pair["vehicle_class"] for pair in pairs], dtype="U8")})
    path = output_dir / "highd_author_full_25hz.npz"
    np.savez_compressed(path, **arrays)
    per_recording = {str(recording): int(np.sum(arrays["recording_id"] == recording)) for recording in np.unique(arrays["recording_id"])}
    manifest = {"source": "all local highD raw recordings", "recordings_scanned": recordings, "native_fps": 25,
                "minimum_following_seconds": minimum_seconds, "minimum_following_frames": minimum_frames,
                "selection": "no lane changes; dominant preceding vehicle; continuous reciprocal relation; all qualifying pairs",
                "velocity_policy": velocity_policy,
                "lane_change_policy": lane_change_policy,
                "pair_count": len(pairs), "per_recording_pair_count": per_recording,
                "author_reference": "IDM_Bayesian_Calibration data_loader.load_CF_pairs, with 25 Hz retained"}
    (output_dir / "full_dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def load_ragged_pairs(path: str | Path) -> tuple[dict[str, np.ndarray], list[dict[str, np.ndarray]]]:
    """Load full-cohort arrays and expose one causal trajectory per qualifying pair."""
    data = load_dataset(path)
    pairs: list[dict[str, np.ndarray]] = []
    for index, (start, stop) in enumerate(zip(data["offsets"][:-1], data["offsets"][1:])):
        pair = {key: np.asarray(data[key][start:stop], float) for key in ("follower_x", "follower_v", "leader_x", "leader_v", "gap")}
        pair.update({key: data[key][index] for key in ("recording_id", "follower_id", "leader_id", "length_sum", "vehicle_class")})
        pairs.append(pair)
    return data, pairs
