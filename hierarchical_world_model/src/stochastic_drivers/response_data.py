"""Full-recording causal response observations, including short gaps and entries."""

import ast
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from process_highD.src.io_utils import load_config, resolve_data_path
from process_highD.src.lane_utils import parse_lane_markings
from process_highD.src.preprocess import prepare_recording

from ..core.utils import file_sha256, save_json
from ..motion import observed_acceleration
from ..scene_geometry import rectangle_separation
from .driver_policy import anticipated_entry_features, physical_leader_observation
from .raw_data import DRIVER_DATA_DIR, ROOT, canonical_lateral, recording_splits

RESPONSE_DATA_DIR = DRIVER_DATA_DIR / "interaction_response"
PROTOCOL = "positive_body_gap_and_observed_lateral_trend"


def observation_rows(recording, geometry):
    blocks = []
    for vehicle in recording.vehicle_ids():
        track = recording.get_vehicle_track(vehicle)
        meta = recording.tracks_meta.loc[vehicle]
        selected = track.index.to_numpy() % 5 == 0
        data = track.loc[
            selected,
            [
                "x",
                "y",
                "xVelocity",
                "yVelocity",
                "xAcceleration",
                "yAcceleration",
                "_abnormal",
                "frontSightDistance",
            ],
        ].reset_index()
        direction = int(meta.drivingDirection)
        centers = np.sort(
            [
                geometry["lanes"][lane]["center"]
                for lane in geometry[f"direction_{direction}_lanes"]
            ]
        )
        center = centers[np.abs(track.y.to_numpy()[:, None] - centers).argmin(1)]
        settled = (np.abs(track.yVelocity.to_numpy()) < 0.15) & (
            abs(track.y.to_numpy() - center) < 0.6
        )
        anchor = np.maximum.accumulate(np.where(settled, np.arange(len(track)), 0))
        offset = abs(track.y.to_numpy() - center[anchor])
        lane_width = float(np.median(np.diff(centers))) if len(centers) > 1 else 4.0
        data["changing"] = (
            (abs(track.yVelocity.to_numpy()) >= 0.15) | (offset >= 0.6)
        )[selected]
        data["progress"] = (offset / lane_width).clip(0, 1)[selected]
        data["lateral_speed"] = abs(track.yVelocity.to_numpy())[selected]
        data["vehicle"], data["direction"] = vehicle, direction
        data["length"], data["width"] = float(meta.width), float(meta.height)
        data["class"] = 0 if meta["class"] == "Car" else 1
        previous = (
            track[
                ["x", "y", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]
            ]
            .shift(1)
            .to_numpy(np.float32)
        )
        previous_good = np.r_[
            False,
            (np.diff(track.index.to_numpy()) == 1) & ~track._abnormal.to_numpy()[:-1],
        ]
        for k in range(6):
            data[f"previous_{k}"] = np.nan_to_num(previous[selected, k])
        data["previous_valid"] = previous_good[selected]
        motion = track[
            ["x", "y", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]
        ].to_numpy(np.float32)
        offsets = np.arange(len(track))[None] - np.arange(5, -1, -1)[:, None]
        history = motion[offsets.clip(0)]
        history_valid = (offsets >= 0) & ~track._abnormal.to_numpy()[offsets.clip(0)]
        history_valid &= (
            track.index.to_numpy()[None] - track.index.to_numpy()[offsets.clip(0)]
            == np.arange(5, -1, -1)[:, None]
        )
        acceleration = observed_acceleration(
            torch.from_numpy(motion[None]),
            history=torch.from_numpy(history[None]),
            history_valid=torch.from_numpy(history_valid[None]),
        )[0].numpy()
        data["observed_ax"], data["observed_ay"] = acceleration[selected].T
        blocks.append(data)
    return pd.concat(blocks, ignore_index=True)


def extract_response(recording_id, split, output):
    torch.set_num_threads(1)
    cfg_path = ROOT / "process_highD/config.yaml"
    cfg = load_config(cfg_path)
    recording = prepare_recording(
        resolve_data_path(cfg["paths"]["raw_dir"], cfg_path), recording_id, cfg
    )
    geometry = parse_lane_markings(recording.recording_meta)
    canonical_lateral(recording, geometry)
    rows = observation_rows(recording, geometry)
    order = np.lexsort((rows.vehicle, rows.frame, rows.direction))
    direction, frame = rows.direction.to_numpy()[order], rows.frame.to_numpy()[order]
    boundaries = np.r_[
        0,
        np.flatnonzero((np.diff(direction) != 0) | (np.diff(frame) != 0)) + 1,
        len(order),
    ]
    motion = rows[
        ["x", "y", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]
    ].to_numpy(np.float32)
    sizes = rows[["length", "width"]].to_numpy(np.float32)
    previous = rows[[f"previous_{k}" for k in range(6)]].to_numpy(np.float32)
    observed = rows[["observed_ax", "observed_ay"]].to_numpy(np.float32)
    leader = np.full(len(rows), -1, np.int32)
    gap = np.full(len(rows), 1e6, np.float32)
    leader_speed = np.linalg.norm(motion[:, 2:4], axis=-1)
    entry = np.zeros((len(rows), 4), np.float32)
    collision = np.zeros(len(rows), bool)
    for begin in range(0, len(boundaries) - 1, 32):
        groups = [
            order[lo:hi]
            for lo, hi in zip(
                boundaries[begin : begin + 32], boundaries[begin + 1 : begin + 33]
            )
        ]
        width = max(map(len, groups))
        state = np.zeros((len(groups), width, 6), np.float32)
        dims = np.ones((len(groups), width, 2), np.float32)
        past = np.zeros_like(state)
        acceleration = np.zeros((*state.shape[:2], 2), np.float32)
        valid = np.zeros((len(groups), width), bool)
        past_valid = np.zeros_like(valid)
        for k, index in enumerate(groups):
            state[k, : len(index)], dims[k, : len(index)] = motion[index], sizes[index]
            past[k, : len(index)] = previous[index]
            acceleration[k, : len(index)] = observed[index]
            valid[k, : len(index)] = ~rows._abnormal.to_numpy()[index]
            past_valid[k, : len(index)] = rows.previous_valid.to_numpy()[index]
        state, dims, past = map(torch.from_numpy, (state, dims, past))
        valid, past_valid = map(torch.from_numpy, (valid, past_valid))
        gaps, _, speeds, indices = physical_leader_observation(state, valid, dims)
        entries = anticipated_entry_features(
            state,
            valid,
            dims,
            past,
            past_valid,
            acceleration=torch.from_numpy(acceleration),
        )
        heading = torch.atan2(state[..., 3], state[..., 2])
        separation = rectangle_separation(
            state[:, None, :, :2],
            heading[:, None],
            dims,
            torch.zeros_like(state[:, None, :, :2]),
        )[:, 0]
        pair = (
            valid[:, :, None]
            & valid[:, None, :]
            & ~torch.eye(width, dtype=torch.bool)[None]
        )
        overlap = (separation.le(0) & pair).any(-1)
        for k, index in enumerate(groups):
            n = len(index)
            selected = indices[k, :n].numpy()
            leader[index] = np.where(selected >= 0, index[selected.clip(0)], -1)
            gap[index], leader_speed[index] = gaps[k, :n].numpy(), speeds[k, :n].numpy()
            entry[index], collision[index] = (
                entries[k, :n].numpy(),
                overlap[k, :n].numpy(),
            )
    speed = np.linalg.norm(motion[:, 2:4], axis=-1)
    acceleration = (
        motion[:, 2] * motion[:, 4] + motion[:, 3] * motion[:, 5]
    ) / speed.clip(0.5)
    free = leader < 0
    keep = (
        ~rows._abnormal.to_numpy()
        & ~collision
        & (speed > 5)
        & (speed < 60)
        & (abs(acceleration) < 8)
        & (
            (free & (rows.frontSightDistance.to_numpy() >= 120))
            | ((leader >= 0) & (gap > 0) & (gap < 150))
            | (entry[:, 0] > 0)
        )
    )
    road_lengths = (
        rows.groupby("direction").x.agg(lambda x: float(x.max() - x.min())).to_dict()
    )
    density = (
        rows.groupby(["direction", "frame"]).vehicle.transform("count").to_numpy()
        / np.array(
            [
                road_lengths[d] * len(geometry[f"direction_{d}_lanes"])
                for d in rows.direction
            ]
        )
        * 1000
    )
    payload = {
        "recording": np.full(keep.sum(), recording_id, np.int16),
        "vehicle": rows.vehicle.to_numpy(np.int32)[keep],
        "frame": rows.frame.to_numpy(np.int32)[keep],
        "direction": rows.direction.to_numpy(np.uint8)[keep],
        "class": rows["class"].to_numpy(np.uint8)[keep],
        "density": density[keep].astype(np.float32),
        "gap": gap[keep],
        "speed": speed[keep],
        "leader_speed": leader_speed[keep],
        "acceleration": acceleration[keep],
        "free": free[keep],
        "leader_vehicle": np.where(
            leader >= 0, rows.vehicle.to_numpy()[leader.clip(0)], -1
        )[keep].astype(np.int32),
        "entry_features": entry[keep],
    }
    for name in ("changing", "progress", "lateral_speed"):
        payload[name] = rows[name].to_numpy(np.float32)[keep]
    path = Path(output) / "recordings" / f"{recording_id:02d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    report = {
        "recording": recording_id,
        "split": split,
        "physical_vehicles": len(recording.vehicle_ids()),
        "sampled_states": len(rows),
        "response_states": int(keep.sum()),
        "short_gap_states": int((keep & (gap <= 2)).sum()),
        "anticipated_entries": int((entry[keep, 0] > 0).sum()),
        "actual_overlap_excluded": int(collision.sum()),
    }
    save_json(report, path.with_suffix(".json"))
    print("Physical response", report, flush=True)
    return report


def observation_signature():
    definitions = {}
    for filename, names in (
        (Path(__file__), ("observation_rows", "extract_response")),
        (
            Path(__file__).with_name("driver_policy.py"),
            (
                "body_relations",
                "physical_leader_observation",
                "anticipated_entry_features",
            ),
        ),
        (
            Path(__file__).parents[1] / "scene_geometry.py",
            ("rectangle_separation", "paired_rectangle_separation"),
        ),
        (
            Path(__file__).parents[1] / "motion.py",
            ("observed_acceleration", "advance_motion"),
        ),
    ):
        for node in ast.parse(filename.read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                definitions[node.name] = ast.dump(node, include_attributes=False)
    definitions["preprocessing_config"] = file_sha256(
        ROOT / "process_highD/config.yaml"
    )
    return hashlib.sha256(json.dumps(definitions, sort_keys=True).encode()).hexdigest()


def prepare_response_data(workers=8):
    output = RESPONSE_DATA_DIR
    output.mkdir(parents=True, exist_ok=True)
    signature = {
        name: file_sha256(Path(__file__).with_name(name))
        for name in ("response_data.py", "driver_policy.py")
    }
    manifest = output / "manifest.json"
    if manifest.exists():
        stored = json.loads(manifest.read_text())
        if stored["definition_sha256"] != observation_signature():
            raise ValueError(
                "Physical response cache source changed; rebuild its generated cache explicitly"
            )
        if stored["complete"]:
            return stored["recordings"]
    jobs = [
        (r, split, output)
        for split, recordings in recording_splits().items()
        for r in recordings
    ]
    needed = [
        job
        for job in jobs
        if not (output / "recordings" / f"{job[0]:02d}.json").exists()
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_extract, needed))
    reports = [
        json.loads((output / "recordings" / f"{r:02d}.json").read_text())
        for r, _, _ in jobs
    ]
    save_json(
        {
            "protocol": PROTOCOL,
            "source_sha256": signature,
            "definition_sha256": observation_signature(),
            "split_recordings": recording_splits(),
            "fps": 25,
            "state_sampling_hz": 5,
            "entry_horizon_s": 1.5,
            "complete": True,
            "recordings": reports,
        },
        manifest,
    )
    return reports


def _extract(job):
    return extract_response(*job)


def load_response(output=RESPONSE_DATA_DIR, splits=("train", "val", "test")):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["protocol"] != PROTOCOL or not manifest["complete"]:
        raise ValueError(
            "Physical response data are incomplete or have a different observation protocol"
        )
    result = {}
    for split in splits:
        blocks = []
        for recording in manifest["split_recordings"][split]:
            with np.load(output / "recordings" / f"{recording:02d}.npz") as d:
                blocks.append({k: d[k] for k in d.files})
        data = {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
        order = np.lexsort((data["frame"], data["vehicle"], data["recording"]))
        prior = np.roll(order, 1)
        valid = (
            (data["recording"][order] == data["recording"][prior])
            & (data["vehicle"][order] == data["vehicle"][prior])
            & (data["frame"][order] - data["frame"][prior] == 5)
            & ~data["free"][order]
            & ~data["free"][prior]
            & (data["leader_vehicle"][order] == data["leader_vehicle"][prior])
        )
        data["leader_acceleration"] = np.zeros(len(order), np.float32)
        data["leader_history_available"] = np.zeros(len(order), bool)
        data["leader_acceleration"][order[valid]] = (
            data["leader_speed"][order[valid]] - data["leader_speed"][prior[valid]]
        ) / 0.2
        data["leader_history_available"][order] = valid
        result[split] = data
    return result
