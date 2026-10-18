#!/usr/bin/env python3
"""Full-Train signed pose-error trajectories and weighted empirical representatives."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import shutil
import sys

import numpy as np
import torch
from scipy.stats import qmc
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from process_highD.src.io_utils import load_config, resolve_data_path
from process_highD.src.lane_utils import parse_lane_markings
from process_highD.src.preprocess import prepare_recording
from hierarchical_world_model.src.stochastic_drivers.raw_data import (
    canonical_lateral,
    recording_splits,
)
from hierarchical_world_model.src.error_trajectories import (
    GROUPS,
    DT,
    STEPS,
    REPRESENTATIVES,
    MODEL_PATH,
)
from hierarchical_world_model.src.motion import (
    motion_forecast,
    observed_acceleration,
    lateral_direction,
)
from hierarchical_world_model.src.core.utils import load_yaml, save_json, file_sha256
from hierarchical_world_model.src.data import prepare_experiment_data
from hierarchical_world_model.src.tracking import (
    TrackingConfig,
    tracking_reference,
    tracking_actions,
)
from hierarchical_world_model.src.core.dynamics import (
    KinematicTrafficDynamics,
    DynamicsConfig,
)
from hierarchical_world_model.src.scene_geometry import (
    torch_map_lane_geometry,
    torch_nearest_lane_geometry,
)

OUTPUT = ROOT / "results/hierarchical_world_model/bridge"
CACHE = OUTPUT / ".cache" / "error_trajectories"
SEED = 20261015
WORKERS = 8


def signed_angle(value):
    return np.arctan2(np.sin(value), np.cos(value))


def raw_errors(recording_id):
    torch.set_num_threads(1)
    threadpool_limits(1)
    directory = CACHE / f"{recording_id:02d}"
    report_path = directory / "report.json"
    if report_path.exists():
        return json.loads(report_path.read_text())
    cfg_path = ROOT / "process_highD/config.yaml"
    cfg = load_config(cfg_path)
    recording = prepare_recording(
        resolve_data_path(cfg["paths"]["raw_dir"], cfg_path), recording_id, cfg
    )
    geometry = parse_lane_markings(recording.recording_meta)
    canonical_lateral(recording, geometry)
    groups = [[] for _ in range(6)]
    vehicles = windows = 0
    times = torch.arange(1, STEPS + 1) * DT
    for vehicle in recording.vehicle_ids():
        track = recording.get_vehicle_track(vehicle)
        if len(track) < STEPS + 2:
            continue
        frames = track.index.to_numpy()
        good = ~track._abnormal.to_numpy()
        bad = np.r_[0, np.cumsum(~good)]
        anchors = np.arange(1, len(track) - STEPS)
        keep = (bad[anchors + STEPS + 1] == bad[anchors - 1]) & (
            frames[anchors + STEPS] - frames[anchors - 1] == STEPS + 1
        )
        anchors = anchors[keep]
        if not len(anchors):
            continue
        vehicles += 1
        windows += len(anchors)
        state = track[
            ["x", "y", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]
        ].to_numpy(np.float32)
        current = torch.from_numpy(state[anchors][None])
        previous = torch.from_numpy(state[anchors - 1][None])
        indices = anchors[None] - np.arange(5, -1, -1)[:, None]
        history = torch.from_numpy(state[indices.clip(0)][None])
        history_valid = (indices >= 0) & good[indices.clip(0)]
        history_valid &= (
            frames[anchors][None] - frames[indices.clip(0)]
            == np.arange(5, -1, -1)[:, None]
        )
        history_valid = torch.from_numpy(history_valid[None])
        acceleration = observed_acceleration(
            current, history=history, history_valid=history_valid
        )
        future = state[anchors[:, None] + np.arange(1, STEPS + 1)]
        heading = np.arctan2(future[..., 3], future[..., 2])
        lateral = abs(state[anchors, 3]) > 0.15
        braking = (state[anchors, 2] - state[anchors - 1, 2]) / DT < -0.5
        for cold in (False, True):
            predicted, angle = motion_forecast(
                current,
                None if cold else previous,
                None if cold else torch.ones(current.shape[:2], dtype=torch.bool),
                times,
                history=None if cold else history,
                history_valid=None if cold else history_valid,
            )
            error = np.concatenate(
                (
                    future[..., :2] - predicted[0].transpose(0, 1).numpy(),
                    signed_angle(heading - angle[0].transpose(0, 1).numpy())[..., None],
                ),
                -1,
            )
            sign = lateral_direction(
                current, torch.zeros_like(acceleration) if cold else acceleration
            )[0].numpy()
            error[..., 1:] *= sign[:, None, None]
            index = (
                4 + lateral.astype(int)
                if cold
                else braking.astype(int) + 2 * lateral.astype(int)
            )
            for group in np.unique(index):
                groups[group].append(error[index == group])
    directory.mkdir(parents=True, exist_ok=True)
    counts = {}
    for group, blocks in enumerate(groups):
        values = (
            np.concatenate(blocks) if blocks else np.empty((0, STEPS, 3), np.float32)
        )
        np.save(directory / f"{GROUPS[group]}.npy", values)
        counts[GROUPS[group]] = len(values)
    report = dict(
        recording=recording_id,
        vehicles=vehicles,
        unique_physical_windows=windows,
        counts=counts,
    )
    save_json(report, report_path)
    print("Forecast trajectories", report, flush=True)
    return report


def tracking_errors(config):
    report_path = CACHE / "tracking_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text())
    torch.set_num_threads(1)
    data = prepare_experiment_data(config, ROOT)
    rows = data.train_rows
    with np.load(
        ROOT / "results/hierarchical_world_model/diffusion_plans/train/positions.npz"
    ) as d:
        if not np.array_equal(
            d["sequence_id"].astype(str),
            np.asarray(data.bundle.arrays["sequence_id"])[rows].astype(str),
        ):
            raise ValueError("Train plans are not aligned with physical scene identity")
        plans = d["positions"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracker = TrackingConfig(**config["tracking"])
    plant = KinematicTrafficDynamics(DynamicsConfig(speed_max_mps=100))
    counts = np.zeros(2, np.int64)
    actor_windows = 0
    for offset in range(0, len(rows), 512):
        directory = CACHE / f"tracking_{offset}"
        partial = directory / "report.json"
        if partial.exists():
            saved = json.loads(partial.read_text())
            counts += saved["counts"]
            actor_windows += saved["actor_windows"]
            continue
        chosen = rows[offset : offset + 512]
        initial = np.asarray(data.bundle.arrays["agent_states"][chosen, 24, 1:])
        states = torch.tensor(initial, device=device)
        valid = torch.tensor(
            np.asarray(data.bundle.arrays["agent_valid"][chosen, 24, 1:]), device=device
        )
        plan = plans[offset : offset + len(chosen)]
        reference = torch.tensor(
            tracking_reference(
                initial[..., :2], plan, window=tracker.derivative_window
            ),
            device=device,
        )
        positions = [states[..., :2].cpu().numpy()]
        headings = [torch.atan2(states[..., 3], states[..., 2]).cpu().numpy()]
        for frame in range(149):
            states = plant.step(
                states,
                tracking_actions(states, reference[:, frame], tracker),
                valid,
                DT,
            )
            positions.append(states[..., :2].cpu().numpy())
            headings.append(torch.atan2(states[..., 3], states[..., 2]).cpu().numpy())
        positions = np.stack(positions, 1)
        headings = np.stack(headings, 1)
        path = np.concatenate((initial[:, None, :, :2], plan), 1)
        difference = np.diff(path, axis=1)
        plan_heading = np.arctan2(difference[..., 1], difference[..., 0])
        error_position = positions - path
        valid_np = valid.cpu().numpy()
        maps = torch.tensor(data.bundle.arrays["map_polylines"][chosen])
        present = torch.tensor(data.bundle.arrays["map_polyline_valid"][chosen])
        lanes, lane_present, widths = torch_map_lane_geometry(maps, present)
        plan_lane = torch_nearest_lane_geometry(
            torch.tensor(plan[..., 1]), lanes, lane_present, widths
        )[0].numpy()
        actual_lane = torch_nearest_lane_geometry(
            torch.tensor(positions[..., 1]), lanes, lane_present, widths
        )[0].numpy()
        groups = [[], []]
        for frame in range(150 - STEPS):
            epos = (
                error_position[:, frame + 1 : frame + STEPS + 1]
                - error_position[:, frame : frame + 1]
            )
            angle = signed_angle(
                headings[:, frame + 1 : frame + STEPS + 1]
                - plan_heading[:, frame : frame + STEPS]
                - (headings[:, frame] - plan_heading[:, frame])[:, None]
            )
            values = np.concatenate((epos, angle[..., None]), -1).transpose(0, 2, 1, 3)
            displacement = (
                path[:, frame + 1 : frame + STEPS + 1, :, 1]
                - path[:, frame : frame + 1, :, 1]
            )
            index = np.abs(displacement).argmax(1)
            trend = np.take_along_axis(displacement, index[:, None], axis=1)[:, 0]
            sign = np.where(trend < 0, -1, 1)
            values[..., 1:] *= sign[..., None, None]
            remaining = plan_lane[:, frame:]
            changing = (remaining != remaining[:, :1]).any(1) | (
                actual_lane[:, frame] != remaining[:, -1]
            )
            for k in (0, 1):
                groups[k].append(values[valid_np & (changing == bool(k))])
        directory.mkdir(parents=True, exist_ok=True)
        local = []
        for k, blocks in enumerate(groups):
            values = np.concatenate(blocks)
            np.save(directory / f"{GROUPS[6+k]}.npy", values)
            local.append(len(values))
        actors = int(valid_np.sum())
        save_json(dict(counts=local, actor_windows=actors), partial)
        counts += local
        actor_windows += actors
        print(
            "Tracking trajectories",
            offset + len(chosen),
            "/",
            len(rows),
            local,
            flush=True,
        )
    report = dict(
        scenes=len(rows),
        actor_windows=actor_windows,
        counts=counts.tolist(),
        all_valid_anchor_prefixes=True,
        definition="future pose error minus current observed error",
    )
    save_json(report, report_path)
    return report


def chunks(group):
    for path in sorted(CACHE.glob(f"*/{group}.npy")):
        values = np.load(path, mmap_mode="r").reshape(-1, STEPS * 3)
        for begin in range(0, len(values), 8192):
            yield path, begin, np.asarray(values[begin : begin + 8192])


def fit_group(group):
    threadpool_limits(1)
    output = CACHE / "representatives" / f"{group}.npz"
    if output.exists():
        return str(output)
    count = 0
    total = np.zeros(STEPS * 3)
    squares = np.zeros_like(total)
    for _, _, values in chunks(group):
        count += len(values)
        total += values.sum(0, dtype=np.float64)
        squares += np.square(values, dtype=np.float64).sum(0)
    if count < REPRESENTATIVES:
        raise ValueError(f"Insufficient physical windows for {group}: {count}")
    mean = total / count
    scale = np.sqrt(np.maximum(squares / count - mean**2, 1e-10))
    model = MiniBatchKMeans(
        n_clusters=REPRESENTATIVES,
        batch_size=8192,
        random_state=SEED,
        n_init=1,
        reassignment_ratio=0,
    )
    initial = []
    ready = False
    for _, _, values in chunks(group):
        if not len(values):
            continue
        normalized = ((values - mean) / scale).astype(np.float32)
        if not ready:
            initial.append(normalized)
            if sum(len(v) for v in initial) < REPRESENTATIVES:
                continue
            normalized = np.concatenate(initial)
            ready = True
        model.partial_fit(normalized)
    representatives = np.zeros((32, STEPS * 3), np.float32)
    minimum = np.full(32, np.inf)
    weights = np.zeros(32, np.int64)
    identities = [None] * 32
    for path, begin, values in chunks(group):
        if not len(values):
            continue
        normalized = ((values - mean) / scale).astype(np.float32)
        distances = ((normalized[:, None] - model.cluster_centers_[None]) ** 2).sum(-1)
        labels = distances.argmin(1)
        weights += np.bincount(labels, minlength=32)
        best = distances.argmin(0)
        for k, index in enumerate(best):
            if distances[index, k] < minimum[k]:
                minimum[k] = distances[index, k]
                representatives[k] = values[index]
                identities[k] = [str(path.relative_to(CACHE)), int(begin + index)]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        trajectories=representatives.reshape(32, STEPS, 3),
        weights=weights / count,
        count=count,
        mean=mean,
        scale=scale,
        source_identity=json.dumps(identities),
    )
    print("Empirical representatives", group, count, flush=True)
    return str(output)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    config = load_yaml(ROOT / "hierarchical_world_model/config.yaml")
    recordings = recording_splits()["train"]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        reports = list(pool.map(raw_errors, recordings))
    tracking = tracking_errors(config)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        paths = list(pool.map(fit_group, GROUPS))
    blocks = [dict(np.load(path)) for path in paths]
    np.savez_compressed(
        MODEL_PATH,
        trajectories=np.stack([b["trajectories"] for b in blocks]),
        weights=np.stack([b["weights"] for b in blocks]),
        points=qmc.Sobol(2, scramble=True, seed=SEED).random_base2(4),
        groups=np.array(GROUPS),
        dt_s=DT,
        protocol="observed_lateral_trend_direction_normalized",
    )
    save_json(
        dict(
            complete=True,
            fit_split="train",
            test_used=False,
            split_recordings=recordings,
            unique_physical_forecast_windows=sum(
                r["unique_physical_windows"] for r in reports
            ),
            groups={g: int(b["count"]) for g, b in zip(GROUPS, blocks)},
            representative_sources={
                g: json.loads(str(b["source_identity"])) for g, b in zip(GROUPS, blocks)
            },
            tracking=tracking,
            temporal_sampling="whole x/y/heading error; lateral and heading signs normalized by observed or own-plan lateral direction",
            representatives_per_group=32,
            pair_combinations=16,
            horizon_s=1.5,
            dt_s=DT,
            source_sha256=file_sha256(__file__),
            recordings=reports,
        ),
        OUTPUT / "calibration.json",
    )
    shutil.rmtree(CACHE)


if __name__ == "__main__":
    main()
