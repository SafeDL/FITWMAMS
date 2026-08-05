"""Batched all-EVT-tail Flow x QR-WM composition evaluation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from world_model.src.core.initial_behavior_anchor import FrozenLegacyFlowSchema
from world_model.src.core.flow_composition import (
    FLOW_COMPOSITION_SEED,
    INNER_WORLD_SAMPLES,
    load_flow_tail_starts,
    translated_ego_replay,
    write_flow_composition_report,
)
from world_model.src.core.sequential_dataset import is_canonical_qr_manifest, sequence_manifest_path
from world_model.src.core.utils import ensure_dir, file_sha256, load_json, select_device

from .train import load_qr_checkpoint, require_canonical_qr_checkpoint
from .environment import BatchedQRWorldModelEnvironment


def replay_states_to_ego_controls(
    initial_ego: np.ndarray, future_ego: np.ndarray, *, dt_s: float,
) -> np.ndarray:
    """Recover 25 Hz unicycle controls from consecutive ego velocity states.

    Flow composition has logged ego states rather than an ADS control trace.
    This adapter reconstructs the control which realizes the observed speed and
    heading changes, then supplies it only to the environment's ego dynamics.
    It is deliberately outside QR-WM's model interfaces.
    """
    initial = np.asarray(initial_ego, np.float32)
    future = np.asarray(future_ego, np.float32)
    if initial.ndim != 2 or initial.shape[1] != 6 or future.ndim != 3 or future.shape[0] != len(initial) or future.shape[2] != 6:
        raise ValueError("initial_ego must be [batch, 6] and future_ego must be [batch, frames, 6]")
    if float(dt_s) <= 0.0:
        raise ValueError("dt_s must be positive")
    states = np.concatenate((initial[:, None], future), axis=1)
    velocity, next_velocity = states[:, :-1, 2:4], states[:, 1:, 2:4]
    speed, next_speed = np.linalg.norm(velocity, axis=-1), np.linalg.norm(next_velocity, axis=-1)
    dot = np.sum(velocity * next_velocity, axis=-1)
    cross = velocity[..., 0] * next_velocity[..., 1] - velocity[..., 1] * next_velocity[..., 0]
    return np.stack(((next_speed - speed) / float(dt_s), np.arctan2(cross, dot) / float(dt_s)), axis=-1).astype(np.float32)


def evaluate_flow_composition(
    *,
    checkpoint: Path,
    output_dir: Path,
    flow_start_batch_size: int = 96,
    sequence_cache_owner: Path | None = None,
    model: Any | None = None,
    prepared_starts: dict[str, np.ndarray] | None = None,
    prepared_cache: dict[str, np.ndarray] | None = None,
    prepared_donors: np.ndarray | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate Flow STARTs with 25 Hz replay-derived ego controls.

    The optional prepared inputs let the complete long-tail audit reuse the
    same Flow starts, replay cache, QR model, and device for its paired and
    unpaired studies.  Supplying one prepared input requires supplying all.
    """
    if int(flow_start_batch_size) < 1:
        raise ValueError("flow_start_batch_size must be positive")
    device = select_device("auto") if device is None else device
    repo_root = Path(__file__).resolve().parents[3]
    cache_owner = sequence_cache_owner or (
        repo_root / "results/highd_world_model/training_data/qr_sequence_cache"
    )
    cache_owner = Path(cache_owner).resolve()
    cache_manifest = load_json(sequence_manifest_path(cache_owner))
    if not is_canonical_qr_manifest(cache_manifest):
        raise RuntimeError(
            "Flow × QR-WM evaluation requires the QR raw-150-state cache "
            "(1.00 s START + 4.96 s ROLL); refusing a different replay cache."
        )
    prepared = (prepared_starts, prepared_cache, prepared_donors)
    if any(item is not None for item in prepared) and not all(item is not None for item in prepared):
        raise ValueError("prepared_starts, prepared_cache, and prepared_donors must be supplied together")
    preparation_started = time.perf_counter()
    if prepared_starts is None:
        starts, cache, donors = load_flow_tail_starts(
            repo_root, device=device, replay_scope="all_evt_tail", sequence_cache_owner=cache_owner,
        )
    else:
        starts, cache, donors = prepared_starts, prepared_cache, np.asarray(prepared_donors, np.int64)
    model = load_qr_checkpoint(checkpoint, device=device) if model is None else model
    require_canonical_qr_checkpoint(model)
    flow_schema = FrozenLegacyFlowSchema.load(repo_root / "results/highd_tail_flow/dataset_schema.json")
    if model.flow_schema_sha256 != flow_schema.schema_sha256:
        raise ValueError("QR-WM checkpoint Flow schema differs from the frozen Flow START sampler")
    rollout_frames = min(
        int(model.cfg.rollout_frames),
        int(np.asarray(cache["actions_highd"]).shape[1]),
        int(np.asarray(cache["agent_states"]).shape[1]) - 25,
    )
    if rollout_frames != int(model.cfg.rollout_frames):
        raise RuntimeError(
            "QR Flow evaluation requires the canonical START+ROLL cache with "
            f"{model.cfg.rollout_frames} transitions; found {rollout_frames}"
        )
    preparation_seconds = time.perf_counter() - preparation_started
    generated_rows: list[np.ndarray] = []
    ego_rows: list[np.ndarray] = []
    valid_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    target_ego_rows: list[np.ndarray] = []
    target_valid_rows: list[np.ndarray] = []
    audit_rows: dict[str, list[np.ndarray]] = {}
    batch_size = int(flow_start_batch_size)
    environment = BatchedQRWorldModelEnvironment(model, device=device)
    rollout_started = time.perf_counter()
    for start in range(0, len(donors), batch_size):
        stop = min(start + batch_size, len(donors))
        rows = donors[start:stop]
        repeat = int(INNER_WORLD_SAMPLES)
        features = np.repeat(np.asarray(starts["features"][start:stop], np.float32), repeat, axis=0)
        slots = np.repeat(np.asarray(starts["slot_mask"][start:stop], bool), repeat, axis=0)
        map_polylines = np.repeat(np.asarray(cache["map_polylines"][rows], np.float32), repeat, axis=0)
        map_valid = np.repeat(np.asarray(cache["map_polyline_valid"][rows], bool), repeat, axis=0)
        lane_edges = np.repeat(np.asarray(cache["lane_graph_edges"][rows]), repeat, axis=0)
        # Flow C0 has its own ego state, so translate each donor replay to that sampled origin.
        sampled_ego = np.zeros((len(rows), 6), np.float32)
        sampled_ego[:, 2:6] = np.asarray(starts["features"][start:stop, :4], np.float32)
        ego = np.stack([
            translated_ego_replay(
                cache["agent_states"][row],
                sampled_ego[index],
                rollout_frames=rollout_frames,
            )
            for index, row in enumerate(rows)
        ])
        ego = np.repeat(ego, repeat, axis=0)
        initial_ego = np.repeat(sampled_ego, repeat, axis=0)
        ego_controls = replay_states_to_ego_controls(
            initial_ego, ego, dt_s=model.cfg.simulation_dt_s,
        )
        # Each future is one independently controlled world.  The seed is
        # derived from its Flow-start row and inner-world index, never from the
        # incidental batching order, so the same world can be replayed alone.
        world_seed = (
            int(FLOW_COMPOSITION_SEED) + 10_000_000
            + np.repeat(np.arange(start, stop, dtype=np.int64), repeat) * repeat
            + np.tile(np.arange(repeat, dtype=np.int64), len(rows))
        )
        flow_metadata = {
            key: np.repeat(np.asarray(value[start:stop]), repeat, axis=0)
            for key, value in starts.items() if key != "features"
        }
        flow_metadata["donor_sequence_index"] = np.repeat(rows, repeat)
        flow_metadata["world_seed"] = world_seed
        flow_metadata["ego_control_source"] = np.full(
            len(features), "25hz_replay_velocity_transition", dtype="U40",
        )
        flow_metadata["ego_control_dt_s"] = np.full(
            len(features), model.cfg.simulation_dt_s, dtype=np.float32,
        )
        environment.reset_from_flow_batch(
            features, slots, map_polylines, map_valid, lane_edges,
            deterministic=False, world_randomness=[int(seed) for seed in world_seed],
        )
        # Keep all response intervals and their five physical ticks on the
        # accelerator.  QR-WM is still invoked only once per response.
        frames = []
        for frame in range(0, rollout_frames, model.cfg.execute_frames):
            stop_frame = min(frame + model.cfg.execute_frames, rollout_frames)
            frames.append(
                environment.advance_response(ego_controls[:, frame:stop_frame])["agent_state_frames"]
            )
        joint_frames = torch.cat(frames, dim=1)
        generated_rows.append(joint_frames[:, :, 1:].cpu().numpy())
        ego_rows.append(joint_frames[:, :, 0].cpu().numpy())
        valid_rows.append(np.repeat(slots[:, None], rollout_frames, axis=1))
        target_rows.append(np.repeat(np.asarray(cache["agent_states"][rows, 25:25 + rollout_frames, 1:], np.float32), repeat, axis=0))
        target_ego_rows.append(np.repeat(np.asarray(cache["agent_states"][rows, 25:25 + rollout_frames, 0], np.float32), repeat, axis=0))
        target_valid_rows.append(np.repeat(np.asarray(cache["agent_valid"][rows, 25:25 + rollout_frames, 1:], bool), repeat, axis=0))
        for key, value in flow_metadata.items():
            audit_rows.setdefault(key, []).append(np.asarray(value))
        audit_rows.setdefault("flow_condition", []).append(features)
        print(f"Flow x QR-WM starts {stop}/{len(donors)}", flush=True)
    generated, ego, valid = map(np.concatenate, (generated_rows, ego_rows, valid_rows))
    target, target_ego, target_valid = map(np.concatenate, (target_rows, target_ego_rows, target_valid_rows))
    destination_dir = ensure_dir(output_dir)
    audit_path = destination_dir / "flow_start_audit.npz"
    audit = {key: np.concatenate(value) for key, value in audit_rows.items()}
    np.savez_compressed(audit_path, **audit)
    return write_flow_composition_report(
        checkpoint=checkpoint,
        output_dir=destination_dir,
        protocol={
            "name": "all-highD EVT-tail Flow x QR-WM composition", "outer_flow_samples": 8,
            "inner_world_samples": int(INNER_WORLD_SAMPLES),
            "start_reconstruction_seconds": 1.0,
            "roll_seconds": 4.96,
            "horizon_seconds": rollout_frames * model.cfg.simulation_dt_s,
            "rollout_frames": rollout_frames,
            "sequence_cache": str(cache_owner),
            "sequence_cache_format": cache_manifest["cache_format"],
            "reference_scope": "all highD EVT-tail sequences",
            "supported_evt_tail_replays": int(len(np.unique(donors))), "flow_initial_conditions": int(len(donors)),
            "generated_world_futures": int(len(donors) * INNER_WORLD_SAMPLES), "not_a_paired_reconstruction": True,
            "seed": FLOW_COMPOSITION_SEED, "b0_lifecycle": "START-only",
            "initialization": "encode_start(C0,map) at the first response; later responses use realized joint history",
            "start_semantics": "segment-start behavior reconstruction; a Flow start is not claimed to be a risk-event onset",
            "ego_condition": (
                "25 Hz controls reconstructed from consecutive translated replay velocity states; "
                "applied only by environment ego dynamics"
            ),
            "ego_control_reconstruction": "acceleration=speed_difference/dt; yaw_rate=signed_heading_difference/dt",
            "world_randomness": (
                "each inner QR world receives an auditable derived world_seed that controls "
                "only its START behavior-standard-normal latent"
            ),
            "replay_matching": {
                "event_structure": "exact Flow slot mask and primary risk slot",
                "road_type": "single highD straight-lane cache cohort",
                "initial_ego_speed": "nearest EVT-tail replay longitudinal speed within the event structure",
            },
        },
        generated=generated,
        ego=ego,
        valid=valid,
        target=target,
        target_ego=target_ego,
        target_valid=target_valid,
        extra={
            "performance": {
                "flow_start_batch_size": batch_size,
                "independent_worlds_per_qr_batch": batch_size * int(INNER_WORLD_SAMPLES),
                "flow_sampling_and_replay_matching_seconds": preparation_seconds,
                "batched_qr_evolution_seconds": time.perf_counter() - rollout_started,
                "evolution_world_futures_per_second": (
                    len(donors) * int(INNER_WORLD_SAMPLES)
                    / max(time.perf_counter() - rollout_started, 1.0e-9)
                ),
            },
            "flow_start_audit": {
                "path": str(audit_path), "sha256": file_sha256(audit_path), "samples": int(len(audit["slot_mask"])),
                "fields": sorted(audit), "log_density": "log_prob=conditional_log_prob+event_structure_log_prob",
            },
        },
    )
