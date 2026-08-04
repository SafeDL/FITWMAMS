"""Batched all-EVT-tail Flow x QR-WM composition evaluation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from world_model.src.core.flow_composition import (
    FLOW_COMPOSITION_SEED,
    INNER_WORLD_SAMPLES,
    ROLLOUT_FRAMES,
    load_flow_tail_starts,
    translated_ego_replay,
    write_flow_composition_report,
)
from world_model.src.core.utils import ensure_dir, file_sha256, select_device

from .train import load_qr_checkpoint
from .environment import BatchedQRWorldModelEnvironment


def evaluate_flow_composition(
    *, checkpoint: Path, output_dir: Path, flow_start_batch_size: int = 96
) -> dict[str, Any]:
    """Evaluate Flow STARTs with response-by-response observed ego replay."""
    if int(flow_start_batch_size) < 1:
        raise ValueError("flow_start_batch_size must be positive")
    device = select_device("auto")
    preparation_started = time.perf_counter()
    starts, cache, donors = load_flow_tail_starts(
        Path(__file__).resolve().parents[3], device=device, replay_scope="all_evt_tail"
    )
    model = load_qr_checkpoint(checkpoint, device=device)
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
        ego = np.stack([translated_ego_replay(cache["agent_states"][row], sampled_ego[index]) for index, row in enumerate(rows)])
        ego = np.repeat(ego, repeat, axis=0)
        observed_ego = np.repeat(ego[:, :: model.cfg.execute_frames], model.cfg.execute_frames, axis=1)
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
        environment.reset_from_flow_batch(
            features, slots, map_polylines, map_valid, lane_edges,
            deterministic=False, world_randomness=[int(seed) for seed in world_seed],
        )
        # Keep all 25 response intervals on the accelerator.  Copying every
        # 0.2 s prefix back to NumPy would force 25 GPU synchronizations per
        # world batch; the metrics need the complete five-second tensor only.
        frames = [
            environment.step(observed_ego[:, frame])
            for frame in range(0, ROLLOUT_FRAMES, model.cfg.execute_frames)
        ]
        generated_rows.append(torch.cat(frames, dim=1).cpu().numpy())
        ego_rows.append(observed_ego)
        valid_rows.append(np.repeat(slots[:, None], ROLLOUT_FRAMES, axis=1))
        target_rows.append(np.repeat(np.asarray(cache["agent_states"][rows, 25:, 1:], np.float32), repeat, axis=0))
        target_ego_rows.append(np.repeat(np.asarray(cache["agent_states"][rows, 25:, 0], np.float32), repeat, axis=0))
        target_valid_rows.append(np.repeat(np.asarray(cache["agent_valid"][rows, 25:, 1:], bool), repeat, axis=0))
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
            "inner_world_samples": int(INNER_WORLD_SAMPLES), "horizon_seconds": 5.0,
            "reference_scope": "all highD EVT-tail sequences",
            "supported_evt_tail_replays": int(len(np.unique(donors))), "flow_initial_conditions": int(len(donors)),
            "generated_world_futures": int(len(donors) * INNER_WORLD_SAMPLES), "not_a_paired_reconstruction": True,
            "seed": FLOW_COMPOSITION_SEED, "b0_lifecycle": "START-only",
            "ego_condition": "translated replay states supplied one response at a time and held within the response",
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
