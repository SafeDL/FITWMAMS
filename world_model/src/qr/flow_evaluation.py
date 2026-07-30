"""Formal 8x4 EVT Flow x QR-WM composition test."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from world_model.src.core.flow_composition import (
    FLOW_COMPOSITION_SEED,
    INNER_WORLD_SAMPLES,
    ROLLOUT_FRAMES,
    load_flow_tail_starts,
    tensor,
    translated_ego_replay,
)
from world_model.src.core.long_tail_metrics import (
    collision_metrics,
    distribution_values,
    empirical_distance,
    feature_distribution_distance,
    traffic_fields,
)
from world_model.src.core.utils import ensure_dir, save_json, select_device

from .train import load_qr_checkpoint


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_flow_composition(*, checkpoint: Path, output_dir: Path) -> dict[str, Any]:
    """Evaluate Flow C0+B0 STARTs with separate ADS ego-control replay."""
    device = select_device("auto")
    starts, cache, donors = load_flow_tail_starts(Path(__file__).resolve().parents[3], device=device)
    model = load_qr_checkpoint(checkpoint, device=device)
    generated_rows: list[np.ndarray] = []
    ego_rows: list[np.ndarray] = []
    valid_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    target_ego_rows: list[np.ndarray] = []
    target_valid_rows: list[np.ndarray] = []
    audit_rows: dict[str, list[np.ndarray]] = {}
    batch_size = 16
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
        flow_metadata = {
            key: np.repeat(np.asarray(value[start:stop]), repeat, axis=0)
            for key, value in starts.items() if key != "features"
        }
        flow_metadata["donor_sequence_index"] = np.repeat(rows, repeat)
        ego_tensor = tensor(ego, device)
        ego_controls = model.dynamics.controls_from_highd_actions(ego_tensor[..., 4:6], ego_tensor)
        with torch.no_grad():
            rollout = model.rollout_from_flow(
                tensor(features, device), slot_valid=tensor(slots, device), map_polylines=tensor(map_polylines, device),
                map_polyline_valid=tensor(map_valid, device), lane_graph_edges=tensor(lane_edges, device),
                ego_future_controls=ego_controls, response_steps=ROLLOUT_FRAMES // model.cfg.execute_frames, deterministic=False,
                flow_metadata=flow_metadata,
            )
        generated_rows.append(rollout["predicted_states"][:, :, 1:].cpu().numpy())
        ego_rows.append(ego)
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
    generated_fields, target_fields = traffic_fields(generated, ego, valid), traffic_fields(target, target_ego, target_valid)
    generated_values, target_values = distribution_values(generated_fields), distribution_values(target_fields)
    destination_dir = ensure_dir(output_dir)
    audit_path = destination_dir / "flow_start_audit.npz"
    audit = {key: np.concatenate(value) for key, value in audit_rows.items()}
    np.savez_compressed(audit_path, **audit)
    report = {
        "protocol": {
            "name": "held-out EVT Flow x QR-WM composition", "outer_flow_samples": 8,
            "inner_world_samples": int(INNER_WORLD_SAMPLES), "horizon_seconds": 5.0,
            "supported_held_out_replays": int(len(np.unique(donors))), "flow_initial_conditions": int(len(donors)),
            "generated_world_futures": int(len(donors) * INNER_WORLD_SAMPLES), "not_a_paired_reconstruction": True,
            "seed": FLOW_COMPOSITION_SEED, "b0_lifecycle": "START-only",
            "ego_condition": "translated replay controls with dynamics-propagated ego state",
        },
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "flow_start_audit": {
            "path": str(audit_path), "sha256": _sha256(audit_path), "samples": int(len(audit["slot_mask"])),
            "fields": sorted(audit), "log_density": "log_prob=conditional_log_prob+event_structure_log_prob",
        },
        "closed_loop_distribution": {
            "risk_variable_distribution": {
                key: empirical_distance(target_values[key], generated_values[key])
                for key in ("ttc_s", "drac_mps2", "gap_m", "relative_speed_mps")
            },
            "physical_validity": collision_metrics(generated_fields),
            **feature_distribution_distance(generated, ego, valid, target, target_ego, target_valid, seed=FLOW_COMPOSITION_SEED),
        },
    }
    destination = destination_dir / "flow_composition_evaluation.json"
    save_json(report, destination)
    print(destination)
    return report
