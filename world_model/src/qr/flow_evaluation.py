"""Formal 8x4 EVT Flow x QR-WM composition test."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from world_model.src.core.flow_composition import (
    FLOW_COMPOSITION_SEED,
    INNER_WORLD_SAMPLES,
    ROLLOUT_FRAMES,
    load_flow_tail_starts,
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
from .environment import FlowStartMetadata, QRWorldModelEnvironment


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_flow_composition(*, checkpoint: Path, output_dir: Path) -> dict[str, Any]:
    """Evaluate Flow STARTs with response-by-response observed ego replay."""
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
        observed_ego = np.repeat(ego[:, :: model.cfg.execute_frames], model.cfg.execute_frames, axis=1)
        flow_metadata = {
            key: np.repeat(np.asarray(value[start:stop]), repeat, axis=0)
            for key, value in starts.items() if key != "features"
        }
        flow_metadata["donor_sequence_index"] = np.repeat(rows, repeat)
        environment = QRWorldModelEnvironment(model, device=device)
        generated = []
        for index in range(len(features)):
            metadata = FlowStartMetadata(
                slot_valid=slots[index], map_polylines=map_polylines[index],
                map_polyline_valid=map_valid[index], lane_graph_edges=lane_edges[index],
                primary_slot_index=int(flow_metadata["primary_slot_index"][index]),
                event_structure=flow_metadata["event_structure"][index],
                mask_pattern=int(flow_metadata["mask_pattern"][index]),
                event_structure_id=int(flow_metadata["event_structure_id"][index]),
                event_structure_log_prob=float(flow_metadata["event_structure_log_prob"][index]),
                conditional_log_prob=float(flow_metadata["conditional_log_prob"][index]),
                log_prob=float(flow_metadata["log_prob"][index]),
            )
            environment.reset_from_flow(features[index, :40], features[index, 40:].reshape(6, 6), metadata, deterministic=False)
            frames = [
                environment.step(observed_ego[index, frame])["background_states"]
                for frame in range(0, ROLLOUT_FRAMES, model.cfg.execute_frames)
            ]
            generated.append(np.concatenate(frames, axis=0))
        generated_rows.append(np.stack(generated))
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
            "ego_condition": "translated replay states supplied one response at a time and held within the response",
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
