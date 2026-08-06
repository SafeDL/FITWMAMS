"""Minimal auditable Flow×HiQR long-tail rollout entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from world_model.src.core.flow_composition import load_flow_tail_starts
from world_model.src.core.long_tail_metrics import (
    collision_metrics,
    distribution_values,
    empirical_distance,
    following_error_metrics,
    traffic_fields,
)
from world_model.src.core.utils import ensure_dir, save_json

from .environment import (
    BatchedHiQRWorldModelEnvironment,
    HiQRFlowStartMetadata,
    HiQRWorldRandomness,
)


def replay_states_to_ego_controls(
    states: np.ndarray, valid: np.ndarray, *, dt_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Recover causal ego controls from adjacent logged 25 Hz ego states."""
    source, target = np.asarray(states[:, :-1, 0], np.float32), np.asarray(
        states[:, 1:, 0], np.float32
    )
    source_valid = np.asarray(valid[:, :-1, 0], bool) & np.asarray(
        valid[:, 1:, 0], bool
    )
    source_speed = np.linalg.norm(source[..., 2:4], axis=-1)
    target_speed = np.linalg.norm(target[..., 2:4], axis=-1)
    acceleration = (target_speed - source_speed) / float(dt_s)
    source_vx = np.where(np.abs(source[..., 2]) < 1.0e-4, 1.0e-4, source[..., 2])
    target_vx = np.where(np.abs(target[..., 2]) < 1.0e-4, 1.0e-4, target[..., 2])
    source_heading = np.arctan2(source[..., 3], source_vx)
    target_heading = np.arctan2(target[..., 3], target_vx)
    heading_delta = np.arctan2(
        np.sin(target_heading - source_heading), np.cos(target_heading - source_heading)
    )
    yaw_rate = heading_delta / float(dt_s)
    controls = np.stack((acceleration, yaw_rate), axis=-1).astype(np.float32)
    controls[~source_valid] = 0.0
    return controls, source_valid


def compare_flow_rollouts(
    generated_states: np.ndarray,
    generated_valid: np.ndarray,
    target_states: np.ndarray,
    target_valid: np.ndarray,
) -> dict[str, Any]:
    """Compare a closed-loop HiQR rollout with the aligned replay tail."""
    generated = np.asarray(generated_states, np.float32)
    generated_mask = np.asarray(generated_valid, bool)
    target = np.asarray(target_states, np.float32)
    target_mask = np.asarray(target_valid, bool)
    if (
        generated.shape != target.shape
        or generated_mask.shape != target_mask.shape
        or generated.shape[:-1] != generated_mask.shape
        or generated.shape[2:] != (7, 6)
    ):
        raise ValueError("Flow rollout states must be [batch, ticks, 7, 6]")

    generated_ego, target_ego = generated[:, :, 0].copy(), target[:, :, 0].copy()
    generated_ego[~generated_mask[:, :, 0]] = np.nan
    target_ego[~target_mask[:, :, 0]] = np.nan
    generated_fields = traffic_fields(
        generated[:, :, 1:], generated_ego, generated_mask[:, :, 1:]
    )
    target_fields = traffic_fields(target[:, :, 1:], target_ego, target_mask[:, :, 1:])
    generated_values, target_values = (
        distribution_values(generated_fields),
        distribution_values(target_fields),
    )
    variables = ("gap_m", "ttc_s", "drac_mps2", "relative_speed_mps")
    return {
        "following_error": following_error_metrics(generated_fields, target_fields),
        "risk_variable_distribution": {
            name: empirical_distance(target_values[name], generated_values[name])
            for name in variables
        },
        "generated_collision": collision_metrics(generated_fields),
        "replay_collision": collision_metrics(target_fields),
    }


def evaluate_hiqr_flow_composition(
    model,
    *,
    repo_root: Path,
    cache_owner: Path,
    output_dir: Path,
    max_starts: int = 0,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Run causal Flow starts through HiQR without feeding ADS controls to it.

    The evaluator runs the complete 149-tick ROLL protocol.  Full ADS/AMS
    drivers can call the same batch environment and preserve random controls.
    """
    starts, cache, donors = load_flow_tail_starts(
        repo_root, sequence_cache_owner=cache_owner
    )
    take = len(donors) if max_starts <= 0 else min(int(max_starts), len(donors))
    features = np.asarray(starts["features"][:take], np.float32)
    slots = np.asarray(starts["slot_mask"][:take], bool)
    maps = np.asarray(cache["map_polylines"][donors[:take]], np.float32)
    map_valid = np.asarray(cache["map_polyline_valid"][donors[:take]], bool)
    primary = np.asarray(starts["primary_slot_index"][:take], np.int64)
    metadata = [
        HiQRFlowStartMetadata(
            slot_valid=slots[index],
            map_polylines=maps[index],
            map_polyline_valid=map_valid[index],
            primary_slot_index=int(primary[index]),
            event_structure=np.asarray(starts["event_structure"][index], np.float32),
            mask_pattern=int(starts["mask_pattern"][index]),
            event_structure_id=int(starts["event_structure_id"][index]),
            event_structure_log_prob=float(starts["event_structure_log_prob"][index]),
            conditional_log_prob=float(starts["conditional_log_prob"][index]),
            log_prob=float(starts["log_prob"][index]),
            flow_checkpoint_sha256=str(starts["flow_checkpoint_sha256"][index]),
            flow_schema_sha256=str(starts["flow_schema_sha256"][index]),
            sampling_seed=int(starts["sampling_seed"][index]),
            sampling_temperature=float(starts["sampling_temperature"][index]),
            sampling_rejection={
                "event_structure_sampling": str(
                    starts["sampling_event_structure"][index]
                ),
                "reject_invalid": bool(starts["sampling_reject_invalid"][index]),
                "max_rounds": int(starts["sampling_max_rounds"][index]),
                "oversample_factor": int(starts["sampling_oversample_factor"][index]),
                "min_draw": int(starts["sampling_min_draw"][index]),
                "num_rejected": int(starts["sampling_num_rejected"][index]),
                "rejection_rate": float(starts["sampling_rejection_rate"][index]),
            },
        )
        for index in range(take)
    ]
    environment = BatchedHiQRWorldModelEnvironment(
        model, device=next(model.parameters()).device
    )
    randomness = (
        None
        if deterministic
        else [HiQRWorldRandomness(seed=81_001 + row) for row in range(take)]
    )
    environment.reset_from_flow_batch(
        features,
        slots,
        maps,
        map_valid,
        primary_slot_index=primary,
        flow_metadata=metadata,
        deterministic=deterministic,
        world_randomness=randomness,
    )
    anchor = model.cfg.anchor_state_index
    stop = anchor + model.cfg.rollout_frames + 1
    replay_states = np.asarray(
        cache["agent_states"][donors[:take], anchor:stop], np.float32
    )
    replay_valid = np.asarray(cache["agent_valid"][donors[:take], anchor:stop], bool)
    controls, ego_valid = replay_states_to_ego_controls(
        replay_states, replay_valid, dt_s=model.cfg.simulation_dt_s
    )
    observation: dict[str, Any] | None = None
    generated_states: list[np.ndarray] = []
    generated_valid: list[np.ndarray] = []
    for tick in range(controls.shape[1]):
        observation = environment.step(controls[:, tick], ego_valid[:, tick])
        generated_states.append(
            observation["agent_states"].detach().cpu().numpy().copy()
        )
        generated_valid.append(observation["agent_valid"].detach().cpu().numpy().copy())
    assert observation is not None
    closed_loop_metrics = compare_flow_rollouts(
        np.stack(generated_states, axis=1),
        np.stack(generated_valid, axis=1),
        replay_states[:, 1:],
        replay_valid[:, 1:],
    )
    destination = ensure_dir(output_dir)
    audit_path = destination / "flow_start_audit.npz"
    np.savez_compressed(
        audit_path,
        flow_condition=features,
        donor_sequence_index=donors[:take],
        slot_mask=slots,
        event_structure=np.asarray(starts["event_structure"][:take]),
        mask_pattern=np.asarray(starts["mask_pattern"][:take]),
        primary_slot_index=primary,
        event_structure_id=np.asarray(starts["event_structure_id"][:take]),
        event_structure_log_prob=np.asarray(starts["event_structure_log_prob"][:take]),
        conditional_log_prob=np.asarray(starts["conditional_log_prob"][:take]),
        log_prob=np.asarray(starts["log_prob"][:take]),
        flow_checkpoint_sha256=np.asarray(starts["flow_checkpoint_sha256"][:take]),
        flow_schema_sha256=np.asarray(starts["flow_schema_sha256"][:take]),
        sampling_seed=np.asarray(starts["sampling_seed"][:take]),
        sampling_temperature=np.asarray(starts["sampling_temperature"][:take]),
        sampling_event_structure=np.asarray(starts["sampling_event_structure"][:take]),
        sampling_reject_invalid=np.asarray(starts["sampling_reject_invalid"][:take]),
        sampling_max_rounds=np.asarray(starts["sampling_max_rounds"][:take]),
        sampling_oversample_factor=np.asarray(
            starts["sampling_oversample_factor"][:take]
        ),
        sampling_min_draw=np.asarray(starts["sampling_min_draw"][:take]),
        sampling_num_rejected=np.asarray(starts["sampling_num_rejected"][:take]),
        sampling_rejection_rate=np.asarray(starts["sampling_rejection_rate"][:take]),
    )
    report: dict[str, Any] = {
        "model_type": model.model_type,
        "num_flow_starts": take,
        "deterministic": deterministic,
        "flow_schema_sha256": model.flow_schema_sha256,
        "event_structure": "slot_mask_plus_primary_risk_slot",
        "hiqr_h0_event_structure": "slot_mask_only_causal",
        "flow_start_audit": str(audit_path),
        "executed_ticks": int(controls.shape[1]),
        "scheduled_response_count": int(
            (controls.shape[1] + model.cfg.execute_frames - 1)
            // model.cfg.execute_frames
        ),
        "completed_response_count": int(observation["response_index"]),
        "donor_rows": donors[:take].astype(int).tolist(),
        "ego_control_source": "adjacent_25hz_replay_velocity_transition",
        "closed_loop_metrics": closed_loop_metrics,
    }
    save_json(report, destination / "hiqr_flow_composition_evaluation.json")
    return report
