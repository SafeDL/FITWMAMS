"""Auditable 149-tick Normalizing-Flow × ADS evaluation for HiQR-v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from world_model.src.core.flow_composition import load_flow_tail_starts
from world_model.src.core.utils import ensure_dir, save_json
from world_model.src.hiqr.environment import HiQRFlowStartMetadata, HiQRWorldRandomness
from world_model.src.hiqr.flow_evaluation import (
    compare_flow_rollouts,
    replay_states_to_ego_controls,
)

from .environment import BatchedHiQRV2WorldModelEnvironment

_AUDIT_FIELDS = (
    "event_structure",
    "mask_pattern",
    "event_structure_id",
    "event_structure_log_prob",
    "conditional_log_prob",
    "log_prob",
    "flow_checkpoint_sha256",
    "flow_schema_sha256",
    "sampling_seed",
    "sampling_temperature",
    "sampling_event_structure",
    "sampling_reject_invalid",
    "sampling_max_rounds",
    "sampling_oversample_factor",
    "sampling_min_draw",
    "sampling_num_rejected",
    "sampling_rejection_rate",
)


def _flow_metadata(
    starts: dict[str, np.ndarray],
    slots: np.ndarray,
    maps: np.ndarray,
    map_valid: np.ndarray,
    primary: np.ndarray,
    count: int,
) -> list[HiQRFlowStartMetadata]:
    rows: list[HiQRFlowStartMetadata] = []
    for index in range(count):
        rows.append(
            HiQRFlowStartMetadata(
                slot_valid=slots[index],
                map_polylines=maps[index],
                map_polyline_valid=map_valid[index],
                primary_slot_index=int(primary[index]),
                event_structure=np.asarray(
                    starts["event_structure"][index], np.float32
                ),
                mask_pattern=int(starts["mask_pattern"][index]),
                event_structure_id=int(starts["event_structure_id"][index]),
                event_structure_log_prob=float(
                    starts["event_structure_log_prob"][index]
                ),
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
                    "oversample_factor": int(
                        starts["sampling_oversample_factor"][index]
                    ),
                    "min_draw": int(starts["sampling_min_draw"][index]),
                    "num_rejected": int(starts["sampling_num_rejected"][index]),
                    "rejection_rate": float(starts["sampling_rejection_rate"][index]),
                },
            )
        )
    return rows


def evaluate_hiqr_v2_flow_ads(
    model,
    *,
    repo_root: Path,
    cache_owner: Path,
    output_dir: Path,
    max_starts: int = 0,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Replay logged ADS controls against Normalizing-Flow starts in HiQR-v2.

    The world model receives the realized ego state at every 5 Hz planning
    boundary, but it never receives future ADS actions.  This is the same
    causal interface an online ADS controller uses.
    """
    starts, cache, donors = load_flow_tail_starts(
        repo_root, sequence_cache_owner=cache_owner
    )
    count = len(donors) if max_starts <= 0 else min(int(max_starts), len(donors))
    if count < 1:
        raise ValueError("Flow×ADS evaluation requires at least one start")

    donor_rows = np.asarray(donors[:count], np.int64)
    features = np.asarray(starts["features"][:count], np.float32)
    slots = np.asarray(starts["slot_mask"][:count], bool)
    maps = np.asarray(cache["map_polylines"][donor_rows], np.float32)
    map_valid = np.asarray(cache["map_polyline_valid"][donor_rows], bool)
    primary = np.asarray(starts["primary_slot_index"][:count], np.int64)
    metadata = _flow_metadata(starts, slots, maps, map_valid, primary, count)

    environment = BatchedHiQRV2WorldModelEnvironment(
        model, device=next(model.parameters()).device
    )
    randomness = None
    if not deterministic:
        randomness = [HiQRWorldRandomness(seed=81_001 + row) for row in range(count)]
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

    anchor = int(model.cfg.anchor_state_index)
    stop = anchor + int(model.cfg.rollout_frames) + 1
    replay_states = np.asarray(
        cache["agent_states"][donor_rows, anchor:stop], np.float32
    )
    replay_valid = np.asarray(cache["agent_valid"][donor_rows, anchor:stop], bool)
    controls, ego_valid = replay_states_to_ego_controls(
        replay_states, replay_valid, dt_s=float(model.cfg.simulation_dt_s)
    )

    generated_states: list[np.ndarray] = []
    generated_valid: list[np.ndarray] = []
    observation: dict[str, Any] | None = None
    for tick in range(controls.shape[1]):
        observation = environment.step(controls[:, tick], ego_valid[:, tick])
        generated_states.append(
            observation["agent_states"].detach().cpu().numpy().copy()
        )
        generated_valid.append(observation["agent_valid"].detach().cpu().numpy().copy())
    assert observation is not None

    metrics = compare_flow_rollouts(
        np.stack(generated_states, axis=1),
        np.stack(generated_valid, axis=1),
        replay_states[:, 1:],
        replay_valid[:, 1:],
    )
    destination = ensure_dir(output_dir)
    audit_path = destination / "hiqr_v2_flow_start_audit.npz"
    audit = {name: np.asarray(starts[name][:count]) for name in _AUDIT_FIELDS}
    np.savez_compressed(
        audit_path,
        flow_condition=features,
        donor_sequence_index=donor_rows,
        slot_mask=slots,
        primary_slot_index=primary,
        **audit,
    )
    report: dict[str, Any] = {
        "model_type": model.model_type,
        "composition": "normalizing_flow_x_logged_ads_x_hiqr_v2",
        "num_flow_starts": count,
        "deterministic": deterministic,
        "flow_schema_sha256": model.flow_schema_sha256,
        "flow_start_audit": str(audit_path),
        "executed_ticks": int(controls.shape[1]),
        "scheduled_response_count": int(
            (controls.shape[1] + model.cfg.execute_frames - 1)
            // model.cfg.execute_frames
        ),
        "completed_response_count": int(observation["response_index"]),
        "ego_control_source": "adjacent_25hz_replay_velocity_transition",
        "causal_ads_contract": "current_state_only_no_future_ads_actions",
        "closed_loop_metrics": metrics,
    }
    save_json(report, destination / "hiqr_v2_flow_ads_evaluation.json")
    return report
