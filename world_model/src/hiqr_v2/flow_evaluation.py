"""Auditable 149-tick Normalizing-Flow × ADS evaluation for HiQR-v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from world_model.src.core.flow_composition import load_flow_tail_starts
from world_model.src.core.initial_behavior_anchor import summarize_first_second_states
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


def _paired_ads_intervention(
    model,
    *,
    feature: np.ndarray,
    slots: np.ndarray,
    maps: np.ndarray,
    map_valid: np.ndarray,
    primary: int,
    metadata: HiQRFlowStartMetadata,
    controls: np.ndarray,
    ego_valid: np.ndarray,
    deterministic: bool,
) -> dict[str, Any]:
    """Verify a brake changes only the *next* causal response plan."""
    environment = BatchedHiQRV2WorldModelEnvironment(
        model, device=next(model.parameters()).device
    )
    pair_features = np.repeat(feature[None], 2, axis=0)
    pair_slots = np.repeat(slots[None], 2, axis=0)
    pair_maps = np.repeat(maps[None], 2, axis=0)
    pair_map_valid = np.repeat(map_valid[None], 2, axis=0)
    randomness = None
    if not deterministic:
        # Identical streams isolate the effect of the executed ADS action.
        randomness = [HiQRWorldRandomness(seed=99_001)] * 2
    environment.reset_from_flow_batch(
        pair_features,
        pair_slots,
        pair_maps,
        pair_map_valid,
        primary_slot_index=np.asarray((primary, primary)),
        flow_metadata=[metadata, metadata],
        deterministic=deterministic,
        world_randomness=randomness,
    )
    first_delta = next_delta = 0.0
    for tick in range(int(model.cfg.execute_frames) + 1):
        action = np.repeat(controls[tick : tick + 1], 2, axis=0)
        if tick < int(model.cfg.execute_frames):
            action[1, 0] -= 5.0
        output = environment.step(action, np.full(2, ego_valid[tick], bool))
        plans = output["background_future_actions"].detach().cpu().numpy()
        delta = float(np.abs(plans[0] - plans[1]).max())
        if tick == 0:
            first_delta = delta
        elif tick == int(model.cfg.execute_frames):
            next_delta = delta
    return {
        "brake_delta_mps2": -5.0,
        "current_response_plan_max_abs_delta": first_delta,
        "next_response_plan_max_abs_delta": next_delta,
        "current_response_is_action_causal": first_delta < 1.0e-6,
        "next_response_responds_to_executed_brake": next_delta > 1.0e-6,
    }


def evaluate_hiqr_v2_flow_ads(
    model,
    *,
    repo_root: Path,
    cache_owner: Path,
    output_dir: Path,
    max_starts: int = 0,
    deterministic: bool = True,
    worlds_per_start: int = 1,
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
    worlds = max(1, int(worlds_per_start))
    if count < 1:
        raise ValueError("Flow×ADS evaluation requires at least one start")

    base_donors = np.asarray(donors[:count], np.int64)
    base_features = np.asarray(starts["features"][:count], np.float32)
    base_slots = np.asarray(starts["slot_mask"][:count], bool)
    base_maps = np.asarray(cache["map_polylines"][base_donors], np.float32)
    base_map_valid = np.asarray(cache["map_polyline_valid"][base_donors], bool)
    base_primary = np.asarray(starts["primary_slot_index"][:count], np.int64)
    base_metadata = _flow_metadata(
        starts, base_slots, base_maps, base_map_valid, base_primary, count
    )
    donor_rows = np.repeat(base_donors, worlds)
    features = np.repeat(base_features, worlds, axis=0)
    slots = np.repeat(base_slots, worlds, axis=0)
    maps = np.repeat(base_maps, worlds, axis=0)
    map_valid = np.repeat(base_map_valid, worlds, axis=0)
    primary = np.repeat(base_primary, worlds, axis=0)
    metadata = [base_metadata[index] for index in range(count) for _ in range(worlds)]

    environment = BatchedHiQRV2WorldModelEnvironment(
        model, device=next(model.parameters()).device
    )
    randomness = None
    if not deterministic:
        randomness = [
            HiQRWorldRandomness(seed=81_001 + row)
            for row in range(len(donor_rows))
        ]
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
    intervention = _paired_ads_intervention(
        model,
        feature=base_features[0],
        slots=base_slots[0],
        maps=base_maps[0],
        map_valid=base_map_valid[0],
        primary=int(base_primary[0]),
        metadata=base_metadata[0],
        controls=controls[:worlds][0],
        ego_valid=ego_valid[:worlds][0],
        deterministic=deterministic,
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

    generated = np.stack(generated_states, axis=1)
    generated_mask = np.stack(generated_valid, axis=1)
    metrics = compare_flow_rollouts(
        generated,
        generated_mask,
        replay_states[:, 1:],
        replay_valid[:, 1:],
    )
    destination = ensure_dir(output_dir)
    audit_path = destination / "hiqr_v2_flow_start_audit.npz"
    audit = {
        name: np.repeat(np.asarray(starts[name][:count]), worlds, axis=0)
        for name in _AUDIT_FIELDS
    }
    with torch.no_grad():
        initial, initial_valid, raw_b0 = model.flow_condition_to_scene(
            torch.as_tensor(features, device=next(model.parameters()).device),
            torch.as_tensor(slots, device=next(model.parameters()).device),
        )
        first_second = torch.cat(
            (
                initial[:, None, 1:],
                torch.as_tensor(generated[:, :25, 1:], device=initial.device),
            ),
            dim=1,
        )
        first_second_valid = torch.cat(
            (
                initial_valid[:, None, 1:],
                torch.as_tensor(generated_mask[:, :25, 1:], device=initial.device),
            ),
            dim=1,
        )
        generated_b0, b0_valid = summarize_first_second_states(
            first_second, first_second_valid
        )
        b0_valid &= torch.as_tensor(slots, device=initial.device)
        b0_error = (generated_b0 - raw_b0).abs().mean(dim=-1)
        b0_mae = float(
            (b0_error * b0_valid.float()).sum().div(b0_valid.float().sum().clamp_min(1)).cpu()
        )
    trajectory = generated[:, :, 1:, :2].reshape(count, worlds, -1)
    diversity: list[float] = []
    for left in range(worlds):
        for right in range(left + 1, worlds):
            diversity.append(
                float(np.linalg.norm(trajectory[:, left] - trajectory[:, right], axis=-1).mean())
            )
    finite = np.isfinite(generated).all(axis=(1, 2, 3))
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
        "worlds_per_start": worlds,
        "total_worlds": int(len(donor_rows)),
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
        "finite_rollout_rate": float(finite.mean()),
        "b0_summary_consistency_mae": b0_mae,
        "scene_agent_branch_diversity_m": float(np.mean(diversity)) if diversity else 0.0,
        "flow_audit_complete": bool(set(_AUDIT_FIELDS) <= audit.keys()),
        "paired_ads_brake_intervention": intervention,
        "closed_loop_metrics": metrics,
    }
    save_json(report, destination / "hiqr_v2_flow_ads_evaluation.json")
    return report
