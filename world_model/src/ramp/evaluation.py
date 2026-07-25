"""Deterministic RAMP-WM held-out reconstruction evaluation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from world_model.src.metrics import interaction_metrics, physical_diagnostics
from world_model.src.initial_behavior_anchor import FrozenLegacyFlowSchema
from world_model.src.semi_markov_evaluation import _metrics
from world_model.src.semi_markov_train import _loader, _to_batch
from world_model.src.sequential_dataset import (
    ensure_frozen_flow_behavior_anchor_cache,
    load_sequential_dataset,
    sequence_cache_owner_dir,
)
from world_model.src.utils import save_json, select_device
from .train import load_ramp_checkpoint


def evaluate_ramp_world_model(
    config: dict[str, Any],
    *,
    config_dir: Path,
    checkpoint: Path,
    max_sequences: int = 0,
) -> dict[str, Any]:
    paths, evaluation = config["paths"], config.get("evaluation", {})
    output = Path(paths["output_dir"])
    output = output if output.is_absolute() else (config_dir / output).resolve()
    arrays, manifest = load_sequential_dataset(
        sequence_cache_owner_dir(config, config_dir=config_dir)
    )
    if bool(config.get("model", {}).get("use_start_anchor", True)):
        raw = config["paths"].get("flow_schema")
        if not raw:
            raise ValueError("RAMP START evaluation requires paths.flow_schema")
        schema = FrozenLegacyFlowSchema.load(
            Path(raw) if Path(raw).is_absolute() else (config_dir / raw).resolve()
        )
        arrays.update(
            ensure_frozen_flow_behavior_anchor_cache(
                sequence_cache_owner_dir(config, config_dir=config_dir),
                arrays,
                manifest,
                schema,
            )
        )
    device = select_device(evaluation.get("device", "auto"))
    model = load_ramp_checkpoint(checkpoint, device=device)
    loader = _loader(
        arrays,
        "test",
        batch_size=int(evaluation.get("batch_size", 64)),
        maximum=int(max_sequences or evaluation.get("max_sequences", 0)),
        shuffle=False,
        seed=int(evaluation.get("seed", 123)),
    )
    predicted = []
    target = []
    masks = []
    tails = []
    probabilities = []
    selected = []
    memories = []
    plans = []
    plan_error_sum = plan_error_count = execute_error_sum = execute_error_count = 0.0
    plan_position_sum = plan_velocity_sum = plan_state_count = 0.0
    pair_position_sum = pair_velocity_sum = pair_count = 0.0
    with torch.no_grad():
        for values in loader:
            batch = _to_batch(values, loader.field_names, device)
            rollout = model.rollout_roll_mode(batch, deterministic=True)
            predicted.append(rollout["predicted_states"].cpu().numpy())
            target.append(rollout["target_states"].cpu().numpy())
            masks.append(rollout["target_valid"].cpu().numpy())
            tails.append(batch["is_evt_tail"].cpu().numpy().astype(bool))
            probabilities.append(rollout["candidate_probabilities"].cpu().numpy())
            selected.append(rollout["selected_candidate_index"].cpu().numpy())
            memories.append(rollout["continuous_memory"].cpu().numpy())
            plans.append(rollout["candidate_control_plans"].cpu().numpy())
            # Audit the full one-second plans at every 0.2 s response point,
            # rather than inferring plan quality from the executed prefix.
            candidate_plan = rollout["candidate_control_plans"]
            candidate_states = rollout["predicted_candidate_states"]
            choice = rollout["selected_candidate_index"]
            gather = choice[:, :, None, None, None, None].expand(
                -1,
                -1,
                1,
                candidate_plan.shape[3],
                candidate_plan.shape[4],
                candidate_plan.shape[5],
            )
            selected_plan = candidate_plan.gather(2, gather).squeeze(2)
            state_gather = choice[:, :, None, None, None, None].expand(
                -1,
                -1,
                1,
                candidate_states.shape[3],
                candidate_states.shape[4],
                candidate_states.shape[5],
            )
            selected_states = candidate_states.gather(2, state_gather).squeeze(2)
            controls, control_valid, state_targets, state_valid = [], [], [], []
            for response in range(selected_plan.shape[1]):
                item_control, item_control_valid = model._target_controls(
                    batch, response
                )
                item_state, item_state_valid = model._target_plan_states(
                    batch, response
                )
                controls.append(item_control)
                control_valid.append(item_control_valid)
                state_targets.append(item_state[:, :, 1:])
                state_valid.append(item_state_valid[:, :, 1:])
            target_control = torch.stack(controls, 1)
            valid_control = torch.stack(control_valid, 1)
            target_plan_state = torch.stack(state_targets, 1)
            valid_plan_state = torch.stack(state_valid, 1)
            control_error = (selected_plan - target_control).abs() * valid_control[
                ..., None
            ].float()
            plan_error_sum += float(control_error.sum().cpu())
            plan_error_count += float(valid_control.sum().cpu()) * 2.0
            execute = model.cfg.execute_frames
            prefix_error = control_error[:, :, :execute]
            execute_error_sum += float(prefix_error.sum().cpu())
            execute_error_count += (
                float(valid_control[:, :, :execute].sum().cpu()) * 2.0
            )
            state_error = (
                selected_states[..., :4] - target_plan_state[..., :4]
            ).abs() * valid_plan_state[..., None].float()
            plan_position_sum += float(state_error[..., :2].sum().cpu())
            plan_velocity_sum += float(state_error[..., 2:4].sum().cpu())
            plan_state_count += float(valid_plan_state.sum().cpu()) * 2.0
            agents = selected_states.shape[-2]
            upper = torch.triu(
                torch.ones((agents, agents), dtype=torch.bool, device=device),
                diagonal=1,
            )
            pair_valid = (
                valid_plan_state[..., :, None] & valid_plan_state[..., None, :] & upper
            )
            relative_error = (
                (selected_states[..., :, None, :4] - selected_states[..., None, :, :4])
                - (
                    target_plan_state[..., :, None, :4]
                    - target_plan_state[..., None, :, :4]
                )
            ).abs()
            pair_position_sum += float(
                (relative_error[..., :2] * pair_valid[..., None].float()).sum().cpu()
            )
            pair_velocity_sum += float(
                (relative_error[..., 2:4] * pair_valid[..., None].float()).sum().cpu()
            )
            pair_count += float(pair_valid.sum().cpu()) * 2.0
    pred, tgt, mask, tail = map(
        lambda values: np.concatenate(values, axis=0), (predicted, target, masks, tails)
    )
    full = _metrics(pred, tgt, mask)
    one = _metrics(pred[:, :25], tgt[:, :25], mask[:, :25])
    # Keep the tail report self-contained.  The primary trajectory metrics
    # already include the 1--5 s ADE/FDE curve; adding the matching physical
    # and interaction summaries here prevents callers from having to infer
    # EVT safety quality from the all-test aggregate or a paired-baseline file.
    if tail.any():
        evt = _metrics(pred[tail], tgt[tail], mask[tail])
        evt["physical_diagnostics"] = physical_diagnostics(
            pred[tail, :, 1:],
            mask[tail, :, 1:],
            ego_future_states=tgt[tail, :, 0],
            actions=pred[tail, :, 1:, 4:6],
            slot_names=None,
        )
        evt["interaction_metrics"] = interaction_metrics(
            pred[tail, :, 1:],
            tgt[tail, :, 1:],
            mask[tail, :, 1:],
            ego_future_states=tgt[tail, :, 0],
        )
    else:
        evt = {"available": False}
    probs, choices, memory, plan = map(
        lambda values: np.concatenate(values, axis=0),
        (probabilities, selected, memories, plans),
    )
    physical = physical_diagnostics(
        pred[:, :, 1:],
        mask[:, :, 1:],
        ego_future_states=tgt[:, :, 0],
        actions=pred[:, :, 1:, 4:6],
        slot_names=None,
    )
    interaction = interaction_metrics(
        pred[:, :, 1:], tgt[:, :, 1:], mask[:, :, 1:], ego_future_states=tgt[:, :, 0]
    )
    execute = model.cfg.execute_frames
    overlap = (
        np.abs(
            plan[:, 1:, :, : model.cfg.plan_frames - execute]
            - plan[:, :-1, :, execute:]
        ).mean()
        if plan.shape[1] > 1
        else 0.0
    )
    report = {
        "model_type": model.model_type,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "sequence_cache": manifest,
        "test_sequences": int(len(pred)),
        "one_second_conditional_reconstruction": one,
        "five_second_roll_mode": full,
        "evt_tail": evt,
        "physical_diagnostics": physical,
        "interaction_metrics": interaction,
        "plan_diagnostics": {
            "execution_prefix_control_l1": execute_error_sum
            / max(execute_error_count, 1.0),
            "full_plan_control_l1": plan_error_sum / max(plan_error_count, 1.0),
            "full_plan_state_position_l1": plan_position_sum
            / max(plan_state_count, 1.0),
            "full_plan_state_velocity_l1": plan_velocity_sum
            / max(plan_state_count, 1.0),
            "full_plan_pairwise_position_l1": pair_position_sum / max(pair_count, 1.0),
            "full_plan_pairwise_velocity_l1": pair_velocity_sum / max(pair_count, 1.0),
            "overlap_plan_discontinuity_l1": float(overlap),
            "candidate_switch_rate": (
                float(np.mean(choices[:, 1:] != choices[:, :-1]))
                if choices.shape[1] > 1
                else 0.0
            ),
            "memory_update_norm": (
                float(np.linalg.norm(np.diff(memory, axis=1), axis=-1).mean())
                if memory.shape[1] > 1
                else 0.0
            ),
            "mean_candidate_probability": probs.mean(axis=(0, 1)).tolist(),
        },
        "information_conditions": {
            "start_uses_frozen_flow_behavior_anchor": bool(model.cfg.use_start_anchor),
            "roll_reads_future_ego_or_background": False,
            "baseline_checkpoint_loaded": False,
        },
    }
    save_json(report, output / "ramp_evaluation_summary.json")
    return report
