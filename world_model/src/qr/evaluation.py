"""Formal held-out evaluation for QR-WM reconstruction and multimodality."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from world_model.src.core.initial_behavior_anchor import FrozenLegacyFlowSchema
from world_model.src.core.sequential_dataset import (
    ensure_frozen_flow_behavior_anchor_cache,
    is_canonical_qr_manifest,
    load_sequential_dataset,
    sequence_cache_owner_dir,
)
from world_model.src.core.utils import file_sha256, save_json, select_device
from world_model.src.core.batching import make_sequence_loader, to_device_batch

from .train import load_qr_checkpoint, require_canonical_qr_checkpoint


def _distribution_kl(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float64) + 1.0e-8
    right = right.astype(np.float64) + 1.0e-8
    left /= left.sum(); right /= right.sum()
    return float(np.sum(left * np.log(left / right)))


def evaluate_qr_world_model(
    config: dict[str, Any], *, config_dir: Path, checkpoint: Path | None = None, max_sequences: int = 0
) -> dict[str, Any]:
    paths, evaluation = config["paths"], config.get("evaluation", {})
    output = Path(paths["output_dir"])
    if not output.is_absolute():
        output = (config_dir / output).resolve()
    cache_owner = sequence_cache_owner_dir(config, config_dir=config_dir)
    arrays, manifest = load_sequential_dataset(cache_owner)
    if not is_canonical_qr_manifest(manifest):
        raise RuntimeError(
            "QR-WM evaluation requires the 1.00 s START + 4.96 s ROLL cache. "
            "A different model cache is not a valid substitute."
        )
    schema_value = paths.get("flow_schema")
    if not schema_value:
        raise ValueError("QR-WM evaluation requires paths.flow_schema for the B0 START contract")
    schema_path = Path(schema_value)
    if not schema_path.is_absolute():
        schema_path = (config_dir / schema_path).resolve()
    schema = FrozenLegacyFlowSchema.load(schema_path)
    arrays.update(ensure_frozen_flow_behavior_anchor_cache(cache_owner, arrays, manifest, schema))
    device = select_device(str(evaluation.get("device", "auto")))
    checkpoint = checkpoint or output / "checkpoints" / "best_qr_world_model.pt"
    model = load_qr_checkpoint(checkpoint, device=device)
    require_canonical_qr_checkpoint(model)
    if model.flow_schema_sha256 != schema.schema_sha256:
        raise ValueError("QR-WM checkpoint Flow schema differs from the requested B0 sidecar")
    loader = make_sequence_loader(
        arrays, "test", batch_size=int(evaluation.get("batch_size", 64)),
        maximum=int(max_sequences or evaluation.get("max_sequences", 0)), shuffle=False,
        seed=int(evaluation.get("seed", 123)), num_workers=int(evaluation.get("num_workers", 0)),
    )
    samples = max(2, int(evaluation.get("multimodal_samples", 4)))
    sums = {key: 0.0 for key in ("ade", "ade_count", "fde", "fde_count", "velocity", "acceleration", "min_ade", "min_ade_count", "min_fde", "min_fde_count", "diversity", "diversity_count", "overlap", "overlap_count", "refinement_gain", "refinement_count", "gap", "gap_count", "ttc", "ttc_count", "drac", "drac_count", "start_ade", "start_count", "start_fde", "start_fde_count", "roll_ade", "roll_count", "roll_fde", "roll_fde_count")}
    horizons = {second: [0.0, 0.0, 0.0, 0.0] for second in range(1, 6)}  # ADE sum/count, FDE sum/count
    speed_hist = [np.zeros(100, np.int64), np.zeros(100, np.int64)]
    accel_hist = [np.zeros(100, np.int64), np.zeros(100, np.int64)]
    jerk_hist = [np.zeros(100, np.int64), np.zeros(100, np.int64)]
    episode_count = collision_episodes = 0
    tail_counts = {"sequences": 0, "ade": 0.0, "ade_count": 0.0, "fde": 0.0, "fde_count": 0.0}
    with torch.no_grad():
        for values in loader:
            batch = to_device_batch(values, loader.field_names, device)
            deterministic = model.rollout_reconstruction(batch, deterministic=True, start_mode=True)
            pred = deterministic["predicted_states"][:, :, 1:]
            target = deterministic["target_states"][:, :, 1:]
            valid = deterministic["target_valid"][:, :, 1:]
            distance = torch.linalg.vector_norm(pred[..., :2] - target[..., :2], dim=-1)
            weight = valid.float()
            sums["ade"] += float((distance * weight).sum().cpu()); sums["ade_count"] += float(weight.sum().cpu())
            final_valid = valid[:, -1]
            sums["fde"] += float((distance[:, -1] * final_valid.float()).sum().cpu()); sums["fde_count"] += float(final_valid.sum().cpu())
            start_frames = min(int(model.cfg.start_reconstruction_frames), int(distance.shape[1]))
            start_distance, start_valid = distance[:, :start_frames], valid[:, :start_frames]
            sums["start_ade"] += float((start_distance * start_valid.float()).sum().cpu()); sums["start_count"] += float(start_valid.sum().cpu())
            sums["start_fde"] += float((start_distance[:, -1] * start_valid[:, -1].float()).sum().cpu()); sums["start_fde_count"] += float(start_valid[:, -1].sum().cpu())
            if start_frames < distance.shape[1]:
                roll_distance, roll_valid = distance[:, start_frames:], valid[:, start_frames:]
                sums["roll_ade"] += float((roll_distance * roll_valid.float()).sum().cpu()); sums["roll_count"] += float(roll_valid.sum().cpu())
                sums["roll_fde"] += float((roll_distance[:, -1] * roll_valid[:, -1].float()).sum().cpu()); sums["roll_fde_count"] += float(roll_valid[:, -1].sum().cpu())
            velocity = torch.linalg.vector_norm(pred[..., 2:4] - target[..., 2:4], dim=-1)
            acceleration = torch.linalg.vector_norm(pred[..., 4:6] - target[..., 4:6], dim=-1)
            sums["velocity"] += float((velocity * weight).sum().cpu()); sums["acceleration"] += float((acceleration * weight).sum().cpu())
            for second, row in horizons.items():
                frame = second * 25 - 1
                if frame >= distance.shape[1]:
                    continue
                dist, mask = distance[:, : frame + 1], valid[:, : frame + 1]
                row[0] += float((dist * mask.float()).sum().cpu()); row[1] += float(mask.sum().cpu())
                row[2] += float((distance[:, frame] * valid[:, frame].float()).sum().cpu()); row[3] += float(valid[:, frame].sum().cpu())
            # Directly check collision geometry against the externally replayed ego.
            ego = deterministic["target_states"][:, :, :1]
            dx, dy = (pred[..., 0] - ego[..., 0]).abs(), (pred[..., 1] - ego[..., 1]).abs()
            collided = valid & (dx < 4.5) & (dy < 1.0)
            collision_episodes += int(collided.flatten(start_dim=1).any(dim=1).sum().cpu()); episode_count += int(pred.shape[0])
            gap = (pred[..., 0] - ego[..., 0]).abs()
            target_gap = (target[..., 0] - ego[..., 0]).abs()
            rel_v, target_rel_v = pred[..., 2] - ego[..., 2], target[..., 2] - ego[..., 2]
            sums["gap"] += float(((gap - target_gap).abs() * weight).sum().cpu()); sums["gap_count"] += float(weight.sum().cpu())
            closing, target_closing = (-rel_v).clamp_min(0.0), (-target_rel_v).clamp_min(0.0)
            ttc = torch.where(closing > 1.0e-3, gap / closing.clamp_min(1.0e-3), torch.full_like(gap, 10.0)).clamp_max(10.0)
            target_ttc = torch.where(target_closing > 1.0e-3, target_gap / target_closing.clamp_min(1.0e-3), torch.full_like(target_gap, 10.0)).clamp_max(10.0)
            drac = torch.where(closing > 1.0e-3, closing.square() / (2.0 * gap.clamp_min(1.0e-3)), torch.zeros_like(gap))
            target_drac = torch.where(target_closing > 1.0e-3, target_closing.square() / (2.0 * target_gap.clamp_min(1.0e-3)), torch.zeros_like(target_gap))
            sums["ttc"] += float(((ttc - target_ttc).abs() * weight).sum().cpu()); sums["ttc_count"] += float(weight.sum().cpu())
            sums["drac"] += float(((drac - target_drac).abs() * weight).sum().cpu()); sums["drac_count"] += float(weight.sum().cpu())
            buffers = deterministic["background_future_actions"]
            overlap = (buffers[:, 1:, : -model.cfg.execute_frames] - buffers[:, :-1, model.cfg.execute_frames:]).abs()
            sums["overlap"] += float(overlap.sum().cpu()); sums["overlap_count"] += float(overlap.numel())
            # A complete 25-frame target plan exists for the first 25
            # receding-horizon responses.  The final five responses reach the
            # logged episode boundary and are intentionally excluded from this
            # diagnostic rather than padded with fabricated futures.
            target_plan = batch["agent_states"][:, 25:].unfold(1, model.cfg.plan_frames, model.cfg.execute_frames).permute(0, 1, 4, 2, 3)[:, :, :, 1:]
            target_plan_valid = batch["agent_valid"][:, 25:].unfold(1, model.cfg.plan_frames, model.cfg.execute_frames).permute(0, 1, 3, 2)[:, :, :, 1:]
            initial = deterministic["initial_background_future_states"][:, :target_plan.shape[1]]
            refined = deterministic["refined_background_future_states"][:, :target_plan.shape[1]]
            plan_dist_before = torch.linalg.vector_norm(initial[..., :2] - target_plan[..., :2], dim=-1)
            plan_dist_after = torch.linalg.vector_norm(refined[..., :2] - target_plan[..., :2], dim=-1)
            plan_weight = target_plan_valid.float()
            sums["refinement_gain"] += float(((plan_dist_before - plan_dist_after) * plan_weight).sum().cpu()); sums["refinement_count"] += float(plan_weight.sum().cpu())
            pred_np, target_np, valid_np = pred.cpu().numpy(), target.cpu().numpy(), valid.cpu().numpy()
            for output_hist, expected_hist, field, limits in ((speed_hist[0], speed_hist[1], slice(2, 4), (0.0, 50.0)), (accel_hist[0], accel_hist[1], slice(4, 6), (0.0, 12.0))):
                p = np.linalg.norm(pred_np[..., field], axis=-1)[valid_np]
                t = np.linalg.norm(target_np[..., field], axis=-1)[valid_np]
                output_hist += np.histogram(p, bins=100, range=limits)[0]; expected_hist += np.histogram(t, bins=100, range=limits)[0]
            if pred_np.shape[1] > 1:
                jerk_pred = np.linalg.norm(np.diff(pred_np[..., 4:6], axis=1), axis=-1) / model.cfg.simulation_dt_s
                jerk_target = np.linalg.norm(np.diff(target_np[..., 4:6], axis=1), axis=-1) / model.cfg.simulation_dt_s
                jerk_valid = valid_np[:, 1:] & valid_np[:, :-1]
                jerk_hist[0] += np.histogram(jerk_pred[jerk_valid], bins=100, range=(0.0, 40.0))[0]
                jerk_hist[1] += np.histogram(jerk_target[jerk_valid], bins=100, range=(0.0, 40.0))[0]
            tail = batch["is_evt_tail"].bool()
            if tail.any():
                tail_counts["sequences"] += int(tail.sum().cpu())
                tail_counts["ade"] += float((distance[tail] * weight[tail]).sum().cpu()); tail_counts["ade_count"] += float(weight[tail].sum().cpu())
                tail_counts["fde"] += float((distance[tail, -1] * final_valid[tail].float()).sum().cpu()); tail_counts["fde_count"] += float(final_valid[tail].sum().cpu())
            generated = [
                model.rollout_reconstruction(batch, deterministic=False, start_mode=True)["predicted_states"][:, :, 1:]
                for _ in range(samples)
            ]
            sample_tensor = torch.stack(generated)
            sample_distance = torch.linalg.vector_norm(sample_tensor[..., :2] - target[None, ..., :2], dim=-1)
            denom = weight.sum(dim=(1, 2)).clamp_min(1.0)
            min_ade = ((sample_distance * weight[None]).sum(dim=(2, 3)) / denom[None]).min(dim=0).values
            sums["min_ade"] += float(min_ade.sum().cpu()); sums["min_ade_count"] += float(min_ade.numel())
            final_dist = sample_distance[:, :, -1]
            minimum_fde = final_dist.min(dim=0).values
            sums["min_fde"] += float((minimum_fde * final_valid.float()).sum().cpu()); sums["min_fde_count"] += float(final_valid.sum().cpu())
            pair_total = pair_count = 0.0
            for left in range(samples):
                for right in range(left + 1, samples):
                    branch = torch.linalg.vector_norm(sample_tensor[left, ..., :2] - sample_tensor[right, ..., :2], dim=-1)
                    pair_total += float((branch * weight).sum().cpu()); pair_count += float(weight.sum().cpu())
            sums["diversity"] += pair_total; sums["diversity_count"] += pair_count
    divide = lambda numerator, denominator: float(numerator / max(denominator, 1.0))
    report = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
        "model_type": model.model_type, "sequence_cache": manifest, "test_sequences": int(len(loader.dataset)),
        "closed_loop_trajectory": {
            "ADE_m": divide(sums["ade"], sums["ade_count"]), "FDE_m": divide(sums["fde"], sums["fde_count"]),
            "velocity_mae_mps": divide(sums["velocity"], sums["ade_count"]), "acceleration_mae_mps2": divide(sums["acceleration"], sums["ade_count"]),
            **{f"ADE_{second}s_m": divide(row[0], row[1]) for second, row in horizons.items()},
            **{f"FDE_{second}s_m": divide(row[2], row[3]) for second, row in horizons.items()},
        },
        "start_roll_reconstruction": {
            "start_frames": int(model.cfg.start_reconstruction_frames),
            "start_seconds": float(model.cfg.start_reconstruction_frames * model.cfg.simulation_dt_s),
            "start_ADE_m": divide(sums["start_ade"], sums["start_count"]),
            "start_FDE_m": divide(sums["start_fde"], sums["start_fde_count"]),
            "roll_frames": int(model.cfg.roll_frames),
            "roll_seconds": float(model.cfg.roll_frames * model.cfg.simulation_dt_s),
            "roll_ADE_m": divide(sums["roll_ade"], sums["roll_count"]),
            "roll_FDE_m": divide(sums["roll_fde"], sums["roll_fde_count"]),
            "total_frames": int(model.cfg.rollout_frames),
            "total_seconds": float(model.cfg.rollout_frames * model.cfg.simulation_dt_s),
        },
        "multimodal": {
            "samples_per_condition": samples, "minADE_m": divide(sums["min_ade"], sums["min_ade_count"]),
            "minFDE_m": divide(sums["min_fde"], sums["min_fde_count"]),
            "trajectory_diversity_pairwise_ADE_m": divide(sums["diversity"], sums["diversity_count"]),
        },
        "safety_and_interaction": {
            "collision_episode_rate": divide(collision_episodes, episode_count), "gap_mae_m": divide(sums["gap"], sums["gap_count"]),
            "ttc_error_s": divide(sums["ttc"], sums["ttc_count"]), "drac_error_mps2": divide(sums["drac"], sums["drac_count"]),
        },
        "distribution": {
            "velocity_kl_pred_to_real": _distribution_kl(speed_hist[0], speed_hist[1]),
            "acceleration_kl_pred_to_real": _distribution_kl(accel_hist[0], accel_hist[1]),
            "jerk_kl_pred_to_real": _distribution_kl(jerk_hist[0], jerk_hist[1]),
        },
        "long_horizon_consistency": {
            "background_future_action_overlap_l1": divide(sums["overlap"], sums["overlap_count"]),
            "refinement_position_gain_m": divide(sums["refinement_gain"], sums["refinement_count"]),
        },
        "evt_tail": {
            "test_sequences": int(tail_counts["sequences"]), "ADE_m": divide(tail_counts["ade"], tail_counts["ade_count"]),
            "FDE_m": divide(tail_counts["fde"], tail_counts["fde_count"]),
        },
        "protocol": {
            "full_held_out_split": int(max_sequences or evaluation.get("max_sequences", 0)) == 0,
            "traffic_light_inputs": False, "future_encoder_at_inference": False,
            "ego_condition": "logged ego replay is used only by this reconstruction protocol",
            "flow_interface": "76-D C0+B0 reconstructs the first 1.00 s; raw B0 is absent during the following 4.96 s ROLL",
            "initialization": "encode_start(C0,map) for every held-out deterministic and stochastic rollout; temporal ROLL begins only after generated history exists",
            "start_semantics": "segment-start behavior reconstruction; it does not assert that the natural-window anchor is a risk-event onset",
            "flow_schema_sha256": schema.schema_sha256,
        },
    }
    save_json(report, output / "qr_world_model_evaluation_summary.json")
    return report
