"""Held-out HiQR reconstruction evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from world_model.src.core.initial_behavior_anchor import FrozenLegacyFlowSchema
from world_model.src.core.long_tail_metrics import traffic_fields
from world_model.src.core.sequential_dataset import sequence_cache_owner_dir
from world_model.src.core.utils import ensure_dir, save_json, select_device

from .data import load_hiqr_training_arrays, make_hiqr_loader, to_hiqr_batch
from .train import load_hiqr_checkpoint, require_canonical_hiqr_checkpoint


@torch.no_grad()
def evaluate_hiqr_world_model(
    config: dict[str, Any],
    *,
    config_dir: Path,
    checkpoint: Path | None = None,
    max_sequences: int = 0,
) -> dict[str, Any]:
    paths, evaluation = config["paths"], config.get("evaluation", {})
    output = Path(paths["output_dir"])
    output = output if output.is_absolute() else (config_dir / output).resolve()
    checkpoint = checkpoint or output / "checkpoints" / "best_hiqr_world_model.pt"
    device = select_device(str(evaluation.get("device", "auto")))
    model = load_hiqr_checkpoint(checkpoint, device=device)
    require_canonical_hiqr_checkpoint(model)
    schema_path = Path(paths["flow_schema"])
    schema_path = (
        schema_path
        if schema_path.is_absolute()
        else (config_dir / schema_path).resolve()
    )
    schema = FrozenLegacyFlowSchema.load(schema_path)
    arrays, manifest = load_hiqr_training_arrays(
        cache_owner=sequence_cache_owner_dir(config, config_dir=config_dir),
        output_dir=output,
        flow_schema=schema,
        source_dataset_dir=paths["source_dataset_dir"],
    )
    loader = make_hiqr_loader(
        arrays,
        "test",
        batch_size=int(evaluation.get("batch_size", 64)),
        maximum=int(max_sequences or evaluation.get("max_sequences", 0)),
        shuffle=False,
        seed=int(evaluation.get("seed", 42)),
        num_workers=int(evaluation.get("num_workers", 0)),
    )
    position_sum = final_sum = action_sum = count = final_count = 0.0
    risk_sums = {"gap_m": 0.0, "ttc_s": 0.0, "drac_mps2": 0.0}
    following_count = collision_episodes = episode_count = 0
    for values in loader:
        rollout = model.rollout_reconstruction(
            to_hiqr_batch(values, loader.field_names, device), deterministic=True
        )
        predicted, target, valid = (
            rollout["predicted_states"][:, :, 1:],
            rollout["target_states"][:, :, 1:],
            rollout["target_valid"][:, :, 1:],
        )
        distance = torch.linalg.vector_norm(
            predicted[..., :2] - target[..., :2], dim=-1
        )
        position_sum += float((distance * valid.float()).sum().cpu())
        count += float(valid.sum().cpu())
        last_distance, last_valid = distance[:, -1], valid[:, -1]
        final_sum += float((last_distance * last_valid.float()).sum().cpu())
        final_count += float(last_valid.sum().cpu())
        actions = rollout["background_future_actions"][:, :, :, :]
        action_sum += float(actions.abs().mean().cpu())
        predicted_fields = traffic_fields(
            predicted.detach().cpu().numpy(),
            rollout["target_states"][:, :, 0].detach().cpu().numpy(),
            valid.detach().cpu().numpy(),
        )
        target_fields = traffic_fields(
            target.detach().cpu().numpy(),
            rollout["target_states"][:, :, 0].detach().cpu().numpy(),
            valid.detach().cpu().numpy(),
        )
        following = np.asarray(target_fields["following_valid"], bool)
        following_count += int(following.sum())
        for name in risk_sums:
            difference = np.abs(predicted_fields[name] - target_fields[name])
            risk_sums[name] += float(difference[following].sum())
        collision_episodes += int(
            np.asarray(predicted_fields["collision"], bool).any(axis=(1, 2, 3)).sum()
        )
        episode_count += int(predicted.shape[0])

    def risk_mean(name: str) -> float:
        return (
            float(risk_sums[name] / following_count)
            if following_count
            else float("nan")
        )

    report: dict[str, Any] = {
        "model_type": model.model_type,
        "checkpoint": str(checkpoint),
        "sequence_cache": manifest,
        "test_sequences": int(len(loader.dataset)),
        "ade_m": position_sum / max(count, 1.0),
        "fde_m": final_sum / max(final_count, 1.0),
        "mean_abs_generated_action": action_sum / max(1, len(loader)),
        "interaction_metrics": {
            "following_pair_frames": following_count,
            "gap_mae_m": risk_mean("gap_m"),
            "ttc_mae_s": risk_mean("ttc_s"),
            "drac_mae_mps2": risk_mean("drac_mps2"),
            "generated_collision_episode_rate": (
                collision_episodes / max(episode_count, 1)
            ),
        },
        "protocol": getattr(model, "training_protocol", {}),
    }
    ensure_dir(output)
    save_json(report, output / "hiqr_world_model_evaluation_summary.json")
    return report
