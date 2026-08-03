"""Training loop for the Semi-Markov World Model."""
from __future__ import annotations

import csv
import logging
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from world_model.src.core.batching import (
    OPTIONAL_SEQUENCE_FIELDS,
    SEQUENCE_FIELDS,
    make_sequence_loader,
    select_sequence_indices,
    to_device_batch,
)
from .model import SemiMarkovRelationalWorldModel, SemiMarkovWorldModelConfig
from world_model.src.core.initial_behavior_anchor import FrozenLegacyFlowSchema
from world_model.src.core.sequential_dataset import (
    ensure_frozen_flow_behavior_anchor_cache,
    load_sequential_dataset,
    prepare_sequential_dataset,
    sequence_cache_owner_dir,
)
from world_model.src.core.utils import ensure_dir, file_sha256, save_json, select_device, set_seed

logger = logging.getLogger(__name__)

# Kept as compatibility aliases for external scripts that previously imported
# these Semi-Markov-local names.
FIELDS = SEQUENCE_FIELDS
OPTIONAL_FIELDS = OPTIONAL_SEQUENCE_FIELDS


def _torch():
    import torch
    from torch.utils.data import DataLoader, Dataset
    return torch, DataLoader, Dataset


def _indices(arrays: dict[str, np.ndarray], split: str, maximum: int, seed: int) -> np.ndarray:
    return select_sequence_indices(arrays, split, maximum, seed)


def _loader(
    arrays: dict[str, np.ndarray], split: str, *, batch_size: int, maximum: int, shuffle: bool, seed: int,
    num_workers: int = 0,
):
    return make_sequence_loader(
        arrays,
        split,
        batch_size=batch_size,
        maximum=maximum,
        shuffle=shuffle,
        seed=seed,
        num_workers=num_workers,
    )


def _to_batch(values, names, device):
    return to_device_batch(values, names, device)


def _cfg(config: dict[str, Any]) -> SemiMarkovWorldModelConfig:
    return _model_cfg(config.get("model", {}))


def _model_cfg(source: dict[str, Any]) -> SemiMarkovWorldModelConfig:
    """Build a model config while accepting retired checkpoint fields."""
    valid = {field: source[field] for field in SemiMarkovWorldModelConfig.__dataclass_fields__ if field in source}
    return SemiMarkovWorldModelConfig(**valid)


def _sample_training_rollout_steps(model: SemiMarkovRelationalWorldModel, training: dict[str, Any]) -> int:
    """Choose the closed-loop TBPTT horizon for one optimization batch.

    The posterior still consumes the full six-second sequence.  Sampling only
    the generated portion from one to five seconds prevents the optimizer from
    treating the five-second boundary as a special, always-present training
    event while preserving the fixed 150-frame natural-driving sample.
    """
    minimum_s = float(training.get("rollout_min_seconds", 1.0))
    maximum_s = float(training.get("rollout_max_seconds", 5.0))
    interval = float(model.cfg.response_interval_s)
    minimum = max(1, int(math.ceil(minimum_s / interval - 1.0e-8)))
    maximum = min(model.response_steps, int(math.floor(maximum_s / interval + 1.0e-8)))
    if minimum > maximum:
        raise ValueError("training rollout_min_seconds/rollout_max_seconds are outside the model response horizon")
    torch, _, _ = _torch()
    return int(torch.randint(minimum, maximum + 1, ()).item())


def _teacher_forcing_ratio(training: dict[str, Any], *, epoch: int, schedule_epochs: int, stage_one_epochs: int) -> float:
    """Return either an explicit closed-loop ratio or the two-stage schedule.

    ``min_teacher_forcing=0`` means that the *final* epoch is fully
    closed-loop; it does not make every continuation epoch closed-loop.  An
    explicit fixed setting is therefore necessary for a targeted free-rollout
    recovery experiment and prevents a short continuation from accidentally
    reintroducing teacher forcing.
    """
    fixed = training.get("fixed_teacher_forcing_ratio")
    if fixed is not None:
        value = float(fixed)
        if not 0.0 <= value <= 1.0:
            raise ValueError("training.fixed_teacher_forcing_ratio must be in [0, 1]")
        return value
    if int(epoch) <= int(stage_one_epochs):
        return 1.0
    progress = (int(epoch) - int(stage_one_epochs)) / max(int(schedule_epochs) - int(stage_one_epochs), 1)
    return float(1.0 - (1.0 - float(training.get("min_teacher_forcing", 0.0))) * progress)


def _load_initial_state(model: SemiMarkovRelationalWorldModel, payload: dict[str, Any]) -> dict[str, list[str]]:
    """Load a compatible standalone-model continuation checkpoint.

    The only sanctioned partial loads add or remove the optional conflict
    attention block.
    Hidden sizes, latent vocabulary and all shared parameters remain strict.
    """
    source_cfg = dict(payload.get("model_config", {}))
    source_conflicts = bool(source_cfg.get("use_conflict_zones", False))
    target_conflicts = bool(model.cfg.use_conflict_zones)
    state_dict = dict(payload["state_dict"])
    # The START residual input was expanded from a single, unit-mixed delta to
    # target/prefix/delta in frozen Flow coordinates.  Its input projection is
    # deliberately reinitialised; the zero output layer preserves the
    # deterministic START path while the corrected feedback learns anew.
    expected = model.state_dict()
    for key in list(state_dict):
        if key in expected and tuple(state_dict[key].shape) != tuple(expected[key].shape):
            if key.startswith("anchor_residual."):
                state_dict.pop(key)
            else:
                raise ValueError(f"initial checkpoint tensor shape differs for {key}")
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing, unexpected = list(incompatible.missing_keys), list(incompatible.unexpected_keys)
    conflict_prefixes = ("encoder.conflict_", "encoder.ac_edge.")
    can_add_conflicts = target_conflicts and not source_conflicts
    allowed_missing = all(
        can_add_conflicts and key.startswith(conflict_prefixes)
        for key in missing
    )
    allowed_unexpected = source_conflicts and not target_conflicts and all(
        key.startswith(conflict_prefixes) for key in unexpected
    )
    if (missing and not allowed_missing) or (unexpected and not allowed_unexpected):
        raise ValueError(
            "initial semi-Markov checkpoint is architecturally incompatible; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {"missing": missing, "unexpected": unexpected}


def _fit_descriptor_centroids(arrays: dict[str, np.ndarray], *, num_states: int, seed: int, max_sequences: int = 12000) -> np.ndarray:
    """Fit reproducible k-means prototypes from natural-driving physics only."""
    index = _indices(arrays, "train", max_sequences, seed)
    actions = np.asarray(arrays["actions_highd"][index], np.float32).reshape(len(index), 25, 5, -1, 2).mean(axis=2)
    all_states = np.asarray(arrays["agent_states"][index], np.float32)
    states = all_states[:, 24 : 24 + 25 * 5 : 5]
    valid = np.asarray(arrays["agent_valid"][index, 24 : 24 + 25 * 5 : 5, 1:], bool)
    heading = np.arctan2(states[:, :, 1:, 3], np.where(np.abs(states[:, :, 1:, 2]) < 1.0e-4, 1.0e-4, states[:, :, 1:, 2]))
    speed = np.maximum(np.linalg.norm(states[:, :, 1:, 2:4], axis=-1), 0.5)
    longitudinal = actions[..., 0] * np.cos(heading) + actions[..., 1] * np.sin(heading)
    yaw = (-actions[..., 0] * np.sin(heading) + actions[..., 1] * np.cos(heading)) / speed
    weight = valid.astype(np.float32)
    denom = weight.sum(axis=-1).clip(min=1.0)
    accel = (longitudinal * weight).sum(axis=-1) / denom / 3.0
    yaw_mean = (yaw * weight).sum(axis=-1) / denom / 0.20
    relative_vx = ((states[:, :, 1:, 2] - states[:, :, :1, 2]) * weight).sum(axis=-1) / denom / 10.0
    relative_x = ((states[:, :, 1:, 0] - states[:, :, :1, 0]) * weight).sum(axis=-1) / denom / 40.0
    values = np.tanh(np.stack((accel, yaw_mean, relative_vx, relative_x), axis=-1)).reshape(-1, 4).astype(np.float32)
    rng = np.random.default_rng(int(seed))
    if len(values) < int(num_states):
        raise RuntimeError("not enough response descriptors to fit latent codebook")
    if len(values) > 100000:
        values = values[rng.choice(len(values), size=100000, replace=False)]
    centers = values[rng.choice(len(values), size=int(num_states), replace=False)].copy()
    for _ in range(40):
        squared = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        labels = squared.argmin(axis=1)
        updated = centers.copy()
        for code in range(int(num_states)):
            members = values[labels == code]
            updated[code] = members.mean(axis=0) if len(members) else values[rng.integers(len(values))]
        if np.max(np.abs(updated - centers)) < 1.0e-5:
            centers = updated
            break
        centers = updated
    return centers.astype(np.float32)


def _mean_metrics(model, loader, device, *, teacher_forcing: float) -> dict[str, float]:
    torch, _, _ = _torch()
    model.eval()
    total: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for values in loader:
            result = model.forward_training(_to_batch(values, loader.field_names, device), teacher_forcing_ratio=teacher_forcing)
            for key in ("loss", "recon_loss", "roll_loss", "prior_roll_loss", "endpoint_roll_loss", "prior_endpoint_roll_loss", "first_step_recon", "anchor_loss", "latent_loss", "latent_kl", "duration_nll", "censor_nll", "posterior_boundary_nll", "prototype_reconstruction", "state_bootstrap_nll", "switch_rate", "boundary_target_rate", "prior_control_loss", "late_prior_control_loss", "start_prior_roll_loss", "start_prior_endpoint_loss", "start_prior_control_loss", "plan_loss", "plan_state_loss", "joint_plan_loss", "overlap_loss", "local_residual_loss", "plan_smoothness_loss"):
                total[key] = total.get(key, 0.0) + float(result[key].detach().cpu())
            count += 1
    return {key: value / max(count, 1) for key, value in total.items()}


def _roll_mode_metrics(model, loader, device, *, seed: int) -> dict[str, float]:
    """Validation metrics on the background-only ROLL path used in deployment.

    Posterior reconstruction and differentiable soft state expectations are
    useful optimization signals, but neither is the sampled-duration ROLL
    rollout exposed by ``SemiMarkovBackgroundEnvironment``.  Checkpoints are
    therefore selected by this fixed-seed validation rollout, not by a
    posterior-mixed training loss that can improve while free rollouts drift.
    """
    torch, _, _ = _torch()
    model.eval()
    total_distance = 0.0
    total_valid = 0
    terminal_distance = 0.0
    terminal_valid = 0
    first_terminal_distance = 0.0
    first_terminal_valid = 0
    first_distance = 0.0
    first_valid = 0
    gap_error = 0.0
    gap_count = 0
    with torch.no_grad():
        offset = 0
        for values in loader:
            rollout = model.rollout_roll_mode(
                _to_batch(values, loader.field_names, device), seed=int(seed) + offset, deterministic=True,
            )
            predicted = rollout["predicted_states"]
            target = rollout["target_states"]
            valid = rollout["target_valid"]
            distance = torch.linalg.vector_norm(predicted[..., :2] - target[..., :2], dim=-1)
            total_distance += float((distance * valid.float()).sum().cpu())
            total_valid += int(valid.sum().cpu())
            terminal_distance += float((distance[:, -1] * valid[:, -1].float()).sum().cpu())
            terminal_valid += int(valid[:, -1].sum().cpu())
            first_terminal_distance += float((distance[:, 24] * valid[:, 24].float()).sum().cpu())
            first_terminal_valid += int(valid[:, 24].sum().cpu())
            first_distance += float((distance[:, :25] * valid[:, :25].float()).sum().cpu())
            first_valid += int(valid[:, :25].sum().cpu())
            pred_gap = torch.linalg.vector_norm(predicted[:, :25, 1:, :2] - predicted[:, :25, :1, :2], dim=-1)
            target_gap = torch.linalg.vector_norm(target[:, :25, 1:, :2] - target[:, :25, :1, :2], dim=-1)
            bg_valid = valid[:, :25, 1:]
            gap_error += float((pred_gap.sub(target_gap).abs() * bg_valid.float()).sum().cpu())
            gap_count += int(bg_valid.sum().cpu())
            offset += int(predicted.shape[0])
    return {
        "roll_mode_ADE_m": total_distance / max(total_valid, 1),
        "roll_mode_FDE_m": terminal_distance / max(terminal_valid, 1),
        "roll_mode_1s_ADE_m": first_distance / max(first_valid, 1),
        "roll_mode_1s_FDE_m": first_terminal_distance / max(first_terminal_valid, 1),
        "roll_mode_1s_gap_MAE_m": gap_error / max(gap_count, 1),
    }


def _prototypes(model, loader, device) -> dict[str, Any]:
    torch, _, _ = _torch()
    k = model.cfg.num_latent_states
    usage = np.zeros(k, np.float64)
    weighted_speed = np.zeros(k, np.float64)
    weighted_accel = np.zeros(k, np.float64)
    weighted_relation_change = np.zeros(k, np.float64)
    weighted_lane_change = np.zeros(k, np.float64)
    weighted_primary_change = np.zeros(k, np.float64)
    change_weight = np.zeros(k, np.float64)
    observed_durations: list[int] = []
    boundary_total = 0.0
    boundary_count = 0
    with torch.no_grad():
        for values in loader:
            batch = _to_batch(values, loader.field_names, device)
            out = model.forward_training(batch, teacher_forcing_ratio=1.0)
            q = out["posterior_state_probs"].cpu().numpy()
            # The latent state at response r is encoded from frame 24 + 5r.
            # Derive all prototype descriptors at exactly those response
            # instants, using dynamic map geometry rather than legacy slots.
            response_states = batch["agent_states"][:, 24:149:5]
            response_valid = batch["agent_valid"][:, 24:149:5]
            future, valid = response_states[:, :, 1:], response_valid[:, :, 1:]
            valid_float = valid.float()
            denom = valid_float.sum(dim=-1).clamp_min(1.0)
            speed = (torch.linalg.vector_norm(future[..., 2:4], dim=-1) * valid_float).sum(dim=-1).div(denom).cpu().numpy()
            accel = (torch.linalg.vector_norm(future[..., 4:6], dim=-1) * valid_float).sum(dim=-1).div(denom).cpu().numpy()
            usage += q.sum(axis=(0, 1))
            weighted_speed += (q * speed[..., None]).sum(axis=(0, 1))
            weighted_accel += (q * accel[..., None]).sum(axis=(0, 1))
            b, responses, n, _ = response_states.shape
            flat_states = response_states.reshape(b * responses, n, 6)
            flat_map = batch["map_polylines"][:, None].expand(-1, responses, -1, -1, -1).reshape(b * responses, *batch["map_polylines"].shape[1:])
            flat_map_valid = batch["map_polyline_valid"][:, None].expand(-1, responses, -1, -1).reshape(b * responses, *batch["map_polyline_valid"].shape[1:])
            lane = model._lane_candidates(flat_states, flat_map, flat_map_valid).reshape(b, responses, n, -1)[..., 0]
            lane_valid = response_valid & (lane >= 0)
            if responses > 1:
                lane_change = ((lane[:, 1:, 1:] != lane[:, :-1, 1:]) & lane_valid[:, 1:, 1:] & lane_valid[:, :-1, 1:]).float()
                lane_rate = lane_change.sum(dim=-1) / (lane_valid[:, 1:, 1:] & lane_valid[:, :-1, 1:]).float().sum(dim=-1).clamp_min(1.0)
                ego_lane, background_lane = lane[:, :, :1], lane[:, :, 1:]
                same = ego_lane == background_lane
                adjacent = (ego_lane - background_lane).abs() == 1
                relation = torch.where(same, torch.zeros_like(background_lane), torch.where(adjacent, torch.ones_like(background_lane), torch.full_like(background_lane, 5)))
                relation_valid = lane_valid[:, :, :1] & lane_valid[:, :, 1:]
                relation_change = ((relation[:, 1:] != relation[:, :-1]) & relation_valid[:, 1:] & relation_valid[:, :-1]).float()
                relation_rate = relation_change.sum(dim=-1) / (relation_valid[:, 1:] & relation_valid[:, :-1]).float().sum(dim=-1).clamp_min(1.0)
                distance = torch.linalg.vector_norm(response_states[:, :, 1:, :2] - response_states[:, :, :1, :2], dim=-1)
                distance = distance.masked_fill(~response_valid[:, :, 1:], float("inf"))
                primary = distance.argmin(dim=-1)
                has_primary = response_valid[:, :, 1:].any(dim=-1)
                primary_rate = ((primary[:, 1:] != primary[:, :-1]) & has_primary[:, 1:] & has_primary[:, :-1]).float()
                q_change = torch.as_tensor(q[:, 1:], device=response_states.device, dtype=response_states.dtype)
                for accumulator, values_by_response in (
                    (weighted_relation_change, relation_rate), (weighted_lane_change, lane_rate), (weighted_primary_change, primary_rate),
                ):
                    accumulator += (q_change * values_by_response[..., None]).sum(dim=(0, 1)).cpu().numpy()
                change_weight += q_change.sum(dim=(0, 1)).cpu().numpy()
            boundaries = out["posterior_boundary_probs"].cpu().numpy()[:, 1:]
            boundary_total += float(boundaries.sum())
            boundary_count += int(boundaries.size)
            for sample in boundaries:
                starts = np.flatnonzero(sample >= 0.5) + 1
                edges = np.concatenate(([0], starts, [len(sample) + 1]))
                observed_durations.extend(np.diff(edges).astype(int).tolist())
    frequency = usage / max(float(usage.sum()), 1.0)
    transition_rate = boundary_total / max(boundary_count, 1)
    return {
        "state_usage_frequency": frequency.tolist(),
        "effective_states": int(np.sum(frequency > 0.01)),
        "mean_duration_response_steps": float(1.0 / max(transition_rate, 1.0e-6)),
        "duration_distribution_note": "hazard posterior summaries require held-out evaluation; right-censoring is included in training.",
        "posterior_hard_boundary_duration_histogram": {
            str(step): int(count) for step, count in enumerate(np.bincount(observed_durations, minlength=1)) if step > 0 and count
        },
        "mean_agent_speed_mps_by_state": (weighted_speed / np.maximum(usage, 1.0)).tolist(),
        "mean_agent_acceleration_mps2_by_state": (weighted_accel / np.maximum(usage, 1.0)).tolist(),
        "relation_edge_change_rate_by_state": (weighted_relation_change / np.maximum(change_weight, 1.0)).tolist(),
        "lane_assignment_change_rate_by_state": (weighted_lane_change / np.maximum(change_weight, 1.0)).tolist(),
        "primary_interaction_change_rate_by_state": (weighted_primary_change / np.maximum(change_weight, 1.0)).tolist(),
        "dataset": "highD fixed natural-driving sequence cache",
        "label_permutation_warning": "Latent state numbers are not comparable across independently trained checkpoints.",
    }


def _resolved_path(value: str, config_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_dir / path).resolve()


def _write_reproducibility_artifacts(config: dict[str, Any], output_dir: Path, config_dir: Path) -> dict[str, str]:
    """Persist immutable inputs before any optimizer update."""
    (output_dir / "effective_config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    (output_dir / "command.txt").write_text(" ".join(sys.argv), encoding="utf-8")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True, check=False).stdout.strip() or "unknown"
    (output_dir / "git_commit_sha.txt").write_text(commit + "\n", encoding="utf-8")
    paths = config.get("paths", {})
    values = {"git_commit_sha": commit}
    for name in ("flow_checkpoint", "flow_schema"):
        raw = paths.get(name)
        if raw:
            path = _resolved_path(str(raw), config_dir)
            if not path.exists():
                raise FileNotFoundError(f"frozen Flow artifact missing: {path}")
            values[f"{name}_sha256"] = file_sha256(path)
    return values


def _stage_schedule(training: dict[str, Any], fallback_epochs: int) -> list[dict[str, Any]]:
    stages = [dict(item) for item in training.get("stages", [])]
    if not stages:
        return [{"name": "joint", "epochs": fallback_epochs}]
    if any(int(item.get("epochs", 0)) <= 0 for item in stages):
        raise ValueError("every training stage must have a positive epoch count")
    return stages


def _configure_stage_optimizer(model, stage: str, training: dict[str, Any], torch):
    """Select the explicit training subset and construct its optimizer.

    ``train_all_parameters`` is the reproducible from-scratch protocol.  It
    must not silently inherit one of the plan-head-only fine-tuning policies.
    """
    train_all = bool(training.get("train_all_parameters", False))
    if train_all:
        allowed: tuple[str, ...] = tuple()
    elif model.uses_control_plan:
        plan_heads = (
            "decoder.plan_time", "decoder.intent_time", "decoder.cross_vehicle", "decoder.local_time",
        )
        # The plan heads first learn forecasts and overlap; only joint
        # fine-tuning may alter the decoder/encoder execution path.
        allowed = {
            "anchor": plan_heads,
            "roll": plan_heads if bool(training.get("plan_freeze_latent_in_roll", False)) else (*plan_heads, "latent"),
            "joint": ("decoder",) if bool(training.get("plan_decoder_only_joint", False)) else tuple(),
        }.get(stage, tuple())
    else:
        allowed = {
            "anchor": ("anchor_residual", "decoder"),
            "roll": ("anchor_residual", "decoder", "latent"),
            "joint": tuple(),
        }.get(stage, tuple())
    for name, parameter in model.named_parameters():
        parameter.requires_grad = not allowed or any(name.startswith(prefix) for prefix in allowed)
    groups = []
    covered: set[int] = set()
    for label, prefixes, default_lr in (
        ("encoder", ("encoder",), 5.0e-6),
        ("anchor", ("anchor_residual",), 2.0e-5),
        ("decoder_prior", ("decoder", "latent"), 1.0e-5),
    ):
        values = [
            p for name, p in model.named_parameters()
            if p.requires_grad and id(p) not in covered and any(name.startswith(prefix) for prefix in prefixes)
        ]
        if values:
            groups.append({"params": values, "lr": float(training.get(f"{label}_learning_rate", default_lr))})
            covered.update(id(p) for p in values)
    extra = [p for p in model.parameters() if p.requires_grad and id(p) not in covered]
    if extra:
        groups.append({"params": extra, "lr": float(training.get("learning_rate", 5.0e-5))})
    if not groups:
        raise RuntimeError(f"training stage {stage!r} has no trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=float(training.get("weight_decay", 1.0e-4)))


def _trainable_parameter_summary(model) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if total <= 0 or trainable <= 0:
        raise RuntimeError("Semi-Markov World Model has no trainable parameters")
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_fraction": float(trainable / total),
    }


def train_semi_markov_world_model(
    config: dict[str, Any],
    *,
    config_dir: Path,
    epochs: int | None = None,
    rebuild_dataset: bool = False,
    max_sequences: int | None = None,
    max_train_sequences: int = 0,
    max_val_sequences: int = 0,
    initial_checkpoint: str | Path | None = None,
    stage_offset: int = 0,
) -> dict[str, Any]:
    paths = config["paths"]
    output_dir = Path(paths["output_dir"])
    if not output_dir.is_absolute():
        output_dir = (config_dir / output_dir).resolve()
    ensure_dir(output_dir)
    reproducibility = _write_reproducibility_artifacts(config, output_dir, config_dir)
    manifest = prepare_sequential_dataset(config, config_dir=config_dir, rebuild=rebuild_dataset, max_sequences=max_sequences)
    cache_owner = sequence_cache_owner_dir(config, config_dir=config_dir)
    arrays, loaded_manifest = load_sequential_dataset(cache_owner)
    training = config.get("training", {})
    seed = int(training.get("seed", 42))
    set_seed(seed)
    device = select_device(str(training.get("device", "auto")))
    torch, _, _ = _torch()
    model = SemiMarkovRelationalWorldModel(_cfg(config)).to(device)
    if model.uses_behavior_anchor:
        schema_path = paths.get("flow_schema")
        checkpoint_path = paths.get("flow_checkpoint")
        if not schema_path or not checkpoint_path:
            raise ValueError("Semi-Markov World Model requires paths.flow_schema and paths.flow_checkpoint for the frozen Flow contract")
        frozen = FrozenLegacyFlowSchema.load(_resolved_path(str(schema_path), config_dir))
        expected_schema = paths.get("flow_schema_sha256")
        if expected_schema is not None and frozen.schema_sha256 != str(expected_schema):
            raise ValueError("configured Flow schema SHA256 does not match the frozen schema")
        frozen.verify_checkpoint(_resolved_path(str(checkpoint_path), config_dir), paths.get("flow_checkpoint_sha256"))
        model.set_frozen_flow_schema(frozen)
        arrays.update(ensure_frozen_flow_behavior_anchor_cache(cache_owner, arrays, loaded_manifest, frozen))
    initial_payload: dict[str, Any] | None = None
    if initial_checkpoint is not None:
        initial_path = Path(initial_checkpoint).resolve()
        initial_payload = torch.load(initial_path, map_location=device, weights_only=False)
        if initial_payload.get("model_type") != model.model_type:
            raise ValueError(f"Not a semi_markov_relational checkpoint: {initial_path}")
        transfer_keys = _load_initial_state(model, initial_payload)
        logger.info("continuing from checkpoint=%s missing=%s unexpected=%s", initial_path, transfer_keys["missing"], transfer_keys["unexpected"])
    else:
        # A compatible continuation already contains the audited prototype
        # centroids.  Refitting them here would be both wasted work and a
        # needless source of drift before the checkpoint is loaded.
        centroids = _fit_descriptor_centroids(arrays, num_states=model.cfg.num_latent_states, seed=seed)
        model.latent.set_descriptor_centroids(torch.from_numpy(centroids).to(device))
    loader_workers = int(training.get("num_workers", training.get("num_data_workers", 0)))
    train_loader = _loader(
        arrays, "train", batch_size=int(training.get("batch_size", 16)), maximum=max_train_sequences,
        shuffle=True, seed=seed, num_workers=loader_workers,
    )
    val_loader = _loader(
        arrays, "val", batch_size=int(training.get("batch_size", 16)), maximum=max_val_sequences,
        shuffle=False, seed=seed, num_workers=loader_workers,
    )
    requested_epochs = int(epochs if epochs is not None else training.get("epochs", 80))
    stage_offset = int(stage_offset)
    if requested_epochs <= 0 or stage_offset < 0:
        raise ValueError("epochs must be positive and stage offset cannot be negative")
    stages = _stage_schedule(training, requested_epochs)
    schedule_epochs = sum(int(stage["epochs"]) for stage in stages) if training.get("stages") else requested_epochs
    if stage_offset and initial_checkpoint is None:
        raise ValueError("a non-zero stage offset requires --initial-checkpoint")
    if stage_offset + requested_epochs > schedule_epochs:
        raise ValueError("requested continuation exceeds the configured stage schedule")
    stage_one = int(training.get("stage_one_epochs", max(1, schedule_epochs // 3)))
    grad_clip = float(training.get("grad_clip", 1.0))
    tbptt_response_steps = max(0, int(training.get("tbptt_response_steps", 5)))
    training_protocol = {
        "rollout_min_seconds": float(training.get("rollout_min_seconds", 1.0)),
        "rollout_max_seconds": float(training.get("rollout_max_seconds", 5.0)),
        "tbptt_response_steps": tbptt_response_steps,
        "posterior_full_sequence_frames": 150,
        # Explicitly preserve caller-provided limits so an exploratory subset
        # continuation cannot later be mistaken for a full-cache training run.
        "max_train_sequences": int(max_train_sequences),
        "max_validation_sequences": int(max_val_sequences),
        "stage_offset": stage_offset,
        "run_epochs": requested_epochs,
        "from_scratch": initial_checkpoint is None,
        "train_all_parameters": bool(training.get("train_all_parameters", False)),
    }
    if training.get("fixed_teacher_forcing_ratio") is not None:
        training_protocol["fixed_teacher_forcing_ratio"] = float(training["fixed_teacher_forcing_ratio"])
    history_path = output_dir / "training_history.csv"
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    best_path = checkpoint_dir / "best_semi_markov_relational.pt"
    best = float("inf")
    records: list[dict[str, float]] = []
    optimizer = None
    active_stage = ""
    # A continuation must compete with its starting model.  Otherwise the
    # first optimizer update becomes the "best" checkpoint even when it
    # degrades free-running ROLL performance.
    initial_selection: dict[str, float] | None = None

    def write_history() -> None:
        """Persist every completed epoch so an interrupted run remains auditable."""
        prior_records: list[dict[str, str]] = []
        if stage_offset and history_path.exists():
            with history_path.open(newline="", encoding="utf-8") as handle:
                prior_records = list(csv.DictReader(handle))
        combined_records: list[dict[str, Any]] = [*prior_records, *records]
        fieldnames: list[str] = []
        for item in combined_records:
            for key in item:
                if key not in fieldnames:
                    fieldnames.append(key)
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_records)
    if initial_payload is not None:
        initial_validation = _mean_metrics(model, val_loader, device, teacher_forcing=0.0)
        initial_selection = _roll_mode_metrics(model, val_loader, device, seed=seed + 10_000)
        best = float(initial_validation["loss"])
        if not math.isfinite(best):
            raise FloatingPointError("initial checkpoint has non-finite validation metrics")
        torch.save({
            "model_type": model.model_type, "model_config": model.config_payload(), "state_dict": model.state_dict(),
            "epoch": 0, "validation": initial_validation, "roll_mode_validation": initial_selection,
            "sequence_manifest": manifest, "adapter_version": manifest.get("adapter", "unknown"), "dynamics_version": model.dynamics.version,
            "training_protocol": training_protocol, "selection_origin": "initial_checkpoint",
        }, best_path)
        save_json(_prototypes(model, val_loader, device), output_dir / "latent_state_prototypes.json")
        logger.info(
            "initial val_loss=%.5f roll_mode_fde=%.5f",
            best,
            float(initial_selection["roll_mode_FDE_m"]),
        )
    for local_epoch in range(1, requested_epochs + 1):
        epoch = stage_offset + local_epoch
        remaining_epochs = epoch
        stage_spec = stages[-1]
        for item in stages:
            if remaining_epochs <= int(item["epochs"]):
                stage_spec = item; break
            remaining_epochs -= int(item["epochs"])
        stage_name = str(stage_spec.get("name", "joint")).replace("_realization", "").replace("_rollout", "")
        stage_name = {"A": "anchor", "B": "roll", "C": "joint", "anchor": "anchor", "roll": "roll", "joint": "joint"}.get(stage_name, stage_name)
        if stage_name != active_stage:
            optimizer = _configure_stage_optimizer(model, stage_name, training, torch)
            parameter_summary = _trainable_parameter_summary(model)
            if bool(training.get("train_all_parameters", False)) and parameter_summary["trainable_fraction"] != 1.0:
                raise RuntimeError("from-scratch full training must optimize every Semi-Markov model parameter")
            training_protocol["parameter_summary"] = parameter_summary
            active_stage = stage_name
        teacher = _teacher_forcing_ratio(
            training, epoch=epoch, schedule_epochs=schedule_epochs, stage_one_epochs=stage_one,
        )
        model.train()
        totals: dict[str, float] = {}
        batches = 0
        for values in train_loader:
            optimizer.zero_grad(set_to_none=True)
            # Stage A is deliberately a first-second realization problem.
            # Longer free rollouts belong to Stage B; including them here
            # diluted the START loss and spent most of its compute after B0
            # had already been handed off.
            rollout_steps = (
                model.cfg.behavior_anchor_response_steps
                if active_stage == "anchor"
                else _sample_training_rollout_steps(model, training)
            )
            result = model.forward_training(
                _to_batch(values, train_loader.field_names, device), teacher_forcing_ratio=teacher,
                rollout_response_steps=rollout_steps, tbptt_response_steps=tbptt_response_steps,
            )
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            for key in ("loss", "recon_loss", "roll_loss", "prior_roll_loss", "endpoint_roll_loss", "prior_endpoint_roll_loss", "prior_control_loss", "late_prior_control_loss", "start_prior_roll_loss", "start_prior_endpoint_loss", "start_prior_control_loss", "plan_loss", "plan_state_loss", "joint_plan_loss", "overlap_loss", "local_residual_loss", "plan_smoothness_loss", "anchor_loss", "latent_loss", "switch_rate", "boundary_target_rate", "rollout_response_steps"):
                totals[key] = totals.get(key, 0.0) + float(result[key].detach().cpu())
            batches += 1
        train_metrics = {f"train_{key}": value / max(batches, 1) for key, value in totals.items()}
        val_metrics = _mean_metrics(model, val_loader, device, teacher_forcing=0.0)
        rollout_metrics = _roll_mode_metrics(model, val_loader, device, seed=seed + 10_000)
        row = {
            "epoch": epoch,
            "stage": active_stage,
            "teacher_forcing": teacher,
            **train_metrics,
            **{f"val_{key}": value for key, value in val_metrics.items()},
            **{f"val_{key}": value for key, value in rollout_metrics.items()},
        }
        records.append(row)
        logger.info(
            "epoch=%d train_loss=%.5f val_loss=%.5f val_roll=%.5f roll_mode_fde=%.5f teacher=%.2f",
            epoch,
            row["train_loss"],
            row["val_loss"],
            row["val_roll_loss"],
            row["val_roll_mode_FDE_m"],
            teacher,
        )
        if not math.isfinite(val_metrics["loss"]):
            raise FloatingPointError("non-finite validation loss; no checkpoint was written")
        selection_score = val_metrics["loss"]
        if not math.isfinite(selection_score):
            raise FloatingPointError("non-finite validation loss; no checkpoint was written")
        eligible = True
        row["selection_eligible"] = 1.0
        if eligible and selection_score < best:
            best = selection_score
            payload = {
                "model_type": model.model_type,
                "model_config": model.config_payload(),
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "validation": val_metrics,
                "roll_mode_validation": rollout_metrics,
                "sequence_manifest": manifest,
                "adapter_version": manifest.get("adapter", "unknown"),
                "dynamics_version": model.dynamics.version,
                "training_protocol": training_protocol,
            }
            torch.save(payload, best_path)
            prototype = _prototypes(model, val_loader, device)
            save_json(prototype, output_dir / "latent_state_prototypes.json")
        write_history()
    report = {
        "best_checkpoint": str(best_path) if best_path.exists() else None,
        "checkpoint_sha256": file_sha256(best_path) if best_path.exists() else None,
        "selection_qualified": bool(best_path.exists()),
        "selection_metric": "val_loss",
        "best_validation_loss": best,
        "sequence_manifest": manifest,
        **reproducibility,
    }
    if initial_checkpoint is not None:
        report["initial_checkpoint"] = str(Path(initial_checkpoint).resolve())
        report["initial_roll_mode"] = initial_selection
    save_json(report, output_dir / "training_summary.json")
    return report


def load_semi_markov_checkpoint(path: str | Path, *, device: str | Any = "cpu") -> SemiMarkovRelationalWorldModel:
    torch, _, _ = _torch()
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("model_type") != SemiMarkovRelationalWorldModel.model_type:
        raise ValueError(f"Not a semi_markov_relational checkpoint: {path}")
    model = SemiMarkovRelationalWorldModel(_model_cfg(payload["model_config"]))
    state_dict = dict(payload["state_dict"])
    expected = model.state_dict()
    for key in list(state_dict):
        if key in expected and tuple(state_dict[key].shape) != tuple(expected[key].shape):
            if key.startswith("anchor_residual."):
                state_dict.pop(key)
            else:
                raise ValueError(f"checkpoint tensor shape differs for {key}")
    incompatible = model.load_state_dict(state_dict, strict=False)
    permitted = ("start_mode.", "anchor_residual.")
    if incompatible.unexpected_keys or any(not key.startswith(permitted) for key in incompatible.missing_keys):
        raise ValueError(f"checkpoint is incompatible with the current semi-Markov model: {incompatible}")
    # Environments created from a loaded checkpoint can include this immutable
    # identity in every replay trace without relying on a caller convention.
    model.checkpoint_hash = file_sha256(checkpoint_path)
    return model.to(device).eval()
