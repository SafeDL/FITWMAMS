"""Full-protocol training and checkpoint selection."""

from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch
from scipy.stats import ks_2samp

from world_model.src.core.utils import ensure_dir, save_json, select_device, set_seed

from .config import WorldModelConfig
from .calibration import fit_natural_response_calibrator
from .data import ResponseDataset, prepare_experiment_data, response_loader
from .losses import training_losses
from .model import DiffusionGuidedHiQR
from .planner import frozen_diffusion_plans
from .protocol import STAGED_TRAINING_GATES


CHECKPOINT_SCHEMA = "hierarchical_world_model"


def _stochastic_validation_gates(
    model: DiffusionGuidedHiQR, states: np.ndarray, valid: np.ndarray,
    plans: np.ndarray, maps: np.ndarray, map_valid: np.ndarray, *, device: torch.device, seed: int,
) -> dict[str, float | bool]:
    """Fixed validation probes used before energy-based stochastic selection."""
    from .evaluation import rollout

    count = min(len(states), int(STAGED_TRAINING_GATES["probe_count"]))
    args = (states[:count], valid[:count], plans[:count], maps[:count], map_valid[:count])
    diffusion_only = rollout(model, *args, device=device, history_frames=25, motion_seed=None)
    first = rollout(model, *args, device=device, history_frames=25, motion_seed=seed + 1)
    second = rollout(model, *args, device=device, history_frames=25, motion_seed=seed + 2)
    active = valid[:count, 24, 1:]
    mask = np.broadcast_to(active[:, None], first.states[:, :, 1:, 0].shape)
    distance = np.linalg.norm(first.states[:, :, 1:, :2] - second.states[:, :, 1:, :2], axis=-1)
    trajectory_diversity = float(distance[mask].mean())
    terminal_diversity = float(distance[:, -1][active].mean())
    target_speed = np.linalg.norm(states[:count, 25:174, 1:, 2:4], axis=-1)[mask]
    deterministic_speed = np.linalg.norm(diffusion_only.states[:, :, 1:, 2:4], axis=-1)[mask]
    stochastic_speed = np.linalg.norm(first.states[:, :, 1:, 2:4], axis=-1)[mask]
    deterministic_ks = float(ks_2samp(deterministic_speed, target_speed).statistic)
    stochastic_ks = float(ks_2samp(stochastic_speed, target_speed).statistic)
    target = states[:count, 25:174, 1:, :2]
    first_distance = np.linalg.norm(first.states[:, :, 1:, :2] - target, axis=-1)
    second_distance = np.linalg.norm(second.states[:, :, 1:, :2] - target, axis=-1)
    pair_distance = np.linalg.norm(
        first.states[:, :, 1:, :2] - second.states[:, :, 1:, :2], axis=-1
    )
    # This is the validation Energy Score used for stochastic model
    # selection, not a lexicographic proxy from separate fidelity/diversity
    # columns.
    energy = float(
        (0.5 * (first_distance + second_distance - pair_distance))[mask].mean()
    )
    return {
        "trajectory_diversity_m": trajectory_diversity,
        "terminal_diversity_m": terminal_diversity,
        "diffusion_only_speed_KS": deterministic_ks,
        "stochastic_speed_KS": stochastic_ks,
        "validation_energy_score_m": energy,
        "diversity_gate": (
            trajectory_diversity >= STAGED_TRAINING_GATES["trajectory_diversity_min_m"]
            and terminal_diversity >= STAGED_TRAINING_GATES["terminal_diversity_min_m"]
        ),
        "distribution_gate": stochastic_ks <= deterministic_ks * STAGED_TRAINING_GATES["relative_ks_limit_ratio"],
    }


def _model_config(config: dict[str, Any]) -> WorldModelConfig:
    values = config.get("model", {})
    return WorldModelConfig(
        **{
            name: value
            for name, value in values.items()
            if name in WorldModelConfig.__dataclass_fields__
        }
    )


def _apply_stage_trainable(
    model: DiffusionGuidedHiQR,
    *,
    stage: str,
    stage_config: dict[str, Any],
    training_config: dict[str, Any],
) -> str:
    """Configure requires-grad under the stage contract.

    The plan is: keep one architecture contract, but each stage selects which
    parameter groups are trainable and therefore updated.
    """
    spec = stage_config.get("trainable", None)
    if spec is None:
        raise ValueError(f"stage {stage!r} must declare an explicit trainable set")
    if stage == "stochastic_heads":
        if not (
            model.cfg.graph_coupled_latent_enabled and model.cfg.behavior_mode_decoder_enabled
        ):
            raise ValueError(
                "stochastic_heads requires graph latent and behavior-mode decoder"
            )

    if isinstance(spec, str):
        if spec == "backbone":
            freeze = {
                "latent_transition.",
                "decoder.behavior_mode.",
                "decoder.scene_innovation.",
                "decoder.agent_innovation.",
            }
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(not any(name.startswith(prefix) for prefix in freeze))
            return spec
        if spec == "all":
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            return spec
        if spec == "intervention_adapter_only":
            if not model.cfg.intervention_adapter_enabled:
                raise ValueError(
                    "intervention_adapter_only requires intervention_adapter_enabled"
                )
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model.decoder.intervention_logit.requires_grad_(True)
            return spec
        if spec == "causal_response_field_only":
            if not (model.cfg.causal_response_field_enabled and model.cfg.intervention_adapter_enabled):
                raise ValueError(
                    "causal_response_field_only requires response field and intervention adapter"
                )
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            for name, parameter in model.named_parameters():
                if name.startswith("decoder.response_field."):
                    parameter.requires_grad_(True)
                if (
                    name == "decoder.intervention_logit"
                    and model.cfg.causal_response_field_enabled
                ):
                    parameter.requires_grad_(True)
            return spec
        raise ValueError(f"unsupported trainable spec {spec!r}")

    if not isinstance(spec, (list, tuple)) or not spec:
        raise ValueError(f"unsupported trainable spec for stage {stage!r}")
    prefixes = tuple(str(item) for item in spec)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            parameter.requires_grad_(True)
    return f"prefixes:{','.join(prefixes)}"


def _check_trainable(model: DiffusionGuidedHiQR) -> list[torch.nn.Parameter]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable world-model parameters")
    return trainable


def _to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def _mean_terms(
    model: DiffusionGuidedHiQR,
    loader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
    intervention_interval: int,
    intervention_sequences: int,
    noise_seed: int | None,
    deterministic_forward: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    count = 0
    noise_generator = None
    if noise_seed is not None:
        noise_generator = torch.Generator(device=device).manual_seed(int(noise_seed))
    for batch_index, raw in enumerate(loader):
        batch = _to_device(raw, device)
        scene_noise = torch.randn(
            (len(batch["current"]), model.cfg.scene_latent_dim),
            device=device,
            generator=noise_generator,
        )
        agent_noise = torch.randn(
            (len(batch["current"]), 7, model.cfg.agent_latent_dim),
            device=device,
            generator=noise_generator,
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
            terms = training_losses(
                model,
                batch,
                scene_noise,
                agent_noise,
                include_intervention=batch_index % intervention_interval == 0,
                intervention_sequences=intervention_sequences,
                deterministic_forward=deterministic_forward,
            )
            if not torch.isfinite(terms["loss"]):
                raise FloatingPointError("non-finite diffusion-guided HiQR loss")
            terms["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
        else:
            with torch.no_grad():
                terms = training_losses(
                    model,
                    batch,
                    scene_noise,
                    agent_noise,
                    include_intervention=batch_index % intervention_interval == 0,
                    intervention_sequences=intervention_sequences,
                    deterministic_forward=deterministic_forward,
                )
                if not torch.isfinite(terms["loss"]):
                    raise FloatingPointError(
                        "non-finite diffusion-guided HiQR validation loss"
                    )
        size = len(batch["current"])
        for name, value in terms.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * size
        count += size
    return {name: value / max(count, 1) for name, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: DiffusionGuidedHiQR,
    *,
    epoch: int,
    validation_metric: float,
    experiment_scope: str,
) -> None:
    payload = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        **model.checkpoint_payload(),
        "epoch": int(epoch),
        "validation_metric": float(validation_metric),
        "experiment_scope": str(experiment_scope),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[DiffusionGuidedHiQR, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"not a {CHECKPOINT_SCHEMA} checkpoint: {path}")
    model = DiffusionGuidedHiQR(WorldModelConfig(**payload["model_config"])).to(
        device
    )
    _load_compatible_state_dict(model, payload["state_dict"])
    return model, payload


def _load_compatible_state_dict(
    model: DiffusionGuidedHiQR, state_dict: dict[str, Any]
) -> None:
    """Load E1 weights while allowing opt-in Stochastic Causal HiQR heads.

    The new heads are strictly additive and inactive under their default
    flags, so an older checkpoint remains the exact factual-control baseline
    for a pilot.  Any missing core parameter is still a hard error.
    """
    permitted = {
        "decoder.intervention_logit",
        "response_sensitivity_bounds",
    }
    optional_prefixes = (
        "decoder.response_field.",
        "decoder.behavior_mode.",
        "latent_transition.",
    )
    target = model.state_dict()
    mismatched = {
        name
        for name, value in state_dict.items()
        if name in target
        and hasattr(value, "shape")
        and tuple(value.shape) != tuple(target[name].shape)
    }
    unknown_mismatch = {
        name
        for name in mismatched
        if name not in permitted and not name.startswith(optional_prefixes)
    }
    if unknown_mismatch:
        raise ValueError(
            "checkpoint is incompatible with the maintained response contract: "
            f"shape_mismatch={sorted(unknown_mismatch)}"
        )
    compatible_state = {
        name: value for name, value in state_dict.items() if name not in mismatched
    }
    incompatible = model.load_state_dict(compatible_state, strict=False)
    missing = set(incompatible.missing_keys)
    unknown_missing = {
        name
        for name in missing
        if name not in permitted and not name.startswith(optional_prefixes)
    }
    if unknown_missing or incompatible.unexpected_keys:
        raise ValueError(
            "checkpoint is incompatible with the maintained response contract: "
            f"missing={sorted(unknown_missing)}, unexpected={incompatible.unexpected_keys}"
        )


def train_world_model(
    config: dict[str, Any], *, config_dir: Path, stage: str = "base"
) -> dict[str, Any]:
    from .evaluation import _factual_metrics, rollout

    training = config["training"]
    scope = str(training.get("experiment_scope", "full"))
    if scope not in {"full", "pilot"}:
        raise ValueError("experiment_scope must be 'full' or 'pilot'")
    stages = config.get("stages", {})
    if stage not in {"base", "stochastic_heads"}:
        raise ValueError("training stage must be 'base' or 'stochastic_heads'")
    stage_config = dict(stages.get(stage, {}))
    deterministic_forward = bool(stage_config.get("deterministic_forward", stage == "base"))
    output = ensure_dir(config["paths"]["output_dir"])
    checkpoints = ensure_dir(output / "checkpoints")
    device = select_device(training.get("device", "auto"))
    seed = int(training["seed"])
    set_seed(seed)
    experiment = prepare_experiment_data(config, config_dir)
    cfg = _model_config(config)
    # Frozen Diffusion plans are an in-process training input, not a release
    # artifact.  Keeping them under ``results/`` used to leave hundreds of MB
    # of reproducible preview caches after each staged training run.
    with TemporaryDirectory(prefix="hiqr_diffusion_plans_") as temporary_cache:
        preview_root = Path(temporary_cache)
        train_plans = frozen_diffusion_plans(
            experiment.bundle,
            experiment.train_rows,
            checkpoint=config["paths"]["diffusion_checkpoint"],
            output_dir=preview_root / "train",
            device=device,
            batch_size=256,
            ddim_steps=20,
            experiment_scope=scope,
        )
        validation_plans = frozen_diffusion_plans(
            experiment.bundle,
            experiment.validation_rows,
            checkpoint=config["paths"]["diffusion_checkpoint"],
            output_dir=preview_root / "validation",
            device=device,
            batch_size=256,
            ddim_steps=20,
            experiment_scope=scope,
        )
    response_calibrator, response_report = fit_natural_response_calibrator(
        experiment.bundle.arrays,
        experiment.train_rows,
        minimum_events=int(
            training.get(
                "response_calibration_minimum_events",
                30 if scope == "pilot" else 100,
            )
        ),
        method=str(training.get("response_calibration_method", "exact")),
    )
    train_data = ResponseDataset(
        experiment.bundle,
        experiment.train_rows,
        training=True,
        seed=seed,
        history_choices=cfg.history_choices,
        preview_frames=cfg.preview_frames,
        execute_frames=cfg.execute_frames,
        soft_plans=train_plans,
        response_calibrator=response_calibrator,
    )
    validation_data = ResponseDataset(
        experiment.bundle,
        experiment.validation_rows,
        training=False,
        seed=seed,
        history_choices=cfg.history_choices,
        preview_frames=cfg.preview_frames,
        execute_frames=cfg.execute_frames,
        soft_plans=validation_plans,
        response_calibrator=response_calibrator,
    )
    train_loader = response_loader(
        train_data,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        workers=int(training.get("num_workers", 0)),
        seed=seed,
    )
    validation_loader = response_loader(
        validation_data,
        batch_size=int(training["validation_batch_size"]),
        shuffle=False,
        workers=int(training.get("num_workers", 0)),
        seed=seed,
    )
    # Preview generation may or may not consume RNG depending on cache state.
    # Reset here so model initialization is invariant to that implementation detail.
    set_seed(seed)
    base_checkpoint = checkpoints / "base_best.pt" if stage == "stochastic_heads" else None
    factual_reference = None
    if base_checkpoint is not None:
        if not base_checkpoint.is_file():
            raise FileNotFoundError(f"stochastic_heads requires base checkpoint: {base_checkpoint}")
        loaded, _ = load_checkpoint(base_checkpoint, device=device)
        # Keep an immutable E1 copy for model selection.  The candidate can
        # change its response configuration, but it may not trade factual
        # replay away for an intervention loss on this identical validation
        # cohort.
        factual_reference, _ = load_checkpoint(base_checkpoint, device=device)
        factual_reference.eval()
        if loaded.cfg != cfg:
            raise ValueError("all stages must use the identical model architecture config")
        model = loaded
    else:
        model = DiffusionGuidedHiQR(cfg).to(device)
    model.set_matched_response_bounds(
        torch.from_numpy(response_calibrator.global_bounds).to(device)
    )
    model.set_response_sensitivity_bounds(
        torch.from_numpy(response_calibrator.global_sensitivity_bounds).to(device)
    )
    save_json(response_report, output / "natural_response_calibration.json")
    stage_mode = _apply_stage_trainable(
        model,
        stage=stage,
        stage_config=stage_config,
        training_config=training,
    )
    closed_rows = experiment.validation_rows[:128]
    closed_states = np.asarray(
        experiment.bundle.arrays["agent_states"][closed_rows], np.float32
    )
    closed_valid = np.asarray(
        experiment.bundle.arrays["agent_valid"][closed_rows], bool
    )
    closed_active = closed_valid[:, 24, 1:]
    closed_target = closed_states[:, 25:174]
    closed_maps = np.asarray(
        experiment.bundle.arrays["map_polylines"][closed_rows], np.float32
    )
    closed_map_valid = np.asarray(
        experiment.bundle.arrays["map_polyline_valid"][closed_rows], bool
    )
    closed_plans = validation_plans[: len(closed_rows)]
    baseline_distance = np.linalg.norm(
        closed_plans - closed_target[..., 1:, :2], axis=-1
    )
    baseline_mask = np.broadcast_to(closed_active[:, None], baseline_distance.shape)
    baseline_ade = float(baseline_distance[baseline_mask].mean())
    baseline_fde = float(baseline_distance[:, -1][closed_active].mean())
    baseline_p95 = float(np.quantile(baseline_distance[baseline_mask], 0.95))
    reference_metrics: dict[str, float] | None = None
    if factual_reference is not None:
        factual_reference.set_matched_response_bounds(
            torch.from_numpy(response_calibrator.global_bounds).to(device)
        )
        factual_reference.set_response_sensitivity_bounds(
            torch.from_numpy(response_calibrator.global_sensitivity_bounds).to(device)
        )
        reference_batches = [
            rollout(
                factual_reference,
                closed_states[start : start + 64],
                closed_valid[start : start + 64],
                closed_plans[start : start + 64],
                closed_maps[start : start + 64],
                closed_map_valid[start : start + 64],
                device=device,
                history_frames=25,
                motion_seed=None,
            ).states
            for start in range(0, len(closed_rows), 64)
        ]
        reference_metrics = _factual_metrics(
            np.concatenate(reference_batches), closed_target, closed_active
        )
    factual_relative_tolerance = training.get("factual_relative_tolerance") if stage == "stochastic_heads" else None
    if factual_relative_tolerance is not None and factual_reference is None:
        raise ValueError("factual_relative_tolerance is only valid for stochastic_heads")
    trainable_parameters = _check_trainable(model)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(training["epochs"])
    patience = int(training["patience"])
    best = float("inf")
    best_qualified = False
    stale = 0
    unqualified_epochs = 0
    history: list[dict[str, float | int]] = []
    # Never overwrite the maintained checkpoint until a candidate has passed the
    # factual gate.  This keeps a failed 25 Hz run from being mislabeled as the
    # current model merely because an older checkpoint exists at that path.
    best_path = checkpoints / ("base_best.pt" if stage == "base" else "stochastic_heads_best.pt")
    candidate_path = checkpoints / f"{stage}_candidate.pt"
    if candidate_path.exists():
        candidate_path.unlink()
    for epoch in range(1, epochs + 1):
        train_data.set_epoch(epoch)
        train_terms = _mean_terms(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            grad_clip=float(training["grad_clip"]),
            intervention_interval=int(training.get("intervention_interval", 1)),
            intervention_sequences=int(training.get("intervention_sequences", 8)),
            noise_seed=None,
            deterministic_forward=deterministic_forward,
        )
        validation_terms = _mean_terms(
            model,
            validation_loader,
            device,
            optimizer=None,
            grad_clip=0.0,
            intervention_interval=int(training.get("intervention_interval", 1)),
            intervention_sequences=int(training.get("intervention_sequences", 8)),
            noise_seed=seed + 77_777,
            deterministic_forward=deterministic_forward,
        )
        closed_batches = [
            rollout(
                model,
                closed_states[start : start + 64],
                closed_valid[start : start + 64],
                closed_plans[start : start + 64],
                closed_maps[start : start + 64],
                closed_map_valid[start : start + 64],
                device=device,
                history_frames=25,
                motion_seed=None,
            ).states
            for start in range(0, len(closed_rows), 64)
        ]
        closed_metrics = _factual_metrics(
            np.concatenate(closed_batches), closed_target, closed_active
        )
        if reference_metrics is not None and factual_relative_tolerance is not None:
            relative = float(factual_relative_tolerance)
            qualified = (
                closed_metrics["ADE_m"]
                <= reference_metrics["ADE_m"] * (1.0 + relative)
                and closed_metrics["FDE_m"]
                <= reference_metrics["FDE_m"] * (1.0 + relative)
                and closed_metrics["P95_displacement_error_m"]
                <= reference_metrics["P95_displacement_error_m"] * (1.0 + relative)
            )
        else:
            qualified = (
                closed_metrics["ADE_m"]
                <= baseline_ade + float(training["factual_ade_tolerance_m"])
                and closed_metrics["FDE_m"]
                <= baseline_fde + float(training["factual_fde_tolerance_m"])
                and closed_metrics["P95_displacement_error_m"]
                <= baseline_p95 + float(training["factual_p95_tolerance_m"])
            )
        stochastic_probe: dict[str, float | bool] | None = None
        if stage == "stochastic_heads":
            stochastic_probe = _stochastic_validation_gates(
                model, closed_states, closed_valid, closed_plans,
                closed_maps, closed_map_valid, device=device, seed=seed,
            )
            qualified = bool(
                qualified
                and stochastic_probe["diversity_gate"]
                and stochastic_probe["distribution_gate"]
            )
        factual_score = closed_metrics["ADE_m"] + STAGED_TRAINING_GATES["base_factual_fde_fallback_weight"] * closed_metrics["FDE_m"]
        # Jerk is already part of factual.  Base selection intentionally does
        # not add it a second time.
        selection = factual_score
        if qualified:
            selection = (
                validation_terms["factual"]
                + cfg.closed_loop_factual_weight * validation_terms["closed_loop_factual"]
                if stage == "base"
                else float(stochastic_probe["validation_energy_score_m"])
            )
        row: dict[str, float | int] = {
            "epoch": epoch,
            "selection_metric": selection,
            "validation_closed_loop_ADE_m": closed_metrics["ADE_m"],
            "validation_closed_loop_FDE_m": closed_metrics["FDE_m"],
            "validation_factual_gate_passed": int(qualified),
            **({f"stochastic_{name}": value for name, value in stochastic_probe.items()} if stochastic_probe else {}),
            **{f"train_{name}": value for name, value in train_terms.items()},
            **{f"validation_{name}": value for name, value in validation_terms.items()},
        }
        history.append(row)
        save_json({"stage": stage, "epochs": history}, output / f"training_{stage}_history.json")
        unqualified_epochs = unqualified_epochs + 1 if not qualified else 0
        improved = math.isfinite(selection) and (
            (qualified and not best_qualified)
            or (qualified == best_qualified and selection < best)
        )
        if improved:
            best = selection
            best_qualified = qualified
            stale = 0
            save_checkpoint(
                candidate_path,
                model,
                epoch=epoch,
                validation_metric=selection,
                experiment_scope=scope,
            )
        else:
            stale += 1
        # A pilot exists to rule out factually regressive mechanisms quickly.
        # Do not let tiny movements in an unqualified factual score reset the
        # early-stop clock and consume the complete experimental budget.
        if stale >= patience or unqualified_epochs >= patience:
            break
    if not best_qualified or not candidate_path.is_file():
        if candidate_path.exists():
            candidate_path.unlink()
        summary = {
            "status": "rejected_factual_gate",
            "experiment_scope": scope,
            "epochs_completed": len(history),
            "early_stopped": len(history) < epochs,
            "factual_gate": "no checkpoint was non-inferior to the reference",
            "last_validation": history[-1] if history else None,
            "factual_reference": (
                None
                if reference_metrics is None
                else {
                    "checkpoint": str(base_checkpoint),
                    "metrics": reference_metrics,
                    "relative_tolerance": factual_relative_tolerance,
                }
            ),
        }
        summary["stage"] = stage
        save_json(summary, output / f"training_{stage}_summary.json")
        return summary
    candidate_path.replace(best_path)
    summary = {
        "status": "complete",
        "experiment_scope": scope,
        "best_validation_metric": best,
        "best_checkpoint_factual_gate_passed": best_qualified,
        "epochs_completed": len(history),
        "early_stopped": len(history) < epochs,
        "train_sequences": len(experiment.train_rows),
        "validation_sequences": len(experiment.validation_rows),
        "test_sequences_reserved": len(experiment.test_rows),
        "history_random_truncation": list(cfg.history_choices),
        "response_frequency_hz": 1.0 / cfg.dt_s,
        "response_commit_frames": cfg.execute_frames,
        "scene_latent_refresh_s": cfg.scene_refresh_responses * cfg.dt_s,
        "older_history_mask_probability": 0.5,
        "frozen_diffusion_preview_training": True,
        "stage_trainable": stage_mode,
        "stage": stage,
        "deterministic_forward": deterministic_forward,
        "diffusion_baseline_ADE_m": baseline_ade,
        "diffusion_baseline_FDE_m": baseline_fde,
        "diffusion_baseline_P95_m": baseline_p95,
        "factual_reference": (
            None
            if reference_metrics is None
            else {
                "checkpoint": str(base_checkpoint),
                "metrics": reference_metrics,
                "relative_tolerance": float(factual_relative_tolerance),
            }
        ),
        "selection_metric": (
            "base: factual plus closed-loop factual (jerk is inside factual); "
            "stochastic_heads: minimum validation energy among gated candidates"
        ),
        "factual_error_budget_m": {
            "ADE": float(training["factual_ade_tolerance_m"]),
            "FDE": float(training["factual_fde_tolerance_m"]),
            "P95": float(training["factual_p95_tolerance_m"]),
        },
        "selection_factual_weight": float(training["selection_factual_weight"]),
        "best_checkpoint": str(best_path.relative_to(output)),
    }
    save_json(summary, output / f"training_{stage}_summary.json")
    return summary
