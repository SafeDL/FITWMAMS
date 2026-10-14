"""Train on natural Train and select weights with fixed natural Val supervision."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hierarchical_world_model.src.core.utils import (
    ensure_dir,
    save_json,
    select_device,
    set_seed,
)

from .calibration import fit_natural_response_calibrator
from .config import WorldModelConfig
from .data import (
    ResponseDataset,
    artifact_identity,
    prepare_experiment_data,
    response_loader,
    training_sequence_ids,
)
from .losses import training_losses
from .model import FactualDynamicsModel
from .planner import frozen_diffusion_plans

CHECKPOINT_SCHEMA = "hierarchical_world_model"


def _model_config(config: dict[str, Any]) -> WorldModelConfig:
    return WorldModelConfig(**config["model"])


def _apply_stage_trainable(model, *, stage, stage_config):
    prefixes = (
        "latent_transition.",
        "decoder.behavior_mode.",
        "decoder.scene_innovation.",
        "decoder.agent_innovation.",
    )
    if stage == "base" and stage_config["trainable"] == "backbone":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(not name.startswith(prefixes))
        return "backbone"
    if stage == "stochastic_heads":
        selected = tuple(stage_config["trainable"])
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(selected))
        return "prefixes:" + ",".join(selected)
    raise ValueError("expected base/backbone or stochastic_heads/prefixes")


def _to_device(batch, device):
    return {name: value.to(device) for name, value in batch.items()}


def _mean_terms(
    model,
    loader,
    device,
    *,
    optimizer,
    grad_clip,
    intervention_interval,
    intervention_sequences,
    noise_seed,
    deterministic_forward,
):
    training = optimizer is not None
    model.train(training)
    totals = {}
    count = 0
    generator = None
    if noise_seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(noise_seed))
    for batch_index, raw in enumerate(loader):
        batch = _to_device(raw, device)
        size = len(batch["current"])
        scene = torch.randn(
            size, model.cfg.scene_latent_dim, device=device, generator=generator
        )
        agent = torch.randn(
            size, 7, model.cfg.agent_latent_dim, device=device, generator=generator
        )
        regularize = training and batch_index % intervention_interval == 0
        with torch.set_grad_enabled(training):
            terms = training_losses(
                model,
                batch,
                scene,
                agent,
                include_intervention=regularize,
                include_closed_loop=regularize or not training,
                closed_loop_sequences=intervention_sequences if training else size,
                intervention_sequences=intervention_sequences,
                deterministic_forward=deterministic_forward,
                generator=generator,
            )
            if not torch.isfinite(terms["loss"]):
                raise FloatingPointError("non-finite HiQR loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                terms["loss"].backward()
                terms["gradient_norm"] = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip
                )
                optimizer.step()
        for name, value in terms.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * size
        count += size
    return {name: value / count for name, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: FactualDynamicsModel,
    *,
    epoch: int,
    validation_metric: float,
    experiment_scope: str,
    training_data=None,
):
    payload = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        **model.checkpoint_payload(),
        "epoch": int(epoch),
        "validation_metric": float(validation_metric),
        "experiment_scope": experiment_scope,
        "training_data": training_data,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path, *, device: torch.device):
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload["checkpoint_schema"] != CHECKPOINT_SCHEMA:
        raise ValueError(f"not a {CHECKPOINT_SCHEMA} checkpoint: {path}")
    model = FactualDynamicsModel(WorldModelConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload


def learning_rate_factor(epoch, *, epochs, warmup_epochs, minimum_ratio):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs - 1, 1)
    return minimum_ratio + (1 - minimum_ratio) * 0.5 * (
        1 + math.cos(math.pi * progress)
    )


def should_stop(epoch, last_improvement, *, minimum_epochs, patience):
    return epoch >= minimum_epochs and epoch - last_improvement >= patience


def plot_training_curves(output):
    from .core.utils import load_json
    from tools.plot_style import GENERATED_COLOR, get_pyplot, style_axes

    plt = get_pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    objectives = []
    for axis, stage, title in zip(
        axes, ("base", "stochastic_heads"), ("Base model", "Stochastic heads")
    ):
        history = load_json(output / f"training_{stage}_history.json")["epochs"]
        summary = load_json(output / f"training_{stage}_summary.json")
        epochs = np.asarray([row["epoch"] for row in history])
        objective = np.asarray([row["selection_metric"] for row in history])
        objectives.append(objective)
        axis.plot(epochs, objective, color=GENERATED_COLOR, linewidth=1.7)
        best = summary["best_epoch"]
        axis.scatter(
            [best],
            [objective[best - 1]],
            color=GENERATED_COLOR,
            edgecolor="white",
            zorder=3,
        )
        axis.axvline(best, color=GENERATED_COLOR, linestyle="--", linewidth=0.8)
        axis.axvspan(
            1, min(summary["minimum_epochs"], len(history)), color="#64748b", alpha=0.06
        )
        axis.set(xlabel="Epoch", ylabel="Fixed natural Val objective", title=title)
        axis.text(
            0.97,
            0.93,
            f"Selected epoch {best}\n{len(history)} epochs completed",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
        style_axes(axis)
    lower = min(values.min() for values in objectives)
    upper = max(values.max() for values in objectives)
    padding = (upper - lower) * 0.05
    for axis in axes:
        axis.set_ylim(lower - padding, upper + padding)
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    figure.suptitle("World model training | 72,771 Train / 13,133 Val scenes")
    figure.text(
        0.5,
        0.015,
        "Val includes every scene and one-second closed-loop supervision; Test is not used for selection.\nLight shading: minimum training period; markers: selected weights. No small-sample KS rejection.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.13, 1, 0.92))
    directory = ensure_dir(output / "figures")
    figure.savefig(directory / "training_curves.png", dpi=300)
    plt.close(figure)


def train_world_model(config, *, config_dir: Path, stage="base"):
    if stage not in ("base", "stochastic_heads"):
        raise ValueError("training stage must be base or stochastic_heads")
    training = config["training"]
    stage_config = config["stages"][stage]
    deterministic = stage_config["deterministic_forward"]
    output = ensure_dir(config["paths"]["output_dir"])
    checkpoints = ensure_dir(output / "checkpoints")
    device = select_device(training["device"])
    seed = int(training["seed"])
    set_seed(seed)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    experiment = prepare_experiment_data(config, config_dir)
    cfg = _model_config(config)
    identities = training_sequence_ids(experiment)
    training_data = {
        **identities,
        "source": config["paths"]["sequence_cache_dir"],
        "fit_split": "train",
        "selection_split": "val",
        "test_used_for_fit_or_selection": False,
        "history_selection": "last_observed_frame_index",
        "history_controls": "valid_observed_transitions",
        "diffusion": artifact_identity(config["paths"]["diffusion_checkpoint"]),
        "training_config": training,
    }
    if stage == "base":
        model = FactualDynamicsModel(cfg).to(device)
        training_data["initialization"] = "from_scratch"
    else:
        base = checkpoints / "base_best.pt"
        model, payload = load_checkpoint(base, device=device)
        if (
            model.cfg != cfg
            or any(
                payload["training_data"][key] != value
                for key, value in identities.items()
            )
            or payload["training_data"]["diffusion"] != training_data["diffusion"]
            or payload["training_data"].get("history_controls")
            != "valid_observed_transitions"
        ):
            raise ValueError("stochastic stage requires the current Train/Val base")
        training_data["initialization"] = artifact_identity(base)
    plans = {}
    for split, rows in (
        ("train", experiment.train_rows),
        ("val", experiment.validation_rows),
    ):
        plans[split] = frozen_diffusion_plans(
            experiment.bundle,
            rows,
            checkpoint=config["paths"]["diffusion_checkpoint"],
            output_dir=output.parent / "diffusion_plans" / split,
            device=device,
            batch_size=256,
            ddim_steps=20,
            experiment_scope=training["experiment_scope"],
        )
        print(f"{split}: {len(rows):,} current Diffusion references ready", flush=True)
    calibrator, calibration = fit_natural_response_calibrator(
        experiment.bundle.arrays,
        experiment.train_rows,
        minimum_events=training["response_calibration_minimum_events"],
        method=training["response_calibration_method"],
    )
    model.set_matched_response_bounds(
        torch.from_numpy(calibrator.global_bounds).to(device)
    )
    model.set_response_sensitivity_bounds(
        torch.from_numpy(calibrator.global_sensitivity_bounds).to(device)
    )
    save_json(calibration, output / "natural_response_calibration.json")
    loaders = {}
    datasets = {}
    for split, rows in (
        ("train", experiment.train_rows),
        ("val", experiment.validation_rows),
    ):
        is_training = split == "train"
        datasets[split] = ResponseDataset(
            experiment.bundle,
            rows,
            training=is_training,
            seed=seed,
            history_choices=cfg.history_choices,
            preview_frames=cfg.preview_frames,
            execute_frames=cfg.execute_frames,
            soft_plans=plans[split],
            response_calibrator=calibrator,
        )
        loaders[split] = response_loader(
            datasets[split],
            batch_size=training[
                "batch_size" if is_training else "validation_batch_size"
            ],
            shuffle=is_training,
            workers=training["num_workers"],
            seed=seed,
        )
    set_seed(seed)
    stage_mode = _apply_stage_trainable(model, stage=stage, stage_config=stage_config)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: learning_rate_factor(
            epoch,
            epochs=training["epochs"],
            warmup_epochs=training["warmup_epochs"],
            minimum_ratio=training["minimum_learning_rate"] / training["learning_rate"],
        ),
    )
    best = patience_best = float("inf")
    last_improvement = best_epoch = 0
    history = []
    candidate = checkpoints / f"{stage}_candidate.pt"
    best_path = checkpoints / f"{stage}_best.pt"
    for epoch in range(1, training["epochs"] + 1):
        datasets["train"].set_epoch(epoch)
        values = {}
        rate = optimizer.param_groups[0]["lr"]
        for split in ("train", "val"):
            values[split] = _mean_terms(
                model,
                loaders[split],
                device,
                optimizer=optimizer if split == "train" else None,
                grad_clip=training["grad_clip"],
                intervention_interval=training["intervention_interval"],
                intervention_sequences=training["intervention_sequences"],
                noise_seed=None if split == "train" else seed + 77777,
                deterministic_forward=deterministic,
            )
        score = values["val"]["loss"]
        row = {
            "epoch": epoch,
            "learning_rate": rate,
            "selection_metric": score,
            **{f"train_{k}": v for k, v in values["train"].items()},
            **{f"validation_{k}": v for k, v in values["val"].items()},
        }
        history.append(row)
        save_json(
            {"stage": stage, "epochs": history},
            output / f"training_{stage}_history.json",
        )
        if score < best:
            best, best_epoch = score, epoch
            save_checkpoint(
                candidate,
                model,
                epoch=epoch,
                validation_metric=score,
                experiment_scope=training["experiment_scope"],
                training_data=training_data,
            )
        if score < patience_best - training["min_delta"]:
            patience_best, last_improvement = score, epoch
        print(
            f"{stage} epoch {epoch}/{training['epochs']}: Train={values['train']['loss']:.6f}, Val={score:.6f}, lr={rate:.2g}, best={best_epoch}",
            flush=True,
        )
        scheduler.step()
        if should_stop(
            epoch,
            last_improvement,
            minimum_epochs=training["minimum_epochs"],
            patience=training["patience"],
        ):
            break
    candidate.replace(best_path)
    summary = {
        "status": "complete",
        "stage": stage,
        "experiment_scope": training["experiment_scope"],
        "best_epoch": best_epoch,
        "best_validation_metric": best,
        "epochs_completed": len(history),
        "early_stopped": len(history) < training["epochs"],
        "stop_reason": (
            "validation_patience"
            if len(history) < training["epochs"]
            else "maximum_epochs"
        ),
        "minimum_epochs": training["minimum_epochs"],
        "patience": training["patience"],
        "train_sequences": len(experiment.train_rows),
        "validation_sequences": len(experiment.validation_rows),
        "test_sequences_reserved": len(experiment.test_rows),
        "best_checkpoint": str(best_path.relative_to(output)),
        "data_protocol": {
            k: training_data[k]
            for k in (
                "source",
                "fit_split",
                "selection_split",
                "test_used_for_fit_or_selection",
                "history_selection",
                "diffusion",
                "initialization",
            )
        },
        "stage_trainable": stage_mode,
        "deterministic_forward": deterministic,
        "selection_metric": "full natural Val factual + two independent trajectory samples + one-second closed-loop supervision; no synthetic intervention or KS gate",
        "response_frequency_hz": 25.0,
    }
    save_json(summary, output / f"training_{stage}_summary.json")
    if stage == "stochastic_heads":
        plot_training_curves(output)
    return summary
