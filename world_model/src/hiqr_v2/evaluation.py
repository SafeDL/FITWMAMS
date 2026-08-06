"""Fixed-cohort deterministic and stochastic evaluation for HiQR-v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from world_model.src.core.initial_behavior_anchor import FrozenLegacyFlowSchema
from world_model.src.core.long_tail_metrics import (
    distribution_values,
    empirical_distance,
    traffic_fields,
)
from world_model.src.core.sequential_dataset import sequence_cache_owner_dir
from world_model.src.core.utils import ensure_dir, save_json, select_device, set_seed
from world_model.src.hiqr.train import load_hiqr_checkpoint

from .data import load_hiqr_v2_arrays, make_hiqr_v2_loader, to_hiqr_v2_batch
from .model import HiQRV2WorldModel
from .train import load_hiqr_v2_checkpoint, require_canonical_hiqr_v2_checkpoint

RISK_DISTRIBUTION_NAMES = (
    "gap_m",
    "ttc_s",
    "drac_mps2",
    "relative_speed_mps",
)


def _sequence_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    frame: int | None = None,
) -> np.ndarray:
    distance = torch.linalg.vector_norm(predicted[..., :2] - target[..., :2], dim=-1)
    if frame is None:
        weights = valid.float()
        return (
            (distance * weights)
            .sum(dim=(1, 2))
            .div(weights.sum(dim=(1, 2)).clamp_min(1.0))
            .detach()
            .cpu()
            .numpy()
        )
    weights = valid[:, frame].float()
    return (
        (distance[:, frame] * weights)
        .sum(dim=1)
        .div(weights.sum(dim=1).clamp_min(1.0))
        .detach()
        .cpu()
        .numpy()
    )


def _horizons(frames: int) -> dict[str, int]:
    targets = (
        ("1s", 25),
        ("2s", 50),
        ("3s", 75),
        ("4s", 100),
        ("5s", 125),
        ("5p96s", 149),
    )
    return {
        name: min(frames, requested) - 1
        for name, requested in targets
        if frames >= min(requested, frames)
    }


def _summary_from_rollout(
    rollout: dict[str, Any],
) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, Any]]:
    predicted_all = rollout["predicted_states"]
    target_all = rollout["target_states"]
    predicted = predicted_all[:, :, 1:]
    target = target_all[:, :, 1:]
    valid = rollout["target_valid"][:, :, 1:]
    metrics: dict[str, float] = {
        "ade_m": float(np.mean(_sequence_error(predicted, target, valid)))
    }
    per_sequence: dict[str, np.ndarray] = {
        "ade": _sequence_error(predicted, target, valid)
    }
    for name, frame in _horizons(predicted.shape[1]).items():
        values = _sequence_error(predicted, target, valid, frame)
        metrics[f"fde_{name}_m"] = float(values.mean())
        per_sequence[f"fde_{name}"] = values
        prefix = torch.linalg.vector_norm(
            predicted[:, : frame + 1, ..., :2] - target[:, : frame + 1, ..., :2], dim=-1
        )
        weights = valid[:, : frame + 1].float()
        metrics[f"ade_{name}_m"] = float(
            (
                (prefix * weights).sum(dim=(1, 2))
                / weights.sum(dim=(1, 2)).clamp_min(1.0)
            )
            .mean()
            .cpu()
        )
    predicted_ego = predicted_all[:, :, 0].detach().cpu().numpy()
    target_ego = target_all[:, :, 0].detach().cpu().numpy()
    predicted_numpy = predicted.detach().cpu().numpy()
    target_numpy = target.detach().cpu().numpy()
    valid_numpy = valid.detach().cpu().numpy()
    predicted_fields = traffic_fields(predicted_numpy, predicted_ego, valid_numpy)
    target_fields = traffic_fields(target_numpy, target_ego, valid_numpy)
    following = np.asarray(target_fields["following_valid"], bool)
    predicted_collision = np.asarray(predicted_fields["collision"], bool)
    target_collision = np.asarray(target_fields["collision"], bool)
    predicted_collision_episode = predicted_collision.any(axis=(1, 2, 3))
    target_collision_episode = target_collision.any(axis=(1, 2, 3))
    present = np.asarray(predicted_fields["present"], bool)
    upper = np.triu(np.ones((present.shape[-1], present.shape[-1]), bool), k=1)
    pair_valid = present[:, :, :, None] & present[:, :, None, :] & upper[None, None]
    target_present = np.asarray(target_fields["present"], bool)
    target_pair_valid = (
        target_present[:, :, :, None]
        & target_present[:, :, None, :]
        & upper[None, None]
    )
    predicted_distribution = distribution_values(predicted_fields)
    target_distribution = distribution_values(target_fields)
    finite_episode = np.isfinite(predicted_all.detach().cpu().numpy()).all(
        axis=(1, 2, 3)
    )
    risk = {
        "following_pair_frames": int(following.sum()),
        "gap_absolute_error": np.abs(
            np.asarray(predicted_fields["gap_m"]) - np.asarray(target_fields["gap_m"])
        )[following],
        "ttc_absolute_error": np.abs(
            np.asarray(predicted_fields["ttc_s"]) - np.asarray(target_fields["ttc_s"])
        )[following],
        "drac_absolute_error": np.abs(
            np.asarray(predicted_fields["drac_mps2"])
            - np.asarray(target_fields["drac_mps2"])
        )[following],
        "relative_speed_absolute_error": np.abs(
            np.asarray(predicted_fields["relative_speed_mps"])
            - np.asarray(target_fields["relative_speed_mps"])
        )[following],
        "predicted_collision_pair_points": int(predicted_collision.sum()),
        "predicted_collision_pair_valid_points": int(pair_valid.sum()),
        "target_collision_pair_points": int(target_collision.sum()),
        "target_collision_pair_valid_points": int(target_pair_valid.sum()),
        "collision_episode": predicted_collision_episode,
        "target_collision_episode": target_collision_episode,
        "stable_episode": finite_episode & ~predicted_collision_episode,
    }
    for name in RISK_DISTRIBUTION_NAMES:
        risk[f"predicted_{name}"] = predicted_distribution[name]
        risk[f"target_{name}"] = target_distribution[name]
    terms = rollout.get("terms", {})
    for name in (
        "gate",
        "revision_rate",
        "emergency_rate",
        "prior_scene_std",
        "prior_agent_std",
        "posterior_scene_std",
        "posterior_agent_std",
        "position",
        "posterior_position",
    ):
        if name in terms:
            metrics[name] = float(terms[name].detach().cpu())
    if "posterior_position" in metrics and "position" in metrics:
        metrics["prior_posterior_position_gap_m"] = (
            metrics["position"] - metrics["posterior_position"]
        )
    masks = rollout.get("background_future_action_masks")
    if masks is not None:
        carried = masks["carried"].bool()
        revised = masks["revised"].bool()
        metrics["gate_mean"] = float(
            (rollout["continuation_gate"].squeeze(-1) * carried.float())
            .sum()
            .div(carried.float().sum().clamp_min(1.0))
            .cpu()
        )
        metrics["revision_rate"] = float(
            (revised & carried)
            .float()
            .sum()
            .div(carried.float().sum().clamp_min(1.0))
            .cpu()
        )
    return metrics, per_sequence, risk


def _distribution_summary(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0, "p50": float("nan"), "p95": float("nan")}
    return {
        "count": int(len(finite)),
        "p50": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
    }


def _aggregate_risk(risk_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def values(name: str) -> np.ndarray:
        rows = [np.asarray(row[name]).reshape(-1) for row in risk_rows]
        return np.concatenate(rows) if rows else np.empty(0)

    def safe_mean(items: np.ndarray) -> float:
        return float(items.mean()) if len(items) else float("nan")

    collision = values("collision_episode").astype(bool)
    target_collision = values("target_collision_episode").astype(bool)
    stable = values("stable_episode").astype(bool)
    predicted_pair_points = sum(
        row["predicted_collision_pair_points"] for row in risk_rows
    )
    predicted_pair_valid = sum(
        row["predicted_collision_pair_valid_points"] for row in risk_rows
    )
    target_pair_points = sum(row["target_collision_pair_points"] for row in risk_rows)
    target_pair_valid = sum(
        row["target_collision_pair_valid_points"] for row in risk_rows
    )
    summary: dict[str, Any] = {
        "following_pair_frames": int(
            sum(row["following_pair_frames"] for row in risk_rows)
        ),
        "gap_mae_m": safe_mean(values("gap_absolute_error")),
        "ttc_mae_s": safe_mean(values("ttc_absolute_error")),
        "drac_mae_mps2": safe_mean(values("drac_absolute_error")),
        "relative_speed_mae_mps": safe_mean(values("relative_speed_absolute_error")),
        "collision_pair_point_rate": float(
            predicted_pair_points / max(predicted_pair_valid, 1)
        ),
        "target_collision_pair_point_rate": float(
            target_pair_points / max(target_pair_valid, 1)
        ),
        "collision_episode_rate": safe_mean(collision),
        "target_collision_episode_rate": safe_mean(target_collision),
        "closed_loop_stability_rate": safe_mean(stable),
        "risk_variable_distribution": {},
        "risk_variable_summary": {},
    }
    for name in RISK_DISTRIBUTION_NAMES:
        predicted_values = values(f"predicted_{name}")
        target_values = values(f"target_{name}")
        summary["risk_variable_distribution"][name] = empirical_distance(
            target_values, predicted_values
        )
        summary["risk_variable_summary"][name] = {
            "target": _distribution_summary(target_values),
            "predicted": _distribution_summary(predicted_values),
        }
    return summary


def _bootstrap_difference(
    candidate: np.ndarray, baseline: np.ndarray, *, samples: int, seed: int
) -> dict[str, float]:
    if candidate.shape != baseline.shape or candidate.ndim != 1:
        raise ValueError("bootstrap inputs must be aligned per-sequence arrays")
    delta = candidate - baseline
    generator = np.random.default_rng(seed)
    means = np.empty(int(samples), dtype=np.float64)
    for row in range(len(means)):
        means[row] = delta[generator.integers(0, len(delta), size=len(delta))].mean()
    return {
        "mean_delta_m": float(delta.mean()),
        "ci95_low_m": float(np.quantile(means, 0.025)),
        "ci95_high_m": float(np.quantile(means, 0.975)),
        "candidate_improvement_fraction": float(
            (baseline.mean() - candidate.mean()) / max(baseline.mean(), 1e-12)
        ),
    }


@torch.no_grad()
def _evaluate_model(
    model, loader, device: torch.device, *, stochastic_samples: int
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    deterministic_rows: list[dict[str, float]] = []
    per_sequence: dict[str, list[np.ndarray]] = {}
    risk_rows: list[dict[str, Any]] = []
    stochastic_rows: list[dict[str, np.ndarray]] = []
    batch_sizes: list[int] = []
    model.eval()
    for values in loader:
        batch = to_hiqr_v2_batch(values, loader.field_names, device)
        if hasattr(model, "diagnostic_rollout"):
            deterministic = model.diagnostic_rollout(batch)
        else:
            deterministic = model.rollout_reconstruction(batch, deterministic=True)
            # The immutable V1 model has no isolated diagnostic branch.  Its
            # supervised path is evaluated separately and is never used for
            # the deterministic prior trajectory or risk metrics.
            posterior_terms = model.supervised_terms(batch)
            deterministic["terms"] = dict(deterministic.get("terms", {}))
            deterministic["terms"]["posterior_position"] = posterior_terms["position"]
        summary, sequence, risk = _summary_from_rollout(deterministic)
        deterministic_rows.append(summary)
        risk_rows.append(risk)
        batch_sizes.append(int(deterministic["predicted_states"].shape[0]))
        for name, value in sequence.items():
            per_sequence.setdefault(name, []).append(value)
        samples = [
            model.rollout_reconstruction(batch, deterministic=False)[
                "predicted_states"
            ][:, :, 1:]
            for _ in range(int(stochastic_samples))
        ]
        stack = torch.stack(samples, dim=1)
        target, valid = (
            deterministic["target_states"][:, :, 1:],
            deterministic["target_valid"][:, :, 1:],
        )
        distance = torch.linalg.vector_norm(
            stack[..., :2] - target[:, None, ..., :2], dim=-1
        )
        weights = valid[:, None].float()
        ade_samples = (distance * weights).sum(dim=(2, 3)) / weights.sum(
            dim=(2, 3)
        ).clamp_min(1.0)
        last = distance[:, :, -1]
        last_valid = valid[:, None, -1].float()
        fde_samples = (last * last_valid).sum(dim=2) / last_valid.sum(dim=2).clamp_min(
            1.0
        )
        pair_distance = torch.linalg.vector_norm(
            stack[:, :, None, ..., :2] - stack[:, None, :, ..., :2], dim=-1
        )
        valid_count = valid.float().sum(dim=(1, 2)).clamp_min(1.0)
        energy = (distance * weights).sum(dim=(1, 2, 3)) / (
            valid_count * stochastic_samples
        )
        energy = energy - 0.5 * (pair_distance * valid[:, None, None].float()).sum(
            dim=(1, 2, 3, 4)
        ) / (valid_count * stochastic_samples * stochastic_samples)
        pair_component = (
            (stack[..., :2][:, :, None] - stack[..., :2][:, None, :]).abs().mean(dim=-1)
        )
        component_error = (stack[..., :2] - target[:, None, ..., :2]).abs().mean(dim=-1)
        crps = (component_error * weights).sum(dim=(1, 2, 3)) / (
            valid_count * stochastic_samples
        ) - 0.5 * (pair_component * valid[:, None, None].float()).sum(
            dim=(1, 2, 3, 4)
        ) / (
            valid_count * stochastic_samples * stochastic_samples
        )
        stochastic_rows.append(
            {
                "sample_mean_ade_m": ade_samples.mean(dim=1).cpu().numpy(),
                "sample_min_ade_m": ade_samples.min(dim=1).values.cpu().numpy(),
                "sample_mean_fde_m": fde_samples.mean(dim=1).cpu().numpy(),
                "sample_min_fde_m": fde_samples.min(dim=1).values.cpu().numpy(),
                "energy_score": energy.cpu().numpy(),
                "crps": crps.cpu().numpy(),
            }
        )

    concatenated = {
        name: np.concatenate(values) for name, values in per_sequence.items()
    }
    deterministic_summary = {
        name: float(
            np.average([row[name] for row in deterministic_rows], weights=batch_sizes)
        )
        for name in deterministic_rows[0]
    }
    deterministic_summary["ade_m"] = float(concatenated["ade"].mean())
    for name, values in concatenated.items():
        if name.startswith("fde_"):
            deterministic_summary[f"{name}_m"] = float(values.mean())

    risk_summary = _aggregate_risk(risk_rows)
    stochastic_summary = {
        name: float(np.concatenate([row[name] for row in stochastic_rows]).mean())
        for name in stochastic_rows[0]
    }
    return {
        "deterministic_prior_mean": deterministic_summary,
        "stochastic": stochastic_summary,
        "interaction_metrics": risk_summary,
    }, concatenated


def evaluate_hiqr_v2_world_model(
    config: dict[str, Any],
    *,
    config_dir: Path,
    checkpoint: Path | None = None,
    max_sequences: int = 0,
    baseline_metrics: Path | None = None,
    baseline_label: str = "baseline",
    split: str = "test",
    allow_diagnostic_ablation: bool = False,
) -> dict[str, Any]:
    paths, evaluation = config["paths"], config.get("evaluation", {})
    output = Path(paths["output_dir"])
    output = output if output.is_absolute() else (config_dir / output).resolve()
    device = select_device(str(evaluation.get("device", "auto")))
    set_seed(int(evaluation.get("seed", 42)))
    checkpoint = checkpoint or output / "checkpoints/best_hiqr_v2_world_model.pt"
    model = load_hiqr_v2_checkpoint(checkpoint, device=device)
    require_canonical_hiqr_v2_checkpoint(
        model, allow_diagnostic_ablation=allow_diagnostic_ablation
    )
    schema_path = Path(paths["flow_schema"])
    schema_path = (
        schema_path
        if schema_path.is_absolute()
        else (config_dir / schema_path).resolve()
    )
    schema = FrozenLegacyFlowSchema.load(schema_path)
    arrays, manifest = load_hiqr_v2_arrays(
        cache_owner=sequence_cache_owner_dir(config, config_dir=config_dir),
        v1_sidecar_output_dir=paths["v1_sidecar_output_dir"],
        flow_schema=schema,
        source_dataset_dir=paths["source_dataset_dir"],
    )
    loader = make_hiqr_v2_loader(
        arrays,
        split,
        batch_size=int(evaluation.get("batch_size", 48)),
        maximum=int(max_sequences or evaluation.get("max_sequences", 0)),
        shuffle=False,
        seed=int(evaluation.get("seed", 42)),
        num_workers=int(evaluation.get("num_workers", 0)),
    )
    results, per_sequence = _evaluate_model(
        model,
        loader,
        device,
        stochastic_samples=int(evaluation.get("stochastic_samples", 8)),
    )
    ensure_dir(output / "evaluation")
    metric_path = output / f"evaluation/{split}_per_sequence_metrics.npz"
    np.savez_compressed(metric_path, **per_sequence)
    report: dict[str, Any] = {
        "model_type": model.model_type,
        "checkpoint": str(checkpoint),
        "split": split,
        "sequences": int(len(loader.dataset)),
        "cohort": manifest["hiqr_v2_cohort"],
        "sequence_cache": manifest,
        "per_sequence_metrics": str(metric_path),
        **results,
    }
    if baseline_metrics is not None:
        with np.load(baseline_metrics) as baseline:
            report["bootstrap_vs_baseline"] = _bootstrap_difference(
                per_sequence["fde_5p96s"],
                np.asarray(baseline["fde_5p96s"]),
                samples=int(evaluation.get("bootstrap_samples", 1000)),
                seed=int(evaluation.get("seed", 42)),
            )
        report["bootstrap_baseline_label"] = str(baseline_label)
    save_json(report, output / f"evaluation/{split}_evaluation_summary.json")
    return report


def evaluate_hiqr_v1_fixed_cohort(
    config: dict[str, Any],
    *,
    config_dir: Path,
    checkpoint: Path,
    output: Path,
    max_sequences: int = 0,
    split: str = "test",
) -> dict[str, Any]:
    """Evaluate the immutable V1 checkpoint on the exact V2 stable test cohort."""
    paths, evaluation = config["paths"], config.get("evaluation", {})
    device = select_device(str(evaluation.get("device", "auto")))
    set_seed(int(evaluation.get("seed", 42)))
    schema_path = Path(paths["flow_schema"])
    schema_path = (
        schema_path
        if schema_path.is_absolute()
        else (config_dir / schema_path).resolve()
    )
    schema = FrozenLegacyFlowSchema.load(schema_path)
    arrays, manifest = load_hiqr_v2_arrays(
        cache_owner=sequence_cache_owner_dir(config, config_dir=config_dir),
        v1_sidecar_output_dir=paths["v1_sidecar_output_dir"],
        flow_schema=schema,
        source_dataset_dir=paths["source_dataset_dir"],
    )
    loader = make_hiqr_v2_loader(
        arrays,
        split,
        batch_size=int(evaluation.get("batch_size", 48)),
        maximum=int(max_sequences or evaluation.get("max_sequences", 0)),
        shuffle=False,
        seed=int(evaluation.get("seed", 42)),
        num_workers=int(evaluation.get("num_workers", 0)),
    )
    model = load_hiqr_checkpoint(checkpoint, device=device).eval()
    results, per_sequence = _evaluate_model(
        model,
        loader,
        device,
        stochastic_samples=int(evaluation.get("stochastic_samples", 8)),
    )
    output = ensure_dir(output)
    metric_path = output / f"v1_fixed_{split}_cohort_per_sequence_metrics.npz"
    np.savez_compressed(metric_path, **per_sequence)
    report = {
        "model_type": model.model_type,
        "checkpoint": str(checkpoint),
        "split": split,
        "cohort": manifest["hiqr_v2_cohort"],
        "sequences": int(len(loader.dataset)),
        "per_sequence_metrics": str(metric_path),
        **results,
    }
    save_json(report, output / f"v1_fixed_{split}_cohort_summary.json")
    return report
