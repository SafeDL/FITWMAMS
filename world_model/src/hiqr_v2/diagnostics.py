"""Reproducible, fixed-cohort causal ablations for HiQR-v2.

The diagnostic ladder intentionally changes one causal design decision at a
time.  It is a training launcher, not a claim that an untrained ablation has
already improved the original model.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from world_model.src.core.utils import ensure_dir, save_json

from .config import HiQRV2Config
from .evaluation import (
    _bootstrap_difference,
    evaluate_hiqr_v1_fixed_cohort,
    evaluate_hiqr_v2_world_model,
)
from .train import train_hiqr_v2_world_model

# The five 12-epoch runs form a cumulative causal ladder.  Each row changes
# exactly one decision from its predecessor, so the paired bootstrap answers
# the intended question instead of comparing unrelated combinations.
DIAGNOSTIC_VARIANTS: dict[str, dict[str, Any]] = {
    "R0": {
        "filter_update_mode": "stateless",
        "continuation_mode": "full_replan",
        "scene_mode_responses": 1,
        "physical_mode": "off",
    },
    "R1": {
        "filter_update_mode": "observed",
        "continuation_mode": "full_replan",
        "scene_mode_responses": 1,
        "physical_mode": "off",
    },
    "R2": {
        "filter_update_mode": "observed",
        "continuation_mode": "full_replan",
        "scene_mode_responses": 5,
        "physical_mode": "off",
    },
    "R3": {
        "filter_update_mode": "observed",
        "continuation_mode": "adaptive_5_15_5",
        "scene_mode_responses": 5,
        "physical_mode": "off",
    },
    "R4": {
        "filter_update_mode": "observed",
        "continuation_mode": "adaptive_5_15_5",
        "scene_mode_responses": 5,
        "physical_mode": "target_aware",
    },
}


def diagnostic_variant_names(value: str | Iterable[str]) -> list[str]:
    values = (
        list(DIAGNOSTIC_VARIANTS)
        if value == "all"
        else list(value) if not isinstance(value, str) else [value]
    )
    unknown = sorted(set(values) - set(DIAGNOSTIC_VARIANTS))
    if unknown:
        raise ValueError(f"unknown HiQR-v2 diagnostic variants: {unknown}")
    return values


def _diagnostic_config(
    base: dict[str, Any], *, output: Path, variant: str, max_sequences: int
) -> dict[str, Any]:
    result = deepcopy(base)
    result["paths"] = dict(result["paths"])
    result["paths"]["output_dir"] = str(output)
    result["model"] = {**result.get("model", {}), **DIAGNOSTIC_VARIANTS[variant]}
    result["dataset"] = {
        **result.get("dataset", {}),
        "max_sequences": int(max_sequences),
    }
    training = dict(result["training"])
    # Same 1 s -> 3 s -> 5.96 s course for every intervention, compacted to
    # 12 epochs so decision gates are affordable but still fully closed loop.
    training.update(
        {
            "epochs": 12,
            "stages": [
                {
                    "name": "diagnostic_start",
                    "epochs": 2,
                    "rollout_seconds": 1.0,
                    "learning_rate": 1.0e-4,
                    "batch_size": 48,
                    "val_batch_size": 48,
                },
                {
                    "name": "diagnostic_closed_loop",
                    "epochs": 4,
                    "rollout_seconds": 3.0,
                    "learning_rate": 5.0e-5,
                    "batch_size": 48,
                    "val_batch_size": 48,
                },
                {
                    "name": "diagnostic_full_prior",
                    "epochs": 6,
                    "rollout_seconds": 5.96,
                    "learning_rate": 3.0e-5,
                    "batch_size": 48,
                    "val_batch_size": 48,
                },
            ],
        }
    )
    result["training"] = training
    return result


def decision_from_bootstrap(report: dict[str, Any]) -> dict[str, Any]:
    """Apply the locked >=10% and CI-not-crossing-zero admission gate."""
    result = report.get("bootstrap_vs_baseline")
    if result is None:
        return {
            "eligible_for_v2_default": False,
            "reason": "baseline comparison unavailable",
        }
    eligible = (
        float(result["candidate_improvement_fraction"]) >= 0.10
        and float(result["ci95_high_m"]) < 0.0
    )
    return {
        "eligible_for_v2_default": bool(eligible),
        "improvement_fraction": float(result["candidate_improvement_fraction"]),
        "fde_delta_ci95_m": [float(result["ci95_low_m"]), float(result["ci95_high_m"])],
        "rule": "improvement >= 10% and 95% bootstrap CI entirely below zero",
    }


def continuation_decision(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    """Admit adaptive continuation on smoothness/safety, not just FDE gain."""
    current = candidate["deterministic_prior_mean"]
    reference = baseline["deterministic_prior_mean"]
    risk, reference_risk = candidate["interaction_metrics"], baseline["interaction_metrics"]
    fde_ratio = float(current["fde_5p96s_m"]) / max(
        float(reference["fde_5p96s_m"]), 1.0e-12
    )
    checks = {
        "fde_within_5_percent": fde_ratio <= 1.05,
        "jerk_not_higher": float(current["jerk"]) <= float(reference["jerk"]),
        "plan_discontinuity_lower": float(current["plan_discontinuity"])
        <= float(reference["plan_discontinuity"]),
        "emergency_not_higher": float(current.get("emergency_rate", 0.0))
        <= float(reference.get("emergency_rate", 0.0)) + 0.01,
        "collision_not_higher": float(risk["collision_episode_rate"])
        <= float(reference_risk["collision_episode_rate"]),
        "gap_not_worse": float(risk["gap_mae_m"])
        <= float(reference_risk["gap_mae_m"]) * 1.05,
        "ttc_not_worse": float(risk["ttc_mae_s"])
        <= float(reference_risk["ttc_mae_s"]) * 1.05,
        "drac_not_worse": float(risk["drac_mae_mps2"])
        <= float(reference_risk["drac_mae_mps2"]) * 1.05,
    }
    return {
        "eligible_for_v2_default": bool(all(checks.values())),
        "fde_ratio": fde_ratio,
        "checks": checks,
        "rule": "FDE <= 1.05× full-replan plus lower jerk/discontinuity and no risk regression",
    }


def run_hiqr_v2_diagnostics(
    config: dict[str, Any],
    *,
    config_dir: Path,
    v1_checkpoint: Path,
    variants: str | Iterable[str] = "all",
    max_sequences: int = 0,
    train_variants: bool = False,
) -> dict[str, Any]:
    """Write the plan or execute the five fixed-cohort causal runs."""
    output = Path(config["paths"]["output_dir"])
    output = output if output.is_absolute() else (config_dir / output).resolve()
    root = ensure_dir(output / "diagnostics")
    requested = diagnostic_variant_names(variants)
    selected = list(requested)
    if train_variants:
        final_index = max(list(DIAGNOSTIC_VARIANTS).index(name) for name in selected)
        selected = list(DIAGNOSTIC_VARIANTS)[: final_index + 1]
    baseline = evaluate_hiqr_v1_fixed_cohort(
        config,
        config_dir=config_dir,
        checkpoint=v1_checkpoint,
        output=root / "v1_epoch32_fixed_validation_cohort",
        max_sequences=max_sequences,
        split="val",
    )
    v1_metrics = Path(baseline["per_sequence_metrics"])
    predecessor_metrics: Path | None = None
    runs: dict[str, Any] = {}
    for variant in selected:
        variant_output = root / "runs" / variant
        run_config = _diagnostic_config(
            config, output=variant_output, variant=variant, max_sequences=max_sequences
        )
        item: dict[str, Any] = {
            "variant": variant,
            "model_overrides": DIAGNOSTIC_VARIANTS[variant],
            "model_config": asdict(HiQRV2Config(**run_config["model"])),
            "output_dir": str(variant_output),
            "course": run_config["training"]["stages"],
            "status": "planned",
        }
        if train_variants:
            training = train_hiqr_v2_world_model(run_config, config_dir=config_dir)
            comparison = v1_metrics if predecessor_metrics is None else predecessor_metrics
            report = evaluate_hiqr_v2_world_model(
                run_config,
                config_dir=config_dir,
                checkpoint=Path(training["best_checkpoint"]),
                max_sequences=max_sequences,
                baseline_metrics=comparison,
                baseline_label=(
                    "v1_epoch32" if predecessor_metrics is None else f"previous_{selected[selected.index(variant) - 1]}"
                ),
                split="val",
                allow_diagnostic_ablation=True,
            )
            item.update(
                {"status": "completed", "training": training, "evaluation": report}
            )
            if predecessor_metrics is None:
                item["reference_vs_v1_epoch32"] = report["bootstrap_vs_baseline"]
            elif variant == "R3":
                previous = runs[selected[selected.index(variant) - 1]]["evaluation"]
                item["decision"] = continuation_decision(report, previous)
            else:
                item["decision"] = decision_from_bootstrap(report)
            predecessor_metrics = Path(report["per_sequence_metrics"])
        runs[variant] = item
    physical_decision: dict[str, Any] = {
        "eligible_for_v2_default": False,
        "reason": "R3 and R4 pair not both executed",
    }
    if train_variants and {"R3", "R4"} <= runs.keys():
        no_physical = runs["R3"]
        physical = runs["R4"]
        if no_physical["status"] == physical["status"] == "completed":
            with np.load(physical["evaluation"]["per_sequence_metrics"]) as physical_rows:
                physical_fde = np.asarray(physical_rows["fde_5p96s"])
            with np.load(
                no_physical["evaluation"]["per_sequence_metrics"]
            ) as no_physical_rows:
                no_physical_fde = np.asarray(no_physical_rows["fde_5p96s"])
            comparison = _bootstrap_difference(
                physical_fde,
                no_physical_fde,
                samples=int(
                    config.get("evaluation", {}).get("bootstrap_samples", 1000)
                ),
                seed=int(config["training"].get("seed", 42)),
            )
            physical_decision = {
                **decision_from_bootstrap({"bootstrap_vs_baseline": comparison}),
                "candidate": "target_aware_physical",
                "baseline": "physical_off",
                "bootstrap": comparison,
            }
    report = {
        "diagnostic_name": "hiqr_v2_causal_ablation_fixed_cohort",
        "v1_fixed_cohort_baseline": baseline,
        "fixed_seed": int(config["training"].get("seed", 42)),
        "selection_split": "val",
        "decision_baseline": "cumulative predecessor (R1−R0, R2−R1, R3−R2, R4−R3)",
        "physical_constraint_decision": physical_decision,
        "requested_variants": requested,
        "max_sequences": int(max_sequences),
        "run_training": bool(train_variants),
        "runs": runs,
    }
    save_json(report, root / "diagnostic_manifest.json")
    return report
