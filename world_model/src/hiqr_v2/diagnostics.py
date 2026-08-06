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

# The six 12-epoch runs form a causal ladder.  ``control`` removes the
# persistent observed filter; every candidate is selected against this common
# control, while the named combinations preserve interpretable interventions.
DIAGNOSTIC_VARIANTS: dict[str, dict[str, Any]] = {
    "control": {
        "filter_update_mode": "stateless",
        "continuation_mode": "adaptive_5_15_5",
        "scene_mode_responses": 1,
        "physical_mode": "target_aware",
    },
    "prior_state": {
        "filter_update_mode": "observed",
        "continuation_mode": "adaptive_5_15_5",
        "scene_mode_responses": 1,
        "physical_mode": "target_aware",
    },
    "full_replan": {
        "filter_update_mode": "stateless",
        "continuation_mode": "full_replan",
        "scene_mode_responses": 1,
        "physical_mode": "target_aware",
    },
    "prior_state_full_replan": {
        "filter_update_mode": "observed",
        "continuation_mode": "full_replan",
        "scene_mode_responses": 1,
        "physical_mode": "target_aware",
    },
    "slow_scene": {
        "filter_update_mode": "observed",
        "continuation_mode": "full_replan",
        "scene_mode_responses": 5,
        "physical_mode": "target_aware",
    },
    "no_physical": {
        "filter_update_mode": "observed",
        "continuation_mode": "full_replan",
        "scene_mode_responses": 5,
        "physical_mode": "off",
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


def run_hiqr_v2_diagnostics(
    config: dict[str, Any],
    *,
    config_dir: Path,
    v1_checkpoint: Path,
    variants: str | Iterable[str] = "all",
    max_sequences: int = 0,
    train_variants: bool = False,
) -> dict[str, Any]:
    """Write the experiment plan and optionally execute its six isolated runs."""
    output = Path(config["paths"]["output_dir"])
    output = output if output.is_absolute() else (config_dir / output).resolve()
    root = ensure_dir(output / "diagnostics")
    requested = diagnostic_variant_names(variants)
    selected = list(requested)
    if train_variants and any(name != "control" for name in selected):
        selected = ["control", *[name for name in selected if name != "control"]]
    if train_variants and "no_physical" in selected and "slow_scene" not in selected:
        selected.insert(selected.index("no_physical"), "slow_scene")
    baseline = evaluate_hiqr_v1_fixed_cohort(
        config,
        config_dir=config_dir,
        checkpoint=v1_checkpoint,
        output=root / "v1_epoch32_fixed_validation_cohort",
        max_sequences=max_sequences,
        split="val",
    )
    v1_metrics = Path(baseline["per_sequence_metrics"])
    control_metrics: Path | None = None
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
            comparison = v1_metrics if variant == "control" else control_metrics
            if comparison is None:
                raise RuntimeError("diagnostic control must complete before candidates")
            report = evaluate_hiqr_v2_world_model(
                run_config,
                config_dir=config_dir,
                checkpoint=Path(training["best_checkpoint"]),
                max_sequences=max_sequences,
                baseline_metrics=comparison,
                baseline_label=(
                    "v1_epoch32" if variant == "control" else "diagnostic_control"
                ),
                split="val",
                allow_diagnostic_ablation=True,
            )
            item.update(
                {"status": "completed", "training": training, "evaluation": report}
            )
            if variant == "control":
                control_metrics = Path(report["per_sequence_metrics"])
                item["reference_vs_v1_epoch32"] = report["bootstrap_vs_baseline"]
            else:
                item["decision"] = decision_from_bootstrap(report)
        runs[variant] = item
    physical_decision: dict[str, Any] = {
        "eligible_for_v2_default": False,
        "reason": "slow_scene and no_physical pair not both executed",
    }
    if train_variants and {"slow_scene", "no_physical"} <= runs.keys():
        slow = runs["slow_scene"]
        no_physical = runs["no_physical"]
        if slow["status"] == no_physical["status"] == "completed":
            with np.load(slow["evaluation"]["per_sequence_metrics"]) as physical_rows:
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
        "decision_baseline": "diagnostic_control",
        "physical_constraint_decision": physical_decision,
        "requested_variants": requested,
        "max_sequences": int(max_sequences),
        "run_training": bool(train_variants),
        "runs": runs,
    }
    save_json(report, root / "diagnostic_manifest.json")
    return report
