#!/usr/bin/env python3
"""Compare completed FIRM-WM results with frozen world-model baselines."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.firm.evaluation import _background_metrics
from world_model.src.firm.train import _loader
from world_model.src.core.initial_behavior_anchor import FrozenLegacyFlowSchema
from world_model.src.ramp.train import load_ramp_checkpoint
from world_model.src.semi_markov.train import _to_batch, load_semi_markov_checkpoint
from world_model.src.core.sequential_dataset import (
    ensure_frozen_flow_behavior_anchor_cache,
    load_sequential_dataset,
    sequence_cache_owner_dir,
)
from world_model.src.core.utils import load_json, load_yaml, save_json, select_device


def _pick(payload: dict[str, Any], *path: str) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _firm_row(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "one_second": _pick(payload, "one_second_conditional_reconstruction"),
        "five_second": _pick(payload, "five_second_roll_mode"),
        "evt_tail": _pick(payload, "evt_tail"),
        "physical": _pick(payload, "physical_diagnostics"),
        "interaction": _pick(payload, "interaction_metrics"),
        "information": _pick(payload, "information_conditions"),
    }


def _cat_row(payload: dict[str, Any]) -> dict[str, Any]:
    five_second = _pick(payload, "model_state_reconstruction", "test", "5_chunks")
    return {
        # ``closed_loop.test`` in this archived artifact is only the two
        # chunk (2 s) result.  The separately saved five-chunk reconstruction
        # is the only horizon-aligned CAT reference.
        "one_second": None,
        "five_second": five_second,
        "evt_tail": None,
        "physical": _pick(payload, "closed_loop", "test"),
        "interaction": _pick(payload, "closed_loop", "test"),
        "information": {
            "strictly_information_symmetric": False,
            "reason": "Archived CAT-TopK START consumes a future action summary.",
        },
    }


def _threshold_gate(
    *,
    observed: Any,
    threshold: float,
    relation: str,
    requirement: str,
) -> dict[str, Any]:
    """Record a gate result without treating a missing evaluation as a pass."""
    if not isinstance(observed, (int, float)) or not np.isfinite(float(observed)):
        return {"status": "not_evaluated", "requirement": requirement}
    value = float(observed)
    passed = value <= threshold if relation == "<=" else value >= threshold
    return {
        "status": "passed" if passed else "failed",
        "requirement": requirement,
        "observed": value,
        "threshold": float(threshold),
        "relation": relation,
    }


def _promotion_gate(
    firm: dict[str, Any], replay: dict[str, Any], flow_report: dict[str, Any] | None
) -> dict[str, Any]:
    """Make the goal-document gates explicit in the comparison artifact.

    The gate is deliberately conservative: absent held-out, Flow, or risk
    evidence produces ``not_evaluated`` and therefore cannot promote a
    candidate.  It never alters a checkpoint or hides a failed result.
    """
    firm_fde = _pick(firm, "five_second_roll_mode", "FDE_m")
    ramp_fde = _pick(replay, "ramp_world_model", "five_second", "FDE_m")
    semi_fde = _pick(replay, "semi_markov_world_model", "five_second", "FDE_m")
    requirements: dict[str, dict[str, Any]] = {
        "heldout_fde_vs_ramp": _threshold_gate(
            observed=(None if firm_fde is None or ramp_fde is None else float(firm_fde) - float(ramp_fde)),
            threshold=0.0,
            relation="<=",
            requirement="5 s background-only FDE must not exceed matched frozen RAMP-WM.",
        ),
        "heldout_fde_vs_semi_markov": _threshold_gate(
            observed=(None if firm_fde is None or semi_fde is None else float(firm_fde) - float(semi_fde)),
            threshold=0.0,
            relation="<=",
            requirement="5 s background-only FDE must not exceed matched frozen Semi-Markov.",
        ),
        "heldout_invalid_rate": _threshold_gate(
            observed=_pick(firm, "physical_diagnostics", "invalid_rate"),
            threshold=0.01,
            relation="<=",
            requirement="Held-out replay invalid rate must be below 1%.",
        ),
        "heldout_overlap_rate": _threshold_gate(
            observed=_pick(firm, "physical_diagnostics", "overlap_rate"),
            threshold=0.01,
            relation="<=",
            requirement="Held-out replay overlap rate must be below 1%.",
        ),
    }
    flow_physical = _pick(flow_report or {}, "closed_loop_distribution", "physical_validity")
    requirements["flow_invalid_trajectory_rate"] = _threshold_gate(
        observed=_pick(flow_physical or {}, "invalid_trajectory_rate"),
        threshold=0.01,
        relation="<=",
        requirement="Flow × FIRM invalid-trajectory rate must be below 1%.",
    )
    requirements["flow_collision_overlap_rate"] = _threshold_gate(
        observed=_pick(flow_physical or {}, "collision_overlap_rate"),
        threshold=0.01,
        relation="<=",
        requirement="Flow × FIRM collision-overlap point rate must be below 1%.",
    )
    q90 = _pick(
        flow_report or {},
        "closed_loop_distribution",
        "risk_tail_all_inner_samples",
        "exceedance_at_real_quantiles",
        "q90",
    )
    if isinstance(q90, dict) and isinstance(q90.get("within_highd_bootstrap_95"), bool):
        requirements["flow_risk_q90_calibration"] = {
            "status": "passed" if q90["within_highd_bootstrap_95"] else "failed",
            "requirement": "Generated q90 exceedance must lie in the fixed-highD bootstrap interval.",
            "observed": q90.get("flow_firm"),
            "highd_bootstrap_95": q90.get("highd_bootstrap_95"),
        }
    else:
        requirements["flow_risk_q90_calibration"] = {
            "status": "not_evaluated",
            "requirement": "Generated q90 exceedance must lie in the fixed-highD bootstrap interval.",
        }
    passed = all(value["status"] == "passed" for value in requirements.values())
    return {
        "decision": "eligible_for_world_model_claim" if passed else "not_promoted",
        "requirements": requirements,
        "rule": "All held-out reconstruction, physical-validity, Flow-composition, and q90-calibration gates must pass; missing evidence never passes.",
    }


def _background_replay(
    *,
    model_name: str,
    config_path: Path,
    checkpoint: Path,
) -> dict[str, Any]:
    """Replay a frozen baseline under FIRM's generated-background metric.

    The old summaries include the externally replayed ego in ADE/FDE.  This
    does not change a baseline checkpoint or its archived result: it writes a
    separate, reproducible evaluation with ego removed from the primary error
    denominator.
    """
    config = load_yaml(config_path)
    config_dir = config_path.parent
    arrays, manifest = load_sequential_dataset(
        sequence_cache_owner_dir(config, config_dir=config_dir)
    )
    schema_value = config.get("paths", {}).get("flow_schema")
    if schema_value:
        schema_path = Path(schema_value)
        if not schema_path.is_absolute():
            schema_path = (config_dir / schema_path).resolve()
        schema = FrozenLegacyFlowSchema.load(schema_path)
        arrays.update(
            ensure_frozen_flow_behavior_anchor_cache(
                sequence_cache_owner_dir(config, config_dir=config_dir), arrays, manifest, schema
            )
        )
    evaluation = config.get("evaluation", {})
    device = select_device(str(evaluation.get("device", "auto")))
    if model_name == "ramp_world_model":
        model = load_ramp_checkpoint(checkpoint, device=device)
        needs_seed = False
    elif model_name == "semi_markov_world_model":
        model = load_semi_markov_checkpoint(checkpoint, device=device)
        if model.uses_behavior_anchor:
            if not schema_value:
                raise ValueError("Semi-Markov background replay requires its frozen Flow schema")
            model.set_frozen_flow_schema(schema)
        cold_start = evaluation.get("cold_start_history")
        if cold_start is not None:
            model.cfg = replace(model.cfg, cold_start_history=bool(cold_start))
        needs_seed = True
    else:
        raise ValueError(f"unsupported frozen baseline: {model_name}")
    loader = _loader(
        arrays,
        "test",
        # This replay is no-gradient and deterministic; a larger batch only
        # changes throughput, never the frozen model's trajectory or metric.
        batch_size=max(256, int(evaluation.get("batch_size", 64))),
        maximum=0,
        shuffle=False,
        seed=int(evaluation.get("seed", 123)),
        num_workers=0,
    )
    prediction: list[np.ndarray] = []
    target: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    tail: list[np.ndarray] = []
    ego: list[np.ndarray] = []
    with torch.no_grad():
        for batch_index, values in enumerate(loader):
            batch = _to_batch(values, loader.field_names, device)
            if needs_seed:
                rollout = model.rollout_roll_mode(
                    batch,
                    seed=int(evaluation.get("seed", 123)) + batch_index * int(batch["agent_states"].shape[0]),
                    deterministic=True,
                )
            else:
                rollout = model.rollout_roll_mode(batch, deterministic=True)
            prediction.append(rollout["predicted_states"][:, :, 1:].cpu().numpy())
            target.append(rollout["target_states"][:, :, 1:].cpu().numpy())
            valid.append(rollout["target_valid"][:, :, 1:].cpu().numpy().astype(bool))
            tail.append(batch["is_evt_tail"].cpu().numpy().astype(bool))
            ego.append(rollout["target_states"][:, :, 0].cpu().numpy())
    pred, tgt, mask, is_tail, ego_states = map(
        np.concatenate, (prediction, target, valid, tail, ego)
    )
    # The ego is exactly replayed by the model and is only used for relative
    # metrics; its error is deliberately excluded from every reconstruction
    # average here.
    full = _background_metrics(pred, tgt, mask, ego_states)
    one = _background_metrics(pred[:, :25], tgt[:, :25], mask[:, :25], ego_states[:, :25])
    evt = (
        _background_metrics(pred[is_tail], tgt[is_tail], mask[is_tail], ego_states[is_tail])
        if is_tail.any()
        else {"available": False}
    )
    return {
        "protocol": {
            "split": "full held-out highD test split",
            "horizon_seconds": 5.0,
            "primary_metric_scope": "generated background vehicles only",
            "batch_size": max(256, int(evaluation.get("batch_size", 64))),
            "checkpoint": str(checkpoint),
        },
        "sequence_cache": manifest,
        "one_second": one,
        "five_second": full,
        "evt_tail": evt,
    }


def build(output_dir: Path) -> dict[str, Any]:
    paths = {
        "firm_world_model": output_dir / "evaluation/evaluation_summary.json",
        "ramp_world_model": ROOT / "results/highd_world_model/ramp_world_model/ramp_evaluation_summary.json",
        "semi_markov_world_model": ROOT / "results/highd_world_model/semi_markov_world_model/semi_markov_evaluation_summary.json",
        "cat_topk_world_model": ROOT / "results/highd_world_model/cat_topk_world_model/evaluation_summary.json",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing formal evaluation summaries: " + ", ".join(missing))
    firm, ramp, semi, cat = (load_json(path) for path in paths.values())
    replay_path = output_dir / "evaluation/baseline_background_replay.json"
    if replay_path.exists():
        replay = load_json(replay_path)
    else:
        replay = {
            "ramp_world_model": _background_replay(
                model_name="ramp_world_model",
                config_path=ROOT / "results/highd_world_model/ramp_world_model/configs/highd_ramp_world_model.yaml",
                checkpoint=ROOT / "results/highd_world_model/ramp_world_model/checkpoints/best_ramp_world_model.pt",
            ),
            "semi_markov_world_model": _background_replay(
                model_name="semi_markov_world_model",
                config_path=ROOT / "results/highd_world_model/semi_markov_world_model/configs/highd_semi_markov_world_model.yaml",
                checkpoint=ROOT / "results/highd_world_model/semi_markov_world_model/checkpoints/best_semi_markov_relational.pt",
            ),
        }
        save_json(replay, replay_path)
    flow_path = output_dir / "evaluation/flow_firm_composition.json"
    flow_report = load_json(flow_path) if flow_path.exists() else None
    report = {
        "protocol": {
            "split": "full held-out highD test split",
            "horizon_seconds": 5.0,
            "primary_metrics": ["background-only ADE/FDE", "interaction", "physical validity"],
            "cat_topk_note": "CAT-TopK is shown for reproducibility only and is not a same-information winner/loser claim.",
        },
        "models": {
            "firm_world_model": _firm_row(firm),
            "ramp_world_model": {
                "matched_background_replay": replay["ramp_world_model"],
                "archived_legacy_summary": _firm_row(ramp),
            },
            "semi_markov_world_model": {
                "matched_background_replay": replay["semi_markov_world_model"],
                "archived_legacy_summary": _firm_row(semi),
            },
            "cat_topk_world_model": _cat_row(cat),
        },
        "artifacts": {
            **{name: str(path) for name, path in paths.items()},
            "background_only_replay": str(replay_path),
            "flow_firm_composition": str(flow_path) if flow_path.exists() else None,
        },
        "promotion_gate": _promotion_gate(firm, replay, flow_report),
    }
    save_json(report, output_dir / "evaluation/baseline_comparison.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(ROOT / "results/highd_world_model/firm_world_model"))
    args = parser.parse_args()
    build(Path(args.output_dir).resolve())


if __name__ == "__main__":
    main()
