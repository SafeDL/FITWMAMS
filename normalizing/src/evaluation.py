"""Evaluation, sampling, and visualization for the highD tail flow."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baselines import fit_density_baselines
from .data import load_tail_dataset, output_dir_from_config, split_indices
from .metrics import (
    conditional_nll_by_group,
    correlation_error,
    distribution_match_metrics,
    evaluate_conditional_nll,
    nll_table_for_report,
    occupancy_metrics,
    physical_validity_metrics,
)
from .sampling import (
    load_checkpoint_and_dataset,
    sample_tail_c0,
    samples_to_frame,
    to_ads_initialization,
    to_world_model_start_condition,
)
from .utils import ensure_dir, repo_root_from_file, save_json, select_device
from .visualization import default_checkpoint, write_tail_flow_visual_diagnostics


logger = logging.getLogger(__name__)


def _comparison_flags(main_nll: dict[str, float], baselines: dict[str, Any]) -> dict[str, Any]:
    test = float(main_nll.get("test", np.inf))
    flags = {}
    for name, payload in baselines.items():
        baseline_test = payload.get("nll", {}).get("test")
        flags[name] = bool(baseline_test is not None and test < float(baseline_test))
    return {
        "main_test_nll": test,
        "beats_each_baseline_on_test_nll": flags,
        "beats_all_required_baselines": bool(flags and all(flags.values())),
    }


def _distribution_reference(
    arrays: dict[str, np.ndarray],
    eval_cfg: dict[str, Any],
) -> tuple[str, np.ndarray]:
    split = str(eval_cfg.get("distribution_reference_split") or "all").lower()
    return split, split_indices(arrays, split)


def _resolve_num_generated_samples(raw_value: Any, *, reference_size: int) -> int:
    if raw_value is None:
        return int(reference_size)
    if isinstance(raw_value, str):
        value = raw_value.strip().lower()
        if value in {"match_reference", "reference", "all", "dataset"}:
            return int(reference_size)
    return int(raw_value)


def _write_samples(
    samples: dict[str, np.ndarray],
    schema: dict[str, Any],
    output_dir: Path,
    *,
    output_prefix: str,
) -> dict[str, str]:
    sample_dir = ensure_dir(output_dir / "samples")
    npz_path = sample_dir / f"{output_prefix}.npz"
    csv_path = sample_dir / f"{output_prefix}.csv"
    json_path = sample_dir / f"{output_prefix}_interfaces.json"
    np.savez_compressed(npz_path, **samples)
    samples_to_frame(samples, schema).to_csv(csv_path, index=False)
    interfaces = []
    for i in range(len(samples["features"])):
        primary_slot_name = (
            str(samples["primary_slot_name"][i])
            if "primary_slot_name" in samples
            else None
        )
        interfaces.append(
            {
                "ads_initialization": to_ads_initialization(
                    samples["features"][i],
                    samples["slot_mask"][i],
                    sample_id=i,
                ),
                "world_model_start_condition": to_world_model_start_condition(
                    samples["features"][i],
                    samples["slot_mask"][i],
                    sample_id=i,
                    primary_slot_name=primary_slot_name,
                ),
            }
        )
    save_json(interfaces, json_path)
    return {"npz": str(npz_path), "csv": str(csv_path), "interfaces_json": str(json_path)}


def evaluate_tail_flow(
    config: dict[str, Any],
    *,
    config_dir: str | Path,
    repo_root: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    num_samples: int | None = None,
    output_prefix: str = "generated_samples",
    run_baselines: bool | None = None,
    generate_figures: bool | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    config_dir = Path(config_dir).resolve()
    repo_root = Path(repo_root).resolve() if repo_root else repo_root_from_file(config_dir)
    output_dir = output_dir_from_config(config, config_dir)
    checkpoint = Path(checkpoint_path) if checkpoint_path else default_checkpoint(output_dir)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    device = select_device(str(config.get("device", "auto")))
    flow, arrays, schema, _payload = load_checkpoint_and_dataset(
        checkpoint,
        output_dir,
        repo_root=repo_root,
        device=device,
    )
    eval_cfg = dict(config.get("evaluation", {}))
    batch_size = int(config.get("training", {}).get("batch_size", 256))
    main_nll = {
        split: evaluate_conditional_nll(flow, arrays, split, device, batch_size=batch_size)
        for split in ("train", "val", "test")
    }
    group_nll = conditional_nll_by_group(
        flow,
        arrays,
        schema=schema,
        split="test",
        device=device,
    )

    do_baselines = bool(config.get("baselines", {}).get("enabled", True))
    if run_baselines is not None:
        do_baselines = bool(run_baselines)
    baselines: dict[str, Any] = {}
    if do_baselines:
        baselines = fit_density_baselines(
            arrays,
            cfg=dict(config.get("baselines", {})),
            repo_root=repo_root,
            output_dir=output_dir,
            device=device,
        )

    reference_split, real_idx = _distribution_reference(arrays, eval_cfg)
    requested_num_samples = num_samples
    if requested_num_samples is None:
        requested_num_samples = _resolve_num_generated_samples(
            eval_cfg.get("num_generated_samples", "match_reference"),
            reference_size=len(real_idx),
        )
    event_structure_split = str(
        eval_cfg.get("sample_event_structure_split") or reference_split
    )
    samples = sample_tail_c0(
        flow,
        arrays,
        schema,
        num_samples=int(requested_num_samples),
        device=device,
        seed=int(seed if seed is not None else int(config.get("seed", 42)) + 1000),
        event_structure_split=event_structure_split,
        reject_invalid=bool(eval_cfg.get("reject_invalid_samples", True)),
    )
    sample_paths = _write_samples(
        samples,
        schema,
        output_dir,
        output_prefix=output_prefix,
    )

    compare_idx = real_idx
    if len(compare_idx) > len(samples["features"]):
        rng = np.random.default_rng(int(config.get("seed", 42)))
        compare_idx = rng.choice(compare_idx, size=len(samples["features"]), replace=False)
    distribution_metrics = distribution_match_metrics(
        arrays["features"][compare_idx],
        samples["features"],
        arrays["feature_valid"][compare_idx],
        samples["feature_valid"],
        list(schema["feature_names"]),
    )
    corr_metrics = correlation_error(
        arrays["features_normalized"][compare_idx],
        samples["features_normalized"],
    )
    physical_metrics = physical_validity_metrics(samples["features"], samples["slot_mask"], schema)
    occupancy = occupancy_metrics(arrays["slot_mask"][compare_idx], samples["slot_mask"])
    comparison = _comparison_flags(main_nll, baselines)

    diagnostics_dir = ensure_dir(output_dir / "diagnostics")
    pd.DataFrame(nll_table_for_report(main_nll, baselines)).to_csv(
        diagnostics_dir / "nll_comparison.csv",
        index=False,
    )
    pd.DataFrame(distribution_metrics["per_feature"]).to_csv(
        diagnostics_dir / "feature_distribution_metrics.csv",
        index=False,
    )

    do_figures = bool(eval_cfg.get("generate_figures", True))
    if generate_figures is not None:
        do_figures = bool(generate_figures)
    visual_summary = None
    if do_figures:
        visual_summary = write_tail_flow_visual_diagnostics(
            config,
            config_dir=config_dir,
            repo_root=repo_root,
            checkpoint_path=checkpoint,
            sample_npz=sample_paths["npz"],
            device=str(device),
        )

    report = {
        "checkpoint": str(checkpoint),
        "dataset": schema["dataset_npz"],
        "nll": main_nll,
        "group_nll_test": group_nll,
        "baselines": baselines,
        "nll_comparison": comparison,
        "distribution_reference_split": reference_split,
        "num_real_reference_available": int(len(real_idx)),
        "num_real_reference_samples": int(len(compare_idx)),
        "num_generated_samples": int(len(samples["features"])),
        "distribution_match": distribution_metrics,
        "correlation": corr_metrics,
        "physical_validity": physical_metrics,
        "occupancy": occupancy,
        "sampling_rejection": {
            "num_rejected": int(samples.get("num_rejected", np.asarray([0]))[0]),
            "rejection_rate": float(samples.get("rejection_rate", np.asarray([0.0]))[0]),
        },
        "generated_samples": sample_paths,
        "visual_diagnostics": visual_summary,
    }
    save_json(report, output_dir / "evaluation_summary.json")
    logger.info("Wrote evaluation summary: %s", output_dir / "evaluation_summary.json")
    return report
