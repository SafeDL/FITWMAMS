"""Fit EVT from the canonical highD preprocessing output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import genpareto

from process_highD.src.evt_diagnostics import (
    write_gpd_diagnostic_panel,
)
from process_highD.src.io_utils import load_config, resolve_data_path, write_json
from process_highD.src.natural_segments import (
    validate_natural_segment_contract,
)
from process_highD.src.preprocessing_pipeline import natural_output_paths
from tools.evt import fit_evt_model


def load_evt_context(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    preprocessing_paths = natural_output_paths(config, config_path)
    output_dir = preprocessing_paths["out_dir"]
    sequence_cache = (
        resolve_data_path(config["paths"]["sequence_cache_dir"], config_path)
        / "sequence_cache"
    )
    return {
        "config_path": config_path,
        "config": config,
        "preprocessing_paths": preprocessing_paths,
        "sequence_cache": sequence_cache,
        "output_dir": output_dir,
        "event_bank_dir": output_dir / "event_bank",
        "manifest_dir": output_dir / "manifests",
    }


def evt_output_paths(context: dict[str, Any]) -> dict[str, Path]:
    base = Path(context["output_dir"])
    return {
        "output_dir": base,
        "model": base / "evt_model.json",
        "summary": base / "evt_summary.json",
        "figures": base / "figures",
        "threshold_sensitivity": base / "threshold_sensitivity.csv",
    }


def _gpd_gof(
    values: np.ndarray, *, u: float, xi: float, beta: float
) -> dict[str, float | int]:
    excess = np.asarray(values, dtype=np.float64)
    excess = np.sort(excess[np.isfinite(excess) & (excess > u)] - u)
    count = int(excess.size)
    cdf = genpareto.cdf(excess, c=float(xi), loc=0.0, scale=max(float(beta), 1.0e-12))
    cdf = np.clip(cdf, 1.0e-12, 1.0 - 1.0e-12)
    rank = np.arange(1, count + 1, dtype=np.float64)
    ks = max(
        np.max(np.abs(cdf - (rank - 1.0) / count)),
        np.max(np.abs(rank / count - cdf)),
    )
    cvm = 1.0 / (12.0 * count) + np.sum((cdf - (2.0 * rank - 1.0) / (2.0 * count)) ** 2)
    ad = -count - np.mean((2.0 * rank - 1.0) * (np.log(cdf) + np.log(1.0 - cdf[::-1])))
    return {
        "num_exceedances": count,
        "ks": float(ks),
        "cramer_von_mises": float(cvm),
        "anderson_darling": float(ad),
    }


def _threshold_rows(model: Any) -> list[dict[str, float]]:
    fields = (
        "u",
        "k",
        "exceedance_rate",
        "xi",
        "beta",
        "modified_scale",
        "endpoint",
        "z1000",
    )
    rows = [
        {name: float(candidate[name]) for name in fields}
        for candidate in model.threshold_candidates
    ]
    return sorted(rows, key=lambda row: row["u"])


def _load_segments(context: dict[str, Any]) -> pd.DataFrame:
    segment_path = Path(context["preprocessing_paths"]["segment_csv"])
    segments = pd.read_csv(segment_path)
    validate_natural_segment_contract(
        segments,
        source=str(segment_path),
    )
    return segments


def fit_evt(config_path: Path) -> dict[str, Any]:
    context = load_evt_context(config_path)
    evt_options = context["config"]["evt"]
    segments = _load_segments(context)
    values = pd.to_numeric(segments["event_risk"], errors="coerce").to_numpy(np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("EVT calibration has no finite risks")

    model = fit_evt_model(
        values,
        return_periods=tuple(
            sorted({int(value) for value in evt_options["return_periods"]})
        ),
        min_exceedances=int(evt_options["min_exceedances"]),
        max_tail_fraction=float(evt_options["max_tail_fraction"]),
        max_threshold_candidates=int(evt_options["max_threshold_candidates"]),
        min_threshold_exceedance_rate=float(
            evt_options["min_threshold_exceedance_rate"]
        ),
        bootstrap_samples=int(evt_options["bootstrap_samples"]),
        random_seed=int(evt_options["random_seed"]),
    )
    paths = evt_output_paths(context)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    model.to_json(
        paths["model"],
        model_type="gpd_pot_highd_natural_safety_envelope_intrusion_event_risk",
    )
    threshold_rows = _threshold_rows(model)
    pd.DataFrame(threshold_rows).to_csv(paths["threshold_sensitivity"], index=False)
    figures = write_gpd_diagnostic_panel(
        paths["figures"],
        model=model,
        values=values,
        risk_variable="R_SEI",
        output_filename="gpd_diagnostics.png",
        output_key="gpd_diagnostics",
    )
    segment_path = Path(context["preprocessing_paths"]["segment_csv"])
    summary = {
        "calibration_scope": "all",
        "evaluation_scope": {
            "excluded_risk_slots": [],
            "training_population_modified": False,
        },
        "source": {
            "kind": "process_highD_preprocessed_segments",
            "preprocessing_config": str(context["config_path"]),
            "segment_csv": str(segment_path),
            "all_segments": int(len(segments)),
            "calibration_segments": int(len(values)),
            "calibration_recordings": sorted(
                int(value) for value in segments["recording_id"].unique()
            ),
        },
        "model_path": str(paths["model"]),
        "risk_variable": "event_risk (R_SEI)",
        "u": float(model.u),
        "xi": float(model.xi),
        "beta": float(model.beta),
        "exceedance_rate": float(model.exceedance_rate),
        "num_tail_segments": int(np.sum(values > float(model.u))),
        "return_levels": model.return_levels,
        "return_level_ci": model.return_level_ci,
        "gpd_gof": _gpd_gof(
            values, u=float(model.u), xi=float(model.xi), beta=float(model.beta)
        ),
        "threshold_selection": model.threshold_selection,
        "threshold_sensitivity_csv": str(paths["threshold_sensitivity"]),
        "figures": figures,
    }
    write_json(paths["summary"], summary)
    return summary
