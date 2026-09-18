#!/usr/bin/env python3
"""Cross-record gap-regime split conformal audit for MA-IDM uncertainty bands."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from bayesian_ma_idm.src.data import load_ragged_pairs
from bayesian_ma_idm.src.full_evaluation import evaluate_full_population
from bayesian_ma_idm.src.full_model import fit_full_population


def gap_specs(path: Path, bins: int, quantile: float) -> dict[str, dict[str, list[float]]]:
    """Learn horizon-specific non-negative offsets from training-only anchors."""
    with np.load(path, allow_pickle=False) as data:
        columns = {str(name): index for index, name in enumerate(data["columns"])}
        values = np.asarray(data["values"], float)
    required = {"horizon_s", "gap_m", "position_90_shortfall_q90_m"}
    if not required.issubset(columns):
        raise RuntimeError("calibration diagnostic lacks gap/shortfall columns")
    output = {}
    for horizon in (3., 5.):
        rows = values[np.isclose(values[:, columns["horizon_s"]], horizon)]
        gap, shortfall = rows[:, columns["gap_m"]], rows[:, columns["position_90_shortfall_q90_m"]]
        edges = np.quantile(gap, np.linspace(0., 1., bins + 1)[1:-1])
        index = np.searchsorted(edges, gap, side="right")
        offsets = [float(np.quantile(shortfall[index == group], quantile)) for group in range(bins)]
        output[str(horizon)] = {"gap_bin_edges_m": [float(value) for value in edges], "offsets_m": offsets}
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "bayesian_ma_idm/evidence/highd_cohort/highd_author_full_25hz.npz"))
    parser.add_argument("--output-dir", default=str(ROOT / "bayesian_ma_idm/evidence/calibrated_intervals"))
    parser.add_argument("--holdouts", default="25,26,36")
    parser.add_argument("--futures", type=int, default=64)
    parser.add_argument("--fit-fraction", type=float, default=.60)
    parser.add_argument("--calibration-start", type=float, default=.65)
    parser.add_argument("--calibration-end", type=float, default=.90)
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--shortfall-quantile", type=float, default=.90)
    args = parser.parse_args()
    if args.bins < 2 or not 0 < args.shortfall_quantile < 1:
        raise ValueError("bins must be >=2 and shortfall quantile must be in (0,1)")
    dataset, output = Path(args.dataset), Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    _, pairs = load_ragged_pairs(dataset); recordings = sorted({int(pair["recording_id"]) for pair in pairs})
    for holdout in (int(value) for value in args.holdouts.split(",")):
        train = [recording for recording in recordings if recording != holdout]
        posterior = output / f"ma_idm_outer_{holdout}_early_train.npz"
        fit_full_population(dataset, posterior, model="ma_idm", training_recordings=train, train_fraction=args.fit_fraction)
        diagnostic = output / f"ma_idm_outer_{holdout}_calibration_anchors.npz"
        calibration = output / f"ma_idm_outer_{holdout}_temporal_calibration.json"
        evaluate_full_population(dataset, posterior, calibration, evaluation_recordings=train, futures=args.futures,
                                 anchor_start_fraction=args.calibration_start, anchor_end_fraction=args.calibration_end,
                                 anchor_diagnostics_path=diagnostic)
        specifications = gap_specs(diagnostic, args.bins, args.shortfall_quantile)
        oof = output / f"ma_idm_outer_{holdout}_oof.json"
        evaluate_full_population(dataset, posterior, oof, evaluation_recordings=[holdout], futures=args.futures,
                                 position_interval_offset_by_gap=specifications)
        result = {"extension": "gap-regime temporal split-conformal position band; not paper MA-IDM",
                  "protocol": "outer recording held out; early training fit; later training anchors yield gap-bin offsets",
                  "holdout_recording": holdout, "training_recordings": train, "fit_fraction": args.fit_fraction,
                  "calibration_anchor_range": [args.calibration_start, args.calibration_end], "gap_bins": args.bins,
                  "shortfall_quantile": args.shortfall_quantile, "gap_offset_specifications": specifications,
                  "calibration_report": calibration.name, "calibration_diagnostics": diagnostic.name, "oof_report": oof.name}
        path = output / f"gap_regime_temporal_conformal_holdout_{holdout}.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
