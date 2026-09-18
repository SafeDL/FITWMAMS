#!/usr/bin/env python3
"""Train-only style clustering plus documented finite-HSMM segmentation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesian_ma_idm.src.data import load_ragged_pairs
from multi_regime_bidm.src.hsmm import fit_hsmm
from multi_regime_bidm.src.style import FEATURE_NAMES, decision_grid, kmeans, standardize_train, style_features


def _poisson_tail(lam: float, maximum: int) -> float:
    probability = np.exp(-lam)
    cdf = probability
    for value in range(1, maximum + 1):
        probability *= lam / value
        cdf += probability
    return float(max(0., 1. - cdf))


def run(dataset: str | Path, output_dir: str | Path, recordings: list[int], *, states: int = 3,
        duration_max: int = 50, iterations: int = 12, seed: int = 20260915) -> Path:
    _, all_pairs = load_ragged_pairs(dataset)
    selected = [(index, pair) for index, pair in enumerate(all_pairs) if int(pair["recording_id"]) in set(recordings)]
    if len(selected) < 3:
        raise RuntimeError("stage-A needs at least three train-only highD events")
    features = np.vstack([style_features(pair) for _, pair in selected])
    standardized, feature_mean, feature_scale = standardize_train(features)
    style_id, centers_standardized = kmeans(standardized, clusters=3, seed=seed)
    centers_raw = centers_standardized * feature_scale + feature_mean
    labels_by_pair: list[np.ndarray] = [np.empty(0, dtype=np.int16) for _ in selected]
    style_models: dict[str, dict] = {}
    model_arrays: dict[str, list[np.ndarray]] = {key: [] for key in ("means", "variances", "transition", "initial", "duration_lambda", "observation_mean", "observation_scale")}
    for style in range(3):
        local = np.flatnonzero(style_id == style)
        sequences = [decision_grid(selected[index][1])[0] for index in local]
        if len(sequences) < states:
            raise RuntimeError(f"style {style} has {len(sequences)} events, fewer than {states} regimes")
        model, labels, durations = fit_hsmm(sequences, states=states, duration_max=duration_max,
                                            iterations=iterations, seed=seed + style)
        for index, value in zip(local, labels):
            labels_by_pair[int(index)] = np.asarray(value, dtype=np.int16)
        counts = np.bincount(np.concatenate(labels), minlength=states)
        duration_by_state = [[] for _ in range(states)]
        for value in labels:
            state, length = value[0], 1
            for next_state in value[1:]:
                if next_state == state:
                    length += 1
                else:
                    duration_by_state[int(state)].append(length); state, length = next_state, 1
            duration_by_state[int(state)].append(length)
        style_models[str(style)] = {
            "events": int(len(local)), "decision_points": int(sum(len(sequence) for sequence in sequences)),
            "emission_mean_standardized": model.means.tolist(), "emission_variance_standardized": model.variances.tolist(),
            "transition_no_self": model.transition.tolist(), "initial": model.initial.tolist(),
            "poisson_duration_lambda_ticks": model.duration_lambda.tolist(), "duration_max_ticks": duration_max,
            "poisson_tail_mass_above_duration_max": [_poisson_tail(float(value), duration_max) for value in model.duration_lambda],
            "state_point_count": counts.tolist(), "state_duration_mean_ticks": [float(np.mean(value)) if value else 0. for value in duration_by_state],
            "observation_train_mean": model.observation_mean.tolist(), "observation_train_scale": model.observation_scale.tolist(),
        }
        for key, value in (("means", model.means), ("variances", model.variances), ("transition", model.transition),
                           ("initial", model.initial), ("duration_lambda", model.duration_lambda),
                           ("observation_mean", model.observation_mean), ("observation_scale", model.observation_scale)):
            model_arrays[key].append(np.asarray(value))
    offsets = np.r_[0, np.cumsum([len(value) for value in labels_by_pair])].astype(np.int64)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    archive = output / "stage_a_offline_labels.npz"
    np.savez_compressed(archive, pair_index=np.asarray([index for index, _ in selected], dtype=np.int32),
                        recording_id=np.asarray([pair["recording_id"] for _, pair in selected], dtype=np.int16),
                        follower_id=np.asarray([pair["follower_id"] for _, pair in selected], dtype=np.int32),
                        style_id=style_id.astype(np.int8), style_feature_raw=features,
                        regime_offsets=offsets, offline_regime=np.concatenate(labels_by_pair), decision_dt_s=np.asarray(.2))
    model_archive = output / "stage_a_finite_hsmm_models.npz"
    np.savez_compressed(model_archive, duration_max=np.asarray(duration_max), decision_dt_s=np.asarray(.2),
                        **{key: np.asarray(value) for key, value in model_arrays.items()})
    report = {
        "model_id": "multi_regime_bidm", "stage": "A: train-only style discovery and offline regime segmentation",
        "segmentation_kind": "documented_adaptation_finite_3_state_gaussian_hsmm", "paper_exact_segmentation": "blocked_ambiguity",
        "paper_ambiguity": "paper text states Poisson observations and Gaussian durations although observations are continuous and durations are positive integers",
        "dataset": str(dataset), "train_recordings": recordings, "events": len(selected), "decision_dt_s": .2,
        "style_features": list(FEATURE_NAMES), "feature_standardization": {"fit": "train events only", "mean": feature_mean.tolist(), "scale": feature_scale.tolist()},
        "style_centers_raw_sorted_by_mean_headway": centers_raw.tolist(), "style_event_count": np.bincount(style_id, minlength=3).tolist(),
        "style_name_policy": "IDs are train-only ranks by mean headway; no highD cluster is called emergency/aggressive/timid without separate evidence",
        "offline_only": "HSMM labels use whole sequences and must not be used as online controller states", "style_models": style_models,
        "artifacts": {"offline_labels": archive.name, "finite_hsmm_models": model_archive.name,
                      "online_filter": "src/online_filter.py; must receive only causal observations"},
    }
    path = output / "stage_a_report.json"; path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    leakage = {"style_fit_events": "only listed train recordings", "style_features": "whole-event features permitted for training discovery only",
               "test_event_policy": "must use train style frequencies or a separately implemented observed-prefix classifier; no test future maxima/mean headway",
               "offline_hsmm_policy": "offline labels are an upper-bound diagnostic and never enter online evaluation"}
    (output / "style_leakage_audit.json").write_text(json.dumps(leakage, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "multi_regime_bidm/evidence/dataset/highd_multi_regime_cohort_25hz.npz"))
    parser.add_argument("--output-dir", default=str(ROOT / "multi_regime_bidm/evidence/segmentation"))
    parser.add_argument("--train-recordings", default="25")
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--duration-max", type=int, default=100,
                        help="maximum explicit duration in 5 Hz ticks (default 20 s; report tail mass)")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    print(run(args.dataset, args.output_dir, [int(value) for value in args.train_recordings.split(",")],
              states=args.states, duration_max=args.duration_max, iterations=args.iterations, seed=args.seed))


if __name__ == "__main__":
    main()
