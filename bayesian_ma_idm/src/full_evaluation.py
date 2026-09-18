"""Recording-held-out, rolling-origin evaluation for the full highD cohort."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from .data import load_ragged_pairs
from .evaluation import _rollout, condition_driver_candidates, crps_ensemble
from .model import load_posterior, sample_driver_joint

CALIBRATION_LEVELS = (.50, .80, .90, .95)


def regime_interval_offset(specification: dict[str, object] | None, gap_m: float) -> float:
    """Select a causal position-band offset from fixed training-only gap bins."""
    if specification is None:
        return 0.
    edges = np.asarray(specification["gap_bin_edges_m"], dtype=float)
    offsets = np.asarray(specification["offsets_m"], dtype=float)
    if len(offsets) != len(edges) + 1 or np.any(offsets < 0) or np.any(np.diff(edges) < 0):
        raise ValueError("invalid gap-regime interval-offset specification")
    return float(offsets[np.searchsorted(edges, float(gap_m), side="right")])


def _interval(values: np.ndarray, rng: np.random.Generator, draws: int = 1000) -> list[float]:
    """Cluster (pair) bootstrap interval, retaining correlation between anchors."""
    values = np.asarray(values, float)
    if len(values) < 2:
        return [float(values[0]), float(values[0])]
    means = np.asarray([np.mean(values[rng.integers(0, len(values), len(values))]) for _ in range(draws)])
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def evaluate_full_population(dataset_path: str | Path, posterior_path: str | Path, output_path: str | Path, *,
                             evaluation_recordings: list[int] | None = None, futures: int = 32,
                             prefix_seconds: float = 5., anchor_stride_seconds: float = 10., seed: int = 20260915,
                             anchor_start_fraction: float = 0.,
                             anchor_end_fraction: float = 1., personalize_from_prefix: bool = False,
                             anchor_diagnostics_path: str | Path | None = None,
                             evaluation_vehicle_classes: list[str] | None = None,
                             position_interval_offset_by_gap: dict[str, dict[str, object]] | None = None) -> Path:
    """Evaluate population draws at every rolling origin of every requested pair."""
    _, all_pairs = load_ragged_pairs(dataset_path); posterior = load_posterior(posterior_path)
    model = str(posterior["model"].item()); recordings = None if evaluation_recordings is None else {int(x) for x in evaluation_recordings}
    classes = None if evaluation_vehicle_classes is None else {str(value) for value in evaluation_vehicle_classes}
    pairs = [pair for pair in all_pairs if (recordings is None or int(pair["recording_id"]) in recordings)
             and (classes is None or str(pair["vehicle_class"]) in classes)]
    if not pairs:
        raise RuntimeError("No pairs selected for full evaluation")
    if not 0 <= anchor_start_fraction < anchor_end_fraction <= 1:
        raise ValueError("invalid anchor fractions")
    prefix, stride, horizon = int(prefix_seconds / .04), int(anchor_stride_seconds / .04), int(5. / .04)
    rng = np.random.default_rng(seed)
    per_pair: dict[str, list[dict[str, float]]] = {"3.0": [], "5.0": []}
    shortfalls: dict[str, list[np.ndarray]] = {"3.0": [], "5.0": []}
    anchor_rows: list[tuple[float, ...]] = []
    anchors_total = 0
    for pair in pairs:
        first = max(prefix, int(np.ceil(len(pair["gap"]) * anchor_start_fraction)))
        last = min(len(pair["gap"]) - horizon, int(np.floor(len(pair["gap"]) * anchor_end_fraction)))
        anchors = range(first, last, stride)
        result = {str(seconds): {"pos": [], "speed": [], "crps": [], "coverage": [], "width": [],
                                 "calibration": {str(level): [] for level in CALIBRATION_LEVELS}}
                  for seconds in (3., 5.)}
        for anchor in anchors:
            # The observed prefix is fixed at this rolling origin.  Construct
            # its importance pool once, then generate independent futures from
            # that same conditional joint distribution.  Rebuilding it for
            # every future is both needlessly slow and noisier.
            if personalize_from_prefix:
                conditional_draws, conditional_weight = condition_driver_candidates(
                    posterior, pair, anchor, rng, memory_s=prefix_seconds)
            samples = []
            for _ in range(futures):
                parameters = (conditional_draws[int(rng.choice(len(conditional_draws), p=conditional_weight))]
                              if personalize_from_prefix else sample_driver_joint(posterior, rng))
                samples.append(_rollout(pair, parameters, prefix_index=anchor, horizon_frames=horizon, model=model, rng=rng,
                                        memory_s=5.))
            pos, velocity, acc, _ = (np.asarray([sample[j] for sample in samples]) for j in range(4))
            actual_pos = pair["follower_x"][anchor + 1:anchor + horizon + 1]
            actual_vel = pair["follower_v"][anchor + 1:anchor + horizon + 1]
            actual_acc = np.diff(pair["follower_v"][anchor:anchor + horizon + 1]) / .04
            for seconds in (3., 5.):
                count = int(seconds / .04); low, high = np.quantile(pos[:, :count], [.05, .95], axis=0)
                key = str(seconds); raw_low, raw_high = low.copy(), high.copy()
                shortfalls[key].append(np.maximum(np.maximum(raw_low - actual_pos[:count], actual_pos[:count] - raw_high), 0.))
                offset = 0.
                if position_interval_offset_by_gap is not None:
                    offset += regime_interval_offset(position_interval_offset_by_gap.get(key), float(pair["gap"][anchor]))
                low, high = raw_low - offset, raw_high + offset
                result[key]["pos"].append(float(np.mean((np.mean(pos[:, :count], axis=0) - actual_pos[:count]) ** 2)))
                result[key]["speed"].append(float(np.mean((np.mean(velocity[:, :count], axis=0) - actual_vel[:count]) ** 2)))
                result[key]["crps"].append(crps_ensemble(acc[None, :, :count], actual_acc[None, :count]))
                result[key]["coverage"].append(float(np.mean((actual_pos[:count] >= low) & (actual_pos[:count] <= high))))
                result[key]["width"].append(float(np.mean(high - low)))
                # Report reliability of the raw posterior predictive ensemble
                # at several nominal levels.  This is deliberately measured
                # before any optional conformal band offset.
                for level in CALIBRATION_LEVELS:
                    tail = (1. - level) / 2.
                    level_low, level_high = np.quantile(pos[:, :count], [tail, 1. - tail], axis=0)
                    result[key]["calibration"][str(level)].append(float(np.mean(
                        (actual_pos[:count] >= level_low) & (actual_pos[:count] <= level_high))))
                if anchor_diagnostics_path is not None:
                    mean_error = np.mean(pos[:, :count], axis=0) - actual_pos[:count]
                    anchor_rows.append((float(pair["recording_id"]), float(pair["follower_id"]), seconds,
                                        float(pair["gap"][anchor]), float(pair["follower_v"][anchor]),
                                        float(pair["follower_v"][anchor] - pair["leader_v"][anchor]),
                                        float(np.mean(mean_error)), float(np.sqrt(np.mean(mean_error ** 2))),
                                        float(np.mean((actual_pos[:count] >= low) & (actual_pos[:count] <= high))),
                                        float(np.mean(high - low)), float(np.quantile(shortfalls[key][-1], .90))))
            anchors_total += 1
        for seconds in ("3.0", "5.0"):
            if result[seconds]["pos"]:
                per_pair[seconds].append({"position_rmse_m": float(np.sqrt(np.mean(result[seconds]["pos"]))), "speed_rmse_mps": float(np.sqrt(np.mean(result[seconds]["speed"]))),
                                          "acceleration_crps_mps2": float(np.mean(result[seconds]["crps"])), "position_90_coverage": float(np.mean(result[seconds]["coverage"])), "mean_interval_width_m": float(np.mean(result[seconds]["width"])),
                                          "position_interval_calibration": {level: float(np.mean(values)) for level, values in result[seconds]["calibration"].items()}})
    prefix_likelihood = ("not used" if not personalize_from_prefix else
                         "paper Gaussian SE-GP + iid action covariance")
    report: dict[str, object] = {"model": model, "fit_backend": str(posterior["fit_backend"].item()), "evaluation": "in-sample population rolling-origin prediction" if recordings is None else "recording-held-out rolling-origin population prediction", "native_fps": 25,
                                 "evaluation_recordings": sorted(recordings) if recordings is not None else "all pairs (in-sample)",
                                 "evaluation_vehicle_classes": sorted(classes) if classes is not None else "all classes", "pairs": len(pairs), "anchors": anchors_total, "futures": futures, "prefix_s": prefix_seconds, "anchor_stride_s": anchor_stride_seconds, "anchor_start_fraction": anchor_start_fraction, "anchor_end_fraction": anchor_end_fraction, "personalize_from_observed_prefix": personalize_from_prefix, "prefix_parameter_likelihood": prefix_likelihood, "horizons": {}}
    report["position_interval_offset_by_gap"] = position_interval_offset_by_gap
    bootstrap_rng = np.random.default_rng(seed + 1)
    for seconds, rows in per_pair.items():
        report["horizons"][seconds] = {key: {"mean": float(np.mean([row[key] for row in rows])), "cluster_bootstrap_95": _interval(np.asarray([row[key] for row in rows]), bootstrap_rng)}
                                       for key in rows[0] if key != "position_interval_calibration"}
        report["horizons"][seconds]["position_interval_calibration"] = {
            level: {"mean": float(np.mean([row["position_interval_calibration"][level] for row in rows])),
                    "cluster_bootstrap_95": _interval(np.asarray([row["position_interval_calibration"][level] for row in rows]), bootstrap_rng)}
            for level in map(str, CALIBRATION_LEVELS)}
        report["horizons"][seconds]["position_90_shortfall_q90_m"] = float(np.quantile(np.concatenate(shortfalls[seconds]), .90))
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if anchor_diagnostics_path is not None:
        diagnostic = Path(anchor_diagnostics_path); diagnostic.parent.mkdir(parents=True, exist_ok=True)
        names = np.asarray(("recording_id", "follower_id", "horizon_s", "gap_m", "speed_mps", "closing_speed_mps",
                            "position_mean_error_m", "position_rmse_m", "position_90_coverage", "interval_width_m",
                            "position_90_shortfall_q90_m"))
        np.savez_compressed(diagnostic, columns=names, values=np.asarray(anchor_rows, dtype=float),
                            protocol=np.asarray("per-origin OOF diagnostics; no future states enter rollout"))
    return path
