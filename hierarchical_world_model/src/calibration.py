"""Matched natural-response calibration from recording-isolated highD splits."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def _condition_key(
    gap: float,
    relative_speed: float,
    ttc: float,
    background_acceleration: float,
) -> tuple[int, int, int, int]:
    return (
        int(np.floor(gap / 10.0)),
        int(np.floor((relative_speed + 20.0) / 2.0)),
        int(np.floor(min(ttc, 10.0) / 2.0)),
        int(np.floor((background_acceleration + 8.0) / 0.5)),
    )


@dataclass(frozen=True)
class NaturalResponseCalibrator:
    """Training-only conditional P10/P90 response targets.

    Sparse highD matching cells fall back to the split-level human-response
    interval.  The table is fitted on the training recordings only, then used
    to supervise simulated ego interventions at the same geometric state.
    """

    global_bounds: np.ndarray
    conditional_bounds: tuple[dict[tuple[int, int, int, int], np.ndarray], ...]
    global_sensitivity_bounds: np.ndarray | None = None

    def bounds_for(
        self,
        current: np.ndarray,
        valid: np.ndarray,
    ) -> np.ndarray:
        """Return `[brake/accelerate, slot, P10/P90]` targets."""
        state = np.asarray(current, np.float32)
        present = np.asarray(valid, bool)
        if state.shape != (7, 6) or present.shape != (7,):
            raise ValueError("current/valid must be [7,6]/[7]")
        result = np.broadcast_to(
            self.global_bounds[:, None, :], (2, 6, 2)
        ).copy()
        ego = state[0]
        for slot in range(6):
            if not present[slot + 1]:
                result[:, slot] = 0.0
                continue
            background = state[slot + 1]
            gap = max(float(ego[0] - background[0] - 4.8), 0.0)
            relative_speed = float(background[2] - ego[2])
            closing = max(relative_speed, 0.0)
            ttc = 10.0 if closing <= 1.0e-3 else min(gap / closing, 10.0)
            key = _condition_key(gap, relative_speed, ttc, float(background[4]))
            for kind, table in enumerate(self.conditional_bounds):
                values = table.get(key)
                if values is not None:
                    result[kind, slot] = values
        return result

    def sensitivity_bounds_for(
        self,
        current: np.ndarray,
        valid: np.ndarray,
    ) -> np.ndarray:
        """Return signed-effect bounds per 1 m/s² ego-control change.

        The current implementation keeps this intentionally split-global:
        dose samples are much sparser than the existing geometry-matched
        effect table, and a cell-wise estimate would be overconfident.
        """
        state = np.asarray(current, np.float32)
        present = np.asarray(valid, bool)
        if state.shape != (7, 6) or present.shape != (7,):
            raise ValueError("current/valid must be [7,6]/[7]")
        source = (
            self.global_bounds
            if self.global_sensitivity_bounds is None
            else self.global_sensitivity_bounds
        )
        result = np.broadcast_to(source[:, None, :], (2, 6, 2)).copy()
        result[:, ~present[1:]] = 0.0
        return result


def _serialized_key(key: tuple[int, int, int, int]) -> str:
    return "/".join(str(value) for value in key)


def _parse_key(value: str) -> tuple[int, int, int, int]:
    parts = tuple(int(part) for part in value.split("/"))
    if len(parts) != 4:
        raise ValueError(f"invalid response-calibration key {value!r}")
    return parts


def matched_response_calibration(
    arrays: dict[str, np.ndarray],
    rows: np.ndarray,
    *,
    dt_s: float = 0.04,
    minimum_events: int = 100,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate signed acceleration effects after matched ego action events."""
    states = np.asarray(arrays["agent_states"])[np.asarray(rows, np.int64)]
    valid = np.asarray(arrays["agent_valid"])[np.asarray(rows, np.int64)]
    matched_by_horizon: dict[float, dict[str, np.ndarray]] = {}
    sensitivity_by_horizon: dict[float, dict[str, np.ndarray]] = {}
    conditional_by_horizon: dict[
        float, dict[str, dict[tuple[int, int, int, int], np.ndarray]]
    ] = {}
    for horizon_frames in (5, 10, 20):
        neutral: dict[tuple[int, int, int, int], list[float]] = defaultdict(list)
        events: dict[
            str, list[tuple[tuple[int, int, int, int], float, float]]
        ] = {
            "brake": [],
            "accelerate": [],
            "lane_change": [],
        }
        for response in range(0, 145 - horizon_frames):
            index = 24 + response
            future = index + horizon_frames
            current = states[:, index]
            later = states[:, future]
            present = valid[:, index] & valid[:, future]
            background_acceleration = (
                later[:, 1:, 2] - current[:, 1:, 2]
            ) / (horizon_frames * dt_s)
            dx = current[:, 1:, 0] - current[:, :1, 0]
            dy = np.abs(current[:, 1:, 1] - current[:, :1, 1])
            following = present[:, 1:] & present[:, :1] & (dx < 0.0) & (dy < 1.8)
            gap = np.maximum(-dx - 4.8, 0.0)
            relative_speed = current[:, 1:, 2] - current[:, :1, 2]
            closing = np.maximum(relative_speed, 0.0)
            ttc = np.full_like(gap, 10.0)
            np.divide(gap, closing, out=ttc, where=closing > 1.0e-3)
            previous_vy = states[:, index - 1, 0, 3]
            current_vy = current[:, 0, 3]
            lane_start = (np.abs(previous_vy) < 0.1) & (np.abs(current_vy) >= 0.1)
            previous_action = states[:, index - 1, 0, 4]
            current_action = current[:, 0, 4]
            brake_start = (previous_action > -0.75) & (current_action <= -0.75)
            accelerate_start = (previous_action < 0.75) & (current_action >= 0.75)
            neutral_action = (np.abs(previous_action) < 0.25) & (
                np.abs(current_action) < 0.25
            )
            for sequence, slot in zip(*np.nonzero(following)):
                key = _condition_key(
                    float(gap[sequence, slot]),
                    float(relative_speed[sequence, slot]),
                    float(ttc[sequence, slot]),
                    float(current[sequence, slot + 1, 4]),
                )
                outcome = float(background_acceleration[sequence, slot])
                if lane_start[sequence]:
                    events["lane_change"].append((key, outcome, 1.0))
                elif brake_start[sequence]:
                    events["brake"].append(
                        (key, outcome, max(abs(float(current_action[sequence] - previous_action[sequence])), 0.5))
                    )
                elif accelerate_start[sequence]:
                    events["accelerate"].append(
                        (key, outcome, max(abs(float(current_action[sequence] - previous_action[sequence])), 0.5))
                    )
                elif neutral_action[sequence]:
                    neutral[key].append(outcome)
        matched: dict[str, np.ndarray] = {}
        conditional: dict[
            str, dict[tuple[int, int, int, int], list[float]]
        ] = {name: defaultdict(list) for name in events}
        sensitivity: dict[str, np.ndarray] = {}
        for name, values in events.items():
            sign = {"brake": -1.0, "accelerate": 1.0}.get(name)
            differences = [
                (key, outcome - float(np.median(neutral[key])), dose)
                for key, outcome, dose in values
                if len(neutral[key]) >= 5
            ]
            if sign is None:
                signed = [(key, abs(effect), dose) for key, effect, dose in differences]
            else:
                signed = [(key, sign * effect, dose) for key, effect, dose in differences]
            positive = np.asarray([effect for _, effect, _ in signed], np.float32)
            positive = positive[np.isfinite(positive) & (positive >= 0.0)]
            if len(positive) < int(minimum_events):
                raise RuntimeError(
                    f"only {len(positive)} matched {name} responses at "
                    f"{horizon_frames * dt_s:.1f} s"
                )
            matched[name] = positive
            if name in {"brake", "accelerate"}:
                sensitivity[name] = np.asarray(
                    [
                        effect / dose
                        for _, effect, dose in signed
                        if np.isfinite(effect) and effect >= 0.0
                    ],
                    np.float32,
                )
            for key, effect, _ in signed:
                if np.isfinite(effect) and effect >= 0.0:
                    conditional[name][key].append(float(effect))
        matched_by_horizon[horizon_frames * dt_s] = matched
        sensitivity_by_horizon[horizon_frames * dt_s] = sensitivity
        conditional_by_horizon[horizon_frames * dt_s] = {
            name: {
                key: np.asarray(values, np.float32)
                for key, values in grouped.items()
            }
            for name, grouped in conditional.items()
        }
    matched = matched_by_horizon[0.8]
    bounds = np.asarray(
        [
            np.quantile(matched[name], (0.10, 0.90))
            for name in ("brake", "accelerate")
        ],
        np.float32,
    )
    sensitivity_bounds = np.asarray(
        [
            np.quantile(sensitivity_by_horizon[0.8][name], (0.10, 0.90))
            for name in ("brake", "accelerate")
        ],
        np.float32,
    )
    conditional_minimum = max(5, int(minimum_events) // 20)
    conditional_tables = {
        name: {
            _serialized_key(key): np.quantile(values, (0.10, 0.90)).tolist()
            for key, values in conditional_by_horizon[0.8][name].items()
            if len(values) >= conditional_minimum
        }
        for name in ("brake", "accelerate")
    }
    report = {
        "method": "exact_binned_matching_on_gap_relative_speed_ttc_lane_and_bg_action",
        "horizons_s": [0.2, 0.4, 0.8],
        "split_sequences": int(len(rows)),
        "brake": {
            "matched_events": int(len(matched["brake"])),
            "effect_p10_p50_p90_mps2": np.quantile(
                matched["brake"], (0.1, 0.5, 0.9)
            ).tolist(),
            "effect_samples_mps2": matched["brake"].astype(float).tolist(),
        },
        "accelerate": {
            "matched_events": int(len(matched["accelerate"])),
            "effect_p10_p50_p90_mps2": np.quantile(
                matched["accelerate"], (0.1, 0.5, 0.9)
            ).tolist(),
            "effect_samples_mps2": matched["accelerate"].astype(float).tolist(),
        },
        "dose_sensitivity_p10_p90_per_mps2": sensitivity_bounds.tolist(),
        "lane_change": {
            "matched_events": int(len(matched["lane_change"])),
            "effect_p10_p50_p90_mps2": np.quantile(
                matched["lane_change"], (0.1, 0.5, 0.9)
            ).tolist(),
        },
        "horizon_diagnostics": {
            f"{horizon:.1f}s": {
                name: {
                    "matched_events": int(len(values)),
                    "effect_p10_p50_p90_mps2": np.quantile(
                        values, (0.1, 0.5, 0.9)
                    ).tolist(),
                }
                for name, values in responses.items()
            }
            for horizon, responses in matched_by_horizon.items()
        },
        "conditional_targets": {
            "horizon_s": 0.8,
            "minimum_events_per_cell": conditional_minimum,
            "brake": conditional_tables["brake"],
            "accelerate": conditional_tables["accelerate"],
        },
    }
    return bounds, report


def propensity_matched_response_calibration(
    arrays: dict[str, np.ndarray],
    rows: np.ndarray,
    *,
    dt_s: float = 0.04,
    minimum_events: int = 100,
    neighbors: int = 16,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate natural effects with continuous covariate nearest matching.

    Unlike an exact-bin estimator, this treats every sustained ego
    acceleration/braking frame as a candidate and matches it to nearby neutral
    frames in gap, relative speed, TTC and background acceleration.  It is
    training-split-only and deliberately exposes match-distance diagnostics.
    """
    states = np.asarray(arrays["agent_states"])[np.asarray(rows, np.int64)]
    valid = np.asarray(arrays["agent_valid"])[np.asarray(rows, np.int64)]
    matched_by_horizon: dict[float, dict[str, np.ndarray]] = {}
    sensitivity_by_horizon: dict[float, dict[str, np.ndarray]] = {}
    diagnostics: dict[str, Any] = {}
    for horizon_frames in (5, 10, 20):
        features: list[np.ndarray] = []
        outcomes: list[np.ndarray] = []
        ego_actions: list[np.ndarray] = []
        for response in range(0, 145 - horizon_frames):
            index = 24 + response
            current = states[:, index]
            later = states[:, index + horizon_frames]
            present = valid[:, index] & valid[:, index + horizon_frames]
            dx = current[:, 1:, 0] - current[:, :1, 0]
            dy = np.abs(current[:, 1:, 1] - current[:, :1, 1])
            following = present[:, 1:] & present[:, :1] & (dx < 0.0) & (dy < 1.8)
            if not following.any():
                continue
            gap = np.maximum(-dx - 4.8, 0.0)
            relative_speed = current[:, 1:, 2] - current[:, :1, 2]
            closing = np.maximum(relative_speed, 0.0)
            ttc = np.full_like(gap, 10.0)
            np.divide(gap, closing, out=ttc, where=closing > 1.0e-3)
            outcome = (
                later[:, 1:, 2] - current[:, 1:, 2]
            ) / (horizon_frames * dt_s)
            features.append(
                np.stack(
                    (
                        gap[following] / 30.0,
                        relative_speed[following] / 10.0,
                        ttc[following] / 5.0,
                        current[:, 1:, 4][following] / 2.0,
                    ),
                    axis=-1,
                ).astype(np.float32)
            )
            outcomes.append(outcome[following].astype(np.float32))
            ego_actions.append(
                np.broadcast_to(current[:, :1, 4], dx.shape)[following].astype(
                    np.float32
                )
            )
        feature = np.concatenate(features)
        outcome = np.concatenate(outcomes)
        ego_action = np.concatenate(ego_actions)
        neutral = np.abs(ego_action) < 0.25
        if neutral.sum() < int(neighbors):
            raise RuntimeError("not enough neutral frames for propensity matching")
        tree = cKDTree(feature[neutral])
        response_effect: dict[str, np.ndarray] = {}
        sensitivity: dict[str, np.ndarray] = {}
        horizon_report: dict[str, Any] = {}
        for name, select, sign in (
            ("brake", ego_action <= -0.75, -1.0),
            ("accelerate", ego_action >= 0.75, 1.0),
        ):
            if select.sum() < int(minimum_events):
                raise RuntimeError(
                    f"only {int(select.sum())} {name} treatment frames at "
                    f"{horizon_frames * dt_s:.1f} s"
                )
            distance, indices = tree.query(
                feature[select], k=min(int(neighbors), int(neutral.sum()))
            )
            matched_control = np.median(outcome[neutral][indices], axis=-1)
            effect = sign * (outcome[select] - matched_control)
            keep = np.isfinite(effect) & (effect >= 0.0)
            positive = effect[keep].astype(np.float32)
            if len(positive) < int(minimum_events):
                raise RuntimeError(
                    f"only {len(positive)} positive matched {name} effects at "
                    f"{horizon_frames * dt_s:.1f} s"
                )
            response_effect[name] = positive
            sensitivity[name] = (
                positive / np.abs(ego_action[select][keep]).clip(min=0.75)
            ).astype(np.float32)
            horizon_report[name] = {
                "treatment_frames": int(select.sum()),
                "positive_matched_effects": int(len(positive)),
                "match_distance_p50_p90": np.quantile(
                    np.asarray(distance).reshape(-1), (0.5, 0.9)
                ).tolist(),
                "effect_p10_p50_p90_mps2": np.quantile(
                    positive, (0.1, 0.5, 0.9)
                ).tolist(),
            }
        horizon_s = horizon_frames * dt_s
        matched_by_horizon[horizon_s] = response_effect
        sensitivity_by_horizon[horizon_s] = sensitivity
        diagnostics[f"{horizon_s:.1f}s"] = horizon_report
    bounds = np.asarray(
        [np.quantile(matched_by_horizon[0.8][name], (0.1, 0.9)) for name in ("brake", "accelerate")],
        np.float32,
    )
    sensitivity_bounds = np.asarray(
        [np.quantile(sensitivity_by_horizon[0.8][name], (0.1, 0.9)) for name in ("brake", "accelerate")],
        np.float32,
    )
    report = {
        "method": "continuous_covariate_knn_propensity_matching",
        "neighbors": int(neighbors),
        "horizons_s": [0.2, 0.4, 0.8],
        "split_sequences": int(len(rows)),
        "brake": {
            "matched_events": int(len(matched_by_horizon[0.8]["brake"])),
            "effect_p10_p50_p90_mps2": np.quantile(
                matched_by_horizon[0.8]["brake"], (0.1, 0.5, 0.9)
            ).tolist(),
            "effect_samples_mps2": matched_by_horizon[0.8]["brake"].astype(float).tolist(),
        },
        "accelerate": {
            "matched_events": int(len(matched_by_horizon[0.8]["accelerate"])),
            "effect_p10_p50_p90_mps2": np.quantile(
                matched_by_horizon[0.8]["accelerate"], (0.1, 0.5, 0.9)
            ).tolist(),
            "effect_samples_mps2": matched_by_horizon[0.8]["accelerate"].astype(float).tolist(),
        },
        "dose_sensitivity_p10_p90_per_mps2": sensitivity_bounds.tolist(),
        "horizon_diagnostics": {
            f"{horizon:.1f}s": {
                name: {
                    "matched_events": int(len(values)),
                    "effect_p10_p50_p90_mps2": np.quantile(
                        values, (0.1, 0.5, 0.9)
                    ).tolist(),
                }
                for name, values in responses.items()
            }
            for horizon, responses in matched_by_horizon.items()
        },
        "propensity_match_diagnostics": diagnostics,
        "conditional_targets": {
            "horizon_s": 0.8,
            "minimum_events_per_cell": 0,
            "brake": {},
            "accelerate": {},
        },
    }
    return bounds, report


def fit_natural_response_calibrator(
    arrays: dict[str, np.ndarray],
    rows: np.ndarray,
    *,
    dt_s: float = 0.04,
    minimum_events: int = 100,
    method: str = "exact",
) -> tuple[NaturalResponseCalibrator, dict[str, Any]]:
    """Fit split-isolated response targets for the reactive training loss."""
    if method == "exact":
        bounds, report = matched_response_calibration(
            arrays, rows, dt_s=dt_s, minimum_events=minimum_events
        )
    elif method == "propensity":
        bounds, report = propensity_matched_response_calibration(
            arrays, rows, dt_s=dt_s, minimum_events=minimum_events
        )
    else:
        raise ValueError("response calibration method must be 'exact' or 'propensity'")
    targets = report["conditional_targets"]
    tables = tuple(
        {
            _parse_key(key): np.asarray(value, np.float32)
            for key, value in targets[name].items()
        }
        for name in ("brake", "accelerate")
    )
    return NaturalResponseCalibrator(
        np.asarray(bounds, np.float32), tables, np.asarray(
            report["dose_sensitivity_p10_p90_per_mps2"], np.float32
        )
    ), report
