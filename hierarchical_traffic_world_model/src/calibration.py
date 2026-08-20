"""Matched natural-response calibration from recording-isolated highD splits."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np


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
    conditional_by_horizon: dict[
        float, dict[str, dict[tuple[int, int, int, int], np.ndarray]]
    ] = {}
    for horizon_frames in (5, 10, 20):
        neutral: dict[tuple[int, int, int, int], list[float]] = defaultdict(list)
        events: dict[str, list[tuple[tuple[int, int, int, int], float]]] = {
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
                    events["lane_change"].append((key, outcome))
                elif brake_start[sequence]:
                    events["brake"].append((key, outcome))
                elif accelerate_start[sequence]:
                    events["accelerate"].append((key, outcome))
                elif neutral_action[sequence]:
                    neutral[key].append(outcome)
        matched: dict[str, np.ndarray] = {}
        conditional: dict[
            str, dict[tuple[int, int, int, int], list[float]]
        ] = {name: defaultdict(list) for name in events}
        for name, values in events.items():
            sign = {"brake": -1.0, "accelerate": 1.0}.get(name)
            differences = [
                (key, outcome - float(np.median(neutral[key])) )
                for key, outcome in values
                if len(neutral[key]) >= 5
            ]
            if sign is None:
                signed = [(key, abs(effect)) for key, effect in differences]
            else:
                signed = [(key, sign * effect) for key, effect in differences]
            positive = np.asarray([effect for _, effect in signed], np.float32)
            positive = positive[np.isfinite(positive) & (positive >= 0.0)]
            if len(positive) < int(minimum_events):
                raise RuntimeError(
                    f"only {len(positive)} matched {name} responses at "
                    f"{horizon_frames * dt_s:.1f} s"
                )
            matched[name] = positive
            for key, effect in signed:
                if np.isfinite(effect) and effect >= 0.0:
                    conditional[name][key].append(float(effect))
        matched_by_horizon[horizon_frames * dt_s] = matched
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


def fit_natural_response_calibrator(
    arrays: dict[str, np.ndarray],
    rows: np.ndarray,
    *,
    dt_s: float = 0.04,
    minimum_events: int = 100,
) -> tuple[NaturalResponseCalibrator, dict[str, Any]]:
    """Fit split-isolated response targets for the reactive training loss."""
    bounds, report = matched_response_calibration(
        arrays, rows, dt_s=dt_s, minimum_events=minimum_events
    )
    targets = report["conditional_targets"]
    tables = tuple(
        {
            _parse_key(key): np.asarray(value, np.float32)
            for key, value in targets[name].items()
        }
        for name in ("brake", "accelerate")
    )
    return NaturalResponseCalibrator(np.asarray(bounds, np.float32), tables), report
