#!/usr/bin/env python3
"""Measure the reconstruction ceiling of sparse long-horizon constraints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.evaluation import states_from_actions  # noqa: E402
from tools.plot_style import (
    GENERATED_COLOR,
    REAL_COLOR,
    get_pyplot,
    style_axes,
)  # noqa: E402
from world_model.src.core.sequential_dataset import (
    load_sequential_dataset,
)  # noqa: E402
from world_model.src.core.utils import ensure_dir, load_yaml, save_json  # noqa: E402

CONFIG = ROOT / "diffusion/configs/highd_background_diffusion.yaml"
OUTPUT = ROOT / "results/background_diffusion/design_audit"
DT_S = 0.04
FRAMES = 150


def _interpolate(
    positions: np.ndarray,
    velocities: np.ndarray,
    knot_indices: np.ndarray,
    method: str,
) -> np.ndarray:
    time = np.arange(FRAMES, dtype=np.float64) * DT_S
    knot_time = time[knot_indices]
    values = positions[:, knot_indices].transpose(1, 0, 2, 3)
    if method == "linear":
        flat = values.reshape(len(knot_indices), -1)
        output = np.stack(
            [
                np.interp(time, knot_time, flat[:, index])
                for index in range(flat.shape[1])
            ],
            axis=1,
        ).reshape(FRAMES, len(positions), 6, 2)
    elif method == "cubic":
        output = CubicSpline(knot_time, values, axis=0)(time)
    elif method == "hermite":
        derivatives = velocities[:, knot_indices].transpose(1, 0, 2, 3)
        output = CubicHermiteSpline(knot_time, values, derivatives, axis=0)(time)
    else:
        raise ValueError(f"unknown interpolation method {method!r}")
    return output.transpose(1, 0, 2, 3).astype(np.float32)


def _accumulate(
    prediction: np.ndarray,
    target: np.ndarray,
    active: np.ndarray,
    totals: dict[str, float],
) -> None:
    error = np.linalg.norm(prediction[:, 1:] - target[:, 1:], axis=-1)
    mask = active[:, None]
    totals["absolute_error_sum"] += float((error * mask).sum())
    totals["absolute_error_count"] += int(active.sum()) * (FRAMES - 1)
    totals["final_error_sum"] += float((error[:, -1] * active).sum())
    totals["final_error_count"] += int(active.sum())
    totals["one_second_error_sum"] += float((error[:, 24::25] * mask).sum())
    totals["one_second_error_count"] += int(active.sum()) * error[:, 24::25].shape[1]


def _metrics(totals: dict[str, float]) -> dict[str, float]:
    return {
        "ADE_m": totals["absolute_error_sum"] / totals["absolute_error_count"],
        "FDE_m": totals["final_error_sum"] / totals["final_error_count"],
        "one_second_grid_MAE_m": (
            totals["one_second_error_sum"] / totals["one_second_error_count"]
        ),
    }


def _new_totals() -> dict[str, float]:
    return {
        "absolute_error_sum": 0.0,
        "absolute_error_count": 0,
        "final_error_sum": 0.0,
        "final_error_count": 0,
        "one_second_error_sum": 0.0,
        "one_second_error_count": 0,
    }


def main() -> None:
    config = load_yaml(CONFIG)
    arrays, manifest = load_sequential_dataset(config["paths"]["sequence_cache_dir"])
    rows = np.flatnonzero(np.asarray(arrays["split_index"]) == 2)
    schedules = {
        "position_2s": np.asarray((0, 50, 100, 149)),
        "position_1s": np.asarray((0, 25, 50, 75, 100, 125, 149)),
        "position_0p5s": np.unique(np.r_[np.arange(0, 150, 12), 149]),
    }
    methods = ("linear", "cubic", "hermite")
    totals = {
        "logged_action_reintegration": _new_totals(),
        **{
            f"{schedule}_{method}": _new_totals()
            for schedule in schedules
            for method in methods
        },
    }
    for start in range(0, len(rows), 256):
        take = rows[start : start + 256]
        states = np.asarray(arrays["agent_states"][take, 24:174, 1:], np.float32)
        active = np.asarray(arrays["agent_valid"][take, 24, 1:], bool)
        positions = states[..., :2] - states[:, :1, :, :2]
        velocities = states[..., 2:4]
        reintegrated = states_from_actions(
            states[:, 0], np.asarray(arrays["actions_highd"][take], np.float32)
        )[..., :2]
        reintegrated = np.concatenate(
            (np.zeros_like(reintegrated[:, :1]), reintegrated - states[:, :1, :, :2]),
            axis=1,
        )
        _accumulate(
            reintegrated, positions, active, totals["logged_action_reintegration"]
        )
        for schedule, knot_indices in schedules.items():
            for method in methods:
                prediction = _interpolate(positions, velocities, knot_indices, method)
                _accumulate(
                    prediction, positions, active, totals[f"{schedule}_{method}"]
                )
    report = {
        "experiment_scope": "held_out_constraint_ceiling_audit",
        "test_sequences": int(len(rows)),
        "recording_level_split": True,
        "historical_target": {"ADE_m": 0.059, "FDE_m": 0.103},
        "conditions_use_future_ego": False,
        "metrics": {name: _metrics(value) for name, value in totals.items()},
        "constraint_dimensions": {
            name: int(6 * 2 * (len(indices) - 1)) for name, indices in schedules.items()
        },
        "source_manifest": manifest,
    }
    output = ensure_dir(OUTPUT)
    save_json(report, output / "constraint_ceiling.json")
    plt = get_pyplot()
    names = [
        name
        for name in report["metrics"]
        if name.endswith(("_linear", "_cubic", "_hermite"))
    ]
    ade = [report["metrics"][name]["ADE_m"] for name in names]
    figure, axis = plt.subplots(figsize=(12.0, 5.2))
    colors = [GENERATED_COLOR if "hermite" in name else REAL_COLOR for name in names]
    axis.bar(np.arange(len(names)), ade, color=colors)
    axis.axhline(0.059, color="#B22222", linestyle="--", label="historical ADE target")
    axis.set_xticks(
        np.arange(len(names)),
        [name.replace("position_", "").replace("_", "\n") for name in names],
    )
    axis.set_ylabel("Oracle-reference ADE (m)")
    axis.set_title("Reconstruction ceiling of sparse trajectory constraints")
    axis.legend(frameon=False)
    style_axes(axis)
    figure.tight_layout()
    figure.savefig(output / "constraint_ceiling.png", dpi=300)
    plt.close(figure)
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
