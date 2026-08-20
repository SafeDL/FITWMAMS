#!/usr/bin/env python3
"""Audit smooth position-residual bases before training a generator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import ks_2samp, wasserstein_distance

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import (  # noqa: E402
    extract_trajectory_constraint,
    load_data_bundle,
    pilot_rows,
    trajectory_reference_positions,
)
from world_model.src.core.utils import load_yaml, save_json  # noqa: E402

CONFIG = ROOT / "diffusion/configs/highd_background_diffusion.yaml"
OUTPUT = ROOT / "results/background_diffusion/design_audit/motion_basis.json"


def main() -> None:
    config = load_yaml(CONFIG)
    bundle = load_data_bundle(config, CONFIG.parent)
    rows = pilot_rows(bundle, "test", maximum=1024, seed=20260812)
    states = np.asarray(bundle.arrays["agent_states"][rows, 24:174, 1:], np.float64)
    active = np.asarray(bundle.arrays["agent_valid"][rows, 24, 1:], bool)
    constraint = extract_trajectory_constraint(states.astype(np.float32))
    reference = trajectory_reference_positions(
        states[:, 0].astype(np.float32), constraint
    ).astype(np.float64)
    target = states[:, 1:, :, :2]
    residual = target - reference
    logged = np.asarray(bundle.arrays["actions_highd"][rows], np.float64)
    action_mask = np.broadcast_to(active[:, None, :, None], logged.shape)
    metrics = {}
    for window in (9, 13, 17, 21, 25, 31, 41, 51):
        smooth = savgol_filter(residual, window, 3, axis=1, mode="interp")
        positions = reference + smooth
        error = np.linalg.norm(positions - target, axis=-1)
        acceleration = savgol_filter(
            positions, window, 3, deriv=2, delta=0.04, axis=1, mode="interp"
        )
        jerk = np.diff(acceleration, axis=1) / 0.04
        generated_mask = np.broadcast_to(active[:, None, :, None], acceleration.shape)
        jerk_mask = np.broadcast_to(active[:, None, :, None], jerk.shape)
        fidelity = {}
        for axis, name in enumerate(("ax", "ay")):
            generated = acceleration[..., axis][generated_mask[..., axis]]
            observed = logged[..., axis][action_mask[..., axis]]
            fidelity[name] = {
                "KS": float(ks_2samp(generated, observed).statistic),
                "wasserstein": float(wasserstein_distance(generated, observed)),
            }
        metrics[str(window)] = {
            "ADE_m": float((error * active[:, None]).sum() / (active.sum() * 149)),
            "FDE_m": float((error[:, -1] * active).sum() / active.sum()),
            "action_fidelity": fidelity,
            "ax_outside_minus8_4_rate": float(
                ((acceleration[..., 0] < -8) | (acceleration[..., 0] > 4))[
                    generated_mask[..., 0]
                ].mean()
            ),
            "abs_ay_above4_rate": float(
                (np.abs(acceleration[..., 1]) > 4)[generated_mask[..., 1]].mean()
            ),
            "abs_jx_above12_rate": float(
                (np.abs(jerk[..., 0]) > 12)[jerk_mask[..., 0]].mean()
            ),
            "abs_jy_above8_rate": float(
                (np.abs(jerk[..., 1]) > 8)[jerk_mask[..., 1]].mean()
            ),
        }
    report = {"test_sequences": len(rows), "metrics": metrics}
    save_json(report, OUTPUT)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
