#!/usr/bin/env python3
"""Render paper-style held-out trajectory envelopes and a transparent verdict."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--evaluation-dir", default=str(ROOT / "multi_regime_bidm/evidence/heldout"))
    a = p.parse_args(); directory = Path(a.evaluation_dir)
    values = np.load(directory / "recording_heldout_evaluation.npz", allow_pickle=False)
    actual, filtered, pooled = values["actual_position"], values["filtered_position"], values["pooled_position"]
    time = (np.arange(actual.shape[1]) + 1) * .04; choice = np.linspace(0, len(actual) - 1, min(3, len(actual)), dtype=int)
    fig, axes = plt.subplots(1, len(choice), figsize=(4.5 * len(choice), 3.4), sharey=False)
    axes = np.atleast_1d(axes)
    for axis, row in zip(axes, choice):
        for samples, color, name in ((pooled[row], "#e69f00", "pooled B-IDM"), (filtered[row], "#0072b2", "filtered regime B-IDM")):
            low, high = np.quantile(samples, (.05, .95), axis=0)
            axis.fill_between(time, low, high, color=color, alpha=.16)
            axis.plot(time, np.mean(samples, axis=0), color=color, lw=1.3, label=name)
        axis.plot(time, actual[row], color="black", lw=1.5, label="highD observed")
        axis.set(xlabel="forecast time [s]", ylabel="follower x [m]", title=f"held-out event {row}, prefix style {values['style_from_prefix'][row]}")
    axes[0].legend(fontsize=7); fig.suptitle("25 Hz recording-held-out stochastic forecasts (90% bands)", y=1.03)
    fig.tight_layout(); figures = directory / "figures"; figures.mkdir(exist_ok=True)
    fig.savefig(figures / "fig_recording_heldout_position_envelopes.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    metrics = json.loads((directory / "recording_heldout_metrics.json").read_text(encoding="utf-8"))
    filtered_metrics = metrics["filtered_hierarchical"]
    pooled_metrics = metrics["pooled_b_idm"]
    checks = {
        "filtered_CRPS_better_than_pooled": filtered_metrics["acceleration_crps_mps2"] < pooled_metrics["acceleration_crps_mps2"],
        "filtered_position_RMSE_no_worse_than_pooled": filtered_metrics["position_rmse_m"] <= pooled_metrics["position_rmse_m"],
        "empirical_90pct_position_coverage_at_least_0_8": filtered_metrics["position_90_coverage"] >= .8,
    }
    accepted = all(checks.values())
    audit = {"accepted_calibrated_stochastic_driver": bool(accepted),
             "decision": "accepted" if accepted else "rejected_not_well_calibrated",
             "criteria": {"filtered_CRPS_better_than_pooled": True, "filtered_position_RMSE_no_worse_than_pooled": True,
                          "empirical_90pct_position_coverage_at_least": .8},
             "checks": checks,
             "observed": {"filtered_acceleration_crps_mps2": filtered_metrics["acceleration_crps_mps2"],
                          "pooled_acceleration_crps_mps2": pooled_metrics["acceleration_crps_mps2"],
                          "filtered_position_rmse_m": filtered_metrics["position_rmse_m"],
                          "pooled_position_rmse_m": pooled_metrics["position_rmse_m"],
                          "filtered_position_90_coverage": filtered_metrics["position_90_coverage"]},
             "scope": "36 held-out events from recordings 26 and 36; insufficient for population-significance claims"}
    (directory / "recording_heldout_acceptance_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
