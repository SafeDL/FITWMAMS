#!/usr/bin/env python3
"""Draw posterior intervals from the retained hierarchical Stage-B NUTS fit."""
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-b", default=str(ROOT / "multi_regime_bidm/evidence/posterior"))
    args = parser.parse_args(); stage = Path(args.stage_b)
    names = ("v0 [m/s]", "s0 [m]", "T [s]", "alpha [m/s²]", "beta [m/s²]")
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.3))
    for style in range(3):
        draw = np.load(stage / f"style_{style}_nuts_posterior.npz", allow_pickle=False)["theta_draws"]
        for parameter, axis in enumerate(axes):
            median = np.median(draw[:, :, parameter], axis=0)
            low, high = np.quantile(draw[:, :, parameter], (.05, .95), axis=0)
            x = np.arange(3) + (style - 1) * .12
            axis.errorbar(x, median, yerr=(median - low, high - median), fmt="o", capsize=2,
                          label=f"style {style}" if parameter == 0 else None)
            axis.set(xlabel="offline regime", ylabel=names[parameter], xticks=(0, 1, 2))
            if parameter in (2, 3):
                axis.set_yscale("log")
    axes[0].legend(fontsize=8)
    fig.suptitle("Stage-B hierarchical NUTS: 90% posterior intervals (offline labels; evaluation only)", y=1.04)
    fig.tight_layout(); output = stage / "figures"; output.mkdir(exist_ok=True)
    fig.savefig(output / "fig_stage_b_nuts_posterior_intervals.png", dpi=180, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
