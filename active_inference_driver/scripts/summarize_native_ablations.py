"""Summarize the two required small-budget native rear-end ablations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTCOME_NAMES = {
    0: "collision", 1: "left road", 2: "steer before", 3: "steer after",
    4: "brake before", 5: "brake after", 6: "brake then steer",
}
SMOKE_FULL_EXPERIMENTS = (7, 10, 17, 24)
SMOKE_LABELS = ("1.0 s, 10 m/s", "1.5 s, 15 m/s", "2.5 s, 20 m/s", "3.5 s, 25 m/s")


def _load(run_dir: Path) -> np.ndarray:
    return np.load(run_dir / "Results_rear_end" / "Extracted_data.npy")


def _row(data: np.ndarray, experiment: int) -> dict[str, object]:
    matches = np.flatnonzero(data[6, :, 0].astype(int) == experiment)
    if len(matches) != 1:
        raise RuntimeError(f"expected one extracted row for experiment {experiment}, got {len(matches)}")
    row = int(matches[0])
    outcome = data[5, row]
    valid = np.isfinite(outcome)
    outcome = outcome[valid].astype(int)
    rt = data[3, row][valid]
    acceleration = data[4, row][valid]
    fractions = {name: float(np.mean(outcome == code)) for code, name in OUTCOME_NAMES.items()}
    return {
        "n": int(len(outcome)), "outcome_fractions": fractions,
        "response_time_s": float(np.nanmean(rt)) if np.isfinite(rt).any() else None,
        "response_time_n": int(np.isfinite(rt).sum()),
        "brake_acceleration_mps2": float(np.nanmean(acceleration)) if np.isfinite(acceleration).any() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--no-evidence", type=Path, required=True)
    parser.add_argument("--no-pedal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    full, no_evidence, no_pedal = _load(args.full), _load(args.no_evidence), _load(args.no_pedal)
    rows = []
    for local_index, label in enumerate(SMOKE_LABELS):
        baseline = _row(full, SMOKE_FULL_EXPERIMENTS[local_index])
        without_evidence = _row(no_evidence, local_index)
        without_pedal = _row(no_pedal, local_index)
        rows.append({"condition": label, "full": baseline, "no_evidence": without_evidence,
                     "no_pedal": without_pedal,
                     "no_pedal_minus_full_response_s": (
                         None if baseline["response_time_s"] is None or without_pedal["response_time_s"] is None
                         else without_pedal["response_time_s"] - baseline["response_time_s"]),})
    report = {
        "protocol": "released response extractor; four representative conditions x eight random replicates",
        "full_experiment_indices": list(SMOKE_FULL_EXPERIMENTS),
        "rows": rows,
        "interpretation": {
            "no_evidence": "Changes the outcome mixture and may produce pre-stimulus braking; it is not a fixed braking-delay knob.",
            "no_pedal": "Small stochastic native sample is condition dependent; report the measured deltas rather than assuming exactly 0.2 s.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for name, color in (("full", "tab:blue"), ("no_evidence", "tab:orange"), ("no_pedal", "tab:green")):
        values = [row[name]["response_time_s"] for row in rows]
        axes[0].plot(x, values, marker="o", label=name.replace("_", " "), color=color)
    axes[0].set_xticks(x, SMOKE_LABELS, rotation=25, ha="right")
    axes[0].set_ylabel("extracted response time (s)")
    axes[0].legend()
    width = .25
    for offset, (name, color) in zip((-width, 0, width), (("full", "tab:blue"),
                                                             ("no_evidence", "tab:orange"),
                                                             ("no_pedal", "tab:green"))):
        values = [row[name]["outcome_fractions"]["collision"] for row in rows]
        axes[1].bar(x + offset, values, width=width, label=name.replace("_", " "), color=color)
    axes[1].set_xticks(x, SMOKE_LABELS, rotation=25, ha="right")
    axes[1].set_ylabel("collision fraction")
    axes[1].legend()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
