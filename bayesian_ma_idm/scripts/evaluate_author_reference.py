#!/usr/bin/env python3
"""Reproduce the public stochastic-simulation/Table-II evaluation organisation."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from bayesian_ma_idm.src.author_reference import load_author_reference
from bayesian_ma_idm.src.evaluation import crps_ensemble
from bayesian_ma_idm.src.reference_kernels import GPHistory, idm

DT, HORIZON = .2, 16  # the source uses int(3 / dt) + 1


def _posterior(trace_path: Path, kind: str) -> tuple[np.ndarray, dict[str, float]]:
    trace = az.from_netcdf(trace_path)
    theta = np.asarray(trace.posterior["theta"]).reshape(-1, 20, 5)
    if kind == "ma":
        return theta, {"sigma": float(np.sqrt(np.asarray(trace.posterior["s2_f"]).mean())),
                       "ell_frames": float(np.abs(np.asarray(trace.posterior["ell_frames"])).mean()),
                       "iid": float(np.sqrt(np.asarray(trace.posterior["s2_a"]).mean()))}
    return theta, {"sigma": float(np.asarray(trace.posterior["s_a"]).mean())}


def _simulate(pair: dict, theta: np.ndarray, settings: dict[str, float], anchor: int, kind: str,
              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Faithful source-style 5 Hz rollout over one 3-second window."""
    x, v = float(pair["follower_x"][anchor]), float(pair["follower_v"][anchor])
    pos, speed, acc, gap_values = (np.empty(HORIZON) for _ in range(4))
    pos[0], speed[0], acc[0], gap_values[0] = x, v, 0., float(pair["gap"][anchor])
    history = None
    if kind == "ma":
        memory = max(1, int(3 * settings["ell_frames"]))
        indices = np.arange(anchor - memory, anchor)
        residual = pair["follower_a"][indices] - idm(pair["gap"][indices], pair["follower_v"][indices],
                                                        pair["follower_v"][indices] - pair["leader_v"][indices], theta)
        history = GPHistory(((indices - anchor) * DT).astype(float).tolist(), residual.tolist(), settings["sigma"],
                            settings["ell_frames"] * DT, observation_noise=1.e-3, memory_seconds=None)
    for step in range(1, HORIZON):
        index = anchor + step - 1
        gap = float(pair["leader_x"][index] - x - pair["follower_length"])
        # The public notebook stores the pre-action gap at index ``t`` and
        # compares it to s_real[t]; retain this one-step convention exactly.
        gap_values[step] = gap
        deterministic = float(idm(gap, v, v - pair["leader_v"][index], theta))
        if kind == "ma":
            residual = history.step(float(step - 1) * DT, rng.standard_normal()) + settings["iid"] * rng.standard_normal()
        else:
            residual = settings["sigma"] * rng.standard_normal()
        action = deterministic + residual
        next_v = v + action * DT
        x += .5 * (v + next_v) * DT
        pos[step], speed[step], acc[step - 1] = x, next_v, action
        v = next_v
    acc[-1] = acc[-2]
    return acc, speed, gap_values


def _metrics(samples: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    # The source averages per-rollout RMSE rather than RMSE of the ensemble mean.
    return {"rmse_mean": float(np.mean(np.sqrt(np.mean((samples - actual[None, :]) ** 2, axis=1)))),
            "rmse_std": float(np.std(np.sqrt(np.mean((samples - actual[None, :]) ** 2, axis=1)))),
            "crps": crps_ensemble(samples[None, :, :], actual[None, :])}


def evaluate(dataset_path: Path, ma_trace: Path, b_trace: Path, output_dir: Path, *, anchors: int = 50,
             futures: int = 50, seed: int = 1116) -> Path:
    _, pairs = load_author_reference(dataset_path); output_dir.mkdir(parents=True, exist_ok=True)
    ma_theta, ma_set = _posterior(ma_trace, "ma"); b_theta, b_set = _posterior(b_trace, "b")
    rng = np.random.default_rng(seed); report = {"protocol": "public Stochastic_simulation_GP.ipynb Table-II organisation",
             "fps": 5, "horizon_s": 3., "points": HORIZON, "anchors_per_driver": anchors, "futures_per_anchor": futures,
             "models": {"ma": {"settings": ma_set}, "b": {"settings": b_set}}, "drivers": {}}
    for driver_id in (273, 211):
        index = next(i for i, pair in enumerate(pairs) if int(pair["follower_id"]) == driver_id); pair = pairs[index]
        earliest = max(20, int(3 * ma_set["ell_frames"]))
        candidates = np.arange(earliest, len(pair["gap"]) - HORIZON)
        chosen = rng.choice(candidates, size=min(anchors, len(candidates)), replace=False)
        collected = {kind: {state: [] for state in ("acceleration", "speed", "gap")} for kind in ("ma", "b")}
        for anchor in chosen:
            run = {kind: [] for kind in ("ma", "b")}
            for _ in range(futures):
                run["ma"].append(_simulate(pair, np.mean(ma_theta[:, index], axis=0), ma_set, int(anchor), "ma", rng))
                run["b"].append(_simulate(pair, np.mean(b_theta[:, index], axis=0), b_set, int(anchor), "b", rng))
            actual = {"acceleration": pair["follower_a"][anchor:anchor + HORIZON], "speed": pair["follower_v"][anchor:anchor + HORIZON], "gap": pair["gap"][anchor:anchor + HORIZON]}
            for kind in ("ma", "b"):
                for state, values in zip(("acceleration", "speed", "gap"), zip(*run[kind])):
                    collected[kind][state].append(_metrics(np.asarray(values), actual[state]))
        report["drivers"][str(driver_id)] = {kind: {state: {key: float(np.mean([row[key] for row in results])) for key in results[0]}
                                            for state, results in states.items()} for kind, states in collected.items()}
        # Fig. 8-style qualitative model comparison at the first selected origin.
        anchor = int(chosen[0]); samples = {kind: [] for kind in ("ma", "b")}
        for _ in range(futures):
            samples["ma"].append(_simulate(pair, np.mean(ma_theta[:, index], axis=0), ma_set, anchor, "ma", rng))
            samples["b"].append(_simulate(pair, np.mean(b_theta[:, index], axis=0), b_set, anchor, "b", rng))
        actual = (pair["follower_a"][anchor:anchor + HORIZON], pair["follower_v"][anchor:anchor + HORIZON], pair["gap"][anchor:anchor + HORIZON])
        fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
        for axis, state, observed, label in zip(axes, range(3), actual, ("acceleration [m/s²]", "speed [m/s]", "gap [m]")):
            for kind, color in (("b", "#1f77b4"), ("ma", "#d62728")):
                values = np.asarray(samples[kind])[:, state]
                lo, mid, hi = np.quantile(values, [.05, .5, .95], axis=0)
                axis.fill_between(np.arange(HORIZON) * DT, lo, hi, color=color, alpha=.12)
                axis.plot(np.arange(HORIZON) * DT, mid, color=color, lw=1.2, label=f"{kind.upper()} median")
            axis.plot(np.arange(HORIZON) * DT, observed, color="black", lw=1., label="highD")
            axis.set_ylabel(label)
        axes[0].legend(fontsize=7, ncol=3); axes[-1].set_xlabel("prediction time [s]")
        fig.suptitle(f"Source-protocol 3 s stochastic comparison, driver #{driver_id} (Fig. 8/Table II analogue)")
        fig.savefig(output_dir / f"fig8_table2_author_compare_{driver_id}.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    path = output_dir / "author_reference_table2_analogue.json"; path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "bayesian_ma_idm/evidence/paper_reference/zhang_sun_author_20pairs_5hz.npz"))
    parser.add_argument("--ma-trace", default=str(ROOT / "bayesian_ma_idm/evidence/paper_reference/nuts/author_reference_trace.nc"))
    parser.add_argument("--b-trace", default=str(ROOT / "bayesian_ma_idm/evidence/paper_reference/b_nuts/author_reference_b_trace.nc"))
    parser.add_argument("--output-dir", default=str(ROOT / "bayesian_ma_idm/evidence/paper_reference/evaluation"))
    parser.add_argument("--anchors", type=int, default=50); parser.add_argument("--futures", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1116); args = parser.parse_args()
    print(evaluate(Path(args.dataset), Path(args.ma_trace), Path(args.b_trace), Path(args.output_dir), anchors=args.anchors, futures=args.futures, seed=args.seed))


if __name__ == "__main__":
    main()
