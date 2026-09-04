#!/usr/bin/env python3
"""Held-out, distributional validation for the frozen longitudinal GAIL prior."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.human_prior import HumanActionPrior, build_human_expert_samples, build_human_reference_samples  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from world_model.src.core.utils import ensure_dir, save_json, select_device  # noqa: E402


DEFAULT = ROOT / "hierarchical_world_model/config/reaction_naturalistic.yaml"


def _w1(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.sort(left), np.sort(right)
    q = np.linspace(0., 1., min(len(left), len(right)))
    return float(np.abs(np.quantile(left, q) - np.quantile(right, q)).mean())


def _ks(left: np.ndarray, right: np.ndarray) -> float:
    values = np.sort(np.concatenate((left, right)))
    return float(np.max(np.abs(np.searchsorted(np.sort(left), values, side="right") / len(left) - np.searchsorted(np.sort(right), values, side="right") / len(right))))


def _brake_response_reference(arrays: dict[str, np.ndarray], rows: np.ndarray, *, seed: int, per_group: int = 512) -> dict[str, np.ndarray]:
    """Mine held-out human rear responses to an actually observed front brake.

    This is stricter than comparing a counterfactual emergency policy with
    arbitrary car-following actions: the reference contains a same-lane front
    vehicle whose realised longitudinal acceleration is already <= -2 m/s².
    Rear actions are collected from the following 0.4 s response window.
    """
    rng, buckets = np.random.default_rng(seed), {}
    for start in range(0, len(rows), 512):
        choice = rows[start:start + 512]
        states = np.asarray(arrays["agent_states"])[choice]
        valid = np.asarray(arrays["agent_valid"])[choice]
        # Leave a ten-frame response window after each observed ego brake.
        for frame in range(25, min(163, states.shape[1] - 11)):
            ego, rear = states[:, frame, 0], states[:, frame, 2]
            gap, closing = ego[:, 0] - rear[:, 0] - 4.8, rear[:, 2] - ego[:, 2]
            ego_ax = (states[:, frame + 1, 0, 2] - ego[:, 2]) / .04
            mask = (valid[:, frame, 0] & valid[:, frame, 2] & valid[:, frame + 11, 2] &
                    (gap > .1) & (np.abs(ego[:, 1] - rear[:, 1]) < 1.8) & (ego[:, 0] > rear[:, 0]) & (ego_ax <= -2.))
            for dose_band, upper in ((2, 4.), (4, 8.01)):
                band = mask & ((-ego_ax) >= dose_band) & ((-ego_ax) < upper)
                for low, high, ttc_band in ((0., 2., 0), (2., 4., 1), (4., 10.01, 2)):
                    ttc = np.where(closing > 1.e-4, gap / np.maximum(closing, 1.e-4), 10.)
                    index = band & (ttc >= low) & (ttc < high)
                    selected = np.flatnonzero(index)
                    if not len(selected):
                        continue
                    key = (dose_band, ttc_band)
                    target = buckets.setdefault(key, [])
                    # A reservoir prevents the first temporal segment from
                    # dominating the held-out action distribution.
                    for item in selected.tolist():
                        for lag in range(1, 11):
                            rear_ax = float(np.clip((states[item, frame + lag + 1, 2, 2] - states[item, frame + lag, 2, 2]) / .04, -8., 4.))
                            prev_ax = float(np.clip((states[item, frame + lag, 2, 2] - states[item, frame + lag - 1, 2, 2]) / .04, -8., 4.))
                            value = (rear_ax, abs(rear_ax - prev_ax) / .04, float(-ego_ax[item]), float(ttc[item]))
                            if len(target) < per_group:
                                target.append(value)
                            else:
                                replace = int(rng.integers(0, 1000000))
                                if replace < per_group:
                                    target[replace] = value
    fields = {"final_ax_mps2": [], "jerk_mps3": [], "ego_brake_mps2": [], "ttc_s": [], "dose_band": [], "ttc_band": []}
    for (dose_band, ttc_band), values in buckets.items():
        for action, jerk, ego_brake, ttc in values:
            fields["final_ax_mps2"].append(action); fields["jerk_mps3"].append(jerk)
            fields["ego_brake_mps2"].append(ego_brake); fields["ttc_s"].append(ttc)
            fields["dose_band"].append(dose_band); fields["ttc_band"].append(ttc_band)
    return {name: np.asarray(value, np.float32 if name not in {"dose_band", "ttc_band"} else np.int16) for name, value in fields.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen GAIL human-action prior on a held-out highD split.")
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="optional candidate HumanActionPriorV4 checkpoint")
    parser.add_argument("--label", default="gail", choices=("bc", "gail"),
                        help="output label, allowing BC and GAIL evidence side by side")
    args = parser.parse_args()
    config = load_protocol_config(args.config.resolve())
    base = load_protocol_config(ROOT / config.get("base_config", "hierarchical_world_model/config/release.yaml"))
    experiment = prepare_experiment_data(base, ROOT)
    rows = experiment.validation_rows if args.split == "validation" else experiment.test_rows
    feature, action = build_human_expert_samples(experiment.bundle.arrays, rows, max_samples=args.samples, seed=20260917)
    device = select_device(config["training"].get("device", "auto"))
    checkpoint = config["paths"]["human_prior"] if args.checkpoint is None else args.checkpoint
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("schema") != "longitudinal_gail_human_prior_v4":
        raise ValueError("held-out evaluation requires longitudinal_gail_human_prior_v4")
    prior = HumanActionPrior().to(device); prior.load_state_dict(payload["state_dict"]); prior.eval()
    with torch.no_grad():
        x = torch.from_numpy(feature).to(device)
        predicted, _, _, _ = prior(x, deterministic=True)
        expert_action = torch.from_numpy(action).to(device)
        nll = -prior.action_log_prob(x, expert_action).mean()
        torch.manual_seed(20260917)
        sampled, _, _, _ = prior(x, deterministic=False)
    predicted_np = predicted.cpu().numpy()
    output = ensure_dir(Path(checkpoint).parent)
    suffix = "" if args.label == "gail" else f"_{args.label}"
    np.savez_compressed(output / f"heldout_{args.split}_prior_samples{suffix}.npz", action_mps2=action, predicted_mean_mps2=predicted_np,
                        sampled_action_mps2=sampled.cpu().numpy(),
                        gap_m=feature[:, -4] * 60., closing_mps=feature[:, -3] * 25., ttc_s=feature[:, -2] * 10.)
    report = {
        "schema": "longitudinal_human_prior_heldout_v2", "split": args.split, "samples": int(len(action)),
        "mean_absolute_error_mps2": float(np.abs(predicted_np - action).mean()),
        "conditional_mean_action_w1_mps2": _w1(action, predicted_np),
        "sampled_action_w1_mps2": _w1(action, sampled.cpu().numpy()),
        "sampled_action_ks": _ks(action, sampled.cpu().numpy()),
        "bounded_gaussian_action_nll": float(nll.cpu()), "checkpoint": str(checkpoint),
        "note": "conditional sample selection balances ordinary following and TTC-critical following; distribution distances are reported with that fixed sampling scheme.",
    }
    report["stage"] = args.label
    save_json(report, output / f"heldout_{args.split}_prior_metrics{suffix}.json")
    # A separate balanced context sample contains physical jerk.  It is used
    # by the causal-policy evaluator, whose deliberately rare safety-critical
    # test cases alone cannot supply a statistically adequate human reference.
    reference = build_human_reference_samples(experiment.bundle.arrays, rows, max_samples=args.samples, seed=20260918)
    if args.label == "gail":
        np.savez_compressed(output / f"heldout_{args.split}_human_reference.npz", **reference)
    brake_reference = _brake_response_reference(experiment.bundle.arrays, rows, seed=20260919)
    if args.label == "gail":
        np.savez_compressed(output / f"heldout_{args.split}_brake_response_reference.npz", **brake_reference)
    report["observed_brake_response_reference_samples"] = int(len(brake_reference["final_ax_mps2"]))
    report["observed_brake_response_reference_groups"] = {
        f"dose_{dose}_ttc_{ttc}": int(((brake_reference["dose_band"] == dose) & (brake_reference["ttc_band"] == ttc)).sum())
        for dose in (2, 4) for ttc in (0, 1, 2)
    }
    save_json(report, output / f"heldout_{args.split}_prior_metrics{suffix}.json")
    print(report)


if __name__ == "__main__":
    main()
