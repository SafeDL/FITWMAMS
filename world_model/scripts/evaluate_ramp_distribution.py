#!/usr/bin/env python3
"""Evaluate 32 stochastic RAMP futures on the complete highD EVT-tail split."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.initial_behavior_anchor import FrozenLegacyFlowSchema
from world_model.src.ramp.distribution_evaluation import (
    candidate_calibration,
    empirical_coverage,
    energy_score,
    multivariate_feature_distance,
    one_dimensional_distance,
    pit_histogram,
    temporal_and_relationship_diagnostics,
    trajectory_feature_rows,
    univariate_crps,
)
from world_model.src.ramp.train import load_ramp_checkpoint
from world_model.src.semi_markov_train import _loader, _to_batch
from world_model.src.sequential_dataset import (
    ensure_frozen_flow_behavior_anchor_cache,
    load_sequential_dataset,
    sequence_cache_owner_dir,
)
from world_model.src.utils import load_yaml, save_json, select_device


def _merge_feature_rows(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return (
        {key: np.concatenate([item[key] for item in parts]) for key in parts[0]}
        if parts
        else {}
    )


def _mean_diagnostics(parts: list[dict]) -> dict:
    if not parts:
        return {}
    out = {}
    for key, value in parts[0].items():
        values = [np.asarray(item[key], np.float64) for item in parts]
        out[key] = (
            np.mean(values, axis=0).tolist()
            if np.asarray(value).ndim
            else float(np.nanmean(values))
        )
    return out


def _natural_reference(
    rows: dict[str, np.ndarray], *, seed: int, repetitions: int = 100
) -> dict:
    """Natural-vs-natural split baseline with a bootstrap interval for W1."""
    rng = np.random.default_rng(seed)
    result = {}
    for offset, (name, values) in enumerate(rows.items()):
        values = np.asarray(values, np.float64)
        order = np.random.default_rng(seed + offset).permutation(len(values))
        left, right = values[order[::2]], values[order[1::2]]
        distance = one_dimensional_distance(left, right)
        take = min(len(left), len(right), 1024)
        boots = []
        if take:
            local = np.random.default_rng(seed + 1000 + offset)
            for _ in range(repetitions):
                boots.append(
                    one_dimensional_distance(
                        left[local.integers(len(left), size=take)],
                        right[local.integers(len(right), size=take)],
                    )["wasserstein_1"]
                )
        result[name] = {
            **distance,
            "bootstrap_wasserstein_1_95": (
                [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
                if boots
                else [float("nan"), float("nan")]
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "world_model/scripts/configs/highd_ramp_world_model.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(
            ROOT
            / "results/highd_world_model/ramp_world_model/checkpoints/best_ramp_world_model.pt"
        ),
    )
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=0,
        help="Bounded smoke run only; 0 is complete EVT-tail evaluation.",
    )
    parser.add_argument("--seed", type=int, default=314159)
    args = parser.parse_args()
    if args.samples != 32:
        raise ValueError(
            "formal RAMP EVT distribution protocol requires exactly 32 samples"
        )
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    owner = sequence_cache_owner_dir(config, config_dir=config_path.parent)
    arrays, manifest = load_sequential_dataset(owner)
    raw = config["paths"].get("flow_schema")
    if not raw:
        raise ValueError("RAMP distribution evaluation requires paths.flow_schema")
    path = Path(raw)
    schema = FrozenLegacyFlowSchema.load(
        path if path.is_absolute() else (config_path.parent / path).resolve()
    )
    arrays.update(
        ensure_frozen_flow_behavior_anchor_cache(owner, arrays, manifest, schema)
    )
    # Reuse the normal loader then retain only EVT rows by materializing an index view.
    evt = np.flatnonzero(
        (np.asarray(arrays["split_index"]) == 2)
        & np.asarray(arrays["is_evt_tail"], bool)
    )
    if args.max_sequences:
        evt = evt[: args.max_sequences]
    if not len(evt):
        raise RuntimeError("no EVT-tail test sequences available")

    class View(dict):
        pass

    view = View({key: np.asarray(value)[evt] for key, value in arrays.items()})
    view["split_index"] = np.full(len(evt), 2, np.int64)
    device = select_device(config.get("evaluation", {}).get("device", "auto"))
    model = load_ramp_checkpoint(Path(args.checkpoint).resolve(), device=device)
    loader = _loader(
        view,
        "test",
        batch_size=args.batch_size,
        maximum=0,
        shuffle=False,
        seed=args.seed,
    )
    coverage = []
    score = []
    crps = []
    pit = []
    min_ade = []
    min_fde = []
    probability = []
    candidate_error = []
    choice = []
    generated_rows = []
    observed_rows = []
    generated_temporal = []
    observed_temporal = []
    import torch

    with torch.no_grad():
        for offset, values in enumerate(loader):
            batch = _to_batch(values, loader.field_names, device)
            draws = []
            probs = []
            choices = []
            for sample in range(args.samples):
                rollout = model.rollout_roll_mode(
                    batch,
                    seed=args.seed + offset * args.samples + sample,
                    deterministic=False,
                )
                generated = rollout["predicted_states"][:, :, 1:].cpu().numpy()
                draws.append(generated)
                probability.append(rollout["candidate_probabilities"].cpu().numpy())
                choices.append(rollout["selected_candidate_index"].cpu().numpy())
                generated_rows.append(
                    trajectory_feature_rows(
                        generated,
                        batch["agent_valid"][:, 25:150, 1:].cpu().numpy(),
                        batch["agent_states"][:, 25:150, 0].cpu().numpy(),
                    )
                )
                generated_temporal.append(
                    temporal_and_relationship_diagnostics(
                        generated,
                        batch["agent_valid"][:, 25:150, 1:].cpu().numpy(),
                        batch["agent_states"][:, 25:150, 0].cpu().numpy(),
                    )
                )
                candidate = rollout["predicted_candidate_states"].cpu().numpy()
                target_plan = np.zeros(
                    (candidate.shape[0], candidate.shape[1], candidate.shape[3], 6, 6),
                    np.float32,
                )
                valid_plan = np.zeros(target_plan.shape[:-1], bool)
                for response in range(candidate.shape[1]):
                    start = 25 + response * model.cfg.execute_frames
                    stop = min(
                        start + model.cfg.plan_frames, batch["agent_states"].shape[1]
                    )
                    target_plan[:, response, : stop - start] = (
                        batch["agent_states"][:, start:stop, 1:].cpu().numpy()
                    )
                    valid_plan[:, response, : stop - start] = (
                        batch["agent_valid"][:, start:stop, 1:].cpu().numpy()
                    )
                error = np.linalg.norm(
                    candidate[..., :2] - target_plan[:, :, None, ..., :2], axis=-1
                )
                denom = valid_plan.sum(axis=(-2, -1)).clip(min=1)
                candidate_error.append(
                    (error * valid_plan[:, :, None]).sum(axis=(-2, -1))
                    / denom[:, :, None]
                )
            samples = np.stack(draws)
            target = batch["agent_states"][:, 25:150, 1:].cpu().numpy()
            valid = batch["agent_valid"][:, 25:150, 1:].cpu().numpy().astype(bool)
            ego = batch["agent_states"][:, 25:150, 0].cpu().numpy()
            observed_rows.append(trajectory_feature_rows(target, valid, ego))
            observed_temporal.append(
                temporal_and_relationship_diagnostics(target, valid, ego)
            )
            coverage.append(empirical_coverage(samples[..., :2], target[..., :2]))
            score.append(energy_score(samples, target, valid))
            crps.append(univariate_crps(samples[..., :2], target[..., :2], valid))
            pit.append(pit_histogram(samples[..., :2], target[..., :2], valid))
            distance = np.linalg.norm(samples[..., :2] - target[None, ..., :2], axis=-1)
            denominator = valid.sum(axis=(1, 2)).clip(min=1)
            ade = (distance * valid[None]).sum(axis=(2, 3)) / denominator[None]
            min_ade.extend(ade.min(axis=0).tolist())
            endpoint = distance[:, :, -1]
            endpoint_valid = valid[:, -1]
            endpoint_score = (endpoint * endpoint_valid[None]).sum(
                axis=-1
            ) / endpoint_valid.sum(axis=-1).clip(min=1)[None]
            min_fde.extend(endpoint_score.min(axis=0).tolist())
            choice.append(np.stack(choices))
    generated_feature_rows, observed_feature_rows = _merge_feature_rows(
        generated_rows
    ), _merge_feature_rows(observed_rows)
    pit_counts = np.sum([np.asarray(item["counts"], np.int64) for item in pit], axis=0)
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "samples_per_condition": args.samples,
        "evt_tail_sequences": int(len(evt)),
        "coverage": {
            key: float(np.mean([item[key] for item in coverage])) for key in coverage[0]
        },
        "rank_histogram": {
            "bins": int(pit[0]["bins"]),
            "counts": pit_counts.astype(int).tolist(),
            "chi_square_uniform": float(
                (
                    (pit_counts - pit_counts.mean()) ** 2
                    / np.maximum(pit_counts.mean(), 1.0)
                ).sum()
            ),
        },
        "energy_score": float(np.mean(score)),
        "crps": float(np.mean(crps)),
        "minADE_32": float(np.mean(min_ade)),
        "minFDE_32": float(np.mean(min_fde)),
        "candidate_calibration": candidate_calibration(
            np.concatenate(probability, axis=0), np.concatenate(candidate_error, axis=0)
        ),
        "candidate_probability_mean": np.concatenate(probability, axis=0)
        .mean(axis=(0, 1))
        .tolist(),
        "candidate_switch_rate": float(
            np.mean(
                np.concatenate(choice, axis=1)[:, :, 1:]
                != np.concatenate(choice, axis=1)[:, :, :-1]
            )
        ),
        "two_sample_univariate": {
            key: one_dimensional_distance(
                generated_feature_rows[key], observed_feature_rows[key]
            )
            for key in generated_feature_rows
        },
        "two_sample_multivariate": multivariate_feature_distance(
            generated_feature_rows, observed_feature_rows, seed=args.seed
        ),
        "temporal_and_multi_agent": {
            "generated": _mean_diagnostics(generated_temporal),
            "observed": _mean_diagnostics(observed_temporal),
        },
        "natural_vs_natural_reference": _natural_reference(
            observed_feature_rows, seed=args.seed
        ),
    }
    output = Path(config["paths"]["output_dir"]) / "ramp_distribution_evaluation.json"
    output = output if output.is_absolute() else (config_path.parent / output).resolve()
    save_json(report, output)
    print(output)


if __name__ == "__main__":
    main()
