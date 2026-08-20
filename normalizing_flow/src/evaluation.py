"""Evaluation and paper diagnostics for the direct scenario-condition Flow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ks_2samp

from tools.plot_style import GENERATED_COLOR, REAL_COLOR, get_pyplot, style_axes

from .constraints import KNOT_FEATURE_NAMES, derived_modes
from .data import output_dir_from_config, split_indices
from .features import SLOT_NAMES, feature_index, feature_valid_from_slot_mask
from .metrics import physical_validity_flags
from .sampling import load_checkpoint_and_dataset
from .scenario import ScenarioBatch, inverse_constraint
from .utils import ensure_dir, repo_root_from_file, save_json, select_device


def _terms(
    model, arrays: dict[str, np.ndarray], rows: np.ndarray, device
) -> dict[str, float]:
    import torch

    total: dict[str, float] = {}
    for start in range(0, len(rows), 512):
        selected = rows[start : start + 512]
        with torch.no_grad():
            terms = model.log_prob_tensors(
                c0_normalized=torch.from_numpy(
                    arrays["features_normalized"][selected]
                ).float().to(device),
                slot_mask=torch.from_numpy(arrays["slot_mask"][selected])
                .bool()
                .to(device),
                constraint_normalized=torch.from_numpy(
                    arrays["trajectory_constraint_normalized"][selected]
                ).float().to(device),
            )
        for name, value in terms.items():
            total[name] = total.get(name, 0.0) + float((-value).sum().cpu())
    return {
        name.replace("log_prob", "nll"): value / max(len(rows), 1)
        for name, value in total.items()
    }


def _marginal_metrics(
    real: np.ndarray,
    generated: np.ndarray,
    real_valid: np.ndarray,
    generated_valid: np.ndarray,
) -> dict[str, Any]:
    rows = []
    for agent, slot in enumerate(SLOT_NAMES):
        for feature, name in enumerate(KNOT_FEATURE_NAMES):
            left = real[real_valid[:, agent, feature], agent, feature]
            right = generated[generated_valid[:, agent, feature], agent, feature]
            if len(left) < 5 or len(right) < 5:
                continue
            rows.append(
                {
                    "slot": slot,
                    "feature": name,
                    "ks": float(ks_2samp(left, right).statistic),
                    "real_mean": float(left.mean()),
                    "generated_mean": float(right.mean()),
                }
            )
    return {
        "mean_ks": float(np.mean([row["ks"] for row in rows])),
        "per_feature": rows,
    }


def _derived_mode_tv(real_k, generated_k, real_mask, generated_mask) -> dict[str, Any]:
    real = derived_modes(real_k, real_mask)
    generated = derived_modes(generated_k, generated_mask)
    rows = []
    for agent, slot in enumerate(SLOT_NAMES):
        for axis, name in enumerate(("longitudinal", "lateral")):
            p = np.bincount(
                real[real_mask[:, agent], agent, axis], minlength=3
            ).astype(np.float64) + 1.0e-9
            q = np.bincount(
                generated[generated_mask[:, agent], agent, axis], minlength=3
            ).astype(np.float64) + 1.0e-9
            p /= p.sum()
            q /= q.sum()
            rows.append(
                {
                    "slot": slot,
                    "axis": name,
                    "total_variation": float(0.5 * np.abs(p - q).sum()),
                }
            )
    return {
        "mean_total_variation": float(
            np.mean([row["total_variation"] for row in rows])
        ),
        "per_slot_axis": rows,
    }


def _correlation_error(real: np.ndarray, generated: np.ndarray) -> float:
    left = np.nan_to_num(np.corrcoef(real.reshape(len(real), -1).T), nan=0.0)
    right = np.nan_to_num(np.corrcoef(generated.reshape(len(generated), -1).T), nan=0.0)
    return float(np.mean(np.abs(left - right)))


def _mask_metrics(arrays, rows, generated_mask) -> dict[str, float]:
    real = np.bincount(arrays["mask_pattern"][rows], minlength=64) / len(rows)
    pattern = np.sum(generated_mask * (2 ** np.arange(6)), axis=1).astype(int)
    generated = np.bincount(pattern, minlength=64) / len(pattern)
    return {"generated_vs_test_l1": float(np.abs(real - generated).sum())}


def _baselines(arrays, train, test) -> dict[str, Any]:
    from sklearn.mixture import GaussianMixture

    rng = np.random.default_rng(42)
    fit = train if len(train) <= 20_000 else rng.choice(train, 20_000, replace=False)
    report: dict[str, Any] = {"fit_rows": int(len(fit))}
    for name, key in (
        ("c0", "features_normalized"),
        ("k", "trajectory_constraint_normalized"),
    ):
        x = np.asarray(arrays[key][fit], np.float64).reshape(len(fit), -1)
        y = np.asarray(arrays[key][test], np.float64).reshape(len(test), -1)
        mean = x.mean(0)
        variance = np.maximum(x.var(0), 1.0e-5)
        gaussian = 0.5 * np.sum(
            np.log(2 * np.pi * variance) + (y - mean) ** 2 / variance, axis=1
        )
        gmm = GaussianMixture(
            n_components=8,
            covariance_type="diag",
            reg_covar=1.0e-5,
            max_iter=100,
            random_state=42,
        ).fit(x)
        report[name] = {
            "unconditional_diagonal_gaussian_nll": float(gaussian.mean()),
            "unconditional_diagonal_gmm8_nll": float(-gmm.score(y)),
        }
    return report


def _physical(sample: ScenarioBatch, schema: dict[str, Any]) -> dict[str, Any]:
    invalid, _, _ = physical_validity_flags(sample.c0, sample.slot_mask)
    raw, _ = inverse_constraint(
        sample.constraint_normalized_reference,
        sample.slot_mask,
        schema,
    )
    initial_vx = np.stack(
        [
            sample.c0[:, feature_index(None, "ego_vx_mps")]
            + sample.c0[:, feature_index(name, "rel_vx_mps")]
            for name in SLOT_NAMES
        ],
        axis=1,
    )
    active = sample.slot_mask
    raw_speed = initial_vx[..., None] + raw[..., (2, 6, 10)]
    projected_speed = (
        initial_vx[..., None] + sample.trajectory_constraint[..., (2, 6, 10)]
    )
    monotone = np.diff(sample.trajectory_constraint[..., (0, 4, 8)], axis=-1)
    return {
        "c0_legal_fraction": float((~invalid).mean()),
        "constraint_finite": bool(np.isfinite(sample.trajectory_constraint).all()),
        "raw_knot_speed_legal_fraction": float(
            ((raw_speed >= 0.0) & (raw_speed <= 70.0))[active].mean()
        ),
        "projected_knot_speed_legal_fraction": float(
            ((projected_speed >= 0.0) & (projected_speed <= 70.0))[active].mean()
        ),
        "projected_longitudinal_monotone_fraction": float(
            (monotone[active] >= 0.0).all(axis=-1).mean()
        ),
    }


def _plots(report, arrays, test, generated, generated_valid, output):
    plt = get_pyplot()
    figures = ensure_dir(output / "figures")
    paths = {}
    real_valid = arrays["trajectory_constraint_valid"][test]
    generated_c0_valid = feature_valid_from_slot_mask(
        {"feature_names": report["feature_names"]}, generated.slot_mask
    )
    real_c0_valid = arrays["feature_valid"][test]

    figure, axes = plt.subplots(5, 8, figsize=(18.0, 11.0), constrained_layout=True)
    for feature, axis in enumerate(axes.flat):
        left = arrays["features"][test, feature][real_c0_valid[:, feature]]
        right = generated.c0[:, feature][generated_c0_valid[:, feature]]
        axis.hist(left, bins=50, density=True, alpha=0.45, color=REAL_COLOR)
        axis.hist(right, bins=50, density=True, alpha=0.45, color=GENERATED_COLOR)
        axis.set_title(report["feature_names"][feature], fontsize=8)
        style_axes(axis)
    path = figures / "c0_marginal_distributions.png"
    figure.savefig(path, dpi=300)
    plt.close(figure)
    paths["c0_marginals"] = str(path)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    for axis, values, title in (
        (axes[0], arrays["features_normalized"][test], "Held-out C0 correlation"),
        (axes[1], generated.c0_normalized_reference, "Generated C0 correlation"),
    ):
        image = axis.imshow(
            np.nan_to_num(np.corrcoef(values.T)), vmin=-1, vmax=1, cmap="coolwarm"
        )
        axis.set(title=title, xlabel="C0 coordinate", ylabel="C0 coordinate")
    figure.colorbar(image, ax=axes, shrink=0.8)
    path = figures / "c0_correlation_comparison.png"
    figure.savefig(path, dpi=300)
    plt.close(figure)
    paths["c0_correlation"] = str(path)

    figure, axes = plt.subplots(3, 4, figsize=(13.0, 8.0), constrained_layout=True)
    for feature, axis in enumerate(axes.flat):
        left = arrays["trajectory_constraint"][test, :, feature][
            real_valid[:, :, feature]
        ]
        right = generated.trajectory_constraint[:, :, feature][
            generated_valid[:, :, feature]
        ]
        axis.hist(
            left, bins=60, density=True, alpha=0.45, color=REAL_COLOR, label="test"
        )
        axis.hist(
            right,
            bins=60,
            density=True,
            alpha=0.45,
            color=GENERATED_COLOR,
            label="Flow",
        )
        axis.set_title(KNOT_FEATURE_NAMES[feature])
        style_axes(axis)
    axes[0, 0].legend(frameon=False)
    path = figures / "k_marginal_distributions.png"
    figure.savefig(path, dpi=300)
    plt.close(figure)
    paths["k_marginals"] = str(path)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    real = arrays["trajectory_constraint_normalized"][test].reshape(len(test), -1)
    generated_norm = generated.constraint_normalized_reference.reshape(len(test), -1)
    image = None
    for axis, values, title in (
        (axes[0], real, "Held-out K correlation"),
        (axes[1], generated_norm, "Generated K correlation"),
    ):
        image = axis.imshow(
            np.nan_to_num(np.corrcoef(values.T)), vmin=-1, vmax=1, cmap="coolwarm"
        )
        axis.set(title=title, xlabel="K coordinate", ylabel="K coordinate")
    figure.colorbar(image, ax=axes, shrink=0.8)
    path = figures / "k_correlation_comparison.png"
    figure.savefig(path, dpi=300)
    plt.close(figure)
    paths["k_correlation"] = str(path)

    figure, axis = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    names = ("Direct Flow", "Gaussian", "GMM8")
    values = (
        report["held_out_nll"]["k_nll"],
        report["baselines"]["k"]["unconditional_diagonal_gaussian_nll"],
        report["baselines"]["k"]["unconditional_diagonal_gmm8_nll"],
    )
    axis.bar(names, values, color=(GENERATED_COLOR, "#BDBDBD", REAL_COLOR))
    axis.set(title="Held-out K density", ylabel="NLL (lower is better)")
    style_axes(axis)
    path = figures / "k_nll_baselines.png"
    figure.savefig(path, dpi=300)
    plt.close(figure)
    paths["k_nll"] = str(path)

    real_pattern = np.bincount(arrays["mask_pattern"][test], minlength=64)
    sampled_pattern = np.sum(
        generated.slot_mask * (2 ** np.arange(6)), axis=1
    ).astype(int)
    sampled_pattern = np.bincount(sampled_pattern, minlength=64)
    order = np.argsort(-(real_pattern + sampled_pattern))[:12]
    real_mode = derived_modes(
        arrays["trajectory_constraint"][test], arrays["slot_mask"][test]
    )
    sampled_mode = derived_modes(
        generated.trajectory_constraint, generated.slot_mask
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    location = np.arange(len(order))
    axes[0].bar(location - 0.2, real_pattern[order] / len(test), 0.4, color=REAL_COLOR)
    axes[0].bar(
        location + 0.2,
        sampled_pattern[order] / len(generated.slot_mask),
        0.4,
        color=GENERATED_COLOR,
    )
    axes[0].set(
        title="Most frequent slot masks",
        ylabel="Probability",
        xticks=location,
        xticklabels=[str(value) for value in order],
    )
    labels = ("brake", "steady", "accelerate", "keep", "left", "right")
    real_count = np.concatenate(
        [
            np.bincount(real_mode[..., axis][arrays["slot_mask"][test]], minlength=3)
            for axis in range(2)
        ]
    ).astype(float)
    sampled_count = np.concatenate(
        [
            np.bincount(sampled_mode[..., axis][generated.slot_mask], minlength=3)
            for axis in range(2)
        ]
    ).astype(float)
    real_count /= real_count.sum()
    sampled_count /= sampled_count.sum()
    location = np.arange(6)
    axes[1].bar(location - 0.2, real_count, 0.4, color=REAL_COLOR, label="test")
    axes[1].bar(
        location + 0.2,
        sampled_count,
        0.4,
        color=GENERATED_COLOR,
        label="Flow",
    )
    axes[1].set(
        title="Modes derived after K sampling",
        ylabel="Share",
        xticks=location,
        xticklabels=labels,
    )
    axes[1].legend(frameon=False)
    for axis in axes:
        style_axes(axis)
    path = figures / "mask_and_derived_mode_occupancy.png"
    figure.savefig(path, dpi=300)
    plt.close(figure)
    paths["occupancy"] = str(path)

    history = np.genfromtxt(
        output / "training_history.csv", delimiter=",", names=True
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    axes[0].plot(history["epoch"], history["train_joint_nll"], label="train")
    axes[0].plot(history["epoch"], history["val_joint_nll"], label="validation")
    axes[0].set(title="Joint density", xlabel="Epoch", ylabel="NLL")
    axes[0].legend(frameon=False)
    axes[1].plot(history["epoch"], history["val_c0_nll"], label="C0")
    axes[1].plot(history["epoch"], history["val_k_nll"], label="K")
    axes[1].set(title="Held-out factors", xlabel="Epoch", ylabel="NLL")
    axes[1].legend(frameon=False)
    for axis in axes:
        style_axes(axis)
    path = figures / "training_diagnostics.png"
    figure.savefig(path, dpi=300)
    plt.close(figure)
    paths["training"] = str(path)
    return paths


def evaluate_natural_flow(
    config: dict[str, Any],
    *,
    config_dir: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    config_dir = Path(config_dir).resolve()
    root = Path(repo_root).resolve() if repo_root else repo_root_from_file(config_dir)
    output = output_dir_from_config(config, config_dir)
    checkpoint = output / "checkpoints/best_scenario_condition_flow.pt"
    device = select_device(str(config.get("device", "auto")))
    model, arrays, schema, _ = load_checkpoint_and_dataset(
        checkpoint, output, repo_root=root, device=device
    )
    train = split_indices(arrays, "train")
    test = split_indices(arrays, "test")
    generated = model.sample_scenarios(len(test), int(config.get("seed", 42)) + 1000)
    sample_dir = ensure_dir(output / "samples")
    sample_path = sample_dir / "generated_samples.npz"
    np.savez_compressed(sample_path, **generated.__dict__)
    generated_valid = np.broadcast_to(
        generated.slot_mask[..., None], generated.trajectory_constraint.shape
    )
    real_feature_valid = arrays["feature_valid"][test]
    generated_feature_valid = feature_valid_from_slot_mask(schema, generated.slot_mask)
    c0_ks = []
    for feature in range(40):
        left = arrays["features"][test, feature][real_feature_valid[:, feature]]
        right = generated.c0[:, feature][generated_feature_valid[:, feature]]
        if len(left) >= 5 and len(right) >= 5:
            c0_ks.append(float(ks_2samp(left, right).statistic))
    baselines = _baselines(arrays, train, test)
    held_out = _terms(model, arrays, test, device)
    k_metrics = _marginal_metrics(
        arrays["trajectory_constraint"][test],
        generated.trajectory_constraint,
        arrays["trajectory_constraint_valid"][test],
        generated_valid,
    )
    report = {
        "experiment_scope": "full_clean_natural_driving_flow",
        "probability_factorization": "p(M) p(C0|M) p(K|C0,M)",
        "checkpoint": str(checkpoint),
        "num_train": int(len(train)),
        "num_validation": int(len(split_indices(arrays, "val"))),
        "num_test": int(len(test)),
        "future_leakage_in_c0_context": False,
        "feature_names": list(schema["feature_names"]),
        "held_out_nll": held_out,
        "baselines": baselines,
        "distribution": {
            "c0_mean_ks": float(np.mean(c0_ks)),
            "k": k_metrics,
            "k_correlation_mae": _correlation_error(
                arrays["trajectory_constraint_normalized"][test],
                generated.constraint_normalized_reference,
            ),
            "derived_mode": _derived_mode_tv(
                arrays["trajectory_constraint"][test],
                generated.trajectory_constraint,
                arrays["slot_mask"][test],
                generated.slot_mask,
            ),
            "mask_occupancy": _mask_metrics(arrays, test, generated.slot_mask),
        },
        "physical_validity": _physical(generated, schema),
        "sampling": {
            "scenario_seed": int(config.get("seed", 42)) + 1000,
            "empirical_continuous_sampling": False,
            "pre_projection_coordinates_saved": True,
        },
        "generated_samples": str(sample_path),
    }
    report["quality_gates"] = {
        "c0_mean_ks_le_0p10": report["distribution"]["c0_mean_ks"] <= 0.10,
        "k_mean_ks_le_0p13": k_metrics["mean_ks"] <= 0.13,
        "k_nll_beats_gaussian": held_out["k_nll"]
        < baselines["k"]["unconditional_diagonal_gaussian_nll"],
        "k_nll_beats_gmm8": held_out["k_nll"]
        < baselines["k"]["unconditional_diagonal_gmm8_nll"],
        "projected_physical_valid": (
            report["physical_validity"]["c0_legal_fraction"] == 1.0
            and report["physical_validity"][
                "projected_knot_speed_legal_fraction"
            ]
            == 1.0
            and report["physical_validity"][
                "projected_longitudinal_monotone_fraction"
            ]
            == 1.0
        ),
    }
    report["all_quality_gates_passed"] = all(report["quality_gates"].values())
    report["figures"] = _plots(
        report, arrays, test, generated, generated_valid, output
    )
    save_json(report, output / "evaluation_summary.json")
    save_json(
        {
            "experiment_scope": report["experiment_scope"],
            "model": "direct_scenario_condition_flow",
            "probability_factorization": report["probability_factorization"],
            "samples": int(len(arrays["split_index"])),
            "train_validation_test": [
                report["num_train"],
                report["num_validation"],
                report["num_test"],
            ],
            "sampling_sources": {
                "slot_mask": "training_empirical_categorical_pmf",
                "c0": "conditional_rq_spline_maf",
                "k": "conditional_rq_spline_maf",
            },
            "architecture": {
                "c0_flow": dict(config["model"]["c0_flow"]),
                "k_flow": dict(config["model"]["constraint_flow"]),
            },
            "diffusion_condition": {
                "dimension": 118,
                "groups": {"C0": 40, "M": 6, "K": 72},
                "knot_times_s": [2.0, 4.0, 5.96],
                "contract_changed": False,
            },
            "all_quality_gates_passed": report["all_quality_gates_passed"],
            "figures": [
                str(Path(path).relative_to(output))
                for path in report["figures"].values()
            ],
        },
        output / "manifest.json",
    )
    return report
