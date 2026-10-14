"""Evaluation and paper figures for the all-tail scenario Flow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from tools.plot_style import (
    GENERATED_COLOR,
    PAPER_FIGURE_DPI,
    REAL_COLOR,
    REFERENCE_COLOR,
    SAMPLED_COLOR,
    get_pyplot,
    style_axes,
)

from .constraints import KNOT_FEATURE_NAMES, KNOT_TIMES_S
from .data import dataset_dir_from_config, evt_tail_indices, output_dir_from_config
from .features import SLOT_NAMES, feature_index, feature_valid_from_slot_mask
from .metrics import physical_validity_flags
from .scenario import (
    ScenarioBatch,
    inverse_constraint,
    project_constraint,
    project_physical_c0,
)
from .sampling import inverse_normalize_features, load_checkpoint_and_dataset
from .utils import ensure_dir, load_json, save_json, select_device


def _terms(model, arrays, rows, device) -> dict[str, float]:
    import torch

    totals = {name: 0.0 for name in ("mask", "c0", "k", "joint")}
    for start in range(0, len(rows), 512):
        selected = rows[start : start + 512]
        with torch.no_grad():
            terms = model.log_prob_tensors(
                c0_normalized=torch.from_numpy(arrays["features_normalized"][selected])
                .float()
                .to(device),
                slot_mask=torch.from_numpy(arrays["slot_mask"][selected])
                .bool()
                .to(device),
                constraint_normalized=torch.from_numpy(
                    arrays["trajectory_constraint_normalized"][selected]
                )
                .float()
                .to(device),
            )
        for name in totals:
            totals[name] += float((-terms[f"{name}_log_prob"]).sum().cpu())
    return {f"{name}_nll": value / len(rows) for name, value in totals.items()}


def _marginals(real, generated, real_valid, generated_valid, names):
    rows = []
    for index, name in enumerate(names):
        left = real[:, index][real_valid[:, index]]
        right = generated[:, index][generated_valid[:, index]]
        if len(left) < 5 or len(right) < 5:
            continue
        rows.append(
            {
                "feature": name,
                "ks": float(ks_2samp(left, right).statistic),
                "wasserstein": float(wasserstein_distance(left, right)),
            }
        )
    return {
        "mean_ks": float(np.mean([item["ks"] for item in rows])),
        "mean_wasserstein": float(np.mean([item["wasserstein"] for item in rows])),
        "per_feature": rows,
    }


def _c0_metrics(arrays, rows, sample, schema):
    real_valid = np.asarray(arrays["feature_valid"])[rows]
    generated_valid = feature_valid_from_slot_mask(schema, sample.slot_mask)
    return _marginals(
        np.asarray(arrays["features"])[rows],
        sample.c0,
        real_valid,
        generated_valid,
        list(schema["feature_names"]),
    )


def _k_metrics(arrays, rows, sample):
    real = np.asarray(arrays["trajectory_constraint"])[rows].reshape(len(rows), -1)
    generated = sample.trajectory_constraint.reshape(len(rows), -1)
    real_valid = np.asarray(arrays["trajectory_constraint_valid"])[rows].reshape(
        len(rows), -1
    )
    generated_valid = sample.trajectory_constraint_valid.reshape(len(rows), -1)
    names = [
        f"{slot}_{feature}" for slot in SLOT_NAMES for feature in KNOT_FEATURE_NAMES
    ]
    return _marginals(real, generated, real_valid, generated_valid, names)


def _dimension_metrics(arrays, rows, sample):
    real = np.asarray(arrays["vehicle_dimensions"])[rows]
    real_valid = np.column_stack(
        (np.ones(len(rows), bool), np.asarray(arrays["slot_mask"])[rows])
    )
    generated_valid = np.column_stack(
        (np.ones(len(sample.slot_mask), bool), sample.slot_mask)
    )
    metrics = []
    for agent, name in enumerate(("ego", *SLOT_NAMES)):
        for feature, label in enumerate(("length_m", "width_m")):
            left = real[real_valid[:, agent], agent, feature]
            right = sample.vehicle_dimensions[generated_valid[:, agent], agent, feature]
            metrics.append(
                {
                    "agent": name,
                    "feature": label,
                    "ks": float(ks_2samp(left, right).statistic),
                    "wasserstein": float(wasserstein_distance(left, right)),
                }
            )
    return {
        "sampling": "empirical_dimensions_conditioned_on_mask_pattern",
        "mean_ks": float(np.mean([item["ks"] for item in metrics])),
        "mean_wasserstein": float(np.mean([item["wasserstein"] for item in metrics])),
        "per_feature": metrics,
    }


def _mask_l1(real_mask, generated_mask) -> float:
    powers = 2 ** np.arange(6)
    real = np.bincount((real_mask * powers).sum(1), minlength=64) / len(real_mask)
    generated = np.bincount((generated_mask * powers).sum(1), minlength=64) / len(
        generated_mask
    )
    return float(np.abs(real - generated).sum())


def _joint_vector(c0, mask, k, schema):
    c0_valid = feature_valid_from_slot_mask(schema, mask)
    c0_values = np.asarray(c0, np.float32).copy()
    c0_values[~c0_valid] = 0.0
    k_values = np.asarray(k, np.float32).copy()
    k_values[~np.broadcast_to(mask[..., None], k_values.shape)] = 0.0
    return np.concatenate(
        (mask.astype(np.float32), c0_values, k_values.reshape(len(k), -1)), axis=1
    )


def _multivariate_metrics(real, generated, *, seed: int) -> dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    combined = np.concatenate((real, generated), axis=0)
    labels = np.concatenate((np.zeros(len(real)), np.ones(len(generated))))
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed),
    )
    prediction = cross_val_predict(
        classifier,
        combined,
        labels,
        cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        method="predict_proba",
    )[:, 1]
    auc = float(roc_auc_score(labels, prediction))
    rng = np.random.default_rng(seed)
    real_standard = (real - real.mean(0)) / np.maximum(real.std(0), 1.0e-6)
    generated_standard = (generated - real.mean(0)) / np.maximum(real.std(0), 1.0e-6)
    directions = rng.standard_normal((256, real.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    left = np.sort(real_standard @ directions.T, axis=0)
    right = np.sort(generated_standard @ directions.T, axis=0)
    return {
        "classifier_two_sample_auc": auc,
        "classifier_auc_distance_from_0p5": abs(auc - 0.5),
        "sliced_wasserstein_256": float(np.mean(np.abs(left - right))),
    }


def _physical(sample, schema):
    invalid, reasons, flags = physical_validity_flags(
        sample.c0, sample.slot_mask, sample.vehicle_dimensions
    )
    raw_c0 = inverse_normalize_features(sample.c0_normalized_reference, schema)
    raw_c0[~feature_valid_from_slot_mask(schema, sample.slot_mask)] = 0.0
    raw_invalid, _, _ = physical_validity_flags(
        raw_c0, sample.slot_mask, sample.vehicle_dimensions
    )
    c0_scene_delta = np.abs(raw_c0 - sample.c0).max(axis=1)
    raw_k, _ = inverse_constraint(
        sample.constraint_normalized_reference, sample.slot_mask, schema, raw_c0
    )
    durations = np.diff((0.0, *KNOT_TIMES_S))
    raw_dx = np.diff(
        np.concatenate(
            (np.zeros((*raw_k.shape[:2], 1)), raw_k[..., (0, 2, 4)]), axis=-1
        ),
        axis=-1,
    )
    projected_dx = np.diff(
        np.concatenate(
            (
                np.zeros((*raw_k.shape[:2], 1)),
                sample.trajectory_constraint[..., (0, 2, 4)],
            ),
            axis=-1,
        ),
        axis=-1,
    )
    raw_speed = raw_dx / durations
    projected_speed = projected_dx / durations
    difference = np.abs(raw_k - sample.trajectory_constraint)
    scene_delta = difference.max(axis=(1, 2))
    return {
        "raw_c0_legal_fraction": float((~raw_invalid).mean()),
        "projected_c0_legal_fraction": float((~invalid).mean()),
        "c0_projection_scene_fraction": float((c0_scene_delta > 1.0e-3).mean()),
        "initial_overlap_scenes": int(flags["overlap"].sum()),
        "initial_npc_pair_overlap_instances": int(reasons["npc_pair_box_overlap"]),
        "raw_interval_speed_legal_fraction": float(
            ((raw_speed >= 0.0) & (raw_speed <= 70.0))[sample.slot_mask].mean()
        ),
        "projected_interval_speed_legal_fraction": float(
            ((projected_speed >= 0.0) & (projected_speed <= 70.0))[
                sample.slot_mask
            ].mean()
        ),
        "k_projection_scene_fraction": float((scene_delta > 1.0e-3).mean()),
        "k_projection_p95_m": float(np.quantile(scene_delta, 0.95)),
        "k_projection_max_m": float(scene_delta.max()),
    }


def _baseline_samples(arrays, rows, schema, *, seed: int):
    from sklearn.mixture import GaussianMixture

    rng = np.random.default_rng(seed)
    n = len(rows)
    row_draw = rng.choice(rows, size=n, replace=True)
    masks = np.asarray(arrays["slot_mask"])[row_draw]
    dimensions = np.asarray(arrays["vehicle_dimensions"])[row_draw]
    c0_dim = len(schema["feature_names"])
    k_shape = (6, len(KNOT_FEATURE_NAMES))
    train = np.concatenate(
        (
            np.asarray(arrays["features_normalized"])[rows],
            np.asarray(arrays["trajectory_constraint_normalized"])[rows].reshape(n, -1),
        ),
        axis=1,
    )

    def make(values):
        c0_norm = values[:, :c0_dim].astype(np.float32)
        k_norm = values[:, c0_dim:].reshape(n, *k_shape).astype(np.float32)
        c0 = inverse_normalize_features(c0_norm, schema)
        c0[~feature_valid_from_slot_mask(schema, masks)] = 0.0
        c0 = project_physical_c0(c0, masks, dimensions)
        k, valid = inverse_constraint(k_norm, masks, schema, c0)
        k = project_constraint(k, masks, c0)
        return ScenarioBatch(c0, masks, k, valid, dimensions, c0_norm, k_norm)

    mean = train.mean(0)
    std = np.maximum(train.std(0), 1.0e-4)
    gaussian = make(rng.normal(mean, std, size=(n, train.shape[1])))
    gmm = GaussianMixture(
        n_components=8,
        covariance_type="diag",
        reg_covar=1.0e-5,
        max_iter=200,
        random_state=seed,
    ).fit(train)
    mixture = make(gmm.sample(n)[0])
    bootstrap = ScenarioBatch(
        np.asarray(arrays["features"])[row_draw],
        masks,
        np.asarray(arrays["trajectory_constraint"])[row_draw],
        np.asarray(arrays["trajectory_constraint_valid"])[row_draw],
        dimensions,
    )
    return {"empirical_bootstrap": bootstrap, "gaussian": gaussian, "gmm8": mixture}


def _quality(arrays, rows, sample, schema, seed):
    real_mask = np.asarray(arrays["slot_mask"])[rows]
    c0 = _c0_metrics(arrays, rows, sample, schema)
    k = _k_metrics(arrays, rows, sample)
    real_joint = _joint_vector(
        np.asarray(arrays["features"])[rows],
        real_mask,
        np.asarray(arrays["trajectory_constraint"])[rows],
        schema,
    )
    generated_joint = _joint_vector(
        sample.c0, sample.slot_mask, sample.trajectory_constraint, schema
    )
    return (
        {
            "mask_l1": _mask_l1(real_mask, sample.slot_mask),
            "c0_mean_ks": c0["mean_ks"],
            "k_mean_ks": k["mean_ks"],
            **_multivariate_metrics(real_joint, generated_joint, seed=seed),
        },
        c0,
        k,
    )


def _figures(output, arrays, rows, sample, report):
    plt = get_pyplot()
    figures = ensure_dir(output / "figures")
    history = np.genfromtxt(output / "training_history.csv", delimiter=",", names=True)
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    axis.plot(history["epoch"], history["joint_nll"], color=GENERATED_COLOR)
    axis.set(xlabel="Epoch", ylabel="Training NLL", title="All-tail Flow convergence")
    style_axes(axis)
    figure.tight_layout()
    figure.savefig(figures / "training_diagnostics.png", dpi=PAPER_FIGURE_DPI)
    plt.close(figure)

    cv = report["cross_validation"]
    names = ["Flow", "Gaussian", "GMM-8"]
    values = [
        cv["mean"]["c0_nll"] + cv["mean"]["k_nll"],
        cv["mean"]["diagonal_gaussian_continuous_nll"],
        cv["mean"]["diagonal_gmm8_continuous_nll"],
    ]
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    axis.bar(names, values, color=[GENERATED_COLOR, REFERENCE_COLOR, SAMPLED_COLOR])
    axis.set(ylabel="5-fold held-out continuous NLL", title="Density comparison")
    style_axes(axis, grid=False)
    figure.tight_layout()
    figure.savefig(figures / "cross_validation_nll.png", dpi=PAPER_FIGURE_DPI)
    plt.close(figure)

    comparison = report["method_comparison"]
    methods = list(comparison)
    display_names = {
        "flow": "Flow",
        "empirical_bootstrap": "Bootstrap",
        "gaussian": "Gaussian",
        "gmm8": "GMM-8",
    }
    x = np.arange(len(methods))
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 4.0))
    metrics = (
        ("c0_mean_ks", "Mean C0 KS"),
        ("k_mean_ks", "Mean K KS"),
        ("classifier_auc_distance_from_0p5", "|C2ST AUC − 0.5|"),
    )
    colors = [
        GENERATED_COLOR,
        REAL_COLOR,
        REFERENCE_COLOR,
        SAMPLED_COLOR,
    ]
    for axis, (metric, label) in zip(axes, metrics):
        axis.bar(x, [comparison[name][metric] for name in methods], color=colors)
        axis.set_xticks(x, [display_names[name] for name in methods], rotation=18)
        axis.set_ylabel(label)
        style_axes(axis, grid=False)
    figure.suptitle("Tail-distribution reconstruction")
    figure.tight_layout()
    figure.savefig(figures / "method_comparison.png", dpi=PAPER_FIGURE_DPI)
    plt.close(figure)

    real_mask = np.asarray(arrays["slot_mask"])[rows].mean(0)
    generated_mask = sample.slot_mask.mean(0)
    x = np.arange(6)
    figure, axis = plt.subplots(figsize=(8.2, 4.2))
    axis.bar(x - 0.18, real_mask, 0.36, color=REAL_COLOR, label="EVT tail")
    axis.bar(x + 0.18, generated_mask, 0.36, color=GENERATED_COLOR, label="Flow")
    axis.set_xticks(x, [name.replace("_", "\n") for name in SLOT_NAMES])
    axis.set(ylabel="Occupancy probability", title="Scene occupancy")
    axis.legend(frameon=False)
    style_axes(axis, grid=False)
    figure.tight_layout()
    figure.savefig(figures / "mask_occupancy.png", dpi=PAPER_FIGURE_DPI)
    plt.close(figure)

    same_front = SLOT_NAMES.index("same_front")
    real_mask = np.asarray(arrays["slot_mask"])[rows]
    diagnostics = (
        (
            np.asarray(arrays["features"])[rows, feature_index(None, "ego_vx_mps")],
            sample.c0[:, feature_index(None, "ego_vx_mps")],
            "Ego $v_x$ (m/s)",
        ),
        (
            np.asarray(arrays["features"])[
                rows, feature_index("same_front", "rel_x_m")
            ][real_mask[:, same_front]],
            sample.c0[:, feature_index("same_front", "rel_x_m")][
                sample.slot_mask[:, same_front]
            ],
            r"Front gap $\Delta x$ (m)",
        ),
        (
            np.asarray(arrays["features"])[
                rows, feature_index("same_front", "rel_vx_mps")
            ][real_mask[:, same_front]],
            sample.c0[:, feature_index("same_front", "rel_vx_mps")][
                sample.slot_mask[:, same_front]
            ],
            r"Front $\Delta v_x$ (m/s)",
        ),
        (
            np.asarray(arrays["trajectory_constraint"])[rows, same_front, 0][
                real_mask[:, same_front]
            ],
            sample.trajectory_constraint[:, same_front, 0][
                sample.slot_mask[:, same_front]
            ],
            "$x$ at 2 s (m)",
        ),
        (
            np.asarray(arrays["trajectory_constraint"])[rows, same_front, 4][
                real_mask[:, same_front]
            ],
            sample.trajectory_constraint[:, same_front, 4][
                sample.slot_mask[:, same_front]
            ],
            "$x$ at 5.96 s (m)",
        ),
        (
            np.asarray(arrays["trajectory_constraint"])[rows, same_front, 5][
                real_mask[:, same_front]
            ],
            sample.trajectory_constraint[:, same_front, 5][
                sample.slot_mask[:, same_front]
            ],
            "$y$ at 5.96 s (m)",
        ),
    )
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 6.8))
    for axis, (real, generated, label) in zip(axes.flat, diagnostics):
        for values, color, name in (
            (real, REAL_COLOR, "EVT tail"),
            (generated, GENERATED_COLOR, "Flow"),
        ):
            ordered = np.sort(values)
            axis.plot(
                ordered,
                np.arange(1, len(ordered) + 1) / len(ordered),
                color=color,
                linewidth=1.6,
                label=name,
            )
        axis.set(xlabel=label, ylabel="Empirical CDF")
        style_axes(axis)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Representative tail marginals")
    figure.tight_layout()
    figure.savefig(figures / "marginal_diagnostics.png", dpi=PAPER_FIGURE_DPI)
    plt.close(figure)


def evaluate_tail_flow(
    config: dict[str, Any],
    *,
    config_dir: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    config_dir = Path(config_dir).resolve()
    repo_root = Path(repo_root).resolve()
    output = output_dir_from_config(config, config_dir)
    dataset_dir = dataset_dir_from_config(config, config_dir)
    device = select_device(str(config["device"]))
    checkpoint = output / "checkpoints/scenario_flow.pt"
    model, arrays, schema, payload = load_checkpoint_and_dataset(
        checkpoint, dataset_dir, repo_root=repo_root, device=device
    )
    rows = evt_tail_indices(arrays)
    seed = int(config["evaluation"]["sample_seed"])
    sample = model.sample_scenarios(len(rows), seed)
    samples_dir = ensure_dir(output / "samples")
    sample_path = samples_dir / "generated_samples.npz"
    np.savez_compressed(
        sample_path,
        c0=sample.c0,
        slot_mask=sample.slot_mask,
        trajectory_constraint=sample.trajectory_constraint,
        trajectory_constraint_valid=sample.trajectory_constraint_valid,
        vehicle_dimensions=sample.vehicle_dimensions,
        c0_normalized_reference=sample.c0_normalized_reference,
        constraint_normalized_reference=sample.constraint_normalized_reference,
    )
    flow_quality, c0, k = _quality(arrays, rows, sample, schema, seed)
    comparisons = {"flow": flow_quality}
    for index, (name, candidate) in enumerate(
        _baseline_samples(arrays, rows, schema, seed=seed + 1).items(), start=1
    ):
        comparisons[name] = _quality(arrays, rows, candidate, schema, seed + index)[0]
    cross_validation = load_json(output / "cross_validation.json")
    cross_validation_summary = {
        "protocol": cross_validation["protocol"],
        "folds": len(cross_validation["folds"]),
        "mean": cross_validation["mean"],
        "std": cross_validation["std"],
        "report": "results/flow/cross_validation.json",
    }
    report = {
        "population": "evt_tail",
        "target_events": int(len(rows)),
        "final_fit_events": int(payload["metrics"]["num_fit_events"]),
        "evaluation_role": (
            "goodness_of_fit_to_the_target_tail_distribution; generalization is "
            "reported separately by recording-level cross-validation"
        ),
        "probability_factorization": schema["probability_factorization"],
        "condition_dimensions": {"C0": 26, "M": 6, "K": 36, "D": 14},
        "dimension_handling": "empirical conditional sampler; excluded from Flow loss",
        "in_sample_nll": _terms(model, arrays, rows, device),
        "cross_validation": cross_validation_summary,
        "method_comparison": comparisons,
        "distribution": {
            "c0": c0,
            "k": k,
            "dimensions": _dimension_metrics(arrays, rows, sample),
        },
        "physical": _physical(sample, schema),
        "generated_samples": str(sample_path.relative_to(repo_root)),
        "sample_seed": seed,
        "sampling_device": str(device),
    }
    save_json(report, output / "evaluation_summary.json")
    _figures(output, arrays, rows, sample, report)
    return report
