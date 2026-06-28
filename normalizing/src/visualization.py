"""Visual diagnostics for highD EVT-tail joint-density models."""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .data import load_tail_dataset, output_dir_from_config, split_indices
from .metrics import distribution_match_metrics
from .model import load_maf_checkpoint
from .sampling import event_structure_log_prob
from .utils import ensure_dir, save_json, select_device


logger = logging.getLogger(__name__)

LogProbFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

PREFERRED_FEATURES: tuple[str, ...] = (
    "ego_vx_mps",
    "ego_ax_mps2",
    "same_front_rel_x_m",
    "same_front_rel_vx_mps",
    "same_front_delta_vx_1s_mps",
    "same_front_mean_ax_1s_mps2",
    "same_front_min_ax_1s_mps2",
    "same_front_final_ax_1s_mps2",
    "same_front_other_ax_mps2",
    "same_rear_rel_x_m",
    "left_front_rel_x_m",
    "right_front_rel_x_m",
)

FEATURE_LABELS = {
    "ego_vx_mps": "ego vx (m/s)",
    "ego_ax_mps2": "ego ax (m/s2)",
    "same_front_rel_x_m": "same-lane front dx (m)",
    "same_front_rel_vx_mps": "same-lane front dvx (m/s)",
    "same_front_other_ax_mps2": "same-lane front ax (m/s2)",
    "same_front_delta_vx_1s_mps": "same-front dvx 1s (m/s)",
    "same_front_delta_vy_left_1s_mps": "same-front dvy 1s (m/s)",
    "same_front_mean_ax_1s_mps2": "same-front mean ax 1s (m/s2)",
    "same_front_min_ax_1s_mps2": "same-front min ax 1s (m/s2)",
    "same_front_final_ax_1s_mps2": "same-front final ax 1s (m/s2)",
    "same_front_mean_ay_left_1s_mps2": "same-front mean ay 1s (m/s2)",
    "same_rear_rel_x_m": "same-lane rear dx (m)",
    "left_front_rel_x_m": "left-front dx (m)",
    "right_front_rel_x_m": "right-front dx (m)",
}


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def default_checkpoint(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    return output_dir / "checkpoints" / "best_tail_conditional_maf.pt"


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(Path(path), allow_pickle=False)
    return {key: data[key] for key in data.files}


def _load_log_prob_function(
    checkpoint_path: str | Path,
    *,
    repo_root: str | Path,
    device,
) -> tuple[str, LogProbFn]:
    import torch

    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    payload_type = payload.get("type", "conditional_maf")
    if payload_type != "conditional_maf":
        raise ValueError(f"Unsupported checkpoint type={payload_type!r}; expected conditional_maf")

    flow, _payload = load_maf_checkpoint(
        checkpoint_path,
        repo_root=repo_root,
        map_location=device,
    )

    def log_prob(x: np.ndarray, context: np.ndarray, pattern: np.ndarray) -> np.ndarray:
        import torch

        del pattern
        outs: list[np.ndarray] = []
        flow.eval()
        with torch.no_grad():
            for start in range(0, len(x), 1024):
                stop = start + 1024
                xt = torch.from_numpy(x[start:stop]).float().to(device)
                ct = torch.from_numpy(context[start:stop]).float().to(device)
                outs.append(flow.log_prob(xt, context=ct).detach().cpu().numpy())
        return np.concatenate(outs, axis=0).astype(np.float32)

    return str(payload_type), log_prob


def _label(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


def _finite_valid_values(
    features: np.ndarray,
    valid: np.ndarray,
    feature_idx: int,
) -> np.ndarray:
    mask = valid[:, feature_idx] & np.isfinite(features[:, feature_idx])
    return features[mask, feature_idx].astype(np.float64)


def _select_features(
    schema: dict[str, Any],
    real_features: np.ndarray,
    real_valid: np.ndarray,
    generated_features: np.ndarray,
    generated_valid: np.ndarray,
    *,
    min_count: int = 20,
    max_features: int = 6,
) -> list[str]:
    feature_names = list(schema["feature_names"])
    selected: list[str] = []
    for name in PREFERRED_FEATURES:
        if name not in feature_names:
            continue
        idx = feature_names.index(name)
        real_count = int(np.sum(real_valid[:, idx] & np.isfinite(real_features[:, idx])))
        gen_count = int(np.sum(generated_valid[:, idx] & np.isfinite(generated_features[:, idx])))
        if real_count >= int(min_count) and gen_count >= int(min_count):
            selected.append(name)
        if len(selected) >= int(max_features):
            return selected

    for name in feature_names:
        if name in selected:
            continue
        idx = feature_names.index(name)
        real_count = int(np.sum(real_valid[:, idx] & np.isfinite(real_features[:, idx])))
        gen_count = int(np.sum(generated_valid[:, idx] & np.isfinite(generated_features[:, idx])))
        if real_count >= int(min_count) and gen_count >= int(min_count):
            selected.append(name)
        if len(selected) >= int(max_features):
            break
    return selected


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "p05": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
        }
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _combined_limits(real: np.ndarray, generated: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([real, generated])
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0, 1.0
    lo, hi = np.quantile(values, [0.005, 0.995])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.min(values))
        hi = float(np.max(values))
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    pad = 0.04 * (hi - lo)
    return float(lo - pad), float(hi + pad)


def _plot_marginals(
    *,
    output_path: Path,
    schema: dict[str, Any],
    selected_features: list[str],
    real_features: np.ndarray,
    real_valid: np.ndarray,
    generated_features: np.ndarray,
    generated_valid: np.ndarray,
    real_label: str,
) -> None:
    plt = _matplotlib()
    feature_names = list(schema["feature_names"])
    n = len(selected_features)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 3.4 * nrows))
    axes = np.asarray(axes).reshape(-1)
    for ax, name in zip(axes, selected_features):
        idx = feature_names.index(name)
        real = _finite_valid_values(real_features, real_valid, idx)
        generated = _finite_valid_values(generated_features, generated_valid, idx)
        lo, hi = _combined_limits(real, generated)
        bins = np.linspace(lo, hi, 40)
        ax.hist(real, bins=bins, density=True, alpha=0.45, color="#2f6db3", label=real_label)
        ax.hist(generated, bins=bins, density=True, alpha=0.45, color="#d9822b", label="generated")
        ax.set_title(_label(name), fontsize=11)
        ax.set_xlim(lo, hi)
        ax.grid(alpha=0.22, linewidth=0.7)
    for ax in axes[n:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle("highD EVT-tail target marginal distributions", fontsize=15)
    fig.tight_layout(rect=(0, 0, 0.98, 0.94))
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_all_marginals(
    *,
    output_path: Path,
    schema: dict[str, Any],
    real_features: np.ndarray,
    real_valid: np.ndarray,
    generated_features: np.ndarray,
    generated_valid: np.ndarray,
    real_label: str,
) -> None:
    plt = _matplotlib()
    feature_names = list(schema["feature_names"])
    ncols = 8
    nrows = int(np.ceil(len(feature_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.45 * nrows))
    axes = np.asarray(axes).reshape(-1)
    for ax, name in zip(axes, feature_names):
        idx = feature_names.index(name)
        real = _finite_valid_values(real_features, real_valid, idx)
        generated = _finite_valid_values(generated_features, generated_valid, idx)
        if len(real) < 5 or len(generated) < 5:
            ax.axis("off")
            continue
        lo, hi = _combined_limits(real, generated)
        bins = np.linspace(lo, hi, 30)
        ax.hist(real, bins=bins, density=True, alpha=0.44, color="#2f6db3", label=real_label)
        ax.hist(generated, bins=bins, density=True, alpha=0.44, color="#d9822b", label="generated")
        ax.set_title(_label(name), fontsize=7)
        ax.tick_params(axis="both", labelsize=6)
        ax.grid(alpha=0.16, linewidth=0.5)
    for ax in axes[len(feature_names) :]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle("All 76 feature marginal distributions", fontsize=16)
    fig.tight_layout(rect=(0, 0, 0.985, 0.975))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_all_feature_errors(
    *,
    output_path: Path,
    metrics: dict[str, Any],
) -> None:
    plt = _matplotlib()
    rows = list(metrics.get("per_feature", []))
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame["short_feature"] = frame["feature"].astype(str)
    frame = frame.sort_values("ks", ascending=True)
    height = max(12.0, 0.22 * len(frame))
    fig, axes = plt.subplots(1, 2, figsize=(18.0, height), sharey=True)
    y = np.arange(len(frame))
    axes[0].barh(y, frame["ks"].to_numpy(float), color="#6b8fbf")
    axes[0].set_title("KS statistic by feature")
    axes[0].set_xlabel("KS")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(frame["short_feature"].tolist(), fontsize=6)
    axes[1].barh(y, frame["wasserstein"].to_numpy(float), color="#c9884d")
    axes[1].set_title("Wasserstein distance by feature")
    axes[1].set_xlabel("Wasserstein")
    for ax in axes:
        ax.grid(axis="x", alpha=0.22, linewidth=0.7)
    fig.suptitle("All-feature distribution errors", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_all_correlation(
    *,
    output_path: Path,
    schema: dict[str, Any],
    real_normalized: np.ndarray,
    generated_normalized: np.ndarray,
) -> dict[str, float]:
    plt = _matplotlib()
    feature_names = list(schema["feature_names"])

    def corr(x: np.ndarray) -> np.ndarray:
        out = np.corrcoef(np.nan_to_num(x, nan=0.0), rowvar=False)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    real_corr = corr(real_normalized)
    gen_corr = corr(generated_normalized)
    diff = gen_corr - real_corr
    mask = ~np.eye(len(feature_names), dtype=bool)
    corr_mae = float(np.mean(np.abs(diff[mask]))) if len(feature_names) > 1 else 0.0
    fig, axes = plt.subplots(1, 3, figsize=(22.0, 7.0))
    ticks = np.arange(0, len(feature_names), 8)
    tick_labels = [str(idx) for idx in ticks]
    for ax, matrix, title, vmin, vmax, cmap in (
        (axes[0], real_corr, "tail reference Pearson r", -1, 1, "coolwarm"),
        (axes[1], gen_corr, "generated Pearson r", -1, 1, "coolwarm"),
        (axes[2], diff, "generated - reference", -0.75, 0.75, "coolwarm"),
    ):
        image = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(title)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(tick_labels, fontsize=7)
        ax.set_yticklabels(tick_labels, fontsize=7)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"All-feature correlation check (MAE={corr_mae:.4f})", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return {"all_feature_corr_mae": corr_mae}


def _plot_joint_probability(
    *,
    output_path: Path,
    schema: dict[str, Any],
    selected_features: list[str],
    real_features: np.ndarray,
    real_valid: np.ndarray,
    generated_features: np.ndarray,
    generated_valid: np.ndarray,
    real_label: str,
) -> None:
    plt = _matplotlib()
    from matplotlib.lines import Line2D

    feature_names = list(schema["feature_names"])
    names = selected_features[:6]
    n = len(names)
    fig, axes = plt.subplots(n, n, figsize=(3.2 * n, 3.0 * n))
    if n == 1:
        axes = np.asarray([[axes]])
    limits: dict[str, tuple[float, float]] = {}
    for name in names:
        idx = feature_names.index(name)
        limits[name] = _combined_limits(
            _finite_valid_values(real_features, real_valid, idx),
            _finite_valid_values(generated_features, generated_valid, idx),
        )

    def paired_values(
        features: np.ndarray,
        valid: np.ndarray,
        x_idx: int,
        y_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = (
            valid[:, x_idx]
            & valid[:, y_idx]
            & np.isfinite(features[:, x_idx])
            & np.isfinite(features[:, y_idx])
        )
        return features[mask, x_idx].astype(np.float64), features[mask, y_idx].astype(np.float64)

    def subsample(x: np.ndarray, y: np.ndarray, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        if len(x) <= int(max_points):
            return x, y
        rng = np.random.default_rng(int(seed))
        rows = rng.choice(len(x), size=int(max_points), replace=False)
        return x[rows], y[rows]

    def add_kde_contours(
        ax,
        x: np.ndarray,
        y: np.ndarray,
        *,
        color: str,
        linestyle: str,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
    ) -> None:
        if len(x) < 30 or np.std(x) <= 1.0e-9 or np.std(y) <= 1.0e-9:
            return
        try:
            from scipy.stats import gaussian_kde

            values = np.vstack([x, y])
            kde = gaussian_kde(values)
            gx = np.linspace(xlim[0], xlim[1], 55)
            gy = np.linspace(ylim[0], ylim[1], 55)
            xx, yy = np.meshgrid(gx, gy)
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            sample_density = kde(values)
            levels = np.quantile(sample_density[np.isfinite(sample_density)], [0.35, 0.60, 0.82])
            levels = np.unique(levels[np.isfinite(levels)])
            if len(levels) >= 2:
                ax.contour(
                    xx,
                    yy,
                    zz,
                    levels=np.sort(levels),
                    colors=color,
                    linewidths=1.0,
                    linestyles=linestyle,
                    alpha=0.85,
                )
        except Exception:
            return

    for row, y_name in enumerate(names):
        y_idx = feature_names.index(y_name)
        for col, x_name in enumerate(names):
            x_idx = feature_names.index(x_name)
            ax = axes[row, col]
            xlim = limits[x_name]
            ylim = limits[y_name]
            if row == col:
                real = _finite_valid_values(real_features, real_valid, x_idx)
                generated = _finite_valid_values(generated_features, generated_valid, x_idx)
                bins = np.linspace(xlim[0], xlim[1], 35)
                ax.hist(real, bins=bins, density=True, color="#2f6db3", alpha=0.42)
                ax.hist(generated, bins=bins, density=True, color="#d9822b", alpha=0.42)
            else:
                real_mask = (
                    real_valid[:, x_idx]
                    & real_valid[:, y_idx]
                    & np.isfinite(real_features[:, x_idx])
                    & np.isfinite(real_features[:, y_idx])
                )
                gen_mask = (
                    generated_valid[:, x_idx]
                    & generated_valid[:, y_idx]
                    & np.isfinite(generated_features[:, x_idx])
                    & np.isfinite(generated_features[:, y_idx])
                )
                if np.sum(real_mask) > 5:
                    real_x, real_y = paired_values(real_features, real_valid, x_idx, y_idx)
                    real_x_plot, real_y_plot = subsample(real_x, real_y, 450, seed=1000 + row * 31 + col)
                    ax.scatter(
                        real_x_plot,
                        real_y_plot,
                        s=13,
                        alpha=0.42,
                        color="#2f6db3",
                        linewidths=0,
                    )
                    add_kde_contours(ax, real_x, real_y, color="#1f5fa8", linestyle="solid", xlim=xlim, ylim=ylim)
                if np.sum(gen_mask) > 5:
                    gen_x, gen_y = paired_values(generated_features, generated_valid, x_idx, y_idx)
                    gen_x_plot, gen_y_plot = subsample(gen_x, gen_y, 700, seed=2000 + row * 31 + col)
                    ax.scatter(
                        gen_x_plot,
                        gen_y_plot,
                        s=5.5,
                        alpha=0.23,
                        color="#d9822b",
                        linewidths=0,
                    )
                    add_kde_contours(ax, gen_x, gen_y, color="#c56f1d", linestyle="dashed", xlim=xlim, ylim=ylim)
            if row == n - 1:
                ax.set_xlabel(_label(x_name), fontsize=9)
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(_label(y_name), fontsize=9)
            else:
                ax.set_yticklabels([])
            ax.set_xlim(*xlim)
            if row != col:
                ax.set_ylim(*ylim)
            ax.grid(alpha=0.14, linewidth=0.6)

    fig.suptitle(
        "Joint tail-event target distribution: scatter with KDE contours",
        fontsize=15,
    )
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2f6db3", markersize=6, alpha=0.55, label=real_label),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d9822b", markersize=5, alpha=0.55, label="generated"),
        Line2D([0], [0], color="#1f5fa8", lw=1.2, label=f"{real_label} KDE"),
        Line2D([0], [0], color="#c56f1d", lw=1.2, linestyle="dashed", label="generated KDE"),
    ]
    fig.legend(handles=handles, loc="upper right", frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_correlation(
    *,
    output_path: Path,
    schema: dict[str, Any],
    selected_features: list[str],
    real_features: np.ndarray,
    real_valid: np.ndarray,
    generated_features: np.ndarray,
    generated_valid: np.ndarray,
    real_label: str,
) -> dict[str, float]:
    plt = _matplotlib()
    feature_names = list(schema["feature_names"])
    indices = np.asarray([feature_names.index(name) for name in selected_features], dtype=np.int64)

    def corr(features: np.ndarray, valid: np.ndarray) -> np.ndarray:
        mask = np.all(valid[:, indices], axis=1) & np.all(np.isfinite(features[:, indices]), axis=1)
        if np.sum(mask) < 3:
            return np.zeros((len(indices), len(indices)), dtype=np.float64)
        out = np.corrcoef(features[mask][:, indices], rowvar=False)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    real_corr = corr(real_features, real_valid)
    gen_corr = corr(generated_features, generated_valid)
    diff = gen_corr - real_corr
    mask = ~np.eye(len(indices), dtype=bool)
    corr_mae = float(np.mean(np.abs(diff[mask]))) if len(indices) > 1 else 0.0

    labels = [_label(name) for name in selected_features]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    for ax, matrix, title, vmin, vmax, cmap in (
        (axes[0], real_corr, f"{real_label} Pearson r", -1, 1, "coolwarm"),
        (axes[1], gen_corr, "generated Pearson r", -1, 1, "coolwarm"),
        (axes[2], diff, f"generated - {real_label}", -0.75, 0.75, "coolwarm"),
    ):
        image = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Joint dependence correlation check (MAE={corr_mae:.4f})", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return {"selected_feature_corr_mae": corr_mae}


def _plot_probability_diagnostics(
    *,
    output_path: Path,
    output_dir: Path,
    score_frame: pd.DataFrame,
    real_source: str,
    real_label: str,
) -> None:
    plt = _matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.2))
    real = score_frame.loc[score_frame["source"] == real_source, "log_prob"].to_numpy(float)
    generated = score_frame.loc[score_frame["source"] == "generated", "log_prob"].to_numpy(float)
    bins = np.linspace(
        min(np.nanpercentile(real, 1), np.nanpercentile(generated, 1)),
        max(np.nanpercentile(real, 99), np.nanpercentile(generated, 99)),
        45,
    )
    axes[0, 0].hist(real, bins=bins, density=True, alpha=0.48, color="#2f6db3", label=real_label)
    axes[0, 0].hist(generated, bins=bins, density=True, alpha=0.48, color="#d9822b", label="generated")
    axes[0, 0].set_title("Model joint log probability")
    axes[0, 0].set_xlabel("log p(event | EVT-tail)")
    axes[0, 0].legend(frameon=False)

    for values, label, color in ((-real, real_label, "#2f6db3"), (-generated, "generated", "#d9822b")):
        values = np.sort(values[np.isfinite(values)])
        y = np.arange(1, len(values) + 1) / max(len(values), 1)
        axes[0, 1].plot(values, y, label=label, color=color, linewidth=2)
    axes[0, 1].set_title("NLL empirical CDF")
    axes[0, 1].set_xlabel("-log p(event | EVT-tail)")
    axes[0, 1].set_ylabel("empirical probability")
    axes[0, 1].legend(frameon=False)

    top_patterns = (
        score_frame["mask_pattern"]
        .value_counts()
        .sort_values(ascending=False)
        .head(8)
        .index.astype(int)
        .tolist()
    )
    positions: list[float] = []
    data: list[np.ndarray] = []
    labels: list[str] = []
    for idx, pattern in enumerate(top_patterns):
        for source, offset in ((real_source, -0.18), ("generated", 0.18)):
            values = score_frame.loc[
                (score_frame["source"] == source) & (score_frame["mask_pattern"] == pattern),
                "nll",
            ].to_numpy(float)
            if len(values) == 0:
                continue
            positions.append(idx + offset)
            data.append(values)
            labels.append(source)
    if data:
        bp = axes[1, 0].boxplot(data, positions=positions, widths=0.28, patch_artist=True, showfliers=False)
        for patch, label in zip(bp["boxes"], labels):
            patch.set_facecolor("#2f6db3" if label == real_source else "#d9822b")
            patch.set_alpha(0.48)
        axes[1, 0].set_xticks(np.arange(len(top_patterns)))
        axes[1, 0].set_xticklabels([str(p) for p in top_patterns], rotation=25, ha="right")
    axes[1, 0].set_title("Conditional NLL by slot-mask pattern")
    axes[1, 0].set_ylabel("NLL")

    nll_csv = output_dir / "diagnostics" / "nll_comparison.csv"
    if nll_csv.exists():
        table = pd.read_csv(nll_csv)
        if "test" in table.columns:
            table = table.sort_values("test")
            axes[1, 1].barh(table["model"], table["test"], color="#6b7280")
            axes[1, 1].set_xlabel("held-out test NLL")
            axes[1, 1].set_title("Density model comparison")
    else:
        axes[1, 1].axis("off")
    for ax in axes.reshape(-1):
        ax.grid(alpha=0.22, linewidth=0.7)
    fig.suptitle("Normalizing-flow joint probability diagnostics", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_context_occupancy(
    *,
    output_path: Path,
    real_mask_pattern: np.ndarray,
    generated_mask_pattern: np.ndarray,
    real_primary_slot: np.ndarray,
    generated_primary_slot: np.ndarray,
    real_label: str,
) -> None:
    plt = _matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.0))

    slot_abbrev = {
        "same_front": "SF",
        "same_rear": "SR",
        "left_front": "LF",
        "left_rear": "LR",
        "right_front": "RF",
        "right_rear": "RR",
    }

    def pattern_label(pattern: int) -> str:
        active = [
            slot_abbrev[name]
            for idx, name in enumerate(
                ("same_front", "same_rear", "left_front", "left_rear", "right_front", "right_rear")
            )
            if int(pattern) & (1 << idx)
        ]
        return f"{int(pattern)}\n{'+'.join(active) if active else 'none'}"

    patterns = sorted(set(real_mask_pattern.astype(int).tolist()) | set(generated_mask_pattern.astype(int).tolist()))
    x = np.arange(len(patterns))
    width = 0.38
    real_counts = np.asarray([np.mean(real_mask_pattern.astype(int) == p) for p in patterns])
    gen_counts = np.asarray([np.mean(generated_mask_pattern.astype(int) == p) for p in patterns])
    ax = axes[0]
    ax.bar(x - width / 2, real_counts, width=width, color="#2f6db3", alpha=0.68, label=real_label)
    ax.bar(x + width / 2, gen_counts, width=width, color="#d9822b", alpha=0.68, label="generated")
    ax.set_xticks(x)
    ax.set_xticklabels([pattern_label(p) for p in patterns], rotation=45, ha="right", fontsize=8)
    ax.set_title("Slot-mask pattern occupancy")
    ax.set_xlabel("mask pattern: bit-coded active slots")
    ax.set_ylabel("fraction")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)

    primary_slots = sorted(set(real_primary_slot.astype(str).tolist()) | set(generated_primary_slot.astype(str).tolist()))
    x = np.arange(len(primary_slots))
    real_counts = np.asarray([np.mean(real_primary_slot.astype(str) == slot) for slot in primary_slots])
    gen_counts = np.asarray([np.mean(generated_primary_slot.astype(str) == slot) for slot in primary_slots])
    ax = axes[1]
    ax.bar(x - width / 2, real_counts, width=width, color="#2f6db3", alpha=0.68, label=real_label)
    ax.bar(x + width / 2, gen_counts, width=width, color="#d9822b", alpha=0.68, label="generated")
    ax.set_xticks(x)
    ax.set_xticklabels(primary_slots, rotation=35, ha="right")
    ax.set_title("Primary-slot occupancy")
    ax.set_ylabel("fraction")
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)

    fig.suptitle("Discrete event-structure occupancy", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def write_tail_flow_visual_diagnostics(
    config: dict[str, Any],
    *,
    config_dir: str | Path,
    repo_root: str | Path,
    checkpoint_path: str | Path | None = None,
    sample_npz: str | Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Render audit figures for the learned joint distribution and samples."""
    config_dir = Path(config_dir).resolve()
    output_dir = output_dir_from_config(config, config_dir)
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else default_checkpoint(output_dir)
    sample_npz = Path(sample_npz) if sample_npz else output_dir / "samples" / "generated_samples.npz"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    if not sample_npz.exists():
        raise FileNotFoundError(f"Missing generated sample NPZ: {sample_npz}")

    arrays, schema = load_tail_dataset(output_dir)
    samples = _load_npz(sample_npz)
    selected_device = select_device(device or str(config.get("device", "auto")))
    eval_cfg = dict(config.get("evaluation", {}))
    reference_split = str(eval_cfg.get("distribution_reference_split") or "all").lower()
    real_source = "tail_all" if reference_split in {"all", "full", "dataset"} else f"tail_{reference_split}"
    real_label = real_source.replace("_", " ")
    event_structure_split = str(eval_cfg.get("sample_event_structure_split") or reference_split)
    model_type, log_prob_fn = _load_log_prob_function(
        checkpoint_path,
        repo_root=repo_root,
        device=selected_device,
    )

    real_idx = split_indices(arrays, reference_split)
    real_continuous_log_prob = log_prob_fn(
        arrays["features_normalized"][real_idx].astype(np.float32),
        arrays["contexts"][real_idx].astype(np.float32),
        arrays["mask_pattern"][real_idx].astype(np.int64),
    )
    real_event_log_prob = event_structure_log_prob(
        arrays,
        mask_pattern=arrays["mask_pattern"][real_idx].astype(np.int64),
        primary_slot_index=arrays["primary_slot_index"][real_idx].astype(np.int64),
        split=event_structure_split,
    )
    real_log_prob = real_continuous_log_prob + real_event_log_prob
    generated_continuous_log_prob = log_prob_fn(
        samples["features_normalized"].astype(np.float32),
        samples["contexts"].astype(np.float32),
        samples["mask_pattern"].astype(np.int64),
    )
    generated_event_log_prob = event_structure_log_prob(
        arrays,
        mask_pattern=samples["mask_pattern"].astype(np.int64),
        primary_slot_index=samples["primary_slot_index"].astype(np.int64),
        split=event_structure_split,
    )
    generated_log_prob = generated_continuous_log_prob + generated_event_log_prob

    real_score_frame = pd.DataFrame(
        {
            "source": real_source,
            "row_index": real_idx.astype(np.int64),
            "log_prob": real_log_prob.astype(np.float32),
            "continuous_log_prob": real_continuous_log_prob.astype(np.float32),
            "event_structure_log_prob": real_event_log_prob.astype(np.float32),
            "nll": (-real_log_prob).astype(np.float32),
            "mask_pattern": arrays["mask_pattern"][real_idx].astype(np.int64),
            "primary_slot": arrays["primary_slot_name"][real_idx].astype(str),
            "event_risk": arrays["event_risk"][real_idx].astype(np.float32),
        }
    )
    generated_score_frame = pd.DataFrame(
        {
            "source": "generated",
            "row_index": np.arange(len(generated_log_prob), dtype=np.int64),
            "log_prob": generated_log_prob.astype(np.float32),
            "continuous_log_prob": generated_continuous_log_prob.astype(np.float32),
            "event_structure_log_prob": generated_event_log_prob.astype(np.float32),
            "nll": (-generated_log_prob).astype(np.float32),
            "mask_pattern": samples["mask_pattern"].astype(np.int64),
            "primary_slot": samples["primary_slot_name"].astype(str),
            "event_risk": np.full(len(generated_log_prob), np.nan, dtype=np.float32),
        }
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
            category=FutureWarning,
        )
        score_frame = pd.concat([real_score_frame, generated_score_frame], ignore_index=True)
    diagnostics_dir = ensure_dir(output_dir / "diagnostics")
    figures_dir = ensure_dir(output_dir / "figures")
    score_csv = diagnostics_dir / "joint_probability_scores.csv"
    score_frame.to_csv(score_csv, index=False)

    real_features = arrays["features"][real_idx]
    real_valid = arrays["feature_valid"][real_idx]
    selected_features = _select_features(
        schema,
        real_features,
        real_valid,
        samples["features"],
        samples["feature_valid"],
    )
    if len(selected_features) < 2:
        raise RuntimeError("Not enough valid features for visual diagnostics")

    figure_paths = {
        "tail_c0_marginal_distributions": figures_dir / "tail_c0_marginal_distributions.png",
        "tail_c0_all_marginal_distributions": figures_dir / "tail_c0_all_marginal_distributions.png",
        "tail_c0_all_feature_distribution_errors": figures_dir
        / "tail_c0_all_feature_distribution_errors.png",
        "tail_c0_all_correlation_tail_vs_generated": figures_dir
        / "tail_c0_all_correlation_tail_vs_generated.png",
        "tail_c0_joint_probability_tail_vs_generated": figures_dir
        / "tail_c0_joint_probability_tail_vs_generated.png",
        "tail_c0_correlation_tail_vs_generated": figures_dir
        / "tail_c0_correlation_tail_vs_generated.png",
        "tail_c0_probability_diagnostics": figures_dir / "tail_c0_probability_diagnostics.png",
        "tail_c0_context_occupancy_tail_vs_generated": figures_dir
        / "tail_c0_context_occupancy_tail_vs_generated.png",
    }
    _plot_marginals(
        output_path=figure_paths["tail_c0_marginal_distributions"],
        schema=schema,
        selected_features=selected_features,
        real_features=real_features,
        real_valid=real_valid,
        generated_features=samples["features"],
        generated_valid=samples["feature_valid"],
        real_label=real_label,
    )
    distribution_metrics = distribution_match_metrics(
        real_features,
        samples["features"],
        real_valid,
        samples["feature_valid"],
        list(schema["feature_names"]),
    )
    _plot_all_marginals(
        output_path=figure_paths["tail_c0_all_marginal_distributions"],
        schema=schema,
        real_features=real_features,
        real_valid=real_valid,
        generated_features=samples["features"],
        generated_valid=samples["feature_valid"],
        real_label=real_label,
    )
    _plot_all_feature_errors(
        output_path=figure_paths["tail_c0_all_feature_distribution_errors"],
        metrics=distribution_metrics,
    )
    all_corr_metrics = _plot_all_correlation(
        output_path=figure_paths["tail_c0_all_correlation_tail_vs_generated"],
        schema=schema,
        real_normalized=arrays["features_normalized"][real_idx],
        generated_normalized=samples["features_normalized"],
    )
    _plot_joint_probability(
        output_path=figure_paths["tail_c0_joint_probability_tail_vs_generated"],
        schema=schema,
        selected_features=selected_features,
        real_features=real_features,
        real_valid=real_valid,
        generated_features=samples["features"],
        generated_valid=samples["feature_valid"],
        real_label=real_label,
    )
    corr_metrics = _plot_correlation(
        output_path=figure_paths["tail_c0_correlation_tail_vs_generated"],
        schema=schema,
        selected_features=selected_features,
        real_features=real_features,
        real_valid=real_valid,
        generated_features=samples["features"],
        generated_valid=samples["feature_valid"],
        real_label=real_label,
    )
    _plot_probability_diagnostics(
        output_path=figure_paths["tail_c0_probability_diagnostics"],
        output_dir=output_dir,
        score_frame=score_frame,
        real_source=real_source,
        real_label=real_label,
    )
    _plot_context_occupancy(
        output_path=figure_paths["tail_c0_context_occupancy_tail_vs_generated"],
        real_mask_pattern=arrays["mask_pattern"][real_idx],
        generated_mask_pattern=samples["mask_pattern"],
        real_primary_slot=arrays["primary_slot_name"][real_idx],
        generated_primary_slot=samples["primary_slot_name"],
        real_label=real_label,
    )

    summary = {
        "model_type": model_type,
        "checkpoint": str(checkpoint_path),
        "sample_npz": str(sample_npz),
        "distribution_reference_split": reference_split,
        "num_real_reference": int(len(real_idx)),
        "num_generated": int(len(generated_log_prob)),
        "selected_features": selected_features,
        "joint_probability_score_csv": str(score_csv),
        "log_prob_summary": {
            real_source: _summary_stats(real_log_prob),
            "generated": _summary_stats(generated_log_prob),
        },
        "nll_summary": {
            real_source: _summary_stats(-real_log_prob),
            "generated": _summary_stats(-generated_log_prob),
        },
        "correlation": corr_metrics,
        "all_feature_distribution_match": {
            key: value
            for key, value in distribution_metrics.items()
            if key != "per_feature"
        },
        "all_feature_correlation": all_corr_metrics,
        "figures": {key: str(path) for key, path in figure_paths.items()},
    }
    save_json(summary, diagnostics_dir / "visualization_summary.json")
    logger.info("Wrote visual diagnostics to %s", figures_dir)
    return summary
