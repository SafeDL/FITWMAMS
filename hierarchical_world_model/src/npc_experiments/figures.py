"""Publication PNGs of conditional behavior and autonomous traffic execution."""

from pathlib import Path

import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from tools.plot_style import (
    GENERATED_COLOR,
    PAPER_FIGURE_DPI,
    REAL_COLOR,
    get_pyplot,
    style_axes,
)

from ..core.utils import load_json
from ..stochastic_drivers.raw_data import DRIVER_DATA_DIR
from .metrics import (
    FEATURE_EDGES,
    JOINT_EDGES,
)

STAGES = ("reference",)
COLORS = {
    "real": REAL_COLOR,
    "reference": GENERATED_COLOR,
}
LABELS = {
    "real": "Observed highD",
    "reference": "NPC response model",
}
LINES = {
    "real": "-",
    "reference": "--",
}
FIGURE_FILES = (
    "driver_parameters.png",
    "conditional_behavior.png",
    "traffic_distributions.png",
    "lateral_consistency.png",
    "distribution_stability.png",
    "diagnostics.png",
)


def canvas(
    title,
    subtitle,
    stages=STAGES,
    *,
    shape=(2, 3),
    figsize=(15.6, 9.4),
    include_real=True,
    histogram=False,
):
    plt = get_pyplot()
    plt.rcParams.update(
        {
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "axes.linewidth": 0.8,
        }
    )
    figure, axes = plt.subplots(*shape, figsize=figsize, squeeze=False)
    owners = ("real", *stages) if include_real else stages
    columns = min(3, len(owners))
    legend_rows = int(np.ceil(len(owners) / columns))
    figure.subplots_adjust(
        left=0.07,
        right=0.975,
        bottom=0.10,
        top=0.85 - 0.035 * legend_rows,
        hspace=0.43,
        wspace=0.34,
    )
    figure.suptitle(title, y=0.982, fontsize=18)
    figure.text(0.5, 0.94, subtitle, ha="center", fontsize=11, color="#586879")
    handles = [
        Line2D([], [], color=COLORS[name], linestyle=LINES[name], lw=2.2)
        for name in owners
    ]
    if histogram:
        handles[0] = Patch(facecolor=to_rgba(COLORS["real"], 0.38), edgecolor="white")
    figure.legend(
        handles,
        [LABELS[name] for name in owners],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.908),
        ncol=columns,
        frameon=False,
        handlelength=2.7,
        columnspacing=1.8,
    )
    for index, axis in enumerate(axes.flat):
        style_axes(axis)
        axis.grid(False)
        axis.yaxis.grid(True, color="#DFE5EB", alpha=0.65)
        axis.set_axisbelow(True)
        axis.text(
            -0.12,
            1.05,
            f"({chr(97 + index)})",
            transform=axis.transAxes,
            fontsize=15,
            fontweight="bold",
        )
    return plt, figure, axes


def finish(plt, figure, destination, caption):
    figure.text(0.07, 0.022, caption, ha="left", fontsize=10, color="#586879")
    figure.savefig(destination, dpi=PAPER_FIGURE_DPI)
    plt.close(figure)


def binned_density(counts, edges):
    """Finite-bin density retains underflow and overflow mass in normalization."""
    counts, edges = np.asarray(counts, float), np.asarray(edges)
    return counts[1:-1] / max(counts.sum(), 1) / np.diff(edges[1:-1])


def density_histogram(axis, counts, edges, stages=STAGES, view=None):
    """Touching observed bars and unfilled model histograms on identical physical bins."""
    edges = np.asarray(edges)
    finite_edges = edges[1:-1]
    widths = np.diff(finite_edges)
    for owner in ("real", *stages):
        count = np.asarray(counts[owner], float)
        density = binned_density(count, edges)
        if owner == "real":
            axis.bar(
                finite_edges[:-1],
                density,
                width=widths,
                align="edge",
                facecolor=to_rgba(COLORS[owner], 0.38),
                edgecolor="white",
                linewidth=0.65,
                zorder=2,
            )
        else:
            axis.stairs(
                density,
                finite_edges,
                color=COLORS[owner],
                linestyle=LINES[owner],
                linewidth=1.8,
                zorder=3,
            )
    if view is not None:
        axis.set_xlim(view)
    axis.set_ylim(bottom=0)
    axis.set_ylabel("Probability density")


def heatmaps(
    figure, axes, observed, generated, xedges, yedges, xlabel, ylabel, *, totals=None
):
    xedges, yedges = np.asarray(xedges), np.asarray(yedges)
    totals = (observed.sum(), generated.sum()) if totals is None else totals
    r = observed[1:-1, 1:-1] / max(totals[0], 1)
    m = generated[1:-1, 1:-1] / max(totals[1], 1)
    area = np.diff(xedges[1:-1])[:, None] * np.diff(yedges[1:-1])[None]
    r, m = r / area, m / area
    positive = np.r_[r[r > 0], m[m > 0]]
    minimum = max(float(positive.min()) if len(positive) else 1e-8, 1e-8)
    maximum = max(float(positive.max()) if len(positive) else 1e-7, minimum * 1.01)
    norm = LogNorm(vmin=minimum, vmax=maximum)
    images = []
    for axis, value, title in zip(
        axes[:2],
        (r, m),
        ("Observed interaction density", "Simulated interaction density"),
    ):
        images.append(
            axis.pcolormesh(
                xedges[1:-1],
                yedges[1:-1],
                np.ma.masked_where(value.T <= 0, value.T),
                cmap="Blues",
                norm=norm,
                shading="flat",
                rasterized=True,
            )
        )
        axis.set_title(title, loc="left")
    colorbar = figure.colorbar(images[0], ax=list(axes[:2]), fraction=0.026, pad=0.025)
    colorbar.set_label("Probability density", fontsize=10)
    delta = m - r
    limit = max(float(np.abs(delta).max()), 1e-8)
    image = axes[2].pcolormesh(
        xedges[1:-1],
        yedges[1:-1],
        delta.T,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        shading="flat",
        rasterized=True,
    )
    axes[2].set_title("Model minus observed", loc="left")
    colorbar = figure.colorbar(image, ax=axes[2], fraction=0.05, pad=0.025)
    colorbar.set_label("Density difference", fontsize=10)
    for axis in axes:
        axis.set(xlabel=xlabel, ylabel=ylabel)
        axis.grid(False)


def clustered_ratio(values, recording):
    values, recording = np.asarray(values, float), np.asarray(recording)
    total = values.sum(0)
    if total[1] == 0:
        return np.nan, np.array([np.nan, np.nan])
    unique = np.unique(recording)
    clusters = np.stack([values[recording == key].sum(0) for key in unique])
    draws = np.random.default_rng(20261012).integers(
        len(unique), size=(2000, len(unique))
    )
    sums = clusters[draws].sum(1)
    keep = sums[:, 1] > 0
    return total[0] / total[1], np.quantile(
        sums[keep, 0] / sums[keep, 1], [0.025, 0.975]
    )


def aggregate(numeric, cohorts, owner, field):
    return sum(numeric[f"{key}_{owner}_{field}"].mean(0) for key in cohorts)


def plot_npc_results(output):
    from ..stochastic_drivers.idm_population import sample_theta
    from .evaluation import ROLLOUT_DIRECTORIES, stage_models

    output = Path(output)
    selected, models = stage_models(output)
    stages = tuple(models)
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    report = load_json(output / "autonomous_traffic/report.json")
    numeric = {}
    for section, folder in ROLLOUT_DIRECTORIES.items():
        with np.load(
            output / "autonomous_traffic" / folder / "statistics.npz"
        ) as saved:
            for name in saved.files:
                recording, direction, field = name.split("_", 2)
                owner, _, feature = field.partition("_")
                if owner in ("real", "model"):
                    owner = selected if owner == "model" else owner
                    numeric[f"{section}_{recording}_{direction}_{owner}_{feature}"] = (
                        saved[name]
                    )
    conditional = dict(np.load(output / "conditional_behavior/statistics.npz"))
    keys = report["long"]["cohorts"]
    protocol = load_json(DRIVER_DATA_DIR / "lateral_protocol.json")
    split = "test"

    def panel(title, shape=(2, 3), real=True):
        return canvas(
            title,
            "Natural traffic evaluation · actual highway-env execution · recording-cluster statistics",
            stages,
            shape=shape,
            include_real=real,
            histogram=real,
        )

    plt, figure, axes = panel("Autonomous traffic distributions")
    for axis, (feature, label, view) in zip(
        axes[0],
        (
            ("speed", "Speed [m/s]", (5, 55)),
            ("gap", "Net gap [m]", (0, 150)),
            ("acceleration", "Longitudinal acceleration [m/s²]", (-3, 2)),
        ),
    ):
        counts = {
            owner: aggregate(numeric, keys, owner, feature).sum(0)
            for owner in ("real", *stages)
        }
        density_histogram(axis, counts, FEATURE_EDGES[feature], stages, view)
        axis.set_xlabel(label)
    real = (
        aggregate(numeric, keys, "real", "joint")
        .sum(0)
        .reshape(len(JOINT_EDGES[0]) - 1, -1)
    )
    generated = aggregate(numeric, keys, selected, "joint").sum(0).reshape(real.shape)
    heatmaps(
        figure,
        axes[1],
        real,
        generated,
        *JOINT_EDGES,
        "Net gap [m]",
        "Leader minus follower speed [m/s]",
    )
    finish(
        plt,
        figure,
        directory / "traffic_distributions.png",
        "Identical physical bins, derivative processing and vehicle-time weights. Overflow mass is retained.",
    )

    plt, figure, axes = panel("Lane-change occurrence and execution")
    for axis, feature, label in zip(
        axes[0],
        ("duration", "peak_lateral_speed", "peak_lateral_acceleration"),
        (
            "Duration [s]",
            "Peak lateral speed [m/s]",
            "Peak lateral acceleration [m/s²]",
        ),
    ):
        counts = {
            owner: aggregate(numeric, keys, owner, "lane_" + feature)
            for owner in ("real", *stages)
        }
        density_histogram(axis, counts, protocol["edges"][feature], stages)
        axis.set_xlabel(label)
    recording = [int(key.split("_")[1]) for key in keys]
    for index, owner in enumerate(("real", *stages)):
        exposures = np.stack(
            [numeric[f"{key}_{owner}_exposure"].mean(0).sum(0) for key in keys]
        )
        value, interval = clustered_ratio(exposures[:, [2, 1]], recording)
        axes[1, 0].plot(interval * 1000, [index, index], color=COLORS[owner], lw=2)
        axes[1, 0].plot(value * 1000, index, "o", color=COLORS[owner])
        count = aggregate(numeric, keys, owner, "phase_count")
        curve = aggregate(numeric, keys, owner, "phase_sum") / max(float(count), 1)
        for axis, component in zip(axes[1, 1:], range(2)):
            axis.plot(
                np.linspace(0, 1, 101),
                curve[component],
                color=COLORS[owner],
                ls=LINES[owner],
                lw=1.8,
            )
    axes[1, 0].set(
        yticks=range(1 + len(stages)),
        yticklabels=[LABELS[o] for o in ("real", *stages)],
        xlabel="Completed changes / 1,000 vehicle-km",
    )
    axes[1, 1].set(xlabel="Maneuver phase", ylabel="Lane-normalized lateral position")
    axes[1, 2].set(xlabel="Maneuver phase", ylabel="Directed lateral speed [m/s]")
    finish(
        plt,
        figure,
        directory / "lateral_consistency.png",
        "Completion and censoring use the common trajectory extractor. Repeats do not add observed human events.",
    )
    plt, figure, axes = panel("Persistent driver populations", real=False)
    latent = np.random.default_rng(20261012).standard_normal((10000, 5))
    samples = {stage: sample_theta(model[1], latent) for stage, model in models.items()}
    labels = (
        "Desired speed [m/s]",
        "Standstill gap [m]",
        "Desired time headway [s]",
        "Acceleration parameter [m/s²]",
        "Braking parameter [m/s²]",
    )
    for index, axis in enumerate(axes.flat[:5]):
        values = np.concatenate([sample[:, index] for sample in samples.values()])
        edges = np.linspace(values.min(), values.max(), 36)
        for stage, sample in samples.items():
            counts, _ = np.histogram(sample[:, index], edges)
            axis.stairs(
                counts / counts.sum() / np.diff(edges),
                edges,
                color=COLORS[stage],
                ls=LINES[stage],
                lw=1.8,
            )
        axis.set(xlabel=labels[index], ylabel="Probability density")
    covariance = models[selected][1]["log_cov"]
    scale = np.sqrt(np.diag(covariance))
    im = axes.flat[5].imshow(
        covariance / scale[:, None] / scale[None], vmin=-1, vmax=1, cmap="RdBu_r"
    )
    axes.flat[5].set(
        xticks=range(5),
        xticklabels=("$v_0$", "$s_0$", "T", "a", "b"),
        yticks=range(5),
        yticklabels=("$v_0$", "$s_0$", "T", "a", "b"),
        title="Joint log-parameter correlation",
    )
    figure.colorbar(im, ax=axes.flat[5], fraction=0.045)
    finish(
        plt,
        figure,
        directory / "driver_parameters.png",
        "One persistent style per vehicle; the correlated population is sampled with common base randomness.",
    )

    plt, figure, axes = panel("Conditional behavior and lane-choice calibration")
    for axis, group in zip(
        axes[0], ("free_driving", "steady_following", "approaching")
    ):
        counts = {"real": conditional[f"{split}__{selected}__{group}_real"].sum(0)}
        counts.update(
            {
                stage: conditional[f"{split}__{stage}__{group}_model"].sum(0)
                for stage in stages
            }
        )
        density_histogram(axis, counts, FEATURE_EDGES["acceleration"], stages, (-3, 2))
        axis.set(
            xlabel="Acceleration [m/s²]", title=group.replace("_", " ").capitalize()
        )
    for side, axis in enumerate(axes[1, :2]):
        for owner in ("real", *stages):
            stage = selected if owner == "real" else owner
            curve = conditional[f"{split}__{stage}__lane_side_probability"][side]
            value = curve[:, 1 if owner == "real" else 2]
            axis.plot(
                [7.5, 22.5, 45, 90],
                value,
                marker="o",
                color=COLORS[owner],
                ls=LINES[owner],
            )
        axis.set(
            xlabel="Target rear gap group [m]",
            ylabel="One-second choice probability",
            title=("Lower-y lane", "Higher-y lane")[side],
        )
    joint = conditional[f"{split}__{selected}__start_gap_joint_model_recording"].sum(0)
    real_joint = conditional[
        f"{split}__{selected}__start_gap_joint_real_recording"
    ].sum(0)
    delta = joint / max(joint.sum(), 1) - real_joint / max(real_joint.sum(), 1)
    im = axes[1, 2].imshow(
        delta.T,
        origin="lower",
        cmap="RdBu_r",
        vmin=-max(abs(delta).max(), 1e-8),
        vmax=max(abs(delta).max(), 1e-8),
    )
    axes[1, 2].set(
        xlabel="Target front-gap bin",
        ylabel="Target rear-gap bin",
        title="Start-condition probability difference",
    )
    figure.colorbar(im, ax=axes[1, 2], fraction=0.045)
    finish(
        plt,
        figure,
        directory / "conditional_behavior.png",
        "Observed state conditions are held fixed. The last row/column preserves absence of a visible neighbor.",
    )

    plt, figure, axes = panel("Continuous traffic stability", shape=(2, 2), real=False)
    for axis, metric in zip(axes.flat, ("speed", "joint", "acceleration", "gap")):
        for stage in stages:
            item = report["long"]["models"][stage][metric]
            time = np.arange(len(item["window"])) * 10 + 5
            interval = np.asarray(item["window_CI95"])
            axis.plot(
                time, item["window"], color=COLORS[stage], ls=LINES[stage], lw=1.8
            )
            axis.fill_between(
                time, interval[:, 0], interval[:, 1], color=COLORS[stage], alpha=0.08
            )
        axis.set(
            xlabel="Elapsed simulation time [s]",
            ylabel="Hellinger distance",
            title=metric.replace("joint", "Net gap–relative speed"),
        )
    finish(
        plt,
        figure,
        directory / "distribution_stability.png",
        "Ten-second matched windows; no state reset and no discarded warm-up segment.",
    )

    plt, figure, axes = panel("Motion and interaction diagnostics")
    for axis, feature, label in zip(
        axes[0],
        ("acceleration", "headway", "ttc"),
        ("Acceleration [m/s²]", "Time headway [s]", "Time to collision [s]"),
    ):
        counts = {
            owner: aggregate(numeric, keys, owner, feature).sum(0)
            for owner in ("real", *stages)
        }
        density_histogram(axis, counts, FEATURE_EDGES[feature], stages)
        axis.set_xlabel(label)
    for stage in stages:
        temporal = report["long"]["models"][stage]["temporal"]
        axes[1, 0].plot(
            temporal["lag_seconds"],
            temporal["model"]["correlation"],
            marker="o",
            color=COLORS[stage],
            ls=LINES[stage],
        )
    axes[1, 0].plot(
        temporal["lag_seconds"],
        temporal["real"]["correlation"],
        marker="o",
        color=COLORS["real"],
    )
    axes[1, 0].set(xlabel="Lag [s]", ylabel="Acceleration residual correlation")
    for axis, feature in zip(axes[1, 1:], ("speed", "joint")):
        for owner in ("real", *stages):
            counts = aggregate(numeric, keys, owner, "age_" + feature)
            if owner == "real":
                continue
            observed = aggregate(numeric, keys, "real", "age_" + feature)
            from .lateral import distance

            axis.plot(
                range(4),
                distance(observed, counts),
                marker="o",
                color=COLORS[owner],
                ls=LINES[owner],
            )
        axis.set(
            xticks=range(4),
            xticklabels=("0–2", "2–5", "5–10", ">10"),
            xlabel="Residence time [s]",
            ylabel="Hellinger distance",
            title=feature,
        )
    finish(
        plt,
        figure,
        directory / "diagnostics.png",
        "Non-closing TTC is retained in the overflow category. Acceleration and risk variables are auxiliary diagnostics.",
    )
