"""Shared plotting style for paper-oriented result figures."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


REAL_COLOR = "#4C78A8"
GENERATED_COLOR = "#F58518"
SAMPLED_COLOR = "#54A24B"
REFERENCE_COLOR = "#333333"
CRITICAL_COLOR = "#E45756"
PAPER_FIGURE_DPI = 300
PAPER_SERIF_FONTS = [
    "Times New Roman",
    "Times",
    "Nimbus Roman",
    "Liberation Serif",
    "DejaVu Serif",
]
PAPER_PANEL_LABELSIZE = 16.0
PAPER_ANNOTATION_FONTSIZE = 11.5
PAPER_PROFILE_LABELSIZE = 12.0
PAPER_SINGLE_PANEL_FIGSIZE = (6.8, 4.6)
PAPER_SUBSET_HISTOGRAM_FIGSIZE = (6.8, 4.4)
PAPER_PROFILE_FIGSIZE = (7.7, 5.2)
PAPER_SIX_PANEL_FIGSIZE = (14.4, 8.2)
PAPER_SIX_PANEL_LAYOUT = {"pad": 1.05, "w_pad": 1.35, "h_pad": 1.65}
PAPER_NOTE_BBOX = {
    "boxstyle": "round,pad=0.22",
    "facecolor": "white",
    "edgecolor": "#BDBDBD",
    "linewidth": 0.45,
    "alpha": 0.88,
}
PAPER_PANEL_RC = {
    "font.family": "serif",
    "font.serif": PAPER_SERIF_FONTS,
    "mathtext.fontset": "stix",
    "mathtext.rm": "STIXGeneral",
    "mathtext.it": "STIXGeneral:italic",
    "mathtext.bf": "STIXGeneral:bold",
    "axes.unicode_minus": False,
    "font.size": 13.0,
    "axes.titlesize": 16.0,
    "axes.labelsize": 15.0,
    "xtick.labelsize": 13.5,
    "ytick.labelsize": 13.5,
    "legend.fontsize": 13.0,
    "figure.titlesize": 16.0,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.45,
    "lines.linewidth": 1.5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
}
PAPER_PROFILE_RC = {
    **PAPER_PANEL_RC,
    "savefig.bbox": None,
    "savefig.pad_inches": 0.10,
}


def configure_matplotlib() -> Any:
    """Configure matplotlib for deterministic, serif, mathtext-ready figures."""
    cache_dir = Path(tempfile.gettempdir()) / "tread_matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg", force=True)
    matplotlib.rcParams.update(PAPER_PANEL_RC)
    return matplotlib


def get_pyplot() -> Any:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    return plt


def style_axes(ax: Any, *, grid: bool = True) -> None:
    if grid:
        ax.grid(True, color="#D9D9D9", linewidth=0.45, alpha=0.65)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction="out", length=3.0, width=0.7)


def label_for(name: str) -> str:
    labels = {
        "ego_vx_0": r"$v_{x,\mathrm{ego}}^{0}$ (m/s)",
        "log_initial_gap": r"$\log g_{0}$",
        "initial_gap": r"$g_{0}$ (m)",
        "initial_lateral_offset": r"$d_{y}^{0}$ (m)",
        "initial_delta_v": r"$\Delta v_{x}^{0}$ (m/s)",
        "initial_delta_vx": r"$\Delta v_{x}^{0}$ (m/s)",
        "target_ax_0": r"$a_{x,\mathrm{tar}}^{0}$ (m/s$^2$)",
        "target_vy_0": r"$v_{y,\mathrm{tar}}^{0}$ (m/s)",
        "target_ay_0": r"$a_{y,\mathrm{tar}}^{0}$ (m/s$^2$)",
        "lead_ax_0": r"$a_{x,\mathrm{tar}}^{0}$ (m/s$^2$)",
        "final_lateral_offset": r"$d_{y}^{T}$ (m)",
        "time_to_cross": r"$t_{\mathrm{cross}}$ (s)",
        "target_speed_change": r"$\Delta v_{x,\mathrm{tar}}^{T}$ (m/s)",
        "lead_speed_change": r"$\Delta v_{x,\mathrm{tar}}^{T}$ (m/s)",
        "lead_min_ax": r"$a_{x,\mathrm{tar}}^{\min}$ (m/s$^2$)",
        "lead_braking_duration": r"$\tau_{b,\mathrm{tar}}$ (s)",
        "lead_final_speed": r"$v_{x,\mathrm{tar}}^{T}$ (m/s)",
        "lead_displacement": r"$x_{\mathrm{tar}}(t)-x_{\mathrm{tar}}^{0}$ (m)",
        "lead_mean_abs_ax": r"$\overline{|a_{x,\mathrm{tar}}|}$ (m/s$^2$)",
        "lead_accel_std": r"$\sigma(a_{x,\mathrm{tar}})$ (m/s$^2$)",
        "lead_braking_impulse": r"$I_{\mathrm{brake}}$ (m/s)",
        "lane_entry_time": r"$t_{\mathrm{entry}}$ (s)",
        "longitudinal_displacement": r"$\Delta x$ (m)",
        "total_lateral_displacement": r"$\Delta y$ (m)",
        "lateral_progress_toward_ego_lane": r"$\Delta y_{\mathrm{ego}}$ (m)",
        "final_abs_lateral_offset": r"$|d_{y}^{T}|$ (m)",
        "max_abs_longitudinal_accel": r"$\max |a_x|$ (m/s$^2$)",
        "max_abs_lateral_velocity": r"$\max |v_y|$ (m/s)",
        "mean_abs_lateral_accel": r"$\overline{|a_y|}$ (m/s$^2$)",
        "max_abs_jerk": r"$\max |j_x|$ (m/s$^3$)",
    }
    return labels.get(str(name), str(name).replace("_", " "))


def descriptive_condition_label_for(name: str) -> str:
    labels = {
        "ego_vx_0": r"Initial ego speed, $v_{x,\mathrm{ego}}^{0}$ (m/s)",
        "log_initial_gap": r"Initial gap, $\log g_{0}$",
        "initial_gap": r"Initial gap, $g_{0}$ (m)",
        "initial_lateral_offset": r"Initial lateral offset, $d_{y}^{0}$ (m)",
        "initial_delta_v": r"Initial relative speed, $\Delta v_{x}^{0}$ (m/s)",
        "initial_delta_vx": r"Initial relative speed, $\Delta v_{x}^{0}$ (m/s)",
        "target_ax_0": r"Initial target longitudinal accel., $a_{x,\mathrm{tar}}^{0}$ (m/s$^2$)",
        "target_vy_0": r"Initial target lateral speed, $v_{y,\mathrm{tar}}^{0}$ (m/s)",
        "target_ay_0": r"Initial target lateral accel., $a_{y,\mathrm{tar}}^{0}$ (m/s$^2$)",
        "lead_ax_0": r"Initial lead longitudinal accel., $a_{x,\mathrm{tar}}^{0}$ (m/s$^2$)",
        "final_lateral_offset": r"Final lateral offset, $d_{y}^{T}$ (m)",
        "time_to_cross": r"Lane-crossing time, $t_{\mathrm{cross}}$ (s)",
        "target_speed_change": r"Target speed change, $\Delta v_{x,\mathrm{tar}}^{T}$ (m/s)",
        "lead_speed_change": r"Lead speed change, $\Delta v_{x,\mathrm{tar}}^{T}$ (m/s)",
        "lead_min_ax": r"Minimum lead acceleration, $a_{x,\mathrm{tar}}^{\min}$ (m/s$^2$)",
        "lead_braking_duration": r"Lead braking duration, $\tau_{b,\mathrm{tar}}$ (s)",
    }
    return labels.get(str(name), label_for(name))


def rel_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any, *, force: bool) -> bool:
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=True)
        f.write("\n")
    return True


def fget(mapping: dict[str, Any], key: str, default: Any = math.nan) -> Any:
    return mapping.get(key, default) if isinstance(mapping, dict) else default


def save_figure(
    fig: Any,
    path: Path,
    root: Path,
    *,
    force: bool,
    dpi: int = PAPER_FIGURE_DPI,
) -> list[str]:
    if force or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
    return [rel_path(path, root)]


def gpd_survival(
    y: np.ndarray,
    *,
    u: float,
    xi: float,
    beta: float,
    exceedance_rate: float,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    z = np.maximum(y - u, 0.0)
    if abs(xi) < 1e-10:
        tail = np.exp(-z / beta)
    else:
        tail = np.power(np.maximum(1.0 + xi * z / beta, 1e-300), -1.0 / xi)
    return exceedance_rate * tail


def record(
    manifest: dict[str, Any],
    key: str,
    *,
    status: str,
    outputs: list[str] | None = None,
    sources: list[str] | None = None,
    skipped_reason: str | None = None,
    notes: str | None = None,
) -> None:
    manifest["experiments"][key] = {
        "status": status,
        "outputs": outputs or [],
        "source_artifacts": sources or [],
        "skipped_reason": skipped_reason,
        "notes": notes,
    }


def build_manifest(
    scope: str,
    created_by: str,
    root: Path,
    source_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "experiment_scope": scope,
        "created_by": created_by,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "no_training_or_rerun_statement": (
            "Only existing JSON/CSV/NPZ/PNG results were read; "
            "no training, EVT fitting, or subset simulation was rerun."
        ),
        "source_files": {
            key: rel_path(path, root) for key, path in source_paths.items() if path.exists()
        },
        "missing_source_files": {
            key: rel_path(path, root) for key, path in source_paths.items() if not path.exists()
        },
        "experiments": {},
    }


def write_experiment_readme(
    path: Path,
    manifest: dict[str, Any],
    *,
    title: str,
    description: str,
    no_rerun_note: str,
    interpretation_notes: list[str],
    force: bool,
) -> None:
    if path.exists() and not force:
        return

    generated: list[str] = []
    reused: list[str] = []
    skipped: list[str] = []
    for name, exp in manifest["experiments"].items():
        generated.extend(exp.get("outputs", []))
        if exp.get("source_artifacts"):
            reused.extend(f"{name}: {src}" for src in exp["source_artifacts"] if src)
        if exp.get("skipped_reason"):
            skipped.append(f"{name}: {exp['skipped_reason']}")

    lines = [
        f"# {title}",
        "",
        description,
        no_rerun_note,
        "",
        "## Inputs",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in manifest["source_files"].items())
    lines.extend(["", "## Generated Artifacts", ""])
    lines.extend(f"- `{artifact}`" for artifact in generated)
    lines.extend(["", "## Reused Existing Artifacts", ""])
    lines.extend(f"- reused existing artifact: `{item}`" for item in reused)
    if not reused:
        lines.append("- None")
    lines.extend(["", "## Skipped Artifacts", ""])
    lines.extend(f"- {item}" for item in skipped)
    if not skipped:
        lines.append("- None")
    lines.extend(["", "## Interpretation Notes", ""])
    lines.extend(f"- {note}" for note in interpretation_notes)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
