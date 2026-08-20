"""Shared plotting style for paper-oriented result figures."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any


REAL_COLOR = "#4C78A8"
GENERATED_COLOR = "#F58518"
REFERENCE_COLOR = "#333333"
PAPER_FIGURE_DPI = 300
PAPER_SERIF_FONTS = [
    "Times New Roman",
    "Times",
    "Nimbus Roman",
    "Liberation Serif",
    "DejaVu Serif",
]
PAPER_PANEL_LABELSIZE = 16.0
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
