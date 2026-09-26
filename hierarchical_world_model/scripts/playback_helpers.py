"""Shared world-view drawing helpers for the current semantic ADS playbacks."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

from hierarchical_world_model.src.evaluation import Rollout


EGO_COLOR = "#D62728"
NPC_COLOR = "#1f78b4"
ROAD_COLOR = "#6f7378"
LANE_COLOR = "#ffffff"
FOCUS_COLOR = "#7e22ce"
SOURCE_REAR_COLOR = "#e69f00"
PASSIVE_COLOR = "#8c8c8c"


def _draw_vehicle(
    axis: Any,
    state: np.ndarray,
    *,
    color: str,
    label: str | None = None,
    filled: bool,
    alpha: float,
) -> None:
    x, y, vx, vy = (float(value) for value in state[:4])
    heading = float(np.arctan2(vy, vx)) if np.hypot(vx, vy) > 1.0e-6 else 0.0
    patch = Rectangle(
        (-2.4, -0.9),
        4.5,
        1.8,
        facecolor=color if filled else "none",
        edgecolor="black",
        linewidth=0.8,
        alpha=alpha,
        zorder=6 if filled else 5,
    )
    patch.set_transform(Affine2D().rotate(heading).translate(x, y) + axis.transData)
    axis.add_patch(patch)
    if label is not None:
        axis.text(
            x,
            y + 1.9,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            color="black",
            zorder=7,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.2},
        )


def _draw_lane_markings(axis: Any) -> None:
    for lane in (-7.5, -5.625, -1.875, 1.875, 5.625, 7.5):
        outer = abs(lane) > 7.0
        axis.axhline(
            lane,
            color=LANE_COLOR,
            linewidth=0.8,
            linestyle="-" if outer else "--",
            alpha=0.45 if outer else 0.28,
            zorder=0,
        )


def _draw_world(
    axis: Any,
    *,
    states: np.ndarray,
    valid: np.ndarray,
    frame: int,
    title: str,
    focus_slot: int,
    source_slot: int | None = None,
    focus_role: str = "target rear",
    source_role: str = "source rear",
) -> None:
    axis.clear()
    axis.set_facecolor(ROAD_COLOR)
    _draw_lane_markings(axis)
    center = float(states[frame, 0, 0])
    axis.set(
        xlim=(center - 62.0, center + 62.0),
        ylim=(-8.2, 8.2),
        aspect="equal",
        xlabel="x [m]",
        ylabel="y [m]",
        title=title,
    )
    trail = slice(max(0, frame - 45), frame + 1)
    for slot in np.flatnonzero(valid[1:]):
        agent = int(slot) + 1
        color = (
            FOCUS_COLOR
            if int(slot) == focus_slot
            else SOURCE_REAR_COLOR
            if source_slot is not None and int(slot) == source_slot
            else NPC_COLOR
        )
        label = (
            f"{focus_role} b{agent}"
            if int(slot) == focus_slot
            else f"{source_role} b{agent}"
            if source_slot is not None and int(slot) == source_slot
            else f"b{agent}"
        )
        axis.plot(
            states[trail, agent, 0],
            states[trail, agent, 1],
            color=color,
            linewidth=1.35,
            alpha=0.88,
        )
        _draw_vehicle(
            axis,
            states[frame, agent],
            color=color,
            label=label if int(slot) in {focus_slot, source_slot} else None,
            filled=True,
            alpha=0.68,
        )
    axis.plot(
        states[trail, 0, 0],
        states[trail, 0, 1],
        color=EGO_COLOR,
        linewidth=1.8,
        alpha=0.9,
    )
    _draw_vehicle(
        axis,
        states[frame, 0],
        color=EGO_COLOR,
        label="ADS",
        filled=True,
        alpha=0.92,
    )
    axis.text(
        0.01,
        0.02,
        "red: ADS | purple: " + focus_role
        + (" | orange: " + source_role if source_slot is not None else "")
        + " | blue: other NPCs",
        transform=axis.transAxes,
        fontsize=7,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.2},
    )
    axis.tick_params(labelsize=7)


def _one(value: Rollout, index: int) -> Rollout:
    diagnostics = None
    if value.controller_diagnostics is not None:
        diagnostics = {
            key: item[index : index + 1]
            for key, item in value.controller_diagnostics.items()
        }
    return Rollout(
        states=value.states[index : index + 1],
        background_actions=value.background_actions[index : index + 1],
        ego_actions=value.ego_actions[index : index + 1],
        reference_actions=value.reference_actions[index : index + 1],
        base_background_actions=(
            None
            if value.base_background_actions is None
            else value.base_background_actions[index : index + 1]
        ),
        controller_diagnostics=diagnostics,
    )
