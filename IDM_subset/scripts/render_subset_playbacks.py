#!/usr/bin/env python3
"""Render HighwayEnv playbacks for the retained AMS subset failure cases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from IDM_subset.src.world_subset_runner import _build_evaluator  # noqa: E402
from hierarchical_world_model.src.execution import rollout_world  # noqa: E402
from hierarchical_world_model.src.randomness import (  # noqa: E402
    WorldExogenousState,
)
from hierarchical_world_model.src.visualization import (  # noqa: E402
    DIFFUSION_COLOR,
    EGO_COLOR,
    ROAD_COLOR,
    _draw_lane_markings,
    _draw_vehicle,
)
from tools.plot_style import get_pyplot  # noqa: E402
from world_model.src.core.utils import load_json, load_yaml, save_json  # noqa: E402

CONFIG = ROOT / "IDM_subset/configs/world_subset_idm.yaml"


def _local_coordinates(states: np.ndarray) -> np.ndarray:
    local = np.asarray(states, np.float32).copy()
    local[..., :2] -= local[0:1, 0:1, :2]
    return local


def _collision_mask(states: np.ndarray, valid: np.ndarray) -> np.ndarray:
    ego = states[:, :, :1]
    background = states[:, :, 1:]
    active = np.asarray(valid[:, 1:], bool)[:, None]
    longitudinal = background[..., 0] - ego[..., 0]
    lateral = np.abs(background[..., 1] - ego[..., 1])
    return active & (np.abs(longitudinal) < 4.8) & (lateral < 1.8)


def _render_case(
    *,
    path: Path,
    case: dict[str, Any],
    states: np.ndarray,
    valid: np.ndarray,
    frame_stride: int,
) -> dict[str, Any]:
    local = _local_coordinates(states)
    collision = _collision_mask(states[None], valid[None])[0]
    frames = np.arange(0, len(states), max(int(frame_stride), 1))
    if frames[-1] != len(states) - 1:
        frames = np.append(frames, len(states) - 1)
    collision_frames = np.flatnonzero(collision.any(axis=1))
    first_collision = None if not len(collision_frames) else int(collision_frames[0])
    plt = get_pyplot()
    figure, axis = plt.subplots(figsize=(12.0, 4.8), dpi=100)
    figure.subplots_adjust(left=0.065, right=0.965, bottom=0.18, top=0.84)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        path,
        mode="I",
        duration=40 * max(int(frame_stride), 1),
        loop=0,
    ) as writer:
        for frame in frames:
            axis.clear()
            axis.set_facecolor(ROAD_COLOR)
            center_x = float(local[frame, 0, 0])
            _draw_lane_markings(axis)
            axis.set(
                xlim=(center_x - 60.0, center_x + 60.0),
                ylim=(-8.2, 8.2),
                xlabel="relative longitudinal position [m]",
                ylabel="relative lateral position [m]",
                aspect="equal",
                title=(
                    f"IDM subset {case['case_id']} | t={frame * 0.04:.2f}s | "
                    f"S_EVT={float(case['evt_score']):.2f}"
                ),
            )
            trail_start = max(0, frame - 45)
            axis.plot(
                local[trail_start : frame + 1, 0, 0],
                local[trail_start : frame + 1, 0, 1],
                color=EGO_COLOR,
                linewidth=2.0,
                label="IDM ego",
            )
            for slot in np.flatnonzero(valid[1:]):
                axis.plot(
                    local[trail_start : frame + 1, slot + 1, 0],
                    local[trail_start : frame + 1, slot + 1, 1],
                    color=DIFFUSION_COLOR,
                    linewidth=1.35,
                    alpha=0.85,
                )
                colliding = bool(collision[frame, slot])
                _draw_vehicle(
                    axis,
                    local[frame, slot + 1],
                    color="#ffbf00" if colliding else DIFFUSION_COLOR,
                    label=f"collision background b{slot + 1}" if colliding else None,
                    filled=True,
                    alpha=0.9 if colliding else 0.52,
                )
            _draw_vehicle(
                axis,
                local[frame, 0],
                color="#ffbf00" if collision[frame].any() else EGO_COLOR,
                label="IDM ego",
                filled=True,
                alpha=0.92,
            )
            axis.text(
                0.01,
                0.02,
                "red: IDM ego | blue: HiQR background | yellow: footprint overlap",
                transform=axis.transAxes,
                fontsize=7.5,
                va="bottom",
                ha="left",
                bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
            )
            axis.tick_params(labelsize=8)
            figure.canvas.draw()
            writer.append_data(
                np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()
            )
    plt.close(figure)
    return {
        "case_id": case["case_id"],
        "gif": str(path),
        "world_exogenous_state": str(case["world_exogenous_state"]),
        "event_risk": float(case["event_risk"]),
        "evt_score": float(case["evt_score"]),
        "collision": bool(case["collision"]),
        "first_collision_frame": first_collision,
        "first_collision_time_s": None
        if first_collision is None
        else float(first_collision * 0.04),
        "frame_stride": int(frame_stride),
        "playback_frames": int(len(frames)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render GIFs from the exact retained AMS subset worlds."
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    subset_dir = (config_path.parent / config["subset_simulation"]["output_dir"]).resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else subset_dir / "playbacks"
    )
    cases = load_json(subset_dir / "world_subset_top_cases.json")[: max(int(args.top_k), 0)]
    evaluator, provenance = _build_evaluator(config, config_path.parent)
    episodes = []
    for case in cases:
        exogenous = WorldExogenousState.load(case["world_exogenous_state"])
        rollout = rollout_world(
            evaluator.sampler,
            exogenous,
            evaluator.policy,
            steps=evaluator.steps,
            evt_model=evaluator.evt_model,
        )
        episodes.append(
            _render_case(
                path=output_dir / f"{case['case_id']}.gif",
                case=case,
                states=rollout.states[0],
                valid=rollout.initial_valid[0],
                frame_stride=args.frame_stride,
            )
        )
    save_json(
        {
            "schema": "highway_env_idm_subset_playbacks_v1",
            "role": "playbacks rendered from the retained AMS subset final population",
            "source_summary": str(subset_dir / "world_subset_summary.json"),
            "provenance": provenance,
            "episodes": episodes,
        },
        output_dir / "playback_manifest.json",
    )
    print(f"rendered {len(episodes)} subset playbacks in {output_dir}")


if __name__ == "__main__":
    main()
