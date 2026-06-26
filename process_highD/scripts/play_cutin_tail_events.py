#!/usr/bin/env python3
"""Render diffusion-generated cut-in tail scenarios to GIF."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from process_highD.src.event_playback import render_generated_scenarios_gif
from process_highD.src.idm_ego import load_idm_ego_config


GENERATED_SCENARIOS_PATH = (
    ROOT
    / "results"
    / "highd_cutin_tail"
    / "generated"
    / "diffusion_generated_scenarios.npz"
)
HIGHD_CONFIG_PATH = ROOT / "process_highD" / "scripts" / "configs" / "highd_default.yaml"
IDM_EGO_CONFIG_PATH = ROOT / "tools" / "idm_ego.yaml"
OUTPUT_DIR = (
    ROOT / "results" / "highd_cutin_tail" / "generated" / "event_playbacks"
)
OUTPUT_NAME = "generated_cutin_scenario"

# "all": every generated scenario; int: random sample count; tuple/list: exact indices.
SCENARIO_SELECTION: str | int | tuple[int, ...] = 10
RANDOM_SEED = 42

DT = 0.04
VIEW_WIDTH = 160.0
TRAIL_FRAMES = 50
PLAYBACK_SPEED = 1.0
FPS = 25.0
IDM_EGO_CONFIG = load_idm_ego_config(IDM_EGO_CONFIG_PATH, event_type="cut_in")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    render_generated_scenarios_gif(
        generated_npz_path=GENERATED_SCENARIOS_PATH,
        output_dir=OUTPUT_DIR,
        output_name=OUTPUT_NAME,
        scenario_selection=SCENARIO_SELECTION,
        random_seed=RANDOM_SEED,
        background_config_path=HIGHD_CONFIG_PATH,
        idm_ego_config=IDM_EGO_CONFIG,
        dt=DT,
        view_width=VIEW_WIDTH,
        trail_frames=TRAIL_FRAMES,
        playback_speed=PLAYBACK_SPEED,
        fps=FPS,
    )


if __name__ == "__main__":
    main()
