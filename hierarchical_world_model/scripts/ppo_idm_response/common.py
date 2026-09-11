"""Shared configuration for the active PPO--IDM response workflow."""
from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "hierarchical_world_model/config/ppo_idm_response.yaml"


def load_response_config(config_path: Path = CONFIG) -> tuple[dict, dict]:
    response = yaml.safe_load(config_path.read_text())
    world_model = yaml.safe_load((ROOT / response["base_config"]).read_text())
    return response, world_model


def result_directory(response: dict) -> Path:
    return ROOT / response["paths"]["output_dir"]


def event_directory(response: dict) -> Path:
    return ROOT / response["paths"]["event_reference"]
