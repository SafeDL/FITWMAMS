"""Shared paths for the human-response A2 workflow."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "hierarchical_world_model/config/human_response_a2.yaml"


def load_config() -> tuple[dict, dict]:
    config = yaml.safe_load(CONFIG.read_text())
    base = yaml.safe_load((ROOT / config["base_config"]).read_text())
    return config, base


def result_root(config: dict) -> Path:
    return ROOT / config["paths"]["output_dir"]
