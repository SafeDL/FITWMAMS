"""Shared paths and immutable configuration for A2 human calibration."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "hierarchical_world_model/config/a2_human_calibration.yaml"


def load_config() -> tuple[dict, dict]:
    config = yaml.safe_load(CONFIG.read_text())
    world = yaml.safe_load((ROOT / config["base_config"]).read_text())
    return config, world


def result_root(config: dict) -> Path:
    return ROOT / config["paths"]["output_dir"]


def require_aligned_factual_protocol(config: dict) -> None:
    """Prevent optimization against an evaluation protocol known to be mismatched."""
    if config["method"].get("protocol_status") != "aligned":
        raise RuntimeError(
            "A2 calibration is blocked until the policy-on factual protocol is aligned. "
            "See doc/FITWMAMS_A2_Factual_Protocol_Correction.md."
        )
