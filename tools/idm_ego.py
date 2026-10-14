"""Shared configuration for the fixed highway-env IDM ADS."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


IDM_PARAMETER_KEYS = (
    "ACC_MAX",
    "COMFORT_ACC_MAX",
    "COMFORT_ACC_MIN",
    "DISTANCE_WANTED",
    "TIME_WANTED",
    "DELTA",
    "POLITENESS",
    "LANE_CHANGE_MIN_ACC_GAIN",
    "LANE_CHANGE_MAX_BRAKING_IMPOSED",
    "LANE_CHANGE_DELAY",
)


def load_idm_ego_config(path: str | Path) -> dict[str, Any]:
    """Load the shared fixed-IDM ADS parameters."""
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)
