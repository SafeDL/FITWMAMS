#!/usr/bin/env python3
"""Run the independent IDM Monte Carlo baseline in the maintained world."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from IDM_subset.src.world_subset_runner import run_monte_carlo_from_config  # noqa: E402
from world_model.src.core.utils import load_yaml  # noqa: E402

CONFIG = ROOT / "IDM_subset/configs/world_subset_idm.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the independent IDM Monte Carlo baseline in HighwayEnv."
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    summary = run_monte_carlo_from_config(load_yaml(config_path), config_path.parent)
    print(summary)


if __name__ == "__main__":
    main()
