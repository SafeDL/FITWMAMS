#!/usr/bin/env python3
"""Run independent latent Monte Carlo baseline for cut-in events."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.subset_entrypoints import run_monte_carlo_entrypoint


DEFAULT_CONFIG_PATH = (
    ROOT / "PPO_subset" / "scripts" / "configs" / "latent_subset_cutin.yaml"
)


def main() -> None:
    run_monte_carlo_entrypoint(
        subset_name="PPO_subset",
        default_config_path=DEFAULT_CONFIG_PATH,
        event_type="cut_in",
        max_samples=200_000,
    )


if __name__ == "__main__":
    main()
