#!/usr/bin/env python3
"""Replay final-level cut-in subset scenarios."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from A2C_subset.src.script_entrypoints import play_final_level_entrypoint


DEFAULT_CONFIG_PATH = (
    ROOT / "A2C_subset" / "scripts" / "configs" / "latent_subset_cutin.yaml"
)


def main() -> None:
    play_final_level_entrypoint(
        default_config_path=DEFAULT_CONFIG_PATH,
        event_type="cut_in",
        label="Cut-in",
    )


if __name__ == "__main__":
    main()
