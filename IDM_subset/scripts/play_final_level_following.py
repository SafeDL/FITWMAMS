#!/usr/bin/env python3
"""Replay final-level car-following subset scenarios."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.subset_entrypoints import play_final_level_entrypoint


DEFAULT_CONFIG_PATH = (
    ROOT / "IDM_subset" / "scripts" / "configs" / "latent_subset_following.yaml"
)


def main() -> None:
    play_final_level_entrypoint(
        subset_name="IDM_subset",
        default_config_path=DEFAULT_CONFIG_PATH,
        event_type="following",
        label="Following",
    )


if __name__ == "__main__":
    main()
