#!/usr/bin/env python3
"""Run latent subset simulation for car-following events."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.subset_entrypoints import run_subset_entrypoint


DEFAULT_CONFIG_PATH = (
    ROOT / "SAIRL_subset" / "scripts" / "configs" / "latent_subset_following.yaml"
)


def main() -> None:
    run_subset_entrypoint(
        subset_name="SAIRL_subset",
        default_config_path=DEFAULT_CONFIG_PATH,
        event_type="following",
        label="Following",
        result_filename="sairl_following_result.json",
        compact_result=("SAIRL_subset.src.result_payload", "compact_sairl_result"),
        policy_section="sairl_policy",
        policy_label="SAIRL",
        seed_policy=True,
    )


if __name__ == "__main__":
    main()
