"""Build the shared highD scenario-condition dataset."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from normalizing_flow.src.data import build_scenario_dataset
from normalizing_flow.src.utils import load_yaml, setup_logging

CONFIG = ROOT / "normalizing_flow/config.yaml"


def main() -> None:
    setup_logging()
    build_scenario_dataset(load_yaml(CONFIG), config_dir=CONFIG.parent)


if __name__ == "__main__":
    main()
