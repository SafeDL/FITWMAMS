"""Fit the Train+Val natural-scene Flow."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from normalizing_flow.src.train import train_natural_flow
from normalizing_flow.src.utils import load_yaml, setup_logging

CONFIG = ROOT / "normalizing_flow/natural_config.yaml"


def main():
    setup_logging()
    train_natural_flow(load_yaml(CONFIG), config_dir=CONFIG.parent, repo_root=ROOT)


if __name__ == "__main__":
    main()
