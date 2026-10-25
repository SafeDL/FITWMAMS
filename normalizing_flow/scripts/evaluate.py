"""Evaluate and sample the EVT-tail scenario-condition Flow."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from normalizing_flow.src.evaluation import evaluate_tail_flow
from normalizing_flow.src.utils import load_yaml, setup_logging

CONFIG = ROOT / "normalizing_flow/config.yaml"


def main() -> None:
    setup_logging()
    evaluate_tail_flow(load_yaml(CONFIG), config_dir=CONFIG.parent, repo_root=ROOT)


if __name__ == "__main__":
    main()
