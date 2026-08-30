#!/usr/bin/env python3
"""Run the A0 hard gate before considering a GAIL/PPO extension."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.a0_necessity import run_a0_necessity_diagnostic
from hierarchical_world_model.src.protocol import load_protocol_config
from world_model.src.core.utils import setup_logging


CONFIG = ROOT / "hierarchical_world_model/config/release.yaml"
OUTPUT = ROOT / "results/hierarchical_world_model/a0_necessity"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--maximum-contexts", type=int, default=0)
    args = parser.parse_args()
    if args.maximum_contexts < 0:
        raise ValueError("maximum-contexts must be non-negative")
    setup_logging()
    report = run_a0_necessity_diagnostic(
        load_protocol_config(args.config.resolve()),
        config_dir=ROOT,
        output=args.output.resolve(),
        maximum_contexts=args.maximum_contexts,
    )
    print(report["decision"]["next_step"])


if __name__ == "__main__":
    main()
