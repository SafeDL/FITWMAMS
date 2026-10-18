#!/usr/bin/env python3
"""Run one or more registered world models under the common IDM protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ads.idm.src.multi_world_runner import run_model_suite  # noqa: E402
from hierarchical_world_model.src.core.utils import load_yaml  # noqa: E402

CONFIG = ROOT / "ads/idm/configs/world_models_idm.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run protocol-matched IDM evaluation for multiple world models."
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="world models to execute (default: all enabled models)",
    )
    parser.add_argument(
        "--estimators",
        nargs="+",
        choices=("subset", "monte_carlo"),
        default=None,
    )
    parser.add_argument(
        "--development",
        action="store_true",
        help="mark this run as development output (formal=false)",
    )
    args = parser.parse_args()
    path = args.config.resolve()
    suite = load_yaml(path)
    estimators = args.estimators or suite.get("execution", {}).get(
        "estimators", ["subset", "monte_carlo"]
    )
    for result in run_model_suite(
        suite,
        path.parent,
        selected=args.models,
        estimators=estimators,
        formal=not args.development,
    ):
        print(result)


if __name__ == "__main__":
    main()
