#!/usr/bin/env python3
"""Build the checked, protocol-matched IDM world-model comparison report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from IDM_subset.src.multi_world_runner import (  # noqa: E402
    build_comparison_report,
    build_explicit_subset_comparison,
)
from world_model.src.core.utils import load_yaml  # noqa: E402

CONFIG = ROOT / "IDM_subset/configs/world_models_idm.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare completed IDM results after enforcing shared protocol fields."
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--hierarchical-subset-summary", type=Path, default=None)
    parser.add_argument("--trafficbots-subset-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="report available results without requiring both estimators for every model",
    )
    args = parser.parse_args()
    path = args.config.resolve()
    explicit = (
        args.hierarchical_subset_summary is not None
        or args.trafficbots_subset_summary is not None
    )
    if explicit:
        if (
            args.hierarchical_subset_summary is None
            or args.trafficbots_subset_summary is None
        ):
            parser.error("both explicit subset summaries are required")
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir is not None
            else ROOT / "IDM_subset/results/comparisons/trafficbots_vs_hierarchical_subset"
        )
        print(
            build_explicit_subset_comparison(
                args.hierarchical_subset_summary,
                args.trafficbots_subset_summary,
                output_dir,
            )
        )
        return
    print(
        build_comparison_report(
            load_yaml(path),
            path.parent,
            selected=args.models,
            require_all=not args.allow_incomplete,
        )
    )


if __name__ == "__main__":
    main()
