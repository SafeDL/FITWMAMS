"""Build the checked, protocol-matched IDM world-model comparison report."""

from __future__ import annotations

import argparse
from pathlib import Path

from ads.idm.src.multi_world_runner import (
    build_comparison_report,
    build_explicit_subset_comparison,
)
from hierarchical_world_model.src.core.utils import load_yaml

ROOT = Path(__file__).resolve().parents[3]

CONFIG = ROOT / "ads/idm/configs/world_models_idm.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare completed IDM results after enforcing shared protocol fields."
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--tracking-npc-summary", type=Path, default=None)
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
        args.tracking_npc_summary is not None
        or args.trafficbots_subset_summary is not None
    )
    if explicit:
        if args.tracking_npc_summary is None or args.trafficbots_subset_summary is None:
            parser.error("both explicit subset summaries are required")
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir is not None
            else ROOT / "results/idm_ams/comparisons/trafficbots_vs_tracking_npc"
        )
        print(
            build_explicit_subset_comparison(
                args.tracking_npc_summary,
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
