"""Prepare the prefix-only D3 fixed-stimulus probe manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from npc_behavior_benchmark.data.probes import mine_lateral_probe_manifest, mine_probe_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/benchmark_v1.yaml",
    )
    parser.add_argument(
        "--kind", choices=("longitudinal", "lateral"), default="longitudinal"
    )
    args = parser.parse_args()
    function = (
        mine_lateral_probe_manifest if args.kind == "lateral" else mine_probe_manifest
    )
    print(json.dumps(function(args.config), indent=2))


if __name__ == "__main__":
    main()
