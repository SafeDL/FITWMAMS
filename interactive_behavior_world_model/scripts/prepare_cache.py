"""Materialize the audited real-history benchmark cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from interactive_behavior_world_model.data.causal_cache import build_causal_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/benchmark_v1.yaml",
    )
    args = parser.parse_args()
    print(json.dumps(build_causal_cache(args.config), indent=2))


if __name__ == "__main__":
    main()
