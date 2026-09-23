"""Build the deduplicated D1 interaction-event manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from interactive_behavior_world_model.data.events import mine_event_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/benchmark_v1.yaml",
    )
    args = parser.parse_args()
    print(json.dumps(mine_event_manifest(args.config), indent=2))


if __name__ == "__main__":
    main()
