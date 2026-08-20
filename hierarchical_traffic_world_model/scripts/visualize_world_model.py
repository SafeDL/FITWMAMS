#!/usr/bin/env python3
"""Build final figures and reconstruction playbacks."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_traffic_world_model.src.visualization import (  # noqa: E402
    build_visualizations,
)
from world_model.src.core.utils import load_yaml, save_json  # noqa: E402

CONFIG = (
    ROOT
    / "hierarchical_traffic_world_model/configs/highd_hierarchical_world_model.yaml"
)


def main() -> None:
    config = load_yaml(CONFIG)
    if config["training"].get("experiment_scope") != "full":
        raise ValueError("visualization requires the maintained full protocol")
    report = build_visualizations(config, config_dir=CONFIG.parent)
    report["experiment_scope"] = "full"
    save_json(
        report,
        Path(config["paths"]["output_dir"]) / "visualization_manifest.json",
    )


if __name__ == "__main__":
    main()
