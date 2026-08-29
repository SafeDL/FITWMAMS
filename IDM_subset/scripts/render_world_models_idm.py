#!/usr/bin/env python3
"""Render separate, visually matched AMS playbacks for enabled world models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from IDM_subset.scripts.render_subset_playbacks import (  # noqa: E402
    render_subset_playbacks,
)
from IDM_subset.src.multi_world_runner import load_model_configs  # noqa: E402
from world_model.src.core.utils import load_yaml  # noqa: E402

CONFIG = ROOT / "IDM_subset/configs/world_models_idm.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render model-separated playbacks with one visual contract."
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=None)
    args = parser.parse_args()
    path = args.config.resolve()
    suite = load_yaml(path)
    visual = suite.get("visualization", {})
    top_k = int(args.top_k if args.top_k is not None else visual.get("top_cases", 10))
    stride = int(
        args.frame_stride
        if args.frame_stride is not None
        else visual.get("frame_stride", 2)
    )
    models = load_model_configs(suite, path.parent, args.models)
    for model_id, (_, _, model_config_path) in models.items():
        print(
            render_subset_playbacks(
                model_id=model_id,
                config_path=model_config_path,
                top_k=top_k,
                frame_stride=stride,
            )
        )


if __name__ == "__main__":
    main()
