#!/usr/bin/env python3
"""Run the formal Flow×HiQR long-tail response audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.core.cli_config import materialize_config  # noqa: E402
from world_model.src.core.sequential_dataset import (  # noqa: E402
    sequence_cache_owner_dir,
)
from world_model.src.core.utils import setup_logging  # noqa: E402
from world_model.src.hiqr.flow_evaluation import (  # noqa: E402
    evaluate_hiqr_flow_composition,
)
from world_model.src.hiqr.train import (  # noqa: E402
    load_hiqr_checkpoint,
    require_canonical_hiqr_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs/highd_hiqr_world_model.yaml"),
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "results/highd_world_model/hiqr_world_model")
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max-starts", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    config, path = materialize_config(
        Path(args.config),
        Path(args.output_dir),
        config_name=Path(args.config).name,
        resolve_path_keys=("sequence_cache_dir", "flow_schema", "source_dataset_dir"),
    )
    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path(config["paths"]["output_dir"])
        / "checkpoints/best_hiqr_world_model.pt"
    )
    model = load_hiqr_checkpoint(checkpoint)
    require_canonical_hiqr_checkpoint(model)
    evaluate_hiqr_flow_composition(
        model,
        repo_root=ROOT,
        cache_owner=sequence_cache_owner_dir(config, config_dir=path.parent),
        output_dir=Path(config["paths"]["output_dir"]),
        max_starts=args.max_starts,
        deterministic=not args.stochastic,
    )


if __name__ == "__main__":
    main()
