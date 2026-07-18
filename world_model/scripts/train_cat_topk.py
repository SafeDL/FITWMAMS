#!/usr/bin/env python3
"""Train CAT-TopK from the prepared START/ROLL cache in a new output directory."""
from __future__ import annotations

import argparse
from copy import deepcopy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.train import train_world_model
from world_model.src.utils import load_yaml, setup_logging


def _materialize_config(template_path: Path, output_dir: Path) -> tuple[dict, Path]:
    template_path, output_dir = template_path.resolve(), output_dir.resolve()
    config = deepcopy(load_yaml(template_path))
    config.setdefault("paths", {})["output_dir"] = str(output_dir)
    config_path = output_dir / "configs" / "highd_cat_topk.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config, config_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config_dir = Path(__file__).parent / "configs"
    parser.add_argument("--config", default=str(config_dir / "highd_cat_topk_world_model.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/highd_world_model/catk_topk_reproduction"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    config, config_path = _materialize_config(Path(args.config), Path(args.output_dir))
    summary = train_world_model(
        config,
        config_dir=config_path.parent,
        epochs=args.epochs,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        rebuild_dataset=args.rebuild_dataset,
    )
    print(summary.get("best_checkpoint"))


if __name__ == "__main__":
    main()
