#!/usr/bin/env python3
"""Export auditable latent-state prototypes for a Semi-Markov checkpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.semi_markov_train import _loader, _prototypes, load_semi_markov_checkpoint
from world_model.src.sequential_dataset import load_sequential_dataset, sequence_cache_owner_dir
from world_model.src.utils import ensure_dir, load_yaml, save_json, select_device

CONFIG = Path(__file__).resolve().parent / "configs" / "highd_semi_markov_relational.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    output_dir = Path(config["paths"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else output_dir / "checkpoints" / "best_semi_markov_relational.pt"
    device = select_device(str(config.get("evaluation", {}).get("device", "auto")))
    arrays, _ = load_sequential_dataset(sequence_cache_owner_dir(config, config_dir=config_path.parent))
    loader = _loader(
        arrays, args.split, batch_size=int(config.get("evaluation", {}).get("batch_size", 16)),
        maximum=int(args.max_sequences), shuffle=False, seed=int(config.get("evaluation", {}).get("seed", 123)),
    )
    report = _prototypes(load_semi_markov_checkpoint(checkpoint, device=device), loader, device)
    report.update({"checkpoint": str(checkpoint), "split": args.split, "num_sequences": len(loader.dataset)})
    target = Path(args.output).resolve() if args.output else ensure_dir(output_dir) / "latent_state_prototypes.json"
    save_json(report, target)
    print(target)


if __name__ == "__main__":
    main()
