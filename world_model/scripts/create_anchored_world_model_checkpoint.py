#!/usr/bin/env python3
"""将名义自然驾驶锚点与训练完成的 CAT-K 残差候选合成为 v4 checkpoint。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.data import dataset_dir_from_config, load_world_model_dataset  # noqa: E402
from world_model.src.model import (  # noqa: E402
    AnchoredTopKStartRollWorldModel,
    build_model_from_schema,
    load_checkpoint,
    model_config_payload,
)
from world_model.src.utils import ensure_dir, load_yaml  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--residual-checkpoint", required=True)
    parser.add_argument("--nominal-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch

    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    data_dir = dataset_dir_from_config(config, config_path.parent)
    _arrays, schema = load_world_model_dataset(data_dir)
    device = torch.device("cpu")
    model = build_model_from_schema(schema, config)
    if not isinstance(model, AnchoredTopKStartRollWorldModel):
        raise TypeError("The config must construct AnchoredTopKStartRollWorldModel")
    residual, residual_payload = load_checkpoint(str(Path(args.residual_checkpoint).resolve()), device)
    nominal, _nominal_payload = load_checkpoint(str(Path(args.nominal_checkpoint).resolve()), device)
    missing, unexpected = model.load_state_dict(residual.state_dict(), strict=False)
    if unexpected or any(not key.startswith("nominal_model.") for key in missing):
        raise RuntimeError(f"Residual checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.nominal_model.load_state_dict(nominal.state_dict(), strict=True)
    model.nominal_model.requires_grad_(False)
    model.eval()
    output = Path(args.output).resolve()
    ensure_dir(output.parent)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": model_config_payload(model),
            "schema": schema,
            "config": config,
            "best_epoch": int(residual_payload.get("best_epoch", -1)),
            "best_val_loss": float(residual_payload.get("best_val_loss", float("nan"))),
        },
        output,
    )
    print(output)


if __name__ == "__main__":
    main()
