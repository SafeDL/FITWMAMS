#!/usr/bin/env python3
"""Sample natural lead-car rollouts from a trained diffusion prior."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from diffusion.src.kinematics import integrate_following_actions
from diffusion.src.model import build_model_from_schema
from diffusion.src.types import VehicleBox, VehicleState
from diffusion.src.utils import load_json, load_yaml, save_json, select_device, set_seed, setup_logging


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "natural_following.yaml"
DEFAULT_CHECKPOINT_PATH = "checkpoints/best.pt"


def _resolve_output_dir(config: dict, config_dir: Path) -> Path:
    return (config_dir / config.get("paths", {}).get("output_dir", "../../../data/diffusion_natural/following")).resolve()


def _resolve_checkpoint_path(checkpoint: str | None, output_dir: Path) -> Path:
    path = Path(checkpoint or DEFAULT_CHECKPOINT_PATH)
    if path.is_absolute():
        return path
    cwd_path = path.resolve()
    if cwd_path.exists():
        return cwd_path
    return (output_dir / path).resolve()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _decode_actions(x: np.ndarray, stats: dict) -> np.ndarray:
    mean = np.asarray(stats["actions"]["mean"], dtype=np.float32)
    std = np.asarray(stats["actions"]["std"], dtype=np.float32)
    return (x * std + mean).astype(np.float32)


def _actions_to_ax(actions: np.ndarray, context_states: np.ndarray, schema: dict, config: dict) -> np.ndarray:
    rep = str(schema.get("action_representation", config.get("action", {}).get("representation", "acceleration"))).lower()
    dt = float(schema.get("dt", 0.04))
    ax_min = float(config.get("action", {}).get("ax_min", -8.0))
    ax_max = float(config.get("action", {}).get("ax_max", 4.0))
    if rep == "jerk":
        prev_ax = context_states[:, -1, 1, 4].astype(np.float32)
        ax = prev_ax[:, None] + np.cumsum(actions[:, :, 0], axis=1) * dt
    else:
        ax = actions[:, :, 0]
    return np.clip(ax, ax_min, ax_max).astype(np.float32)


def _integrate(ax: np.ndarray, context_states: np.ndarray, adv_length: np.ndarray, schema: dict) -> np.ndarray:
    dt = float(schema.get("dt", 0.04))
    trajectories: list[np.ndarray] = []
    for i in range(ax.shape[0]):
        lead0 = context_states[i, -1, 1]
        initial = VehicleState(
            x=float(lead0[0]),
            y=float(lead0[1]),
            vx=float(lead0[2]),
            vy=float(lead0[3]),
            ax=float(lead0[4]),
            ay=float(lead0[5]),
            box=VehicleBox(length=float(adv_length[i])),
        )
        trajectories.append(integrate_following_actions(initial, ax[i, :, None], dt)[1:])
    return np.stack(trajectories, axis=0)


def sample_rollouts(config: dict, config_dir: Path, checkpoint: str | None, split: str, num_samples: int) -> Path:
    output_dir = _resolve_output_dir(config, config_dir)
    schema = load_json(output_dir / "feature_schema.json")
    stats = load_json(output_dir / "normalization_stats.json")
    arrays = _load_npz(output_dir / "dataset_normalized.npz")
    raw = _load_npz(output_dir / "dataset.npz")
    set_seed(int(config.get("evaluation", {}).get("seed", config.get("training", {}).get("seed", 42))))
    device = select_device(config.get("training", {}).get("device", "auto"))
    model = build_model_from_schema(schema, config).to(device)
    state = torch.load(_resolve_checkpoint_path(checkpoint, output_dir), map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    split_index = {"train": 0, "val": 1, "test": 2}[split]
    idx = np.where(arrays["split_index"] == split_index)[0][: int(num_samples)]
    if len(idx) == 0:
        raise RuntimeError(f"No samples for split={split}")
    history = torch.from_numpy(arrays["context_states"][idx]).float().to(device)
    context = torch.from_numpy(arrays["context_features"][idx]).float().to(device)
    relative = torch.from_numpy(arrays["relative_history"][idx]).float().to(device)
    with torch.no_grad():
        normalized_actions = model.sample(len(idx), history, context, relative).detach().cpu().numpy()
    actions = _decode_actions(normalized_actions, stats)
    ax = _actions_to_ax(actions, raw["context_states"][idx], schema, config)
    trajectories = _integrate(ax, raw["context_states"][idx], raw["adv_length"][idx], schema)

    out_path = output_dir / "natural_rollouts.npz"
    np.savez_compressed(
        out_path,
        sample_index=idx.astype(np.int64),
        actions=actions.astype(np.float32),
        acceleration=ax.astype(np.float32),
        lead_trajectory=trajectories.astype(np.float32),
    )
    save_json(
        {
            "path": str(out_path),
            "num_samples": int(len(idx)),
            "split": split,
            "action_representation": schema.get("action_representation"),
        },
        output_dir / "natural_rollouts_summary.json",
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH, help="Checkpoint path.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val", help="Dataset split.")
    parser.add_argument("--num-samples", type=int, default=64, help="Number of contexts to sample.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    args = parser.parse_args()
    setup_logging(args.log_level)
    cfg_path = Path(args.config).resolve()
    sample_rollouts(load_yaml(cfg_path), cfg_path.parent, args.checkpoint, args.split, args.num_samples)


if __name__ == "__main__":
    main()
