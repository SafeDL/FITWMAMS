"""Fine-tune response models or an equal-update human-only control."""

from __future__ import annotations

import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from npc_behavior_benchmark.data.dataset import CausalBCDataset, EventAnchoredBCDataset
from npc_behavior_benchmark.data.manifest import load_benchmark_config
from npc_behavior_benchmark.data.response_dataset import ResponsePairDataset
from npc_behavior_benchmark.policies.response_delta_policy import (
    student_paired_response,
    energy_distance_loss,
)
from npc_behavior_benchmark.policies.shared_bc import SharedBCModel, gaussian_nll


def _to(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def train(args):
    config, _ = load_benchmark_config(args.config)
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    source = torch.load(args.base_checkpoint, map_location="cpu")
    model = SharedBCModel(**source["model_config"])
    model.load_state_dict(source["model_state"])
    model.to(device)
    style_resampling = (
        source.get("style_resampling", "persistent")
        if args.style_resampling == "auto"
        else args.style_resampling
    )
    if style_resampling == "none":
        style_resampling = "persistent"
    method_id = args.method_id or f"response_{args.ablation}"
    if args.include_d1 and "_d0d1" not in method_id:
        method_id += (
            "_d0d1"
            if args.d1_fraction == 1.0
            else f"_d0d1f{int(round(args.d1_fraction * 100)):03d}"
        )
    feature_mean = torch.as_tensor(source["feature_mean"], device=device)
    feature_std = torch.as_tensor(source["feature_std"], device=device)
    response_train = ResponsePairDataset(args.config, "train")
    response_val = (
        ResponsePairDataset(args.config, "validation")
        if args.ablation != "a0c"
        else None
    )
    if args.max_response:
        response_train = Subset(
            response_train, range(min(args.max_response, len(response_train)))
        )
        if response_val is not None:
            response_val = Subset(
                response_val,
                range(min(max(args.max_response // 4, 1), len(response_val))),
            )
    d0_human = CausalBCDataset(args.config, "train", seed=seed)
    d0_human_val = CausalBCDataset(args.config, "validation", seed=seed)
    d1_human = (
        EventAnchoredBCDataset(
            args.config, "train", seed=seed, fraction=args.d1_fraction
        )
        if args.include_d1
        else None
    )
    d1_human_val = (
        EventAnchoredBCDataset(args.config, "validation", seed=seed)
        if args.include_d1
        else None
    )
    human = ConcatDataset((d0_human, d1_human)) if d1_human is not None else d0_human
    if d1_human_val is not None:
        human_val = ConcatDataset(
            (
                Subset(d0_human_val, range(min(1024, len(d0_human_val)))),
                Subset(d1_human_val, range(min(1024, len(d1_human_val)))),
            )
        )
    else:
        human_val = Subset(d0_human_val, range(min(2048, len(d0_human_val))))
    response_loader = DataLoader(
        response_train,
        batch_size=args.response_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    response_val_loader = (
        None
        if response_val is None
        else DataLoader(
            response_val,
            batch_size=args.response_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    )
    human_loader = DataLoader(
        human,
        batch_size=args.human_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    human_val_loader = DataLoader(
        human_val,
        batch_size=args.human_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    human_iterator = iter(human_loader)
    pair_file = (
        Path(config["paths"]["output_dir"]) / "data/response_pairs_ma_idm_train_v3.npz"
    )
    if args.ablation != "a0c":
        with np.load(pair_file, allow_pickle=False) as pairs:
            key = (
                "teacher_response_delta"
                if args.ablation == "a2"
                else "teacher_reactive_features"
            )
            values = np.asarray(pairs[key], np.float64)
        scale = np.std(values, axis=(0, 1, 2)).clip(0.05).astype(np.float32)
        scale_t = torch.as_tensor(scale, device=device).reshape(1, 1, 1, 4)
    else:
        scale = np.ones(4, np.float32)
        scale_t = None
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    history = []
    best = float("inf")
    stale = 0
    output = Path(config["paths"]["output_dir"]) / f"methods/{method_id}" / str(seed)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        d0_human.set_epoch(epoch)
        totals = {"loss": 0.0, "human": 0.0, "response": 0.0}
        count = 0
        epoch_start = time.time()
        for response_raw in response_loader:
            try:
                human_raw = next(human_iterator)
            except StopIteration:
                human_iterator = iter(human_loader)
                human_raw = next(human_iterator)
            rb = _to(response_raw, device)
            hb = _to(human_raw, device)
            optimizer.zero_grad(set_to_none=True)
            human_style = (
                torch.randn(
                    (len(hb["features"]), hb["features"].shape[2], model.style_dim),
                    device=device,
                )
                if model.style_dim
                else None
            )
            mean, log_std = model(
                (hb["features"] - feature_mean) / feature_std,
                hb["history_valid"],
                human_style,
            )
            human_loss = gaussian_nll(mean, log_std, hb["target"], hb["target_valid"])
            if args.ablation == "a0c":
                response_loss = human_loss.new_zeros(())
                loss = human_loss
            else:
                student_delta, student_reactive = student_paired_response(
                    model,
                    feature_mean,
                    feature_std,
                    rb["history"],
                    rb["history_valid"],
                    rb["logged_future"],
                    rb["intervention_stimulus"],
                    rb["stimulus_index"],
                    rb["response_index"],
                    rb["lengths"],
                    rb["widths"],
                    rb["map_polylines"],
                    rb["map_valid"],
                    futures=args.student_futures,
                    style_resampling=style_resampling,
                )
                student = student_delta if args.ablation == "a2" else student_reactive
                teacher = (
                    rb["teacher_delta"]
                    if args.ablation == "a2"
                    else rb["teacher_reactive"]
                )
                response_loss = energy_distance_loss(student, teacher, scale_t)
                loss = human_loss + args.response_weight * response_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            size = len(rb["history"])
            count += size
            totals["loss"] += float(loss.detach()) * size
            totals["human"] += float(human_loss.detach()) * size
            totals["response"] += float(response_loss.detach()) * size
        model.eval()
        torch.manual_seed(12345)
        val_response = 0.0
        val_count = 0
        with torch.no_grad():
            if response_val_loader is not None:
                for raw in response_val_loader:
                    rb = _to(raw, device)
                    sd, sr = student_paired_response(
                        model,
                        feature_mean,
                        feature_std,
                        rb["history"],
                        rb["history_valid"],
                        rb["logged_future"],
                        rb["intervention_stimulus"],
                        rb["stimulus_index"],
                        rb["response_index"],
                        rb["lengths"],
                        rb["widths"],
                        rb["map_polylines"],
                        rb["map_valid"],
                        futures=args.student_futures,
                        style_resampling=style_resampling,
                    )
                    value = energy_distance_loss(
                        sd if args.ablation == "a2" else sr,
                        (
                            rb["teacher_delta"]
                            if args.ablation == "a2"
                            else rb["teacher_reactive"]
                        ),
                        scale_t,
                    )
                    val_response += float(value) * len(rb["history"])
                    val_count += len(rb["history"])
            val_human = 0.0
            human_count = 0
            for raw in human_val_loader:
                hb = _to(raw, device)
                validation_style = (
                    torch.zeros(
                        (len(hb["features"]), hb["features"].shape[2], model.style_dim),
                        device=device,
                    )
                    if model.style_dim
                    else None
                )
                mean, ls = model(
                    (hb["features"] - feature_mean) / feature_std,
                    hb["history_valid"],
                    validation_style,
                )
                value = gaussian_nll(mean, ls, hb["target"], hb["target_valid"])
                val_human += float(value) * len(hb["features"])
                human_count += len(hb["features"])
        val_response /= max(val_count, 1)
        val_human /= max(human_count, 1)
        criterion = (
            val_human
            if args.ablation == "a0c"
            else val_human + args.response_weight * val_response
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": totals["loss"] / count,
            "train_human_nll": totals["human"] / count,
            "train_response_energy": totals["response"] / count,
            "validation_human_nll": val_human,
            "validation_response_energy": val_response,
            "validation_criterion": criterion,
            "seconds": time.time() - epoch_start,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if criterion < best - 1e-5:
            best = criterion
            stale = 0
            torch.save(
                {
                    **source,
                    "model_state": model.state_dict(),
                    "fit_seed": seed,
                    "method_id": method_id,
                    "training_data": "D0+D1" if args.include_d1 else "D0",
                    "d1_training_fraction": (
                        args.d1_fraction if args.include_d1 else 0.0
                    ),
                    "style_resampling": style_resampling if model.style_dim else "none",
                    "response_feature_scale": scale,
                },
                output / "best.pt",
            )
        else:
            stale += 1
            if stale >= 4:
                break
    report = {
        "method_id": method_id,
        "fit_seed": seed,
        "training_data": "D0+D1" if args.include_d1 else "D0",
        "d1_training_fraction": args.d1_fraction if args.include_d1 else 0.0,
        "d0_human_rows": len(d0_human),
        "d1_human_rows": len(d1_human) if d1_human is not None else 0,
        "style_dim": model.style_dim,
        "style_resampling": style_resampling if model.style_dim else "none",
        "response_pairs": len(response_train),
        "validation_pairs": 0 if response_val is None else len(response_val),
        "best_validation_criterion": best,
        "epochs_completed": len(history),
        "wall_time_seconds": time.time() - started,
        "history": history,
    }
    (output / "training_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/benchmark_v1.yaml",
    )
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--ablation", choices=("a0c", "a1", "a2"), required=True)
    p.add_argument("--method-id", default="")
    p.add_argument(
        "--style-resampling",
        choices=("auto", "persistent", "per_decision"),
        default="auto",
    )
    p.add_argument("--seed", type=int, default=20260919)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--response-batch-size", type=int, default=32)
    p.add_argument("--human-batch-size", type=int, default=128)
    p.add_argument("--student-futures", type=int, default=2)
    p.add_argument("--response-weight", type=float, default=0.25)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-response", type=int, default=0)
    p.add_argument("--include-d1", action="store_true")
    p.add_argument("--d1-fraction", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    print(json.dumps(train(a), indent=2))


if __name__ == "__main__":
    main()
