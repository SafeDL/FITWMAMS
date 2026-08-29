from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from world_model.trafficbots.config import load_config
from world_model.trafficbots.data import TrafficBotsHighDDataset, make_loader
from world_model.trafficbots.module import HighDTrafficBotsModule
from world_model.src.core.utils import set_seed

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path, help="Lightning last.ckpt to resume exactly")
    args = parser.parse_args()
    config = load_config(args.config) if args.config else load_config()
    train = config["training"]
    seed = int(train.get("seed", config["experiment"]["seed"]))
    set_seed(seed)
    dataset = TrafficBotsHighDDataset(config["paths"]["sequence_cache_dir"], "train", seed=seed)
    validation_limit = int(train.get("validation_sequences", 0))
    validation = TrafficBotsHighDDataset(
        config["paths"]["sequence_cache_dir"], "val", seed=seed, maximum=validation_limit
    )
    output = Path(config["paths"]["output_dir"])
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    best = ModelCheckpoint(
        dirpath=checkpoints, filename="best", monitor="val/loss", mode="min",
        save_top_k=1, save_last=False, every_n_epochs=1,
    )
    # ``last.ckpt`` is overwritten every fixed number of optimiser updates.
    # It is intentionally separate from the validation-selected best checkpoint
    # so a machine interruption never discards an unfinished full epoch.
    resume = ModelCheckpoint(
        dirpath=checkpoints, save_top_k=0, save_last=True,
        every_n_train_steps=int(train.get("checkpoint_every_n_train_steps", 250)),
    )
    callbacks = [
        best, resume,
        EarlyStopping(monitor="val/loss", mode="min", patience=int(train.get("early_stopping_patience", 4)), strict=True),
    ]
    config_sha256 = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    (output / "training_manifest.json").write_text(json.dumps({
        "experiment_scope": "full", "seed": seed, "config_sha256": config_sha256,
        "train_sequences": len(dataset), "validation_sequences": len(validation),
        "test_sequences_reserved": len(TrafficBotsHighDDataset(config["paths"]["sequence_cache_dir"], "test", seed=seed)),
        "training_detach_model_input": bool(train["training_detach_model_input"]),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    trainer = Trainer(
        max_epochs=int(train["max_epochs"]), default_root_dir=config["paths"]["output_dir"],
        accelerator="auto", devices=1, deterministic=True, benchmark=False,
        precision=int(train.get("precision", 32)),
        gradient_clip_val=float(train.get("gradient_clip_val", 0.0)),
        log_every_n_steps=200, num_sanity_val_steps=0,
        callbacks=callbacks, logger=True,
    )
    automatic_resume = checkpoints / "last.ckpt"
    resume_path = args.resume or (automatic_resume if automatic_resume.is_file() else None)
    trainer.fit(
        HighDTrafficBotsModule(config),
        make_loader(dataset, batch_size=int(train["batch_size"]), shuffle=True, workers=int(train.get("num_workers", 0)), seed=seed),
        make_loader(validation, batch_size=int(train.get("validation_batch_size", train["batch_size"])), shuffle=False, workers=int(train.get("num_workers", 0)), seed=seed),
        ckpt_path=str(resume_path) if resume_path else None,
    )

if __name__ == "__main__": main()
