"""Released-configuration TrafficBots training wrapper for canonical highD."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from . import upstream  # registers vendored ``models`` and ``utils`` imports
from .data import DT_S

try:  # Keep cache-contract tests importable before the isolated env is created.
    from pytorch_lightning import LightningModule
except ModuleNotFoundError:  # pragma: no cover - exercised only without optional runtime
    LightningModule = nn.Module  # type: ignore[misc,assignment]


def _upstream_model_config(config: dict[str, Any]):
    from omegaconf import OmegaConf

    released = OmegaConf.load(Path(__file__).parent / "upstream" / "sim_agent.yaml")
    model = config["model"]
    released.time_step_gt = 149
    released.hidden_dim = int(model["hidden_dim"])
    released.model.hidden_dim = int(model["hidden_dim"])
    released.model.temp_window_size = int(model["policy_history_steps"])
    released.model.n_tgt_knn = int(model["n_tgt_knn"])
    released.model.ag_encoder.k_tgt_knn_ag2mp = float(model["ag2mp_knn"]) / float(model["n_tgt_knn"])
    released.model.ag_encoder.k_tgt_knn_ag2ag = float(model["ag2ag_knn"]) / float(model["n_tgt_knn"])
    released.model.ag_encoder.k_tgt_knn_ag2tl = float(model["ag2tl_knn"]) / float(model["n_tgt_knn"])
    released.model.tl_encoder.k_tgt_knn_tl2mp = float(model["tl2mp_knn"]) / float(model["n_tgt_knn"])
    released.model.tl_encoder.k_tgt_knn_tl2tl = float(model["tl2tl_knn"]) / float(model["n_tgt_knn"])
    released.model.latent_encoder.temporal_down_sample_rate = 1
    released.model.latent_encoder.temporal_indices = list(model["posterior_temporal_indices"])
    # Upstream HPTR accesses nested settings with attribute syntax (for
    # example ``input_encoder.mode``), so these must remain DictConfig values.
    # Converting them to plain dictionaries makes construction fail before the
    # first forward pass.
    OmegaConf.resolve(released)
    return released.model


class HighDTrafficBotsModule(LightningModule):
    """HighD wrapper retaining released detach and shared-policy semantics."""
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        from models.traffic_bots import TrafficBots
        from models.modules.distributions import DestCategorical
        from models.metrics.loss import BalancedKL
        from utils.dynamics import MultiPathPP

        options = _upstream_model_config(config)
        options.pop("_target_", None)
        self.model = TrafficBots(
            **options,
            mp_attr_dim=11,
            tl_state_dim=5,
            ag_attr_dim=6,
            ag_motion_dim=3,
            navi_mode="dest",
            navi_dim=None,
            time_step_gt=149,
            n_mp_pl_node=8,
            tl_mode="stop",
            action_dim=2,
        )
        self._dest_class = DestCategorical
        self.plant = MultiPathPP(dt=DT_S, max_acc=5.0, max_yaw_rate=1.5)
        train = config["training"]
        self.kl = BalancedKL(float(train["kl_balance_scale"]), float(train["kl_free_nats"]))

    @staticmethod
    def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
        return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

    def _tokens(self, batch: dict[str, torch.Tensor]):
        mp_pose = torch.cat((batch["map/pos"][..., :2], torch.atan2(batch["map/dir"][..., 1], batch["map/dir"][..., 0]).unsqueeze(-1)), -1)
        mp_tokens = self.model.mp_encoder(batch["map/valid"], batch["map/type"].float(), mp_pose, batch["map/type"])
        tl_pose = torch.cat((batch["tl_stop/pos"][..., :2], torch.atan2(batch["tl_stop/dir"][..., 1], batch["tl_stop/dir"][..., 0]).unsqueeze(-1)), -1)
        tl_tokens = self.model.tl_encoder.pre_compute(batch["tl_stop/valid"].any(-1), None, tl_pose, **mp_tokens)
        return mp_tokens, tl_tokens

    def _dest_distribution(self, batch: dict[str, torch.Tensor], mp_tokens: dict[str, torch.Tensor]):
        valid = batch["agent/valid"][..., :1]
        pose = torch.cat((batch["agent/pos"][..., :1, :2], batch["agent/yaw_bbox"][..., :1, :]), -1)
        motion = torch.cat((batch["agent/spd"][..., :1, :], batch["agent/acc"][..., :1, :], batch["agent/yaw_rate"][..., :1, :]), -1)
        attributes = torch.cat((batch["agent/size"], batch["agent/type"].float()), -1)
        distribution = self.model.navi_predictor(valid, attributes, motion, pose, ag_type=batch["agent/type"], **mp_tokens)
        logits = distribution.distribution.logits.masked_fill(~batch["destination_candidate_valid"], float("-inf"))
        all_invalid = ~batch["destination_candidate_valid"].any(-1)
        logits = logits.masked_fill(all_invalid.unsqueeze(-1), 0.0)
        return self._dest_class(logits=logits, valid=valid.any(-1))

    def _latent(self, batch: dict[str, torch.Tensor], mp_tokens, tl_tokens, posterior: bool):
        pose = torch.cat((batch["agent/pos"][..., :2], batch["agent/yaw_bbox"]), -1)
        motion = torch.cat((batch["agent/spd"], batch["agent/acc"], batch["agent/yaw_rate"]), -1)
        attributes = torch.cat((batch["agent/size"], batch["agent/type"].float()), -1)
        # The posterior is an inference network and may use the complete GT
        # scene during training.  The standard-Gaussian prior must only learn
        # which agents exist at S0; passing the full validity sequence would be
        # a future-presence side channel for datasets with spawning agents.
        latent_valid = (
            batch["agent/valid"]
            if posterior
            else batch["agent/valid"][..., :1]
        )
        return self.model.latent_encoder(latent_valid, attributes, motion, pose, batch["agent/type"], batch["tl_stop/state"], mp_tokens, tl_tokens, posterior)

    def _loss_rollout(
        self, batch: dict[str, torch.Tensor], *, epoch: int, training: bool = True
    ) -> dict[str, torch.Tensor]:
        mp_tokens, tl_tokens = self._tokens(batch)
        posterior, prior = self._latent(batch, mp_tokens, tl_tokens, True), self._latent(batch, mp_tokens, tl_tokens, False)
        # The released 90/10 posterior/prior rollout mixture is sampled per
        # scene, rather than coupling every item in a minibatch to one draw.
        if training:
            use_prior = torch.rand(len(batch["agent/valid"]), 1, 1, device=batch["agent/valid"].device) < float(self.config["training"]["p_rollout_prior"])
            latent = torch.where(use_prior, prior.sample(False), posterior.sample(False))
        else:
            # A validation loss must be a comparable checkpoint-selection
            # criterion, not a fresh Monte-Carlo draw each epoch.  The model is
            # in eval mode here, so posterior mode is deterministic while still
            # measuring the released CVAE reconstruction objective.
            latent = posterior.sample(True)
        posterior_valid = posterior.valid
        prior_valid = prior.valid
        if training:
            latent_valid = torch.where(
                use_prior.squeeze(-1), prior_valid, posterior_valid
            )
        else:
            latent_valid = posterior_valid
        destination_dist = self._dest_distribution(batch, mp_tokens)
        gt_destination = batch["agent/dest"]
        destination = gt_destination
        loss_valid = batch["agent/valid"].any(-1)
        destination_loss = -destination_dist.log_prob(gt_destination)[loss_valid].mean()
        valid = batch["agent/valid"][..., 0].clone()
        pose = torch.cat((batch["agent/pos"][..., 0, :2], batch["agent/yaw_bbox"][..., 0, :]), -1)
        motion = torch.cat((batch["agent/spd"][..., 0, :], batch["agent/acc"][..., 0, :], batch["agent/yaw_rate"][..., 0, :]), -1)
        attributes = torch.cat((batch["agent/size"], batch["agent/type"].float()), -1)
        forcing_probability = (
            max(0.0, float(self.config["training"]["prob_forcing_agent"]) - epoch * float(self.config["training"]["prob_forcing_agent_decrease_per_epoch"]))
            if training
            else 0.0
        )
        force_agents = torch.rand_like(valid.float()) < forcing_probability
        self.model.init(); navi_updated = True
        losses: list[torch.Tensor] = []
        for step in range(149):
            # This is intentionally before every policy call: released V1.5 detach semantics.
            policy_pose = pose.detach() if self.config["training"]["training_detach_model_input"] else pose
            policy_motion = motion.detach() if self.config["training"]["training_detach_model_input"] else motion
            action_dist, _ = self.model(valid, policy_pose, policy_motion, attributes, batch["agent/type"], latent, latent_valid, destination, latent_valid, navi_updated, batch["tl_stop/state"][:, :, step], tl_tokens, mp_tokens)
            navi_updated = False
            controls = self.plant.process_action(action_dist.mean)
            predicted_pose, predicted_motion = self.plant.update(pose, motion, controls)
            target_valid = batch["agent/valid"][..., step + 1]
            target_pose = torch.cat((batch["agent/pos"][..., step + 1, :2], batch["agent/yaw_bbox"][..., step + 1, :]), -1)
            target_motion = torch.cat((batch["agent/spd"][..., step + 1, :], batch["agent/acc"][..., step + 1, :], batch["agent/yaw_rate"][..., step + 1, :]), -1)
            mask = target_valid
            position = F.smooth_l1_loss(predicted_pose[..., :2], target_pose[..., :2], reduction="none").mean(-1)
            yaw = 0.5 * (1 - torch.cos(predicted_pose[..., 2] - target_pose[..., 2]))
            speed = F.smooth_l1_loss(predicted_motion[..., 0], target_motion[..., 0], reduction="none")
            losses.append((0.1 * position[mask] + 10.0 * yaw[mask] + 0.1 * speed[mask]).mean())
            # Teacher forcing is background-only; ego is externally GT-overridden for training.
            override = (force_agents & target_valid); override[:, 0] = target_valid[:, 0]
            pose = torch.where(override.unsqueeze(-1), target_pose, predicted_pose)
            motion = torch.where(override.unsqueeze(-1), target_motion, predicted_motion)
            valid = target_valid
        # ``Independent`` latent distributions return one balanced-KL value
        # per [scene, agent], matching the released metric accumulator.
        kl = self.kl.compute(posterior.distribution, prior.distribution)
        # Released ``kl_for_unseen_agent=True`` uses posterior validity for KL.
        kl_loss = kl[posterior_valid].mean()
        reconstruction = torch.stack(losses).mean()
        total = reconstruction + kl_loss + destination_loss
        return {"loss": total, "reconstruction": reconstruction, "kl": kl_loss, "destination": destination_loss}

    def training_step(self, batch: dict[str, Any], batch_idx: int):
        losses = self._loss_rollout(
            self._move(batch, self.device), epoch=int(getattr(self, "current_epoch", 0))
        )
        if hasattr(self, "log_dict"): self.log_dict({f"train/{key}": value for key, value in losses.items()}, prog_bar=True)
        return losses["loss"]

    def validation_step(self, batch: dict[str, Any], batch_idx: int):
        """A fixed held-out closed-loop objective for checkpoint selection.

        Validation deliberately disables teacher forcing.  It does not touch the
        test split and keeps the released detached-policy rollout semantics.
        """
        losses = self._loss_rollout(
            self._move(batch, self.device), epoch=int(getattr(self, "current_epoch", 0)), training=False
        )
        if hasattr(self, "log_dict"):
            self.log_dict({f"val/{key}": value for key, value in losses.items()}, prog_bar=True, sync_dist=False)
        return losses["loss"]

    def configure_optimizers(self):
        train = self.config["training"]
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=float(train["learning_rate"]),
            weight_decay=float(train["weight_decay"]), betas=(0.9, 0.95),
        )
        # Match the released V1.5 optimizer schedule instead of silently
        # training an improved or differently annealed highD variant.
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=int(train.get("lr_scheduler_step_size", 7)),
                    gamma=float(train.get("lr_scheduler_gamma", 0.5)),
                ),
                "interval": "epoch",
            },
        }
