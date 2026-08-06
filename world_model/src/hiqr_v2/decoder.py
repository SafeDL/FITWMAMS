"""State-aware 5/15/5 plan continuation for HiQR-v2."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import HiQRV2Config


class StateAwarePlanContinuationDecoder(nn.Module):
    """Preserve only the imminent action prefix and replan stale long horizons."""

    def __init__(self, cfg: HiQRV2Config) -> None:
        super().__init__()
        self.cfg = cfg
        h, z, g = (
            int(cfg.hidden_dim),
            int(cfg.agent_residual_dim),
            int(cfg.scene_latent_dim),
        )
        self.agent = nn.Sequential(nn.Linear(2 * h + z, h), nn.SiLU(), nn.Linear(h, h))
        self.context = nn.Sequential(nn.Linear(h + g, h), nn.SiLU(), nn.Linear(h, h))
        self.carry = nn.Sequential(nn.Linear(21, h), nn.SiLU(), nn.Linear(h, h))
        self.time = nn.Parameter(torch.zeros(int(cfg.plan_frames), h))
        self.blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                h,
                int(cfg.num_heads),
                h * 2,
                float(cfg.dropout),
                batch_first=True,
                activation="gelu",
                norm_first=True,
            ),
            num_layers=max(1, int(cfg.decoder_layers)),
        )
        self.action = nn.Linear(h, 2)
        self.gate = nn.Linear(h, 1)
        self.register_buffer(
            "gate_time_bias", torch.linspace(-2.0, 1.0, int(cfg.carry_replan_frames))
        )
        nn.init.normal_(self.time, std=0.02)
        nn.init.zeros_(self.action.bias)

    def _direct_actions(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.action(tokens)
        acceleration = torch.where(
            logits[..., 0] >= 0.0,
            float(self.cfg.max_acceleration) * torch.tanh(logits[..., 0]),
            -float(self.cfg.min_acceleration) * torch.tanh(logits[..., 0]),
        )
        yaw_rate = float(self.cfg.max_yaw_rate) * torch.tanh(logits[..., 1])
        return torch.stack((acceleration, yaw_rate), dim=-1)

    def forward(
        self,
        agents: torch.Tensor,
        global_hidden: torch.Tensor,
        agent_hidden: torch.Tensor,
        scene_latent: torch.Tensor,
        agent_residual: torch.Tensor,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        previous_buffer: torch.Tensor | None = None,
        previous_background_states: torch.Tensor | None = None,
        previous_expected_ego: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, backgrounds, horizon = (
            current.shape[0],
            current.shape[1] - 1,
            int(self.cfg.plan_frames),
        )
        if backgrounds != 6:
            raise ValueError("HiQR-v2 requires ego plus six background slots")
        base = self.agent(
            torch.cat(
                (agents[:, 1:], agent_hidden[:, 1:], agent_residual[:, 1:]), dim=-1
            )
        )
        context = self.context(torch.cat((global_hidden, scene_latent), dim=-1))
        token = base[:, None] + context[:, None, None] + self.time[None, :, None]
        valid = current_valid[:, None, 1:].expand(-1, horizon, -1)
        gate = token.new_zeros((batch, horizon, backgrounds, 1))
        emergency = torch.zeros_like(valid)
        carried = torch.zeros_like(valid)
        if (
            previous_buffer is not None
            and self.cfg.continuation_mode == "adaptive_5_15_5"
        ):
            if previous_background_states is None or previous_expected_ego is None:
                raise ValueError(
                    "HiQR-v2 continuation requires prior predicted states and ego expectation"
                )
            execute = int(self.cfg.execute_frames)
            carry_action = previous_buffer[:, execute:]
            carry_state = previous_background_states[:, execute:]
            background_error = (
                current[:, 1:] - previous_background_states[:, execute - 1]
            )
            ego_error = current[:, 0] - previous_expected_ego[:, execute - 1, 0]
            elapsed = torch.linspace(
                0.0, 1.0, horizon - execute, device=current.device, dtype=current.dtype
            )[None, :, None, None]
            staleness = torch.cat(
                (
                    carry_action,
                    carry_state,
                    background_error[:, None].expand(-1, horizon - execute, -1, -1),
                    ego_error[:, None, None].expand(
                        -1, horizon - execute, backgrounds, -1
                    ),
                    elapsed.expand(batch, -1, backgrounds, -1),
                ),
                dim=-1,
            )
            token[:, : horizon - execute] = token[:, : horizon - execute] + self.carry(
                staleness
            )
            carried[:, : horizon - execute] = valid[:, : horizon - execute]
        flat = token.reshape(batch, horizon * backgrounds, -1)
        padding = ~valid.reshape(batch, horizon * backgrounds)
        empty = padding.all(dim=1)
        if empty.any():
            padding = padding.clone()
            padding[empty, 0] = False
        direct = self._direct_actions(
            self.blocks(flat, src_key_padding_mask=padding).reshape(
                batch, horizon, backgrounds, -1
            )
        )
        if previous_buffer is None or self.cfg.continuation_mode == "full_replan":
            actions = direct
        else:
            execute = int(self.cfg.execute_frames)
            hard = int(self.cfg.hard_carry_frames)
            replan_end = hard + int(self.cfg.carry_replan_frames)
            carry_action = previous_buffer[:, execute:]
            ego_position_ratio = torch.linalg.vector_norm(
                ego_error[..., :2], dim=-1
            ) / float(self.cfg.emergency_ego_position_error_m)
            ego_velocity_ratio = torch.linalg.vector_norm(
                ego_error[..., 2:4], dim=-1
            ) / float(self.cfg.emergency_ego_velocity_error_mps)
            emergency_severity = torch.maximum(ego_position_ratio, ego_velocity_ratio)
            emergency_alpha = 1.0 - torch.exp(
                -float(self.cfg.emergency_gate_sharpness)
                * torch.relu(emergency_severity - 1.0)
            )
            emergency_alpha = emergency_alpha[:, None, None, None].expand(
                -1, hard, backgrounds, -1
            )
            alpha = torch.sigmoid(
                self.gate(token[:, hard:replan_end])
                + self.gate_time_bias[None, :, None, None]
            )
            gate[:, :hard] = emergency_alpha
            gate[:, hard:replan_end] = alpha
            hard_actions = (1.0 - emergency_alpha) * carry_action[
                :, :hard
            ] + emergency_alpha * direct[:, :hard]
            mixed = (1.0 - alpha) * carry_action[:, hard:replan_end] + alpha * direct[
                :, hard:replan_end
            ]
            actions = torch.cat((hard_actions, mixed, direct[:, replan_end:]), dim=1)
            emergency[:, :hard] = emergency_alpha.squeeze(-1) > 1.0e-4
        actions = (
            torch.stack(
                (
                    actions[..., 0].clamp(
                        float(self.cfg.min_acceleration),
                        float(self.cfg.max_acceleration),
                    ),
                    actions[..., 1].clamp(
                        -float(self.cfg.max_yaw_rate), float(self.cfg.max_yaw_rate)
                    ),
                ),
                dim=-1,
            )
            * valid[..., None].float()
        )
        revised = torch.zeros_like(valid)
        if (
            previous_buffer is not None
            and self.cfg.continuation_mode == "adaptive_5_15_5"
        ):
            carry_action = previous_buffer[:, int(self.cfg.execute_frames) :]
            revised[:, :20] = (actions[:, :20] - carry_action[:, :20]).abs().amax(
                dim=-1
            ) > 1.0e-4
        return {
            "background_future_actions": actions,
            "continuation_gate": gate,
            "background_future_action_masks": {
                "carried": carried,
                "revised": revised & valid,
                "emergency": emergency & valid,
                "valid": valid,
            },
        }
