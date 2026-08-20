"""Configuration for the diffusion-guided HiQR response layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from world_model.src.hiqr.config import HiQRConfig


@dataclass(frozen=True)
class WorldModelConfig:
    """One maintained 25 Hz, six-background response contract."""

    hidden_dim: int = 128
    num_heads: int = 4
    temporal_layers: int = 1
    relation_layers: int = 2
    decoder_layers: int = 2
    dropout: float = 0.1
    scene_latent_dim: int = 16
    agent_latent_dim: int = 16
    scene_refresh_responses: int = 25
    scene_noise_scale: float = 0.17
    agent_noise_scale: float = 0.06
    agent_noise_correlation: float = 0.999
    jerk_knots: int = 5
    history_frames: int = 25
    history_choices: tuple[int, ...] = (5, 10, 15, 25)
    execute_frames: int = 1
    preview_frames: int = 25
    dt_s: float = 0.04
    min_acceleration_mps2: float = -8.0
    max_acceleration_mps2: float = 4.0
    max_yaw_rate_rps: float = 0.6
    max_residual_jerk_mps3: float = 2.0
    max_residual_yaw_acceleration_rps2: float = 0.2
    min_acceleration_std_mps2: float = 0.015
    min_yaw_rate_std_rps: float = 0.0002
    initial_acceleration_std_mps2: float = 0.20
    initial_yaw_rate_std_rps: float = 0.01
    stochastic_longitudinal_jerk_mps3: float = 0.60
    stochastic_yaw_acceleration_rps2: float = 0.006
    velocity_rebase_horizon_s: float = 0.40
    # The maintained formal YAML has always specified 0.50.  This default is
    # also the migration value for checkpoints created before the field was
    # serialized in ``model_config``.
    soft_anchor_final_weight: float = 0.50
    causal_response_min_gain: float = 1.00
    causal_response_max_gain: float = 1.40
    causal_response_initial_gain: float = 1.10
    causal_response_scale: float = 1.0
    intervention_adapter_enabled: bool = False
    intervention_trigger_threshold_mps2: float = 0.5
    intervention_trigger_history_frames: int = 5
    intervention_adapter_max_gain: float = 0.5
    intervention_memory_decay: float = 0.85
    use_soft_plan: bool = True
    stochastic_latents: bool = True
    action_weight: float = 1.0
    position_weight: float = 1.0
    velocity_weight: float = 0.25
    nll_weight: float = 0.05
    boundary_weight: float = 0.08
    jerk_weight: float = 0.10
    angular_weight: float = 0.05
    energy_weight: float = 1.0
    locality_weight: float = 0.10
    monotonicity_weight: float = 0.08
    response_strength_weight: float = 2.0
    closed_loop_factual_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.execute_frames != 1 or self.history_frames != 25:
            raise ValueError("the response contract is one 25 Hz frame with 25-frame history")
        if self.preview_frames != 25:
            raise ValueError("HiQR uses a one-second soft preview")
        if set(self.history_choices) - {5, 10, 15, 25}:
            raise ValueError("history choices must be 5, 10, 15 or 25")
        if self.scene_refresh_responses != 25:
            raise ValueError("the 16-D scene latent updates once per second")
        if not (
            0.0 <= self.causal_response_min_gain
            < self.causal_response_initial_gain
            < self.causal_response_max_gain
        ):
            raise ValueError("causal gains must satisfy 0 <= min < initial < max")
        if not 0.5 <= self.causal_response_scale <= 4.0:
            raise ValueError("causal response scale must lie inside [0.5, 4.0]")
        if self.intervention_trigger_history_frames not in {1, 2, 5}:
            raise ValueError("intervention trigger history must be 1, 2 or 5 frames")
        if not 0.25 <= self.intervention_trigger_threshold_mps2 <= 2.0:
            raise ValueError("intervention trigger threshold must lie inside [0.25, 2]")
        if not 0.0 < self.intervention_adapter_max_gain <= 2.0:
            raise ValueError("intervention adapter maximum gain must lie inside (0, 2]")
        if not 0.5 <= self.intervention_memory_decay < 1.0:
            raise ValueError("intervention memory decay must lie inside [0.5, 1)")
        if not 0.0 < self.velocity_rebase_horizon_s <= 1.0:
            raise ValueError("velocity rebase horizon must lie inside (0, 1] s")
        if not 0.0 <= self.soft_anchor_final_weight < 1.0:
            raise ValueError("soft anchor final weight must lie inside [0, 1)")
        if not 0.0 <= self.agent_noise_correlation < 1.0:
            raise ValueError("agent noise correlation must lie inside [0, 1)")

    def hiqr_config(self) -> HiQRConfig:
        """Configure the reused, validated HiQR relational/filtering core."""
        return HiQRConfig(
            hidden_dim=self.hidden_dim,
            scene_latent_dim=self.scene_latent_dim,
            agent_residual_dim=self.agent_latent_dim,
            temporal_layers=self.temporal_layers,
            attention_layers=self.relation_layers,
            decoder_layers=self.decoder_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            jerk_knots=self.jerk_knots,
            max_longitudinal_jerk=self.max_residual_jerk_mps3,
            max_yaw_jerk=self.max_residual_yaw_acceleration_rps2,
            scene_noise_scale=self.scene_noise_scale,
            residual_noise_scale=self.agent_noise_scale,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
