"""Assembly contract for the Causal Influence Hierarchical World Model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from .model import FactualDynamicsModel
from .reaction_controller import CausalInfluenceResponsePolicy
from .rule_models import RuleModelBundle


METHOD_NAME = "Causal Influence Hierarchical World Model"
METHOD_ACRONYM = "CIH-WM"
METHOD_SCHEMA = "causal_influence_hierarchical_world_model"


def load_cih_method_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the public CIH-WM component contract."""
    source = Path(path)
    config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if config.get("schema_name") != METHOD_SCHEMA:
        raise ValueError(f"not a {METHOD_ACRONYM} configuration: {source}")
    declared = config.get("method", {})
    if declared.get("name") != METHOD_NAME or declared.get("acronym") != METHOD_ACRONYM:
        raise ValueError("CIH-WM method identity is incomplete")
    if declared.get("controller") != "causal_influence_response":
        raise ValueError("CIH-WM requires the causal influence response policy")
    lateral = config.get("response_channels", {}).get("lateral", {})
    if bool(lateral.get("enabled", False)):
        raise ValueError(
            "causal lateral response has no accepted training evidence; keep it disabled"
        )
    if bool(config.get("influence", {}).get("enable_secondary", False)):
        raise ValueError(
            "CIH-WM's maintained scope is direct follower response; keep chained influence disabled"
        )
    response_prior = declared.get("response_prior", {})
    if response_prior.get("name") != "mechanism_guided_response_prior":
        raise ValueError("CIH-WM requires the merged mechanism-guided response prior")
    if response_prior.get("frozen") is not True:
        raise ValueError("the merged response prior must remain frozen")
    paths = config.get("paths", {})
    if not paths.get("response_prior_checkpoint") or not paths.get("response_prior_lineage"):
        raise ValueError("CIH-WM requires response-prior checkpoint and lineage paths")
    return config


def build_response_policy(
    config: dict[str, Any],
    *,
    root: str | Path,
    device: torch.device,
    checkpoint: str | Path | None = None,
) -> CausalInfluenceResponsePolicy:
    """Build the maintained response policy, optionally loading its calibrator."""
    workspace = Path(root)
    paths = config["paths"]
    mechanism = RuleModelBundle.load(
        workspace / paths["longitudinal_mechanism_prior"]
    )
    method, supervised, calibrator = (
        config["method"], config["supervised"], config["calibrator"]
    )
    policy = CausalInfluenceResponsePolicy(
        mechanism,
        response_prior_checkpoint=str(
            workspace / paths["response_prior_checkpoint"]
        ),
        device=device,
        calibrator_hidden_dim=int(calibrator.get("hidden_dim", 64)),
        calibrator_initial_log_std=float(calibrator.get("initial_log_std", -2.0)),
        calibration_scale_radius=float(calibrator.get("scale_radius", 0.75)),
        calibration_residual_max_mps2=float(
            calibrator.get("residual_max_mps2", 2.0)
        ),
        event_calibration_strength=float(
            supervised.get("event_calibration_strength", 1.0)
        ),
        event_gate_temperature=float(supervised.get("event_gate_temperature", 1.0)),
        correction_jerk_limit_mps3=float(
            method.get("correction_jerk_limit_mps3", 12.0)
        ),
        reaction_trigger_threshold_mps2=float(
            method.get("reaction_trigger_threshold_mps2", 0.25)
        ),
    ).to(device)
    if checkpoint is not None:
        payload = torch.load(Path(checkpoint), map_location=device, weights_only=False)
        if payload.get("schema") != "cih_world_model":
            raise ValueError("response-policy checkpoint is not a CIH-WM artifact")
        state = payload.get("calibrator_state_dict", payload.get("adapter_state_dict"))
        if state is None:
            raise ValueError("response-policy checkpoint has no calibrator state")
        policy.adapter.load_state_dict(state, strict=True)
    return policy


@dataclass(frozen=True)
class CausalInfluenceHierarchicalWorldModel:
    """One deployable CIH-WM assembled from its factual and response layers.

    The factual checkpoint remains independent so its released reconstruction
    can be audited directly.  This wrapper defines the single runtime method
    consumed by evaluation and future ADS integration.
    """

    factual_dynamics: FactualDynamicsModel
    response_policy: CausalInfluenceResponsePolicy
    influence_graph_config: dict[str, float | int]
    response_channels: dict[str, dict[str, Any]]
    factual_checkpoint: Path
    response_checkpoint: Path | None

    @classmethod
    def load(
        cls,
        config: dict[str, Any],
        *,
        root: str | Path,
        device: torch.device,
        response_checkpoint: str | Path | None = None,
    ) -> tuple["CausalInfluenceHierarchicalWorldModel", dict[str, Any]]:
        from .train import load_checkpoint

        workspace = Path(root)
        world_config_path = workspace / config["base_config"]
        world_config = yaml.safe_load(world_config_path.read_text(encoding="utf-8"))
        factual_checkpoint = workspace / world_config["paths"]["evaluation_checkpoint"]
        factual, payload = load_checkpoint(factual_checkpoint, device=device)
        factual.eval()
        for parameter in factual.parameters():
            parameter.requires_grad_(False)
        response_path = None if response_checkpoint is None else Path(response_checkpoint)
        policy = build_response_policy(
            config,
            root=workspace,
            device=device,
            checkpoint=response_path,
        )
        return cls(
            factual_dynamics=factual,
            response_policy=policy,
            influence_graph_config={
                str(key): value for key, value in config["influence"].items()
            },
            response_channels=config["response_channels"],
            factual_checkpoint=factual_checkpoint,
            response_checkpoint=response_path,
        ), payload

    def rollout_options(self) -> dict[str, Any]:
        """Arguments shared by offline evaluation and ADS closed-loop rollout."""
        return {
            "controller": self.response_policy,
            "influence_graph_config": dict(self.influence_graph_config),
        }

    def contract(self) -> dict[str, Any]:
        """Return a JSON-ready description of the deployable method."""
        return {
            "schema": METHOD_SCHEMA,
            "name": METHOD_NAME,
            "acronym": METHOD_ACRONYM,
            "algorithmic_modules": {
                "factual_transition": {
                    "role": "frozen factual reconstruction foundation",
                    "formulation": "diffusion-plan-conditioned traffic state transition",
                },
                "causal_influence": {
                    "role": "select which realized follower relations may respond",
                    "formulation": "direct ego-to-follower and attenuated one-hop follower routing",
                    "forbidden_inputs": "future logged state, pending ego action, intervention label",
                },
                "human_response_calibration": {
                    "role": "turn authority into a realistic longitudinal action adjustment",
                    "formulation": "frozen mechanism-guided response prior plus learned constrained calibrator",
                },
                "mechanism_guided_response_prior": {
                    "role": "frozen learned longitudinal response proposal inside CIH-WM",
                    "formulation": "calibrated car-following mechanism and learned residual response",
                    "lineage": "results/hierarchical_world_model/cih_wm/response_prior_lineage.json",
                },
            },
            "implementation_note": "observation encoding, belief filtering, jerk decoding, integration, PPO, and action bounds support these modules but are not separate algorithmic claims",
            "response_channels": self.response_channels,
            "nominal_reference_is_model_input": False,
        }
