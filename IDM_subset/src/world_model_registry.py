"""Lazy registry for world models supported by the IDM evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorldModelSpec:
    """Execution and replay hooks for one background world model."""

    model_id: str
    display_name: str
    background_label: str
    background_color: str
    background_color_name: str
    default_config: Path
    random_blocks: tuple[str, ...]
    provenance_keys: tuple[str, ...]

    def validate_formal_provenance(
        self, config: dict[str, Any], config_dir: Path
    ) -> dict[str, Any]:
        """Preflight formal artifacts before a multi-model run starts."""
        if self.model_id == "hierarchical":
            from .world_subset_runner import _require_formal_provenance

            return _require_formal_provenance(config, config_dir)
        if self.model_id == "trafficbots":
            from .trafficbots_runner import _require_trafficbots_provenance

            return _require_trafficbots_provenance(config, config_dir)
        raise AssertionError(f"unhandled registered model: {self.model_id}")

    def run_subset(
        self, config: dict[str, Any], config_dir: Path, *, formal: bool = True
    ) -> Path:
        if self.model_id == "hierarchical":
            from .world_subset_runner import run_subset_from_config

            return run_subset_from_config(config, config_dir, formal=formal)
        if self.model_id == "trafficbots":
            from .trafficbots_runner import run_trafficbots_subset_from_config

            return run_trafficbots_subset_from_config(config, config_dir, formal=formal)
        raise AssertionError(f"unhandled registered model: {self.model_id}")

    def run_monte_carlo(
        self, config: dict[str, Any], config_dir: Path, *, formal: bool = True
    ) -> Path:
        if self.model_id == "hierarchical":
            from .world_subset_runner import run_monte_carlo_from_config

            return run_monte_carlo_from_config(config, config_dir, formal=formal)
        if self.model_id == "trafficbots":
            from .trafficbots_runner import run_trafficbots_monte_carlo_from_config

            return run_trafficbots_monte_carlo_from_config(config, config_dir, formal=formal)
        raise AssertionError(f"unhandled registered model: {self.model_id}")

    def build_evaluator(
        self, config: dict[str, Any], config_dir: Path
    ) -> tuple[Any, dict[str, Any]]:
        if self.model_id == "hierarchical":
            from .world_subset_runner import _build_evaluator

            return _build_evaluator(config, config_dir)
        if self.model_id == "trafficbots":
            from .trafficbots_runner import _build_evaluator

            return _build_evaluator(config, config_dir)
        raise AssertionError(f"unhandled registered model: {self.model_id}")

    def replay_case(self, evaluator: Any, path: str | Path) -> tuple[Any, Any]:
        """Return ``(rollout, exogenous_state)`` for an exact stored case."""
        if self.model_id == "hierarchical":
            from hierarchical_world_model.src.execution import rollout_world

            world = self.load_exogenous(path)
            rollout = rollout_world(
                evaluator.sampler,
                world,
                evaluator.policy,
                steps=evaluator.steps,
                evt_model=evaluator.evt_model,
            )
            return rollout, world
        if self.model_id == "trafficbots":
            world = self.load_exogenous(path)
            return evaluator.rollout(world), world
        raise AssertionError(f"unhandled registered model: {self.model_id}")

    def load_exogenous(self, path: str | Path) -> Any:
        if self.model_id == "hierarchical":
            from hierarchical_world_model.src.randomness import WorldExogenousState
            from hierarchical_world_model.src.empirical_context import EmpiricalKWorldState

            try:
                return WorldExogenousState.load(path)
            except (KeyError, ValueError):
                return EmpiricalKWorldState.load(path)
        if self.model_id == "trafficbots":
            from .trafficbots_randomness import TrafficBotsExogenousState

            return TrafficBotsExogenousState.load(path)
        raise AssertionError(f"unhandled registered model: {self.model_id}")


ROOT = Path(__file__).resolve().parents[2]

_MODELS = {
    "hierarchical": WorldModelSpec(
        model_id="hierarchical",
        display_name="Hierarchical Flow–Diffusion–HiQR",
        background_label="HiQR background",
        background_color="#3274a1",
        background_color_name="blue",
        default_config=ROOT / "IDM_subset/configs/world_subset_idm.yaml",
        random_blocks=(
            "scenario_uniform",
            "c0_base_latent",
            "k_base_latent",
            "diffusion_noise",
            "scene_innovations",
            "agent_response_innovations",
        ),
        provenance_keys=(
            "flow_checkpoint_sha256",
            "diffusion_checkpoint_sha256",
            "response_checkpoint_sha256",
            "evt_model_sha256",
            "idm_ego_config_sha256",
            "execution_backend",
            "hiqr_vehicle_dynamics_contract",
        ),
    ),
    "trafficbots": WorldModelSpec(
        model_id="trafficbots",
        display_name="TrafficBots V1.5-HighD",
        background_label="TrafficBots background",
        background_color="#6a3d9a",
        background_color_name="purple",
        default_config=ROOT / "IDM_subset/configs/trafficbots_subset_idm.yaml",
        random_blocks=(
            "scenario_uniform",
            "c0_base_latent",
            "personality_latent",
            "destination_uniform",
        ),
        provenance_keys=(
            "flow_checkpoint_sha256",
            "trafficbots_config_sha256",
            "trafficbots_checkpoint_sha256",
            "evt_model_sha256",
            "idm_ego_config_sha256",
            "execution_backend",
            "background_dynamics_contract",
        ),
    ),
}

_ALIASES = {
    "hierarchical": "hierarchical",
    "hiqr": "hierarchical",
    "flow_diffusion_hiqr": "hierarchical",
    "trafficbots": "trafficbots",
    "trafficbot": "trafficbots",
    "tb": "trafficbots",
}


def world_model_ids() -> tuple[str, ...]:
    return tuple(_MODELS)


def get_world_model(model_id: str) -> WorldModelSpec:
    normalized = str(model_id).strip().lower().replace("-", "_")
    canonical = _ALIASES.get(normalized)
    if canonical is None:
        supported = ", ".join(world_model_ids())
        raise ValueError(f"unknown IDM world model {model_id!r}; choose one of: {supported}")
    return _MODELS[canonical]
