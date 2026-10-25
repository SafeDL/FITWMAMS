"""End-to-end Flow, diffusion and reactive-world sampling interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diffusion.src.data import decode_residual_trajectory, prepare_flow_condition
from diffusion.src.train import load_checkpoint as load_diffusion_checkpoint
from hierarchical_world_model.src.core.evaluation_scope import (
    EXCLUDED_EVALUATION_SLOTS,
    evaluation_scope_contract,
    scoped_slot_mask,
)
from hierarchical_world_model.src.core.utils import load_json, load_yaml
from normalizing_flow.src.features import feature_valid_from_slot_mask
from normalizing_flow.src.scenario import load_checkpoint as load_flow_checkpoint

from .config import WorldModelConfig
from .data import artifact_identity
from .highway import HighwayEnvClosedLoopWorld
from .randomness import WorldExogenousState
from .stochastic_drivers.idm_population import (
    MODEL_PATH as DRIVER_MODEL_PATH,
)
from .stochastic_drivers.idm_population import (
    sample_theta,
)
from .stochastic_drivers.lane_trajectory import MODEL_PATH as LANE_MODEL_PATH
from .takeover import TakeoverConfig, TakeoverController
from .tracking import TrackingConfig

DEFAULT_WORLD_SELECTION = (
    Path(__file__).resolve().parents[2]
    / "results/hierarchical_world_model/model_assets/selection.json"
)


def load_selected_components():
    """Load the published chain and reject the known old/new asset mismatch."""
    from .shared_evaluation import load_shared_components

    selection = load_json(DEFAULT_WORLD_SELECTION)
    population, fit = load_shared_components()
    if (
        selection["status"] != "selected_after_shared_val_review"
        or selection["mobil"] != fit["mobil"]
        or selection["selected_model"] != "mobil"
        or selection["diffusion"] != artifact_identity(selection["diffusion"]["path"])
        or Path(selection["driver_parameter_source"]).resolve()
        != DRIVER_MODEL_PATH.resolve()
        or selection["driver_population"]
        != {name: population[name].tolist() for name in ("log_mean", "log_cov")}
        or Path(selection["lane_trajectory_model"]).resolve()
        != LANE_MODEL_PATH.resolve()
        or selection["lane_trajectory_model_mtime_ns"]
        != LANE_MODEL_PATH.stat().st_mtime_ns
    ):
        raise ValueError("world components differ from the published shared chain")
    return selection, population


def split_motion_seed(seed: int) -> tuple[int, int]:
    """Derive diffusion and reactive streams from one path-level seed."""
    sequence = np.random.SeedSequence(int(seed))
    left, right = sequence.spawn(2)
    return int(left.generate_state(1)[0]), int(right.generate_state(1)[0])


@dataclass(frozen=True)
class SampledWorldBatch:
    """Concrete Flow/Diffusion sample plus its response-layer provenance."""

    scenario: Any
    initial_states: np.ndarray
    initial_valid: np.ndarray
    soft_plan: np.ndarray
    scenario_seed: int
    motion_seed: int
    response_seed: int
    exogenous_state: WorldExogenousState | None = None
    vehicle_dimensions: np.ndarray | None = None


class HierarchicalWorldSampler:
    """Compose the three trained probability and simulation layers."""

    def __init__(
        self,
        *,
        flow_checkpoint: str | Path,
        diffusion_checkpoint: str | Path,
        diffusion_contract: str | Path,
        repo_root: str | Path,
        device: str | torch.device = "cpu",
        ddim_steps: int = 20,
        excluded_slots: tuple[str, ...] = EXCLUDED_EVALUATION_SLOTS,
        flow_population: str = "evt_tail",
    ) -> None:
        self.device = torch.device(device)
        world_config = load_yaml(
            Path(repo_root) / "hierarchical_world_model/config.yaml"
        )
        self.generated_road = world_config["generated_road"]
        self.tracking_config = TrackingConfig(**world_config["tracking"])
        self.takeover_config = TakeoverConfig(**world_config["takeover"])
        self.flow, flow_payload = load_flow_checkpoint(
            flow_checkpoint,
            repo_root=repo_root,
            device=self.device,
        )
        if flow_population not in ("evt_tail", "natural"):
            raise ValueError(f"unsupported Flow population: {flow_population}")
        if flow_payload.get("metrics", {}).get("population") != flow_population:
            raise ValueError(
                f"generated worlds require a {flow_population} Flow checkpoint"
            )
        self.flow_schema = self.flow.schema
        self.diffusion, diffusion_payload = load_diffusion_checkpoint(
            diffusion_checkpoint, device=self.device
        )
        self.diffusion_contract = diffusion_payload["dataset_contract"]
        declared = load_json(diffusion_contract)
        if declared != self.diffusion_contract:
            raise ValueError("diffusion checkpoint and declared contract differ")
        self.cfg = WorldModelConfig(**world_config["dynamics"])
        self.selection, self.driver_population = load_selected_components()
        self.flow.eval()
        self.diffusion.eval()
        self.ddim_steps = int(ddim_steps)
        self.excluded_slots = tuple(str(slot) for slot in excluded_slots)
        self.evaluation_scope = (
            evaluation_scope_contract()
            if self.excluded_slots == EXCLUDED_EVALUATION_SLOTS
            else {
                "schema": "explicit_local_scope_v1",
                "excluded_background_slots": list(self.excluded_slots),
                "training_population_modified": False,
            }
        )

    def _scope_condition(
        self, c0: np.ndarray, slot_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remove excluded agents before sampling K or invoking downstream models."""
        slots = np.asarray(
            scoped_slot_mask(slot_mask, excluded_slots=self.excluded_slots), bool
        )
        values = np.asarray(c0, np.float32).copy()
        values[~feature_valid_from_slot_mask(self.flow_schema, slots)] = 0.0
        return values, slots

    @torch.no_grad()
    def sample_world_exogenous(
        self, n: int, *, seed: int, response_steps: int = 149
    ) -> WorldExogenousState:
        """Sample the explicit prior variables required by one complete world."""
        return WorldExogenousState.sample(
            n,
            seed=seed,
            response_steps=response_steps,
        )

    def _compose(
        self,
        scenario: Any,
        *,
        scenario_seed: int,
        motion_seed: int,
        exogenous_state: WorldExogenousState | None = None,
        diffusion_noise: np.ndarray | None = None,
    ) -> SampledWorldBatch:
        prepared = [
            prepare_flow_condition(
                scenario,
                index,
                flow_schema=self.flow_schema,
                diffusion_contract=self.diffusion_contract,
            )
            for index in range(len(scenario.c0))
        ]
        condition = (
            torch.from_numpy(np.stack([item["condition"] for item in prepared]))
            .float()
            .to(self.device)
        )
        target_mask = torch.from_numpy(
            np.stack([item["target_mask"] for item in prepared])
        ).to(self.device)
        diffusion_seed, response_seed = split_motion_seed(motion_seed)
        initial_noise = None
        generator = torch.Generator(device=self.device).manual_seed(diffusion_seed)
        if diffusion_noise is not None and exogenous_state is not None:
            raise ValueError("provide diffusion_noise or exogenous_state, not both")
        if diffusion_noise is not None:
            initial_noise = torch.as_tensor(
                diffusion_noise, dtype=condition.dtype, device=self.device
            )
            if tuple(initial_noise.shape) != tuple(target_mask.shape):
                raise ValueError(
                    f"diffusion_noise must have shape {tuple(target_mask.shape)}"
                )
        if exogenous_state is not None:
            if exogenous_state.batch_size != len(scenario.c0):
                raise ValueError("exogenous state and scenario batch sizes differ")
            exogenous_state.validate()
            initial_noise = torch.from_numpy(exogenous_state.diffusion_noise).to(
                self.device
            )
        normalized = self.diffusion.sample_ddim(
            condition,
            target_mask,
            inference_steps=self.ddim_steps,
            initial_noise=initial_noise,
            generator=generator,
        )
        active = np.asarray(scenario.slot_mask, bool)
        initial = np.stack([item["c0_states"] for item in prepared])
        soft = decode_residual_trajectory(
            normalized.cpu().numpy(),
            np.stack([item["trajectory_reference"] for item in prepared]),
            active,
            self.diffusion_contract,
        )
        valid = np.concatenate((np.ones((len(active), 1), bool), active), axis=1)
        dimensions = np.where(
            valid[..., None],
            np.asarray(scenario.vehicle_dimensions, np.float32),
            np.asarray((4.8, 1.8), np.float32),
        )
        if dimensions.shape != (len(active), 7, 2):
            raise ValueError("vehicle_dimensions must have shape [batch,7,2]")
        if not np.isfinite(dimensions).all() or np.any(dimensions <= 0.0):
            raise ValueError("vehicle_dimensions must be finite and positive")
        return SampledWorldBatch(
            scenario=scenario,
            initial_states=initial,
            initial_valid=valid,
            soft_plan=soft.astype(np.float32),
            scenario_seed=int(scenario_seed),
            motion_seed=int(motion_seed),
            response_seed=response_seed,
            exogenous_state=exogenous_state,
            vehicle_dimensions=dimensions,
        )

    def compose_exogenous(
        self, exogenous_state: WorldExogenousState
    ) -> SampledWorldBatch:
        """Turn an explicit base-randomness state into a concrete traffic world."""
        exogenous_state.validate()
        c0, slot_mask, dimensions = (
            self.flow.sample_initial_conditions_from_base_randomness(
                exogenous_state.scenario_uniform,
                exogenous_state.c0_base_latent,
            )
        )
        c0, slot_mask = self._scope_condition(c0, slot_mask)
        scenario = self.flow.sample_constraints_from_base_randomness(
            c0,
            slot_mask,
            exogenous_state.k_base_latent,
            dimensions,
        )
        return self._compose(
            scenario,
            scenario_seed=0,
            motion_seed=0,
            exogenous_state=exogenous_state,
        )

    def sample_scenarios(
        self, n: int, *, scenario_seed: int, motion_seed: int
    ) -> SampledWorldBatch:
        """Sample unconstrained scenarios and attach frozen response plans."""
        sampled = self.flow.sample_scenarios(int(n), int(scenario_seed))
        c0, slot_mask = self._scope_condition(sampled.c0, sampled.slot_mask)
        scenario = self.flow.sample_constraints(
            c0,
            slot_mask,
            sampled.vehicle_dimensions,
            int(n),
            int(scenario_seed) + 1,
        )
        return self._compose(
            scenario, scenario_seed=scenario_seed, motion_seed=motion_seed
        )

    def sample_constraints(
        self,
        c0: np.ndarray,
        slot_mask: np.ndarray,
        vehicle_dimensions: np.ndarray,
        n: int,
        *,
        scenario_seed: int,
        motion_seed: int,
    ) -> SampledWorldBatch:
        """Sample scenarios conditioned on fixed C0 and slot-presence masks."""
        c0, slot_mask = self._scope_condition(c0, slot_mask)
        scenario = self.flow.sample_constraints(
            c0,
            slot_mask,
            vehicle_dimensions,
            int(n),
            int(scenario_seed),
        )
        return self._compose(
            scenario, scenario_seed=scenario_seed, motion_seed=motion_seed
        )

    def create_world(
        self,
        sample: SampledWorldBatch,
        *,
        idm_config: dict[str, Any] | None = None,
        controller: Any = "auto",
        tracking_config: TrackingConfig | None = None,
        takeover_config: TakeoverConfig | None = None,
    ) -> HighwayEnvClosedLoopWorld:
        """Use the published shared IDM, MOBIL and actual-state continuation."""
        if sample.exogenous_state is None:
            raise ValueError("HighwayEnv execution requires explicit world randomness")
        if isinstance(controller, str):
            if controller != "auto":
                raise ValueError(
                    "controller must be auto, a controller, or None for pure tracking"
                )
            from .stochastic_drivers.online import calibrated_mobil_parameters

            controller = TakeoverController(
                sample_theta(
                    self.driver_population, sample.exogenous_state.driver_latent
                ),
                maneuver_latent=sample.exogenous_state.maneuver_latent,
                takeover_config=takeover_config or self.takeover_config,
                **calibrated_mobil_parameters(self.selection["mobil"]),
            )
        batch = len(sample.initial_states)
        x = np.linspace(-200.0, 200.0, 8, dtype=np.float32)
        lane_offsets = np.asarray(self.generated_road["lane_centers_m"], np.float32)
        lanes = len(lane_offsets)
        maps = np.zeros((batch, 8, 8, 6), np.float32)
        maps[:, :lanes, :, 0] = x[None, None]
        maps[:, :lanes, :, 1] = (
            sample.initial_states[:, None, 0, 1, None] + lane_offsets[None, :, None]
        )
        maps[:, :lanes, :, 2] = 1.0
        maps[:, :lanes, :, 4] = self.generated_road["lane_width_m"]
        map_valid = np.zeros((batch, 8, 8), bool)
        map_valid[:, :lanes] = True
        world = HighwayEnvClosedLoopWorld(
            cfg=self.cfg,
            device=self.device,
            idm_config=idm_config,
            controller=controller,
            tracking_config=tracking_config or self.tracking_config,
        )
        world.reset(
            torch.from_numpy(sample.initial_states),
            torch.from_numpy(sample.initial_valid),
            torch.from_numpy(sample.soft_plan),
            torch.from_numpy(maps),
            torch.from_numpy(map_valid),
            exogenous_state=sample.exogenous_state,
            vehicle_dimensions=sample.vehicle_dimensions,
        )
        return world
