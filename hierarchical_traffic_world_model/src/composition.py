"""End-to-end Flow, diffusion and reactive-world sampling interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diffusion.src.data import prepare_flow_condition, smooth_position_residual
from diffusion.src.train import load_checkpoint as load_diffusion_checkpoint
from normalizing_flow.src.sampling import (
    load_checkpoint_and_dataset,
    sample_constraints,
    sample_scenarios,
)
from world_model.src.core.utils import load_json

from .environment import ClosedLoopWorld
from .train import load_checkpoint as load_response_checkpoint


def split_motion_seed(seed: int) -> tuple[int, int]:
    """Derive diffusion and reactive streams from one path-level seed."""
    sequence = np.random.SeedSequence(int(seed))
    left, right = sequence.spawn(2)
    return int(left.generate_state(1)[0]), int(right.generate_state(1)[0])


@dataclass(frozen=True)
class SampledWorldBatch:
    scenario: Any
    initial_states: np.ndarray
    initial_valid: np.ndarray
    soft_plan: np.ndarray
    state_knot_reference: np.ndarray
    scenario_seed: int
    motion_seed: int
    response_seed: int


class HierarchicalWorldSampler:
    """Compose the three trained probability and simulation layers."""

    def __init__(
        self,
        *,
        flow_checkpoint: str | Path,
        flow_output_dir: str | Path,
        diffusion_checkpoint: str | Path,
        diffusion_contract: str | Path,
        response_checkpoint: str | Path,
        repo_root: str | Path,
        device: str | torch.device = "cpu",
        ddim_steps: int = 20,
    ) -> None:
        self.device = torch.device(device)
        self.flow, _, self.flow_schema, _ = load_checkpoint_and_dataset(
            flow_checkpoint,
            flow_output_dir,
            repo_root=repo_root,
            device=self.device,
        )
        self.diffusion, diffusion_payload = load_diffusion_checkpoint(
            diffusion_checkpoint, device=self.device
        )
        self.diffusion_contract = diffusion_payload["dataset_contract"]
        declared = load_json(diffusion_contract)
        if declared != self.diffusion_contract:
            raise ValueError("diffusion checkpoint and declared contract differ")
        self.response, _ = load_response_checkpoint(
            response_checkpoint, device=self.device
        )
        self.flow.eval()
        self.diffusion.eval()
        self.response.eval()
        self.ddim_steps = int(ddim_steps)

    @torch.no_grad()
    def _compose(
        self,
        scenario: Any,
        *,
        scenario_seed: int,
        motion_seed: int,
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
        generator = torch.Generator(device=self.device).manual_seed(diffusion_seed)
        normalized = self.diffusion.sample_ddim(
            condition,
            target_mask,
            inference_steps=self.ddim_steps,
            generator=generator,
        )
        residual_mean = np.asarray(
            self.diffusion_contract["position_residual"]["mean"], np.float32
        )
        residual_std = np.asarray(
            self.diffusion_contract["position_residual"]["std"], np.float32
        )
        residual = normalized.cpu().numpy().reshape(-1, 149, 6, 2)
        residual = smooth_position_residual(residual * residual_std + residual_mean)
        macro = np.stack([item["trajectory_reference"] for item in prepared])
        active = np.asarray(scenario.slot_mask, bool)
        soft = (macro + residual) * active[:, None, :, None]
        initial = np.stack([item["c0_states"] for item in prepared])
        valid = np.concatenate((np.ones((len(active), 1), bool), active), axis=1)
        return SampledWorldBatch(
            scenario=scenario,
            initial_states=initial,
            initial_valid=valid,
            soft_plan=soft.astype(np.float32),
            state_knot_reference=macro.astype(np.float32),
            scenario_seed=int(scenario_seed),
            motion_seed=int(motion_seed),
            response_seed=response_seed,
        )

    def sample_scenarios(
        self, n: int, *, scenario_seed: int, motion_seed: int
    ) -> SampledWorldBatch:
        scenario = sample_scenarios(self.flow, int(n), scenario_seed)
        return self._compose(
            scenario, scenario_seed=scenario_seed, motion_seed=motion_seed
        )

    def sample_constraints(
        self,
        c0: np.ndarray,
        slot_mask: np.ndarray,
        n: int,
        *,
        scenario_seed: int,
        motion_seed: int,
    ) -> SampledWorldBatch:
        scenario = sample_constraints(
            self.flow,
            c0,
            slot_mask,
            int(n),
            scenario_seed,
        )
        return self._compose(
            scenario, scenario_seed=scenario_seed, motion_seed=motion_seed
        )

    def create_world(self, sample: SampledWorldBatch) -> ClosedLoopWorld:
        batch = len(sample.initial_states)
        x = np.linspace(-200.0, 200.0, 8, dtype=np.float32)
        lane_offsets = np.arange(-3, 5, dtype=np.float32) * 3.6
        maps = np.zeros((batch, 8, 8, 6), np.float32)
        maps[..., 0] = x[None, None]
        maps[..., 1] = (
            sample.initial_states[:, None, 0, 1, None]
            + lane_offsets[None, :, None]
        )
        maps[..., 2] = 1.0
        maps[..., 4] = 3.6
        map_valid = np.ones((batch, 8, 8), bool)
        world = ClosedLoopWorld(self.response, device=self.device)
        world.reset(
            torch.from_numpy(sample.initial_states),
            torch.from_numpy(sample.initial_valid),
            torch.from_numpy(sample.soft_plan),
            torch.from_numpy(maps),
            torch.from_numpy(map_valid),
            motion_seed=sample.response_seed,
        )
        return world
