"""Shared traffic events, recorded highD sources and frozen NPC execution."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hierarchical_world_model.src.config import WorldModelConfig
from hierarchical_world_model.src.core.utils import load_yaml
from hierarchical_world_model.src.highway import HighwayEnvClosedLoopWorld
from hierarchical_world_model.src.randomness import WorldExogenousState
from hierarchical_world_model.src.scene_geometry import SceneGeometry
from hierarchical_world_model.src.shared_evaluation import load_shared_components
from hierarchical_world_model.src.stochastic_drivers.idm_population import sample_theta
from hierarchical_world_model.src.stochastic_drivers.online import (
    calibrated_mobil_parameters,
)
from hierarchical_world_model.src.takeover import TakeoverConfig, TakeoverController
from hierarchical_world_model.src.tracking import TrackingConfig

ANCHOR_INDEX = 24
DEFAULT_WORLD_CONFIG = (
    Path(__file__).resolve().parents[2] / "hierarchical_world_model/config.yaml"
)


@dataclass(frozen=True)
class WorldEvent:
    event_id: str
    source_event_id: str
    initial_states: np.ndarray
    valid: np.ndarray
    soft_reference: np.ndarray
    map_polylines: np.ndarray
    map_polyline_valid: np.ndarray
    exogenous_state: WorldExogenousState
    vehicle_dimensions: np.ndarray | None = None
    split: str = "train"
    initial_history: np.ndarray | None = None
    initial_history_valid: np.ndarray | None = None

    @property
    def geometry(self) -> SceneGeometry:
        return SceneGeometry.from_map_polylines(
            self.map_polylines, self.map_polyline_valid
        )


class HighDEventStore(Sequence[WorldEvent]):
    def __init__(
        self,
        event_csv: str | Path,
        sequence_cache: str | Path,
        *,
        world_seed: int = 0,
    ) -> None:
        frame = pd.read_csv(event_csv)
        required = {"event_id", "source_event_id", "split"}
        if missing := required - set(frame.columns):
            raise KeyError(f"event manifest is missing {sorted(missing)}")
        self.frame = frame.reset_index(drop=True)
        root = Path(sequence_cache)
        self.sequence_ids = np.load(
            root / "sequence_id.npy", mmap_mode="r", allow_pickle=False
        )
        self.states = np.load(
            root / "agent_states.npy", mmap_mode="r", allow_pickle=False
        )
        self.valid = np.load(
            root / "agent_valid.npy", mmap_mode="r", allow_pickle=False
        )
        self.maps = np.load(
            root / "map_polylines.npy", mmap_mode="r", allow_pickle=False
        )
        self.map_valid = np.load(
            root / "map_polyline_valid.npy", mmap_mode="r", allow_pickle=False
        )
        dimensions_path = root / "vehicle_dimensions.npy"
        self.vehicle_dimensions = (
            np.load(dimensions_path, mmap_mode="r", allow_pickle=False)
            if dimensions_path.exists()
            else None
        )
        if self.vehicle_dimensions is not None:
            if self.vehicle_dimensions.shape != (len(self.sequence_ids), 7, 2):
                raise ValueError(
                    "vehicle_dimensions cache must have shape [sequences,7,2]"
                )
            if not np.isfinite(self.vehicle_dimensions).all() or np.any(
                self.vehicle_dimensions <= 0.0
            ):
                raise ValueError("vehicle_dimensions cache must be finite and positive")
        self.row_by_id = {
            str(value): index for index, value in enumerate(self.sequence_ids)
        }
        self.world_seed = int(world_seed)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> WorldEvent:
        item = self.frame.iloc[int(index)]
        event_id = str(item["event_id"])
        base_event_id = str(item.get("base_event_id", event_id))
        if base_event_id not in self.row_by_id:
            raise KeyError(
                f"base event {base_event_id!r} is absent from the sequence cache"
            )
        row = self.row_by_id[base_event_id]
        material = f"ads_event:{self.world_seed}:{event_id}".encode()
        seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
        exogenous_path = item.get("exogenous_path")
        exogenous = (
            WorldExogenousState.load(str(exogenous_path))
            if exogenous_path is not None
            and not pd.isna(exogenous_path)
            and str(exogenous_path)
            else WorldExogenousState.sample(
                1,
                seed=seed,
                response_steps=149,
            )
        )
        return WorldEvent(
            event_id=event_id,
            source_event_id=str(item["source_event_id"]),
            initial_states=np.asarray(
                self.states[row, ANCHOR_INDEX], np.float32
            ).copy(),
            valid=np.asarray(self.valid[row, ANCHOR_INDEX], bool).copy(),
            soft_reference=np.asarray(
                self.states[row, ANCHOR_INDEX + 1 :, 1:, :2], np.float32
            ).copy(),
            map_polylines=np.asarray(self.maps[row], np.float32).copy(),
            map_polyline_valid=np.asarray(self.map_valid[row], bool).copy(),
            exogenous_state=exogenous,
            vehicle_dimensions=(
                None
                if self.vehicle_dimensions is None
                else np.asarray(self.vehicle_dimensions[row], np.float32).copy()
            ),
            split=str(item["split"]),
            initial_history=np.asarray(
                self.states[row, : ANCHOR_INDEX + 1], np.float32
            ).copy(),
            initial_history_valid=np.asarray(
                self.valid[row, : ANCHOR_INDEX + 1], bool
            ).copy(),
        )


class FrozenWorldFactory:
    def __init__(
        self,
        config_path=DEFAULT_WORLD_CONFIG,
        *,
        device="cpu",
        responsive_npcs=True,
        behavior_randomization="W2",
        controller_kwargs=None,
        tracking_config=None,
        takeover_config=None,
    ):
        self.device = torch.device(device)
        self.responsive_npcs = bool(responsive_npcs)
        self.behavior_randomization = str(behavior_randomization)
        if self.behavior_randomization not in {"W0", "W1", "W2"}:
            raise ValueError("behavior_randomization must be W0, W1, or W2")
        settings = load_yaml(config_path)
        self.cfg = WorldModelConfig(**settings["dynamics"])
        self.population, fit = load_shared_components()
        self.controller_kwargs = {
            **calibrated_mobil_parameters(fit["mobil"]),
            **(controller_kwargs or {}),
        }
        self.tracking_config = tracking_config or TrackingConfig(**settings["tracking"])
        self.takeover_config = takeover_config or TakeoverConfig(**settings["takeover"])

    def __call__(self, event):
        controller = None
        if self.responsive_npcs:
            latent = event.exogenous_state.driver_latent
            theta = sample_theta(
                self.population,
                (
                    latent
                    if self.behavior_randomization != "W0"
                    else np.zeros_like(latent)
                ),
            )
            controller = TakeoverController(
                theta,
                maneuver_latent=(
                    event.exogenous_state.maneuver_latent
                    if self.behavior_randomization == "W2"
                    else None
                ),
                takeover_config=self.takeover_config,
                **self.controller_kwargs,
            )
        return HighwayEnvClosedLoopWorld(
            cfg=self.cfg,
            device=self.device,
            controller=controller,
            tracking_config=self.tracking_config,
        )
