"""Stepwise 25 Hz bridge for the externally installed official POMDP.

The official repository is never copied into this project.  Instead this
module imports a pinned v1.0.0 checkout at runtime and reuses its released
initialization plus ``POMDPAgent.choose_action`` boundary.  One wrapper step
is one native 0.2 s decision; callers hold that request for five 25 Hz plant
ticks.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import types
from typing import Any

import numpy as np


OFFICIAL_COMMIT = "56655de845644c45f01ab2898544e55316a3279b"
NATIVE_DT_S = .2
PLANT_DT_S = .04
NATIVE_TICKS_PER_DECISION = 5


@dataclass(frozen=True)
class OfficialDecision:
    """The actual action and diagnostic state emitted by the official agent."""

    acceleration_mps2: np.ndarray
    steering_rate_rps: np.ndarray
    belief: np.ndarray
    weights: np.ndarray
    evidence: np.ndarray
    selected_return: np.ndarray


def _parameters() -> dict[str, object]:
    """Literal full-model parameters used by the released rear-end script."""
    return {
        "a_sd_model": 3.0, "Loom_perc": True, "d_phi_thres": .00215,
        "Loom_change_obs": -1, "perc_noise_factor": .01,
        "noise_pred_fac": .2, "num_plan": 100, "a_sd_plan": 5.0,
        "sample_steering_rate": True, "use_pedals": True, "H": 30,
        "plan_ignore_w": True, "plan_smooth_delta": True,
        "pref_v_sd": .5, "pref_a_sd": .1, "pref_w_sd": .02,
        "lane_cost": -15000, "lane_change_cost": -1000, "coll_cost": -10000,
        "road_pref": 0, "Loom_reward": "V7", "weigh_particles": .001,
        "full_violation_factor": .01, "unpunished_heading": 85,
        "collision_cost_adjusted": True, "N_norm": 32, "H_norm": 20,
        "alpha": 1.0, "EA_mode": "Surprise", "EA_fac": -5.95,
        "EA_init": False,
    }


def _state(*, gap_m: float, ego_speed_mps: float, target_speed_mps: float) -> dict[str, float]:
    return {
        "lane_width": 3.65, "d": 1.72, "lf": 2.1, "lr": 2.1,
        "a_max": 8.0, "w_max": 1.22, "x_ego": 0.0, "y_ego": 0.0,
        "theta_ego": 0.0, "delta_ego": 0.0, "v_ego": ego_speed_mps,
        "x_tar": gap_m + 4.2, "y_tar": 0.0, "theta_tar": 0.0,
        "delta_tar": 0.0, "v_tar": target_speed_mps,
    }


def _numpy(value: Any) -> np.ndarray:
    """Copy either a torch diagnostic or the scalar used before evidence starts."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


class OfficialPOMDP25Hz:
    """Pinned official POMDP, stepped by external current-only observations."""

    def __init__(self, *, source_dir: str | Path, following_dir: str | Path,
                 device: str = "cuda", batch_size: int = 1) -> None:
        self.source_dir = Path(source_dir).resolve()
        self.following_dir = Path(following_dir).resolve()
        self.device_name = device
        self.batch_size = batch_size
        self._agent: Any | None = None
        self._rng_clock_env: Any | None = None
        self._torch: Any | None = None
        self._decision_count = 0

    def _load_api(self) -> tuple[Any, Any, Any, Any]:
        commit = subprocess.check_output(
            ["git", "-C", str(self.source_dir), "rev-parse", "HEAD"], text=True
        ).strip()
        if commit != OFFICIAL_COMMIT:
            raise RuntimeError(f"official source must be {OFFICIAL_COMMIT}, found {commit}")
        if not self.following_dir.is_dir():
            raise RuntimeError(f"missing Results_following: {self.following_dir}")
        if str(self.source_dir) not in sys.path:
            sys.path.insert(0, str(self.source_dir))
        # The released module imports its analysis scripts at import time.
        # Suppress only those side-effect imports; all model classes remain
        # untouched and are imported from the official checkout.
        sys.modules.setdefault("Analysis_rear_end", types.ModuleType("Analysis_rear_end"))
        sys.modules.setdefault("visualization_rear_end", types.ModuleType("visualization_rear_end"))
        import torch
        import simulation_rear_end
        from src.utils import simulation
        return torch, simulation_rear_end, simulation, simulation_rear_end.set_config

    def reset(self, *, gap_m: float, ego_speed_mps: float, target_speed_mps: float,
              seed: int = 0) -> None:
        """Initialize the official agent for an external vehicle state."""
        torch, rear_end, simulation, set_config = self._load_api()
        if self.device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("official CUDA wrapper requested but CUDA is unavailable")
        device = torch.device("cuda", 0) if self.device_name == "cuda" else torch.device("cpu")
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.cuda.empty_cache()
        parameters = _parameters()
        state = _state(gap_m=gap_m, ego_speed_mps=ego_speed_mps, target_speed_mps=target_speed_mps)
        old_cwd = Path.cwd()
        try:
            # ``find_parameters`` is a released lookup helper with a relative
            # Results_following dependency.  It is used unchanged.
            os.chdir(self.following_dir.parent)
            thw = state["x_tar"] / max(target_speed_mps, 1e-6)
            v_diff, target_minimum_acceleration = rear_end.find_parameters(
                target_speed_mps, parameters["EA_fac"], parameters["noise_pred_fac"],
                parameters["H"], parameters["d_phi_thres"], thw,
            )
        finally:
            os.chdir(old_cwd)
        parameters["v_diff"] = v_diff
        parameters["a_tar_min_intensity"] = -target_minimum_acceleration / state["a_max"]
        _, config = set_config(state, parameters, -6.0)
        config["rollout_batch_size"] = self.batch_size
        config["T"] = 60

        captured: dict[str, Any] = {}
        original_run = rear_end.run_simulation

        def capture(_: dict[str, Any], agent: Any, env: Any, eta: Any, b: Any, w: Any) -> dict[str, Any]:
            captured.update(agent=agent, env=env, eta=eta, b=b, w=w)
            return {}

        rear_end.run_simulation = capture
        try:
            rear_end.simulate(config, device)
        finally:
            rear_end.run_simulation = original_run
        if not captured:
            raise RuntimeError("official initialization did not reach run_simulation")
        captured["agent"].reset(captured["b"], captured["w"])
        # The released single-stream implementation advances PyTorch's global
        # stochastic stream when its environment integrates a just-selected
        # action, before the next particle update.  Keep that *random-stream
        # clock* in a discarded official environment so external plant state
        # and observations never leak into the agent, yet the published
        # stochastic driver remains bitwise replayable at the POMDP boundary.
        captured["env"].reset(captured["eta"])
        self._agent, self._torch = captured["agent"], torch
        self._rng_clock_env = captured["env"]
        self._decision_count = 0

    @staticmethod
    def observation(*, ego_x_m: float, ego_speed_mps: float, target_x_m: float,
                    target_speed_mps: float, target_acceleration_mps2: float = 0.0,
                    ego_y_m: float = 0.0, target_y_m: float = 0.0,
                    ego_heading_rad: float = 0.0, target_heading_rad: float = 0.0,
                    ego_steering_rad: float = 0.0, target_steering_rad: float = 0.0,
                    target_steering_rate_rps: float = 0.0) -> np.ndarray:
        """Build the official rear-end decoder observation from current state only."""
        return np.asarray((ego_x_m, ego_y_m, ego_heading_rad, ego_steering_rad, ego_speed_mps,
                           target_x_m, target_y_m, target_heading_rad, target_steering_rad,
                           target_speed_mps, target_acceleration_mps2,
                           target_steering_rate_rps), dtype=np.float32)

    def step(self, observation: np.ndarray) -> OfficialDecision:
        """Consume one current observation and return one 0.2 s official action."""
        if self._agent is None or self._torch is None:
            raise RuntimeError("call reset before step")
        value = np.asarray(observation, dtype=np.float32)
        if value.ndim == 1:
            value = np.broadcast_to(value, (self.batch_size, value.shape[0])).copy()
        if value.shape != (self.batch_size, 12):
            raise ValueError(f"expected ({self.batch_size}, 12) observation, got {value.shape}")
        torch = self._torch
        with torch.no_grad():
            a_disc, a_cont = self._agent.choose_action(torch.as_tensor(value, device=self._agent.planner.device))
        acceleration = a_cont[0, :, 0].detach().cpu().numpy().copy()
        steering_rate = a_cont[0, :, 1].detach().cpu().numpy().copy()
        decision = OfficialDecision(
            acceleration_mps2=acceleration,
            steering_rate_rps=steering_rate,
            belief=_numpy(self._agent.encoder.b),
            weights=_numpy(self._agent.encoder.w),
            # Official planner initializes evidence as a Python scalar until
            # its first reference-plan comparison, so preserve that released
            # behavior rather than forcing an artificial tensor state.
            evidence=_numpy(self._agent.planner.evidence),
            selected_return=_numpy(self._agent.planner.returns_optimized),
        )
        # Deliberately discard this native simulator output: the next caller
        # observation still comes only from the external 25 Hz plant.  See
        # reset() for why this call is needed to preserve the official RNG
        # interleaving between decisions.
        self._rng_clock_env.step(a_disc[0], a_cont[0])
        self._decision_count += 1
        return decision
