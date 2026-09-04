"""Globally calibrated highD IDM and diagnostic MOBIL references.

The rule reference is deliberately a *single* population model. Earlier
experiments split following samples into arbitrary headway ``styles`` and
therefore did not constitute an IDM calibration. This module follows the
coarse-calibration role used in the naturalistic-prior literature: fit one
causal IDM from all eligible highD training observations, validate it on a
held-out split, and use it only as an optional same-rear reference. MOBIL is
independently calibrated and reported, but cannot change yaw-rate here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from world_model.src.core.utils import load_json, save_json


IDM_PARAMETER_NAMES = (
    "desired_speed_mps", "max_acceleration_mps2", "comfortable_brake_mps2",
    "minimum_gap_m", "desired_headway_s", "delta",
)


@dataclass(frozen=True)
class RuleModelBundle:
    """Serializable globally calibrated IDM and diagnostic MOBIL parameters."""

    idm_parameters: tuple[float, float, float, float, float, float]
    mobil_politeness: float
    mobil_incentive_threshold_mps2: float
    mobil_safe_brake_mps2: float
    calibration_method: str = "full-training-observation-idm"
    schema: str = "highd_global_rule_models_v2"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RuleModelBundle":
        if value.get("schema") != "highd_global_rule_models_v2":
            raise ValueError("unsupported rule-model artifact; recalibrate the single global highD IDM/MOBIL model")
        return cls(
            idm_parameters=tuple(float(item) for item in value["idm_parameters"]),
            mobil_politeness=float(value["mobil_politeness"]),
            mobil_incentive_threshold_mps2=float(value["mobil_incentive_threshold_mps2"]),
            mobil_safe_brake_mps2=float(value["mobil_safe_brake_mps2"]),
            calibration_method=str(value.get("calibration_method", "full-training-observation-idm")),
        )

    def save(self, path: str | Path) -> None:
        save_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> "RuleModelBundle":
        return cls.from_dict(load_json(path))

    def style_posterior(self, history: torch.Tensor, *, target_slot_index: int = 1) -> torch.Tensor:
        """Compatibility diagnostic: the global model has one style with mass one."""
        del target_slot_index
        return history.new_ones((len(history), 1))

    def idm_reference(
        self, history: torch.Tensor, current: torch.Tensor, current_valid: torch.Tensor,
        *, target_slot_index: int = 1, min_acceleration: float = -8.0, max_acceleration: float = 4.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a causal IDM action for the designated same-rear follower."""
        del history
        leader, follower = current[:, 0], current[:, target_slot_index + 1]
        desired_speed, a_max, comfort, s0, headway, delta = current.new_tensor(self.idm_parameters).unbind()
        gap = (leader[:, 0] - follower[:, 0] - 4.8).clamp_min(.1)
        follower_speed = torch.linalg.vector_norm(follower[:, 2:4], dim=-1).clamp_min(0.)
        leader_speed = torch.linalg.vector_norm(leader[:, 2:4], dim=-1).clamp_min(0.)
        closing = follower_speed - leader_speed
        desired_gap = s0 + torch.relu(follower_speed * headway + follower_speed * closing / (2.0 * torch.sqrt((a_max * comfort).clamp_min(1.e-5))))
        acceleration = a_max * (1.0 - torch.pow(follower_speed / desired_speed.clamp_min(1.), delta) - torch.square(desired_gap / gap))
        same_lane = (leader[:, 1] - follower[:, 1]).abs() < 1.8
        valid = current_valid[:, 0] & current_valid[:, target_slot_index + 1] & same_lane & (leader[:, 0] > follower[:, 0])
        action = acceleration.clamp(float(min_acceleration), float(max_acceleration))
        return torch.where(valid, action, torch.zeros_like(action)), self.style_posterior(current[:, None])


def _idm_numpy(parameters: np.ndarray, gap: np.ndarray, follower_speed: np.ndarray, leader_speed: np.ndarray) -> np.ndarray:
    desired_speed, a_max, comfort, s0, headway, delta = parameters
    closing = follower_speed - leader_speed
    desired_gap = s0 + np.maximum(follower_speed * headway + follower_speed * closing / (2.0 * np.sqrt(max(a_max * comfort, 1.e-5))), 0.)
    return a_max * (1.0 - np.power(np.maximum(follower_speed, 0.) / max(desired_speed, 1.), delta) - np.square(desired_gap / np.maximum(gap, .1)))


def _physical_parameters(raw: torch.Tensor) -> torch.Tensor:
    low = raw.new_tensor((8., .3, .5, .2, .2, 2.))
    high = raw.new_tensor((45., 5., 7., 8., 3.5, 6.))
    return low + (high - low) * torch.sigmoid(raw)


def _collect_following_observations(arrays: dict[str, np.ndarray], rows: np.ndarray) -> dict[str, np.ndarray]:
    """Extract every valid same-rear following observation from supplied rows."""
    states_all, valid_all = np.asarray(arrays["agent_states"]), np.asarray(arrays["agent_valid"])
    buckets = {key: [] for key in ("gap", "follower_speed", "leader_speed", "observed_ax", "lane_shift")}
    lane_observations = 0
    for row in np.asarray(rows, np.int64):
        states, valid = states_all[int(row)], valid_all[int(row)]
        leader, follower = states[:-1, 0], states[:-1, 2]
        present = valid[:-1, 0] & valid[:-1, 2] & valid[1:, 2]
        gap = leader[:, 0] - follower[:, 0] - 4.8
        same_lane = np.abs(leader[:, 1] - follower[:, 1]) < 1.8
        follower_speed = np.linalg.norm(follower[:, 2:4], axis=-1)
        leader_speed = np.linalg.norm(leader[:, 2:4], axis=-1)
        mask = present & same_lane & (leader[:, 0] > follower[:, 0]) & (gap > .1) & (follower_speed > .5)
        if np.any(mask):
            next_speed = np.linalg.norm(states[1:, 2, 2:4], axis=-1)
            action = np.clip((next_speed - follower_speed) / .04, -8., 4.)
            buckets["gap"].append(gap[mask].astype(np.float32))
            buckets["follower_speed"].append(follower_speed[mask].astype(np.float32))
            buckets["leader_speed"].append(leader_speed[mask].astype(np.float32))
            buckets["observed_ax"].append(action[mask].astype(np.float32))
        horizon = 80
        if len(states) > horizon:
            observed = valid[:-horizon, 1:] & valid[horizon:, 1:]
            shift = np.abs(states[horizon:, 1:, 1] - states[:-horizon, 1:, 1])
            buckets["lane_shift"].append(shift[observed].astype(np.float32))
            lane_observations += int(observed.sum())
    if not buckets["gap"]:
        raise RuntimeError("insufficient same-rear highD following observations for IDM calibration")
    result = {key: np.concatenate(value) if value else np.empty(0, np.float32) for key, value in buckets.items()}
    result["lane_observations"] = np.asarray((lane_observations,), np.int64)
    return result


def fit_rule_models(
    arrays: dict[str, np.ndarray], rows: np.ndarray, *, epochs: int = 24, batch_size: int = 32768, seed: int = 20260902,
) -> tuple[RuleModelBundle, dict[str, Any]]:
    """Fit one IDM on all supplied highD training observations.

    Every eligible observation is visited in each epoch. The objective is a
    robust, gap-weighted longitudinal acceleration residual; validation also
    reports separate held-out action and trajectory diagnostics.
    """
    sample = _collect_following_observations(arrays, rows)
    gap, vf, vl, observed = (sample[key] for key in ("gap", "follower_speed", "leader_speed", "observed_ax"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.nn.Parameter(torch.tensor((1.0, -.2, .2, -1.0, -.5, .7), device=device))
    optimizer = torch.optim.Adam((raw,), lr=.08)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    data = tuple(torch.from_numpy(value) for value in (gap, vf, vl, observed))
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        order = torch.randperm(len(gap), generator=generator)
        losses: list[float] = []
        for start in range(0, len(order), int(batch_size)):
            index = order[start:start + int(batch_size)]
            g, speed, leader_speed, target = (value[index].to(device, non_blocking=True) for value in data)
            desired_speed, a_max, comfort, s0, headway, delta = _physical_parameters(raw)
            closing = speed - leader_speed
            desired_gap = s0 + torch.relu(speed * headway + speed * closing / (2. * torch.sqrt((a_max * comfort).clamp_min(1.e-5))))
            prediction = a_max * (1. - torch.pow(speed / desired_speed, delta) - torch.square(desired_gap / g.clamp_min(.1)))
            weight = 1. + .75 * (3. / g.clamp_min(3.)).square()
            loss = (weight * torch.nn.functional.smooth_l1_loss(prediction.clamp(-8., 4.), target, reduction="none", beta=.75)).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": float(epoch + 1), "robust_acceleration_loss": float(np.mean(losses))})
    parameters = tuple(float(value) for value in _physical_parameters(raw).detach().cpu())
    lane_shift = sample["lane_shift"]
    lane_change_rate = float(np.mean(lane_shift > 1.2)) if len(lane_shift) else 0.
    bundle = RuleModelBundle(
        idm_parameters=parameters,
        mobil_politeness=float(np.clip(.15 + 2. * lane_change_rate, .1, .6)),
        mobil_incentive_threshold_mps2=float(np.clip(np.quantile(np.abs(observed), .35), .1, 1.5)),
        mobil_safe_brake_mps2=float(np.clip(-np.quantile(observed, .05), 1., 6.)),
    )
    prediction = _idm_numpy(np.asarray(parameters), gap, vf, vl).clip(-8., 4.)
    error = prediction - observed
    return bundle, {
        "schema": "highd_global_rule_calibration_report_v2", "training_rows": int(len(rows)),
        "following_observations": int(len(gap)), "optimizer": "Adam; robust gap-weighted full-observation IDM objective",
        "epochs": int(epochs), "batch_size": int(batch_size), "history": history,
        "training_acceleration_mae_mps2": float(np.abs(error).mean()),
        "training_acceleration_rmse_mps2": float(np.sqrt(np.mean(error ** 2))),
        "training_correlation": float(np.corrcoef(prediction, observed)[0, 1]) if np.std(prediction) > 1.e-8 else 0.,
        "idm_parameter_names": IDM_PARAMETER_NAMES, "mobil_lane_observations": int(sample["lane_observations"][0]),
        "mobil_committed_lane_change_rate": lane_change_rate,
        "mobil_scope": "single global diagnostic calibration; no yaw-rate or lane-change action in the longitudinal controller",
    }
