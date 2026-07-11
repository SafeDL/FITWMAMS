"""Future-action feature builders for the naturalness discriminator."""
from __future__ import annotations

from typing import Sequence

import numpy as np


FUTURE_FEATURE_KEYS: tuple[str, ...] = (
    "future_jx",
    "future_ax",
    "future_vx",
    "future_dx",
    "future_gap_proxy",
)

SUMMARY_FEATURE_KEYS: tuple[str, ...] = (
    "future_ax_min",
    "future_ax_max",
    "future_ax_mean",
    "future_ax_std",
    "future_jerk_abs_mean",
    "future_jerk_abs_max",
    "future_vx_min",
    "future_vx_final",
    "future_displacement",
    "future_gap_min_proxy",
    "future_gap_reduction_proxy",
    "action_clip_violation",
    "speed_negative_indicator",
    "jerk_violation_indicator",
)


def _action_cfg(config: dict) -> dict:
    return config.get("action", {})


def _future_cfg(config: dict) -> dict:
    return config.get("future_features", {})


def _representation(schema: dict, config: dict) -> str:
    return str(schema.get("action_representation", _action_cfg(config).get("representation", "jerk"))).lower()


def _dt(schema: dict, config: dict) -> float:
    return float(schema.get("dt", config.get("sampling", {}).get("dt", 0.04)))


def _feature_keys(config: dict) -> list[str]:
    cfg = _future_cfg(config)
    keys: list[str] = []
    if bool(cfg.get("include_action", True)):
        keys.append("future_jx")
    if bool(cfg.get("include_acceleration", True)):
        keys.append("future_ax")
    if bool(cfg.get("include_velocity", True)):
        keys.append("future_vx")
    if bool(cfg.get("include_displacement", True)):
        keys.append("future_dx")
    if bool(cfg.get("include_gap_proxy", True)):
        keys.append("future_gap_proxy")
    return keys


def selected_future_feature_keys(config: dict) -> tuple[str, ...]:
    return tuple(_feature_keys(config))


def _numpy_action_kinematics(
    actions: np.ndarray,
    context_states: np.ndarray,
    ego_length: np.ndarray,
    adv_length: np.ndarray,
    schema: dict,
    config: dict,
) -> dict[str, np.ndarray]:
    actions = np.asarray(actions, dtype=np.float32)
    context_states = np.asarray(context_states, dtype=np.float32)
    if actions.ndim != 3 or actions.shape[-1] < 1:
        raise ValueError(f"Expected actions shape [B,H,1+], got {actions.shape}")
    dt = _dt(schema, config)
    rep = _representation(schema, config)
    action_cfg = _action_cfg(config)
    ax_min = float(action_cfg.get("ax_min", -8.0))
    ax_max = float(action_cfg.get("ax_max", 4.0))
    jerk_abs_max = float(action_cfg.get("jerk_abs_max", 12.0))

    lead0 = context_states[:, -1, 1]
    ego0 = context_states[:, -1, 0]
    prev_ax = lead0[:, 4]
    if rep == "jerk":
        jx = actions[:, :, 0]
        ax = prev_ax[:, None] + np.cumsum(jx, axis=1) * dt
    elif rep == "acceleration":
        ax = actions[:, :, 0]
        prev = np.concatenate([prev_ax[:, None], ax[:, :-1]], axis=1)
        jx = (ax - prev) / max(dt, 1e-6)
    else:
        raise ValueError(f"Unsupported action representation: {rep}")

    v0 = np.maximum(lead0[:, 2], 0.0)
    vx = v0[:, None] + np.cumsum(ax, axis=1) * dt
    v_before = np.concatenate([v0[:, None], vx[:, :-1]], axis=1)
    dx = np.cumsum(v_before * dt + 0.5 * ax * dt * dt, axis=1)
    half_lengths = 0.5 * (np.asarray(ego_length, dtype=np.float32) + np.asarray(adv_length, dtype=np.float32))
    gap0 = lead0[:, 0] - ego0[:, 0] - half_lengths
    ego_dx = ego0[:, 2:3] * (np.arange(1, actions.shape[1] + 1, dtype=np.float32)[None, :] * dt)
    gap_proxy = gap0[:, None] + dx - ego_dx
    return {
        "future_jx": jx.astype(np.float32),
        "future_ax": ax.astype(np.float32),
        "future_vx": vx.astype(np.float32),
        "future_dx": dx.astype(np.float32),
        "future_gap_proxy": gap_proxy.astype(np.float32),
        "action_clip_violation": ((ax < ax_min) | (ax > ax_max)).mean(axis=1).astype(np.float32),
        "speed_negative_indicator": (vx < 0.0).mean(axis=1).astype(np.float32),
        "jerk_violation_indicator": (np.abs(jx) > jerk_abs_max).mean(axis=1).astype(np.float32),
    }


def _numpy_summary(parts: dict[str, np.ndarray]) -> np.ndarray:
    ax = parts["future_ax"]
    jx = parts["future_jx"]
    vx = parts["future_vx"]
    dx = parts["future_dx"]
    gap = parts["future_gap_proxy"]
    summary = np.stack(
        [
            np.min(ax, axis=1),
            np.max(ax, axis=1),
            np.mean(ax, axis=1),
            np.std(ax, axis=1),
            np.mean(np.abs(jx), axis=1),
            np.max(np.abs(jx), axis=1),
            np.min(vx, axis=1),
            vx[:, -1],
            dx[:, -1],
            np.min(gap, axis=1),
            gap[:, 0] - gap[:, -1],
            parts["action_clip_violation"],
            parts["speed_negative_indicator"],
            parts["jerk_violation_indicator"],
        ],
        axis=1,
    )
    return np.nan_to_num(summary.astype(np.float32), nan=0.0, posinf=1e6, neginf=-1e6)


def build_future_features_numpy(
    actions: np.ndarray,
    context_states: np.ndarray,
    relative_history: np.ndarray | None,
    ego_length: np.ndarray,
    adv_length: np.ndarray,
    schema: dict,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Build discriminator future sequence and summary features with numpy.

    ``relative_history`` is accepted for API symmetry with the torch builder. The
    current implementation computes the gap proxy from raw context states and
    vehicle lengths.
    """
    del relative_history
    parts = _numpy_action_kinematics(actions, context_states, ego_length, adv_length, schema, config)
    keys = _feature_keys(config)
    future = np.stack([parts[key] for key in keys], axis=-1)
    return future.astype(np.float32), _numpy_summary(parts)


def normalize_numpy(x: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    return ((x - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)).astype(np.float32)


def denormalize_torch(x, mean: Sequence[float], std: Sequence[float]):
    import torch

    mean_t = torch.as_tensor(mean, dtype=x.dtype, device=x.device)
    std_t = torch.as_tensor(std, dtype=x.dtype, device=x.device)
    return x * std_t + mean_t


def normalize_torch(x, mean: Sequence[float], std: Sequence[float]):
    import torch

    mean_t = torch.as_tensor(mean, dtype=x.dtype, device=x.device)
    std_t = torch.as_tensor(std, dtype=x.dtype, device=x.device)
    return (x - mean_t) / std_t


def build_future_features_torch(
    future_actions,
    context_states,
    relative_history,
    ego_length,
    adv_length,
    schema: dict,
    config: dict,
) -> tuple[object, object]:
    """Torch equivalent of :func:`build_future_features_numpy`.

    This function intentionally uses only torch operations so Stage 3 can
    backpropagate naturalness guidance into ``future_actions``.
    """
    del relative_history
    import torch

    if future_actions.ndim != 3 or future_actions.shape[-1] < 1:
        raise ValueError(f"Expected future_actions shape [B,H,1+], got {tuple(future_actions.shape)}")
    dt = _dt(schema, config)
    rep = _representation(schema, config)
    action_cfg = _action_cfg(config)
    ax_min = float(action_cfg.get("ax_min", -8.0))
    ax_max = float(action_cfg.get("ax_max", 4.0))
    jerk_abs_max = float(action_cfg.get("jerk_abs_max", 12.0))

    lead0 = context_states[:, -1, 1]
    ego0 = context_states[:, -1, 0]
    prev_ax = lead0[:, 4]
    if rep == "jerk":
        jx = future_actions[:, :, 0]
        ax = prev_ax[:, None] + torch.cumsum(jx, dim=1) * dt
    elif rep == "acceleration":
        ax = future_actions[:, :, 0]
        prev = torch.cat([prev_ax[:, None], ax[:, :-1]], dim=1)
        jx = (ax - prev) / max(dt, 1e-6)
    else:
        raise ValueError(f"Unsupported action representation: {rep}")

    v0 = torch.clamp(lead0[:, 2], min=0.0)
    vx = v0[:, None] + torch.cumsum(ax, dim=1) * dt
    v_before = torch.cat([v0[:, None], vx[:, :-1]], dim=1)
    dx = torch.cumsum(v_before * dt + 0.5 * ax * dt * dt, dim=1)
    half_lengths = 0.5 * (ego_length.to(dtype=future_actions.dtype) + adv_length.to(dtype=future_actions.dtype))
    gap0 = lead0[:, 0] - ego0[:, 0] - half_lengths
    steps = torch.arange(1, future_actions.shape[1] + 1, dtype=future_actions.dtype, device=future_actions.device)
    ego_dx = ego0[:, 2:3] * (steps[None, :] * dt)
    gap_proxy = gap0[:, None] + dx - ego_dx

    parts = {
        "future_jx": jx,
        "future_ax": ax,
        "future_vx": vx,
        "future_dx": dx,
        "future_gap_proxy": gap_proxy,
    }
    keys = _feature_keys(config)
    future = torch.stack([parts[key] for key in keys], dim=-1)
    summary = torch.stack(
        [
            torch.min(ax, dim=1).values,
            torch.max(ax, dim=1).values,
            torch.mean(ax, dim=1),
            torch.std(ax, dim=1, unbiased=False),
            torch.mean(torch.abs(jx), dim=1),
            torch.max(torch.abs(jx), dim=1).values,
            torch.min(vx, dim=1).values,
            vx[:, -1],
            dx[:, -1],
            torch.min(gap_proxy, dim=1).values,
            gap_proxy[:, 0] - gap_proxy[:, -1],
            ((ax < ax_min) | (ax > ax_max)).to(future_actions.dtype).mean(dim=1),
            (vx < 0.0).to(future_actions.dtype).mean(dim=1),
            (torch.abs(jx) > jerk_abs_max).to(future_actions.dtype).mean(dim=1),
        ],
        dim=1,
    )
    return future, summary
