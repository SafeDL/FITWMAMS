#!/usr/bin/env python3
"""Compare highD risk with HighwayEnv human and IDM world replays."""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from IDM_subset.src.idm_policy import HighwayEnvIDMPolicy  # noqa: E402
from hierarchical_world_model.src.data import (  # noqa: E402
    ego_controls,
    prepare_experiment_data,
)
from hierarchical_world_model.src.highway import (  # noqa: E402
    HIGHWAY_ENV_HIQR_DYNAMICS_CONTRACT,
    HighwayEnvClosedLoopWorld,
    HighwayEnvTraffic,
)
from hierarchical_world_model.src.planner import frozen_diffusion_plans  # noqa: E402
from hierarchical_world_model.src.train import load_checkpoint  # noqa: E402
from hierarchical_world_model.src.visualization import (  # noqa: E402
    DIFFUSION_COLOR,
    EGO_COLOR,
    LOGGED_REFERENCE_COLOR,
    ROAD_COLOR,
    _draw_lane_markings,
    _draw_vehicle,
)
from hierarchical_world_model.src.execution import trajectory_event_risk  # noqa: E402
from hierarchical_world_model.src.randomness import WorldExogenousState  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from process_highD.src.safety_envelope_risk import SafetyEnvelopeRiskOptions  # noqa: E402
from tools.evt import load_evt_model  # noqa: E402
from tools.idm_ego import load_idm_ego_config  # noqa: E402
from tools.plot_style import get_pyplot  # noqa: E402
from world_model.src.core.utils import (  # noqa: E402
    file_sha256,
    load_yaml,
    save_json,
    select_device,
)

from diffusion.src.data import ANCHOR_INDEX  # noqa: E402
from world_model.src.core.dynamics import KinematicTrafficDynamics  # noqa: E402


DEFAULT_CONFIG = ROOT / "hierarchical_world_model/config/release.yaml"
IDM_CONFIG = ROOT / "tools/idm_ego.yaml"


def _continuous_valid(valid: np.ndarray) -> np.ndarray:
    return np.asarray(valid[:, ANCHOR_INDEX:174].all(axis=1), bool)


def _prepend_anchor(states: np.ndarray, initial: np.ndarray) -> np.ndarray:
    return np.concatenate((initial[:, None], states), axis=1).astype(np.float32)


def _highway_replay(
    model: Any,
    logged_states: np.ndarray,
    logged_valid: np.ndarray,
    soft_plans: np.ndarray,
    maps: np.ndarray,
    map_valid: np.ndarray,
    exogenous: WorldExogenousState,
    *,
    device: Any,
    idm_policy: HighwayEnvIDMPolicy | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(logged_states), max(int(batch_size), 1)):
        stop = min(start + max(int(batch_size), 1), len(logged_states))
        outputs.append(
            _highway_replay_batch(
                model,
                logged_states[start:stop],
                logged_valid[start:stop],
                soft_plans[start:stop],
                maps[start:stop],
                map_valid[start:stop],
                exogenous.select(slice(start, stop)),
                device=device,
                idm_policy=idm_policy,
            )
        )
    return np.concatenate(outputs, axis=0)


def _highway_replay_batch(
    model: Any,
    logged_states: np.ndarray,
    logged_valid: np.ndarray,
    soft_plans: np.ndarray,
    maps: np.ndarray,
    map_valid: np.ndarray,
    exogenous: WorldExogenousState,
    *,
    device: Any,
    idm_policy: HighwayEnvIDMPolicy | None = None,
) -> np.ndarray:
    """Replay factual highD windows under the offline factual conditioning contract."""
    initial = np.asarray(logged_states[:, ANCHOR_INDEX], np.float32)
    initial_valid = np.asarray(logged_valid[:, ANCHOR_INDEX], bool)
    history_start = ANCHOR_INDEX - model.cfg.history_frames + 1
    history = np.asarray(logged_states[:, history_start : ANCHOR_INDEX + 1], np.float32)
    history_valid = np.asarray(
        logged_valid[:, history_start : ANCHOR_INDEX + 1], bool
    )
    historical_ego_controls = ego_controls(
        logged_states[:, history_start:ANCHOR_INDEX, 0],
        logged_states[:, history_start + 1 : ANCHOR_INDEX + 1, 0],
        0.04,
    )
    historical_ego_valid = (
        logged_valid[:, history_start:ANCHOR_INDEX, 0]
        & logged_valid[:, history_start + 1 : ANCHOR_INDEX + 1, 0]
    )
    historical_ego_controls[~historical_ego_valid] = 0.0
    world = HighwayEnvClosedLoopWorld(
        model,
        device=device,
        idm_config=None if idm_policy is None else idm_policy.highway_env_idm_config,
    )
    world.reset(
        torch.from_numpy(initial),
        torch.from_numpy(initial_valid),
        torch.from_numpy(np.asarray(soft_plans, np.float32)),
        torch.from_numpy(np.asarray(maps, np.float32)),
        torch.from_numpy(np.asarray(map_valid, bool)),
        exogenous_state=exogenous,
        initial_history=torch.from_numpy(history),
        initial_history_valid=torch.from_numpy(history_valid),
        committed_ego_controls=torch.from_numpy(historical_ego_controls),
        deterministic_response=True,
    )
    frames = [world.observe()["agent_states"].cpu().numpy()]
    logged_actions = ego_controls(
        logged_states[:, ANCHOR_INDEX:173, 0],
        logged_states[:, ANCHOR_INDEX + 1 : 174, 0],
        0.04,
    )
    for frame in range(logged_actions.shape[1]):
        action = (
            world.idm_actions()
            if idm_policy is not None
            else torch.from_numpy(logged_actions[:, frame]).to(device)
        )
        transition = world.advance_response(action)
        frames.append(transition["agent_state_frames"][:, 0].cpu().numpy())
    return np.stack(frames, axis=1).astype(np.float32)


def _highway_highd_control_replay(
    logged_states: np.ndarray,
    logged_valid: np.ndarray,
    highd_actions: np.ndarray,
) -> np.ndarray:
    """Replay logged controls in HighwayEnv without invoking the world model.

    This is the physical-transfer baseline: any reconstruction error here is
    caused by the state/control representation or HighwayEnv integration, not
    by diffusion or HiQR.
    """
    factual_valid = _continuous_valid(logged_valid)
    ego_actions = ego_controls(
        logged_states[:, ANCHOR_INDEX:173, 0],
        logged_states[:, ANCHOR_INDEX + 1 :174, 0],
        0.04,
    )
    output: list[np.ndarray] = []
    for index in range(len(logged_states)):
        traffic = HighwayEnvTraffic(dt_s=0.04, seed=index)
        traffic.reset(
            logged_states[index, ANCHOR_INDEX],
            factual_valid[index],
        )
        frames = [traffic.states()]
        for frame in range(ego_actions.shape[1]):
            source = torch.from_numpy(
                logged_states[index, ANCHOR_INDEX + frame, 1:]
            )[None]
            actions = torch.from_numpy(highd_actions[index, frame])[None]
            controls = KinematicTrafficDynamics.controls_from_highd_actions(
                actions, source
            )[0].numpy()
            frames.append(
                traffic.step(controls, ego_action=ego_actions[index, frame]).states
            )
        output.append(np.stack(frames))
    return np.stack(output).astype(np.float32)


def _kinematic_highd_control_replay(
    logged_states: np.ndarray,
    logged_valid: np.ndarray,
    highd_actions: np.ndarray,
) -> np.ndarray:
    """Replay the identical logged controls with the offline unicycle plant."""
    valid = _continuous_valid(logged_valid)
    current = torch.from_numpy(logged_states[:, ANCHOR_INDEX].copy())
    present = torch.from_numpy(valid)
    ego_actions = ego_controls(
        logged_states[:, ANCHOR_INDEX:173, 0],
        logged_states[:, ANCHOR_INDEX + 1 :174, 0],
        0.04,
    )
    dynamics = KinematicTrafficDynamics()
    frames = [current.numpy()]
    for frame in range(ego_actions.shape[1]):
        source = torch.from_numpy(
            logged_states[:, ANCHOR_INDEX + frame, 1:]
        )
        actions = torch.from_numpy(highd_actions[:, frame])
        background = KinematicTrafficDynamics.controls_from_highd_actions(
            actions, source
        )
        controls = torch.cat((torch.from_numpy(ego_actions[:, frame, None]), background), dim=1)
        current = dynamics.step(current, controls, present, dt=0.04)
        frames.append(current.numpy())
    return np.stack(frames, axis=1).astype(np.float32)


def _collision_fraction(states: np.ndarray, valid: np.ndarray) -> float:
    # Report the fraction of sequences with at least one collision.  Reducing
    # only over time would yield a background-slot rate because the mask keeps
    # its six-slot axis; that quantity is not a sample-level collision rate.
    return float(_collision_mask(states, valid).any(axis=(1, 2)).mean())


def _collision_mask(states: np.ndarray, valid: np.ndarray) -> np.ndarray:
    ego = states[:, :, :1]
    background = states[:, :, 1:]
    active = np.asarray(valid[:, None, 1:], bool)
    return (
        active
        & (np.abs(background[..., 0] - ego[..., 0]) < 4.8)
        & (np.abs(background[..., 1] - ego[..., 1]) < 1.8)
    )


def _summary(
    states: np.ndarray,
    valid: np.ndarray,
    *,
    evt_model: Any,
    threshold: float,
) -> dict[str, Any]:
    risk = trajectory_event_risk(
        states,
        valid,
        options=SafetyEnvelopeRiskOptions(),
    )
    score = np.asarray(evt_model.score(risk), np.float64)
    failed = score >= float(threshold)
    return {
        "risk": risk,
        "score": score,
        "failed": failed,
        "report": {
            "sequences": int(len(score)),
            "failure_count": int(failed.sum()),
            "failure_probability": float(failed.mean()),
            "collision_fraction": _collision_fraction(states, valid),
            "event_risk_mean": float(risk.mean()),
            "event_risk_p95": float(np.quantile(risk, 0.95)),
            "evt_score_mean": float(score.mean()),
            "evt_score_p95": float(np.quantile(score, 0.95)),
        },
    }


def _paired_change(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, int]:
    return {
        "both_safe": int((~reference & ~candidate).sum()),
        "human_safe_generated_failure": int((~reference & candidate).sum()),
        "human_failure_generated_safe": int((reference & ~candidate).sum()),
        "both_failure": int((reference & candidate).sum()),
    }


def _factual_fidelity(
    generated: np.ndarray,
    factual: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float]:
    """Report HighwayEnv replay displacement against the same highD window."""
    displacement = np.linalg.norm(generated[..., :2] - factual[..., :2], axis=-1)
    background_mask = np.broadcast_to(valid[:, None, 1:], displacement[:, :, 1:].shape)
    background = displacement[:, :, 1:][background_mask]
    ego = displacement[:, :, 0]
    return {
        "ego_ADE_m": float(ego.mean()),
        "ego_FDE_m": float(ego[:, -1].mean()),
        "background_ADE_m": float(background.mean()),
        "background_FDE_m": float(displacement[:, -1, 1:][valid[:, 1:]].mean()),
        "background_P95_displacement_error_m": float(np.quantile(background, 0.95)),
    }


def _select_rows(rows: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if maximum < 1:
        raise ValueError("max_sequences must be positive")
    if len(rows) <= maximum:
        return np.asarray(rows, np.int64)
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(rows, size=maximum, replace=False)).astype(np.int64)


def _local_coordinates(states: np.ndarray) -> np.ndarray:
    local = np.asarray(states, np.float32).copy()
    local[..., :2] -= local[:, :1, :1, :2]
    return local


def _render_collision_playbacks(
    *,
    output_dir: Path,
    rows: np.ndarray,
    valid: np.ndarray,
    factual: np.ndarray,
    idm_states: np.ndarray,
    observed: dict[str, Any],
    idm: dict[str, Any],
    top_k: int,
    frame_stride: int,
) -> list[dict[str, Any]]:
    """Render the highest-risk IDM-only collisions in the natural-playback style."""
    collision = _collision_mask(idm_states, valid)
    selected = np.flatnonzero(
        collision.any(axis=(1, 2)) & ~np.asarray(observed["failed"], bool)
    )
    selected = selected[np.argsort(np.asarray(idm["score"])[selected])[::-1]]
    selected = selected[: max(int(top_k), 0)]
    if not len(selected):
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            {
                "role": "IDM-only collision playbacks from risk calibration",
                "episodes": [],
            },
            output_dir / "playback_manifest.json",
        )
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = get_pyplot()
    factual_local = _local_coordinates(factual)
    idm_local = _local_coordinates(idm_states)
    manifest: list[dict[str, Any]] = []
    for rank, index in enumerate(selected, start=1):
        collision_frames = np.flatnonzero(collision[index].any(axis=1))
        first_collision = int(collision_frames[0])
        slots = np.flatnonzero(collision[index, first_collision]).astype(int)
        path = output_dir / f"idm_collision_{rank:03d}_row_{int(rows[index])}.gif"
        figure, axis = plt.subplots(figsize=(12.0, 4.8), dpi=100)
        figure.subplots_adjust(left=0.065, right=0.965, bottom=0.18, top=0.84)
        frames = np.arange(0, idm_local.shape[1], max(int(frame_stride), 1))
        if frames[-1] != idm_local.shape[1] - 1:
            frames = np.append(frames, idm_local.shape[1] - 1)
        with imageio.get_writer(
            path,
            mode="I",
            duration=40 * max(int(frame_stride), 1),
            loop=0,
        ) as writer:
            for frame in frames:
                axis.clear()
                axis.set_facecolor(ROAD_COLOR)
                center_x = float(idm_local[index, frame, 0, 0])
                _draw_lane_markings(axis)
                axis.set(
                    xlim=(center_x - 60.0, center_x + 60.0),
                    ylim=(-8.2, 8.2),
                    xlabel="relative longitudinal position [m]",
                    ylabel="relative lateral position [m]",
                    aspect="equal",
                    title=(
                        f"IDM collision | highD row {int(rows[index])} | "
                        f"t={frame * 0.04:.2f}s | "
                        f"S_EVT={float(idm['score'][index]):.2f}"
                    ),
                )
                trail_start = max(0, frame - 45)
                axis.plot(
                    factual_local[index, trail_start : frame + 1, 0, 0],
                    factual_local[index, trail_start : frame + 1, 0, 1],
                    color=LOGGED_REFERENCE_COLOR,
                    linestyle=":",
                    linewidth=1.8,
                    alpha=0.9,
                    label="highD factual ego",
                )
                axis.plot(
                    idm_local[index, trail_start : frame + 1, 0, 0],
                    idm_local[index, trail_start : frame + 1, 0, 1],
                    color=EGO_COLOR,
                    linewidth=2.0,
                    alpha=0.9,
                    label="IDM ego",
                )
                for slot in np.flatnonzero(valid[index, 1:]):
                    axis.plot(
                        factual_local[index, trail_start : frame + 1, slot + 1, 0],
                        factual_local[index, trail_start : frame + 1, slot + 1, 1],
                        color="#d9d9d9",
                        linestyle=":",
                        linewidth=1.0,
                        alpha=0.85,
                    )
                    axis.plot(
                        idm_local[index, trail_start : frame + 1, slot + 1, 0],
                        idm_local[index, trail_start : frame + 1, slot + 1, 1],
                        color=DIFFUSION_COLOR,
                        linewidth=1.35,
                        alpha=0.85,
                    )
                    is_collision = bool(collision[index, frame, slot])
                    _draw_vehicle(
                        axis,
                        idm_local[index, frame, slot + 1],
                        color="#ffbf00" if is_collision else DIFFUSION_COLOR,
                        label=(
                            f"collision background b{slot + 1}"
                            if is_collision
                            else None
                        ),
                        filled=True,
                        alpha=0.9 if is_collision else 0.52,
                    )
                _draw_vehicle(
                    axis,
                    factual_local[index, frame, 0],
                    color=LOGGED_REFERENCE_COLOR,
                    label="highD factual ego",
                    filled=False,
                    alpha=0.9,
                )
                _draw_vehicle(
                    axis,
                    idm_local[index, frame, 0],
                    color="#ffbf00" if collision[index, frame].any() else EGO_COLOR,
                    label="IDM ego",
                    filled=True,
                    alpha=0.92,
                )
                axis.text(
                    0.01,
                    0.02,
                    "red: IDM ego | blue: HiQR background | dotted: highD factual | "
                    "yellow: footprint overlap",
                    transform=axis.transAxes,
                    fontsize=7.5,
                    va="bottom",
                    ha="left",
                    bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
                )
                axis.tick_params(labelsize=8)
                figure.canvas.draw()
                rgba = np.asarray(figure.canvas.buffer_rgba())
                writer.append_data(np.asarray(rgba[:, :, :3], dtype=np.uint8).copy())
        plt.close(figure)
        manifest.append(
            {
                "rank": rank,
                "row": int(rows[index]),
                "gif": str(path),
                "first_collision_frame": first_collision,
                "first_collision_time_s": float(first_collision * 0.04),
                "first_collision_slots": (slots + 1).tolist(),
                "frame_stride": int(frame_stride),
                "playback_frames": int(len(frames)),
                "event_risk": float(idm["risk"][index]),
                "evt_score": float(idm["score"][index]),
                "highd_evt_score": float(observed["score"][index]),
            }
        )
    save_json(
        {
            "role": "IDM-only collision playbacks from risk calibration",
            "episodes": manifest,
        },
        output_dir / "playback_manifest.json",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare highD factual controls with HighwayEnv HiQR and IDM replays; "
            "this is a calibration diagnostic, not subset simulation."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--idm-config", type=Path, default=IDM_CONFIG)
    parser.add_argument("--max-sequences", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--render-collisions", action="store_true")
    parser.add_argument("--top-k-collisions", type=int, default=5)
    parser.add_argument("--playback-frame-stride", type=int, default=2)
    parser.add_argument("--collision-output-dir", type=Path, default=None)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_protocol_config(config_path)
    device = select_device(str(config["training"].get("device", "auto")))
    experiment = prepare_experiment_data(config, ROOT)
    rows = _select_rows(experiment.test_rows, args.max_sequences, args.seed)
    arrays = experiment.bundle.arrays
    logged_states = np.asarray(arrays["agent_states"][rows], np.float32)
    logged_valid = np.asarray(arrays["agent_valid"][rows], bool)
    maps = np.asarray(arrays["map_polylines"][rows], np.float32)
    map_valid = np.asarray(arrays["map_polyline_valid"][rows], bool)
    valid = _continuous_valid(logged_valid)
    if not valid[:, 0].all():
        raise RuntimeError("the selected factual windows must keep the ego valid")

    checkpoint_path = Path(config["paths"]["evaluation_checkpoint"])
    model, _ = load_checkpoint(checkpoint_path, device=device)
    model.eval()
    with tempfile.TemporaryDirectory(prefix="risk_calibration_") as cache:
        plans = frozen_diffusion_plans(
            experiment.bundle,
            rows,
            checkpoint=config["paths"]["diffusion_checkpoint"],
            output_dir=cache,
            device=device,
            batch_size=32,
            ddim_steps=20,
            experiment_scope=str(config["training"]["experiment_scope"]),
        )
    factual = logged_states[:, ANCHOR_INDEX:174]
    highd_controls = np.asarray(arrays["actions_highd"])[rows].astype(np.float32)
    direct_highd_states = _highway_highd_control_replay(
        logged_states,
        logged_valid,
        highd_controls,
    )
    kinematic_highd_states = _kinematic_highd_control_replay(
        logged_states,
        logged_valid,
        highd_controls,
    )
    exogenous = WorldExogenousState.sample(
        len(rows),
        seed=int(args.seed) + 1,
        response_steps=149,
        scene_dim=model.cfg.scene_latent_dim,
        agent_dim=model.cfg.agent_latent_dim,
    )
    human_states = _highway_replay(
        model,
        logged_states,
        logged_valid,
        plans,
        maps,
        map_valid,
        exogenous,
        device=device,
    )
    policy = HighwayEnvIDMPolicy.from_dict(
        load_idm_ego_config(args.idm_config.resolve())
    )
    idm_states = _highway_replay(
        model,
        logged_states,
        logged_valid,
        plans,
        maps,
        map_valid,
        exogenous,
        device=device,
        idm_policy=policy,
    )
    evt_path = Path(config["paths"]["evt_model"])
    evt_model = load_evt_model(evt_path)
    risk_level = float(evt_model.return_level(100))
    score_level = float(evt_model.score(risk_level))
    observed = _summary(factual, valid, evt_model=evt_model, threshold=score_level)
    generated_human = _summary(
        human_states,
        valid,
        evt_model=evt_model,
        threshold=score_level,
    )
    generated_idm = _summary(
        idm_states,
        valid,
        evt_model=evt_model,
        threshold=score_level,
    )
    output = Path(config["paths"]["output_dir"])
    collision_playbacks = []
    if args.render_collisions:
        collision_output = (
            args.collision_output_dir.resolve()
            if args.collision_output_dir is not None
            else ROOT / "IDM_subset/results/risk_calibration_diagnostic"
        )
        collision_playbacks = _render_collision_playbacks(
            output_dir=collision_output,
            rows=rows,
            valid=valid,
            factual=factual,
            idm_states=idm_states,
            observed=observed,
            idm=generated_idm,
            top_k=args.top_k_collisions,
            frame_stride=max(int(args.playback_frame_stride), 1),
        )
    report = {
        "schema": "highd_highway_env_risk_calibration_diagnostic",
        "purpose": (
            "Compare the same held-out highD windows under observed human "
            "motion, HighwayEnv human-control replay, and HighwayEnv IDM replay."
        ),
        "threshold": {
            "evt_return_period_segments": 100,
            "event_risk": risk_level,
            "evt_score": score_level,
        },
        "sample": {
            "split": "test",
            "selected_sequences": int(len(rows)),
            "selection_seed": int(args.seed),
            "row_index_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
            "continuous_background_valid_fraction": float(valid[:, 1:].mean()),
        },
        "results": {
            "observed_highd_human": observed["report"],
            "world_replay_logged_human_controls": generated_human["report"],
            "world_replay_idm_controls": generated_idm["report"],
        },
        "highway_env_factual_fidelity": {
            "conditioning": {
                "initial_history": "logged_highd_25_frames",
                "committed_ego_controls": "logged_highd_history_controls",
                "soft_reference": "frozen_diffusion_plan",
                "hiqr_response": "deterministic",
            },
            "logged_human_controls": _factual_fidelity(human_states, factual, valid),
            "direct_logged_controls_highway_env": _factual_fidelity(
                direct_highd_states, factual, valid
            ),
            "direct_logged_controls_offline_unicycle": _factual_fidelity(
                kinematic_highd_states, factual, valid
            ),
            "highway_env_vs_offline_unicycle_same_controls": _factual_fidelity(
                direct_highd_states, kinematic_highd_states, valid
            ),
        },
        "paired_failure_changes": {
            "logged_human_to_world_human_controls": _paired_change(
                observed["failed"], generated_human["failed"]
            ),
            "logged_human_to_world_idm_controls": _paired_change(
                observed["failed"], generated_idm["failed"]
            ),
        },
        "collision_playbacks": collision_playbacks,
        "provenance": {
            "execution_backend": "local_HighwayEnv",
            "hiqr_vehicle_dynamics_contract": HIGHWAY_ENV_HIQR_DYNAMICS_CONTRACT,
            "world_checkpoint": str(checkpoint_path),
            "world_checkpoint_sha256": file_sha256(checkpoint_path),
            "diffusion_checkpoint": str(config["paths"]["diffusion_checkpoint"]),
            "diffusion_checkpoint_sha256": file_sha256(
                config["paths"]["diffusion_checkpoint"]
            ),
            "evt_model": str(evt_path),
            "evt_model_sha256": file_sha256(evt_path),
            "idm_config": str(args.idm_config.resolve()),
            "idm_config_sha256": file_sha256(args.idm_config.resolve()),
        },
        "interpretation": (
            "This is a held-out factual-calibration diagnostic, not proof of "
            "counterfactual human behavior under IDM intervention."
        ),
    }
    path = output / "risk_calibration_diagnostic.json"
    save_json(report, path)
    print(path)


if __name__ == "__main__":
    main()
