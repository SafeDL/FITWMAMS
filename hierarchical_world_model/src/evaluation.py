"""Factual, stochastic and paired-intervention evaluation."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from diffusion.src.data import ANCHOR_INDEX
from world_model.src.core.dynamics import KinematicTrafficDynamics
from world_model.src.core.highd_metrics import factual_metrics as _shared_factual_metrics
from world_model.src.core.highd_metrics import temporal_factual_metrics as _shared_temporal_factual_metrics
from world_model.src.core.highd_metrics import distribution_metrics as _shared_distribution_metrics
from world_model.src.core.highd_metrics import intervention_metrics as _shared_intervention_metrics
from world_model.src.core.highd_metrics import intervention_dose_response as _shared_intervention_dose_response
from world_model.src.core.highd_metrics import semantic_cutin_agents
from world_model.src.core.evaluation_scope import (
    evaluation_scope_contract,
    scoped_canonical_trajectory,
)
from world_model.src.core.utils import ensure_dir, load_json, save_json, select_device, set_seed

from .data import ego_controls, prepare_experiment_data
from .calibration import evaluation_response_calibration
from .planner import frozen_diffusion_plans, stochastic_diffusion_plan_samples


@dataclass(frozen=True)
class Rollout:
    """Numpy rollout tensors and controls returned by the offline evaluator."""

    states: np.ndarray
    background_actions: np.ndarray
    ego_actions: np.ndarray
    reference_actions: np.ndarray


EgoActionPolicy = Callable[[dict[str, torch.Tensor | int]], torch.Tensor | np.ndarray]


def _logged_ego_actions(
    states: np.ndarray, valid: np.ndarray | None = None
) -> np.ndarray:
    actions = ego_controls(
        states[:, ANCHOR_INDEX:173, 0],
        states[:, ANCHOR_INDEX + 1 : 174, 0],
        0.04,
    )
    if valid is not None:
        present = valid[:, ANCHOR_INDEX:173, 0] & valid[:, ANCHOR_INDEX + 1 : 174, 0]
        actions = actions.copy()
        actions[~present] = 0.0
    return actions


def _intervene(
    actions: torch.Tensor,
    kind: str | None,
    dose: float,
) -> torch.Tensor:
    result = actions.clone()
    if kind is None:
        return result
    start, stop = 25, 50
    if kind == "brake":
        result[:, start:stop, 0] = (result[:, start:stop, 0] - dose).clamp_min(-8.0)
    elif kind == "accelerate":
        result[:, start:stop, 0] = (result[:, start:stop, 0] + dose).clamp_max(4.0)
    elif kind == "left":
        result[:, start:stop, 1] = (result[:, start:stop, 1] + dose).clamp_max(0.6)
    else:
        raise ValueError(f"unknown intervention {kind!r}")
    return result


@torch.no_grad()
def rollout(
    model,
    logged_states: np.ndarray,
    logged_valid: np.ndarray,
    soft_plans: np.ndarray,
    map_polylines: np.ndarray,
    map_polyline_valid: np.ndarray,
    *,
    device: torch.device,
    history_frames: int,
    motion_seed: int | None,
    intervention: str | None = None,
    dose: float = 0.0,
    ads_policy: EgoActionPolicy | None = None,
) -> Rollout:
    """Run one causal offline response rollout for a matched batch."""
    logged_states, logged_valid = scoped_canonical_trajectory(
        logged_states, logged_valid
    )
    states = torch.from_numpy(logged_states[:, ANCHOR_INDEX].copy()).to(device)
    valid = torch.from_numpy(logged_valid[:, ANCHOR_INDEX].copy()).to(device)
    history = torch.from_numpy(
        logged_states[:, ANCHOR_INDEX - history_frames + 1 : ANCHOR_INDEX + 1].copy()
    ).to(device)
    history_valid = torch.from_numpy(
        logged_valid[:, ANCHOR_INDEX - history_frames + 1 : ANCHOR_INDEX + 1].copy()
    ).to(device)
    reference = torch.from_numpy(np.asarray(soft_plans, np.float32)).to(device)
    maps = torch.from_numpy(np.asarray(map_polylines, np.float32)).to(device)
    map_valid = torch.from_numpy(np.asarray(map_polyline_valid, bool)).to(device)
    initial_reference = states[:, 1:, :2].clone()
    logged_ego = torch.from_numpy(
        _logged_ego_actions(logged_states, logged_valid)
    ).to(device)
    scheduled_ego = _intervene(logged_ego, intervention, dose)
    historical_start = max(0, ANCHOR_INDEX - history_frames + 1)
    historical_ego_values = ego_controls(
        logged_states[:, historical_start:ANCHOR_INDEX, 0],
        logged_states[:, historical_start + 1 : ANCHOR_INDEX + 1, 0],
        0.04,
    )
    historical_present = (
        logged_valid[:, historical_start:ANCHOR_INDEX, 0]
        & logged_valid[:, historical_start + 1 : ANCHOR_INDEX + 1, 0]
    )
    historical_ego_values[~historical_present] = 0.0
    historical_ego = torch.from_numpy(historical_ego_values).to(device)
    generator = None
    if motion_seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(motion_seed))
    generated: list[torch.Tensor] = []
    background_actions: list[torch.Tensor] = []
    reference_actions: list[torch.Tensor] = []
    filter_state = None
    slow_scene = None
    slow_scene_noise = None
    agent_noise_state = None
    agent_style_state = None
    previous_current = None
    committed_ego_controls = historical_ego
    executed_ego: list[torch.Tensor] = []
    intervention_memory = None
    lateral_intervention_memory = None
    execute = model.cfg.execute_frames
    for start in range(0, 149, execute):
        count = min(execute, 149 - start)
        preview = reference[:, start : start + model.cfg.preview_frames]
        if preview.shape[1] < model.cfg.preview_frames:
            preview = torch.cat(
                (
                    preview,
                    preview[:, -1:].expand(
                        -1, model.cfg.preview_frames - preview.shape[1], -1, -1
                    ),
                ),
                dim=1,
            )
        base = initial_reference if start == 0 else reference[:, start - 1]
        ego_block = scheduled_ego[:, start : start + execute]
        if ads_policy is not None:
            proposed = torch.as_tensor(
                ads_policy(
                    {
                        "agent_states": states.detach().clone(),
                        "agent_valid": valid.detach().clone(),
                        "reference_index": start,
                    }
                ),
                dtype=states.dtype,
                device=device,
            )
            if proposed.shape == (len(states), 2):
                ego_block = proposed[:, None].expand(-1, execute, -1)
            elif proposed.shape == (len(states), execute, 2):
                ego_block = proposed
            else:
                raise ValueError(
                    "ads_policy must return [batch,2] or "
                    "[batch,execute_frames,2] controls"
                )
        if count < execute:
            ego_block = torch.cat(
                (
                    ego_block,
                    ego_block[:, -1:].expand(-1, execute - count, -1),
                ),
                dim=1,
            )
        scene_noise = torch.zeros(
            (len(states), model.cfg.scene_latent_dim),
            device=device,
            dtype=states.dtype,
        )
        agent_noise = torch.zeros(
            (len(states), 7, model.cfg.agent_latent_dim),
            device=device,
            dtype=states.dtype,
        )
        if generator is not None:
            scene_noise.normal_(generator=generator)
            agent_noise.normal_(generator=generator)
        response = model(
            history,
            history_valid,
            states,
            valid,
            preview,
            base,
            maps,
            map_valid,
            filter_state=filter_state,
            previous_current=previous_current,
            slow_scene=slow_scene,
            slow_scene_noise=slow_scene_noise,
            agent_noise_state=agent_noise_state,
            agent_style_state=agent_style_state,
            committed_ego_controls=committed_ego_controls,
            intervention_memory=intervention_memory,
            lateral_intervention_memory=lateral_intervention_memory,
            response_index=start // execute,
            scene_standard_normal=scene_noise,
            agent_standard_normal=agent_noise,
            deterministic=motion_seed is None,
        )
        filter_state = response.filter_state
        slow_scene = response.slow_scene
        slow_scene_noise = response.slow_scene_noise
        agent_noise_state = response.agent_noise_state
        agent_style_state = response.agent_style_state
        intervention_memory = response.intervention_memory
        lateral_intervention_memory = response.lateral_intervention_memory
        previous_current = states
        new_frames: list[torch.Tensor] = []
        for frame in range(count):
            controls = torch.cat(
                (ego_block[:, frame, None], response.actions[:, frame]), dim=1
            )
            states = model.dynamics.step(states, controls, valid, model.cfg.dt_s)
            new_frames.append(states)
        executed_ego.append(ego_block[:, :count])
        committed_ego_controls = torch.cat(
            (committed_ego_controls, ego_block[:, :count]), dim=1
        )[:, -model.cfg.intervention_trigger_history_frames - 1 :]
        block = torch.stack(new_frames, dim=1)
        generated.append(block)
        background_actions.append(response.actions[:, :count])
        reference_actions.append(response.reference_actions[:, :count])
        block_valid = valid[:, None].expand(-1, count, -1)
        history = torch.cat((history, block), dim=1)[:, -history_frames:]
        history_valid = torch.cat((history_valid, block_valid), dim=1)[
            :, -history_frames:
        ]
    return Rollout(
        torch.cat(generated, dim=1).cpu().numpy(),
        torch.cat(background_actions, dim=1).cpu().numpy(),
        torch.cat(executed_ego, dim=1).cpu().numpy(),
        torch.cat(reference_actions, dim=1).cpu().numpy(),
    )


def _factual_metrics(
    generated: np.ndarray,
    target: np.ndarray,
    active: np.ndarray,
) -> dict[str, float]:
    return _shared_factual_metrics(generated, target, active)


def _temporal_factual_metrics(
    generated: np.ndarray,
    target: np.ndarray,
    active: np.ndarray,
) -> dict[str, list[float]]:
    """Return per-horizon errors for drift rather than only end-point summaries."""
    return _shared_temporal_factual_metrics(generated, target, active)


def _factual_event_strata(
    generated: np.ndarray,
    target: np.ndarray,
    active: np.ndarray,
    evt_tail: np.ndarray,
    semantic_cutin: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Report conditional reconstruction errors on disjoint, declared test strata."""
    strata = {
        "all_natural": np.ones(len(generated), dtype=bool),
        "evt_labelled": np.asarray(evt_tail, bool),
        "semantic_cutin": np.asarray(semantic_cutin, bool),
    }
    report = {}
    for name, selected in strata.items():
        if not selected.any():
            raise RuntimeError(f"held-out factual stratum is empty: {name}")
        report[name] = _factual_metrics(
            generated[selected], target[selected], active[selected]
        )
    return report


def _distribution_metrics(
    samples: list[Rollout],
    initial_states: np.ndarray,
    target_states: np.ndarray,
    target_actions: np.ndarray,
    target_highd_actions: np.ndarray,
    active: np.ndarray,
) -> dict[str, Any]:
    return _shared_distribution_metrics(
        samples, initial_states, target_states, target_actions,
        target_highd_actions, active,
    )

def _dose_response_curve(
    baseline: Rollout,
    treatments: dict[float, Rollout],
    initial: np.ndarray,
    active: np.ndarray,
    kind: str,
    natural_calibration: dict[str, Any],
) -> dict[str, Any]:
    """Multi-dose response curves at the same horizons as natural matching."""
    return _shared_intervention_dose_response(
        baseline, treatments, initial, active, kind, natural_calibration,
    )

def _intervention_metrics(
    baseline: Rollout,
    mild: Rollout,
    strong: Rollout,
    initial: np.ndarray,
    active: np.ndarray,
    kind: str,
    natural_effects: np.ndarray | None = None,
) -> dict[str, float]:
    return _shared_intervention_metrics(
        baseline, mild, strong, initial, active, kind, natural_effects,
    )

def evaluate_world_model(config: dict[str, Any], *, config_dir: Path) -> dict[str, Any]:
    """Evaluate the maintained model and persist its complete JSON report."""
    from .calibration import fit_natural_response_calibrator
    from .data import split_rows
    from .model import DiffusionGuidedHiQR
    from .train import load_checkpoint

    output = ensure_dir(config["paths"]["output_dir"])
    device = select_device(config["training"].get("device", "auto"))
    seed = int(config["training"]["seed"])
    set_seed(seed)
    experiment = prepare_experiment_data(config, config_dir)
    scope = str(config["training"].get("experiment_scope", "full"))
    if scope not in {"full", "pilot"}:
        raise ValueError("experiment_scope must be 'full' or 'pilot'")
    checkpoint_path = Path(
        config["paths"].get(
            "evaluation_checkpoint",
            output / "checkpoints/final_world_model.pt",
        )
    )
    model, checkpoint = load_checkpoint(checkpoint_path, device=device)
    evaluation = config.get("evaluation", {})
    if evaluation.get("override_model_config", False):
        raise ValueError(
            "formal evaluation forbids model-config overrides; checkpoint is the contract"
        )
    if "intervention_adapter_logit" in evaluation:
        if not model.cfg.intervention_adapter_enabled:
            raise ValueError("adapter logit override requires intervention adapter")
        with torch.no_grad():
            model.decoder.intervention_logit.fill_(
                float(evaluation["intervention_adapter_logit"])
            )
    if model.cfg.natural_response_kernel_enabled:
        # This optional controller is calibrated only on the training
        # recording split.  In particular, a pilot must not fit it from its
        # small held-out test cohort just because that cohort is convenient.
        reference_path = config["paths"].get("response_calibration_reference")
        if reference_path:
            calibration = load_json(reference_path)
            response_bounds = np.asarray(
                [
                    calibration[name]["effect_p10_p50_p90_mps2"][::2]
                    for name in ("brake", "accelerate")
                ],
                np.float32,
            )
            sensitivity_bounds = np.asarray(
                calibration["dose_sensitivity_p10_p90_per_mps2"], np.float32
            )
        else:
            calibration_rows = split_rows(
                experiment.bundle.arrays, "train", seed=seed
            )
            calibrator, _ = fit_natural_response_calibrator(
                experiment.bundle.arrays,
                calibration_rows,
                minimum_events=int(
                    config["training"].get("response_calibration_minimum_events", 100)
                ),
                method=str(
                    config["training"].get("response_calibration_method", "exact")
                ),
            )
            response_bounds = calibrator.global_bounds
            sensitivity_bounds = calibrator.global_sensitivity_bounds
        model.set_matched_response_bounds(torch.from_numpy(response_bounds).to(device))
        model.set_response_sensitivity_bounds(
            torch.from_numpy(sensitivity_bounds).to(device)
        )
    model.eval()
    rows = experiment.test_rows
    states = np.asarray(experiment.bundle.arrays["agent_states"][rows], np.float32)
    valid = np.asarray(experiment.bundle.arrays["agent_valid"][rows], bool)
    states, valid = scoped_canonical_trajectory(states, valid)
    maps = np.asarray(experiment.bundle.arrays["map_polylines"][rows], np.float32)
    map_valid = np.asarray(experiment.bundle.arrays["map_polyline_valid"][rows], bool)
    active = valid[:, ANCHOR_INDEX, 1:]
    target = states[:, ANCHOR_INDEX + 1 : 174]
    with tempfile.TemporaryDirectory(prefix="hierarchical_wm_eval_") as cache:
        diffusion = frozen_diffusion_plans(
            experiment.bundle,
            rows,
            checkpoint=config["paths"]["diffusion_checkpoint"],
            output_dir=cache,
            device=device,
            batch_size=32,
            ddim_steps=20,
            experiment_scope=scope,
        )
    generated: list[Rollout] = []
    for start in range(0, len(rows), 64):
        generated.append(
            rollout(
                model,
                states[start : start + 64],
                valid[start : start + 64],
                diffusion[start : start + 64],
                maps[start : start + 64],
                map_valid[start : start + 64],
                device=device,
                history_frames=25,
                motion_seed=None,
            )
        )
    diffusion_distance = np.linalg.norm(diffusion - target[..., 1:, :2], axis=-1)
    diffusion_mask = np.broadcast_to(active[:, None], diffusion_distance.shape)
    generated_states = np.concatenate([item.states for item in generated])
    diffusion_states = target.copy()
    diffusion_states[..., 1:, :2] = diffusion
    diffusion_positions = np.concatenate(
        (states[:, ANCHOR_INDEX : ANCHOR_INDEX + 1, 1:, :2], diffusion), axis=1
    )
    diffusion_states[..., 1:, 2:4] = np.diff(diffusion_positions, axis=1) / 0.04
    factual: dict[str, Any] = {
        "open_loop_diffusion": {
            "ADE_m": float(diffusion_distance[diffusion_mask].mean()),
            "FDE_m": float(diffusion_distance[:, -1][active].mean()),
            "P50_displacement_error_m": float(
                np.quantile(diffusion_distance[diffusion_mask], 0.50)
            ),
            "P90_displacement_error_m": float(
                np.quantile(diffusion_distance[diffusion_mask], 0.90)
            ),
            "P95_displacement_error_m": float(
                np.quantile(diffusion_distance[diffusion_mask], 0.95)
            ),
            "P99_displacement_error_m": float(
                np.quantile(diffusion_distance[diffusion_mask], 0.99)
            ),
            "sequences": int(len(rows)),
            "frames": 149,
        },
        "diffusion_guided_hiqr": _factual_metrics(generated_states, target, active),
        "temporal_error": {
            "open_loop_diffusion": _temporal_factual_metrics(
                diffusion_states, target, active
            ),
            "diffusion_guided_hiqr": _temporal_factual_metrics(
                generated_states, target, active
            ),
        },
    }
    factual["event_strata"] = _factual_event_strata(
        generated_states,
        target,
        active,
        np.asarray(experiment.bundle.arrays["is_evt_tail"])[rows],
        semantic_cutin_agents(
            states[:, ANCHOR_INDEX:174], valid[:, ANCHOR_INDEX:174]
        ).any(axis=1),
    )
    factual["history_ablation"] = {}
    for frames in (5, 10, 15):
        generated = [
            rollout(
                model,
                states[start : start + 64],
                valid[start : start + 64],
                diffusion[start : start + 64],
                maps[start : start + 64],
                map_valid[start : start + 64],
                device=device,
                history_frames=frames,
                motion_seed=None,
            ).states
            for start in range(0, len(rows), 64)
        ]
        factual["history_ablation"][str(frames)] = _factual_metrics(
            np.concatenate(generated), target, active
        )
    factual["history_ablation"]["25"] = factual["diffusion_guided_hiqr"]
    ablation_count = min(256, len(rows))
    initial = states[:ablation_count, ANCHOR_INDEX, 1:]
    time = np.arange(1, 150, dtype=np.float32)[None, :, None, None] * 0.04
    constant_velocity = initial[:, None, :, :2] + time * initial[:, None, :, 2:4]
    without_constraint = rollout(
        model,
        states[:ablation_count],
        valid[:ablation_count],
        constant_velocity,
        maps[:ablation_count],
        map_valid[:ablation_count],
        device=device,
        history_frames=25,
        motion_seed=None,
    )
    factual["without_long_horizon_constraint"] = _factual_metrics(
        without_constraint.states,
        target[:ablation_count],
        active[:ablation_count],
    )
    # This is only a bounded evaluation cohort.  It is deliberately not the
    # AMS/subset-simulation population; that estimator lives in IDM_subset.
    stochastic_cohort_size = min(1024, len(rows))
    stochastic_rows = slice(0, stochastic_cohort_size)
    motion_seeds = tuple(seed + sample for sample in range(16))
    sampled_plans = stochastic_diffusion_plan_samples(
        experiment.bundle,
        rows[stochastic_rows],
        checkpoint=config["paths"]["diffusion_checkpoint"],
        device=device,
        batch_size=32,
        ddim_steps=20,
        motion_seeds=motion_seeds,
    )
    sample_rollouts = [
        rollout(
            model,
            states[stochastic_rows],
            valid[stochastic_rows],
            plan,
            maps[stochastic_rows],
            map_valid[stochastic_rows],
            device=device,
            history_frames=25,
            motion_seed=motion_seed + 100_000,
        )
        for plan, motion_seed in zip(sampled_plans, motion_seeds)
    ]
    source_states = states[stochastic_rows, ANCHOR_INDEX:173, 1:]
    target_highd = np.asarray(
        experiment.bundle.arrays["actions_highd"][rows[stochastic_rows]], np.float32
    )
    target_actions = KinematicTrafficDynamics.controls_from_highd_actions(
        torch.from_numpy(target_highd.copy()), torch.from_numpy(source_states.copy())
    ).numpy()
    distribution = _distribution_metrics(
        sample_rollouts,
        states[stochastic_rows, ANCHOR_INDEX],
        target[stochastic_rows],
        target_actions,
        target_highd,
        active[stochastic_rows],
    )
    intervention_count = min(512, len(rows))
    selected = slice(0, intervention_count)
    common = seed + 1000
    baseline = rollout(
        model,
        states[selected],
        valid[selected],
        diffusion[selected],
        maps[selected],
        map_valid[selected],
        device=device,
        history_frames=25,
        motion_seed=common,
    )
    interventions: dict[str, Any] = {}
    # A 1,024-sequence pilot is sufficient for model rollouts but not for all
    # sparse matched human-response cells, especially ego acceleration at
    # 0.8 s.  Its diagnostic reference therefore uses the complete held-out
    # test-recording pool.  No candidate is selected on this reference; full
    # evaluation continues to use exactly its own complete test split.
    natural_reference_rows = (
        split_rows(experiment.bundle.arrays, "test", seed=seed)
        if scope == "pilot"
        else experiment.test_rows
    )
    _, test_response_scale = evaluation_response_calibration(
        experiment.bundle.arrays,
        natural_reference_rows,
        minimum_events=30,
        evaluation_scope=True,
    )
    doses = {
        "brake": (1.5, 2.25, 3.0),
        "accelerate": (1.0, 1.5, 2.0),
        "left": (0.08, 0.12, 0.16),
    }
    for kind, dose_values in doses.items():
        treatments = {
            dose: rollout(
                model,
                states[selected],
                valid[selected],
                diffusion[selected],
                maps[selected],
                map_valid[selected],
                device=device,
                history_frames=25,
                motion_seed=common,
                intervention=kind,
                dose=dose,
            )
            for dose in dose_values
        }
        mild, strong = treatments[dose_values[0]], treatments[dose_values[-1]]
        interventions[kind] = _intervention_metrics(
            baseline,
            mild,
            strong,
            states[selected],
            active[selected],
            kind,
            (
                np.asarray(test_response_scale[kind]["effect_samples_mps2"])
                if kind in {"brake", "accelerate"}
                else None
            ),
        )
        interventions[kind]["dose_response"] = _dose_response_curve(
            baseline,
            treatments,
            states[selected],
            active[selected],
            kind,
            test_response_scale,
        )
    report = {
        "evaluation_schema_version": 3,
        "evaluation_scope": evaluation_scope_contract(),
        "experiment_scope": scope,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "test_sequences": len(rows),
        "factual_fidelity": factual,
        "distribution_stochasticity": distribution,
        "randomness_contract": {
            "z_scenario": (
                "Flow base randomness (u_M,z_C0,z_K); with fixed C0/M only "
                "z_K varies"
            ),
            "z_motion": (
                "one motion seed deterministically addresses epsilon_diff "
                "[149,6,2], the 16-D scene innovation and six 16-D agent "
                "innovations"
            ),
            "paired_interventions_use_common_random_numbers": True,
        },
        "model_contract": {
            "architecture": (
                "Flow p(M)p(C0|M)p(K|C0,M); diffusion (C0,M,K) -> soft plan; "
                "observation-filtered, two-time-scale HiQR jerk response"
            ),
            # One response is committed per simulated frame.  The previous
            # report incorrectly called the 0.2 s diagnostic window the
            # replanning interval, which made the 25 Hz contract look like a
            # 5 Hz policy.
            "response_frequency_hz": float(1.0 / model.cfg.dt_s),
            "response_interval_s": float(model.cfg.execute_frames * model.cfg.dt_s),
            "response_commit_frames": int(model.cfg.execute_frames),
            "preview_horizon_s": float(model.cfg.preview_frames * model.cfg.dt_s),
            "scene_latent_refresh_s": float(
                model.cfg.scene_refresh_responses * model.cfg.dt_s
            ),
            "soft_plan_is_hard_trajectory": False,
            "soft_plan_policy": "rebase plan increments to the realized state",
            "future_ego_action_is_model_input": False,
            "history_training_frames": [5, 10, 15, 25],
            "scene_latent_dim": int(model.cfg.scene_latent_dim),
            "agent_latent_dim_per_vehicle": int(model.cfg.agent_latent_dim),
            "causal_response_scale": float(model.cfg.causal_response_scale),
            "intervention_adapter_enabled": bool(
                model.cfg.intervention_adapter_enabled
            ),
            "natural_response_kernel_enabled": bool(
                model.cfg.natural_response_kernel_enabled
            ),
            "intervention_trigger_threshold_mps2": float(
                model.cfg.intervention_trigger_threshold_mps2
            ),
            "evt_role": "external human-risk scale; excluded from training loss",
        },
        "evaluation_protocol": {
            "factual_reconstruction": (
                "held-out C0 and held-out long-horizon condition with a frozen "
                "diffusion preview; this is conditional reconstruction, not "
                "unconditional scenario-generation accuracy"
            ),
            # Keep the established result schema; these fields are bounded
            # evaluation cohorts, not subset-simulation populations.
            "stochasticity_subset_sequences": int(stochastic_cohort_size),
            "intervention_subset_sequences": int(intervention_count),
            "natural_response_reference_sequences": int(len(natural_reference_rows)),
            "intervention_common_random_numbers": True,
            "jerk_protocol": (
                "raw 0.04 s action jerk is reported together with 0.2 s "
                "windowed jerk because highD raw jerk has a large quantized "
                "zero mass"
            ),
        },
        "limitations": [
            "Within a fixed K constraint, motion diversity must be interpreted "
            "together with its proper score.",
            "Raw 0.04 s jerk KS is dominated by highD's quantized zero mass; "
            "the 0.2 s windowed diagnostic is more representative of continuous "
            "motion but does not replace the raw metric.",
            "Observed highD data and structural intervention tests do not prove "
            "counterfactual correctness for arbitrary ADS policies.",
        ],
        "intervention_effectiveness": interventions,
        "natural_response_calibration": test_response_scale,
        "claims": {
            "counterfactual_correctness_proven": False,
            "official_WOSAC_score": False,
            "interpretation": "highD factual and structural intervention evidence only",
        },
    }
    save_json(report, output / "evaluation.json")
    return report
