#!/usr/bin/env python3
"""Build highD cut-in long-tail contexts and diffusion-generated scenarios."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from process_highD.src.cutin_tail_generation import run_cutin_tail_generation


CUTIN_TAIL_CONTEXT_CONFIG = {
    "event_context_cache_path": (
        ROOT / "results" / "highd_events" / "cutin_event_contexts.npz"
    ),
    "condition_distribution_path": (
        ROOT / "results" / "highd_cutin_tail" / "contexts" / "scenario_condition_distribution.npz"
    ),
    "tail_context_path": (
        ROOT / "results" / "highd_cutin_tail" / "contexts" / "tail_contexts.npz"
    ),
    "independent_tail_peaks_path": (
        ROOT
        / "results"
        / "highd_cutin_tail"
        / "exposure"
        / "highd_independent_tail_peaks.csv"
    ),
    "evt_model_path": (
        ROOT
        / "results"
        / "highd_cutin_tail"
        / "evt"
        / "cutin_peak_evt_model.json"
    ),
    "evt_summary_path": (
        ROOT
        / "results"
        / "highd_cutin_tail"
        / "exposure"
        / "highd_cutin_exposure_summary.json"
    ),
    "num_condition_samples": 5000,
    "num_diffusion_scenarios": 5000,
    "diffusion_config_path": ROOT / "diffusion" / "scripts" / "configs" / "natural_cutin.yaml",
    "diffusion_dataset_dir": ROOT / "results" / "diffusion_natural" / "cutin",
    "diffusion_checkpoint_path": "checkpoints/best_noise_mse_train_val_test.pt",
    "generated_scenarios_path": (
        ROOT
        / "results"
        / "highd_cutin_tail"
        / "generated"
        / "diffusion_generated_scenarios.npz"
    ),
    "diffusion_batch_size": 256,
    "diffusion_inference_steps": 100,
    "diffusion_guidance_scale": 0.5,
    "diffusion_guidance": {
        "guidance_end_y_weight": 1.0,
        # time_to_cross is the lane-boundary crossing time, not the time at
        # which the target center reaches the ego-lane center threshold.
        "guidance_cross_y_weight": 0.0,
        "guidance_post_lane_weight": 1.0,
        "guidance_final_lane_window_seconds": 0.5,
        "guidance_front_at_cross_weight": 1.0,
        "guidance_lateral_jerk_weight": 0.2,
        "lateral_overlap_threshold": 1.0,
        "cutin_lateral_offset": 1.0,
        "post_cutin_window_seconds": 3.0,
    },
    "diffusion_rejection": {
        # Keep semantic post-processing metrics as the model-quality signal.
        # Start with num_diffusion_scenarios sampled conditions; if hard semantic
        # filtering leaves too few outputs, refill from the same condition
        # distribution instead of pre-sampling a fixed oversized candidate pool.
        "enabled": True,
        "enforce_acceptance": True,
        "refill_condition_batch_size": 5000,
        "max_refill_rounds": 20,
        "lateral_overlap_threshold": 1.0,
        "cutin_lateral_offset": 1.0,
        "min_initial_lateral_offset": 1.5,
        "min_lateral_progress": 0.5,
        "min_lateral_approach_speed": 0.05,
        "post_cutin_window_seconds": 3.0,
    },
    "diffusion_device": "auto",
    "diffusion_seed": 42,
    "selection_random_seed": 42,
    "copula_marginal_clip_quantile": 0.01,
    "copula_correlation_regularization": 1.0e-4,
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_cutin_tail_generation(CUTIN_TAIL_CONTEXT_CONFIG)


if __name__ == "__main__":
    main()
