#!/usr/bin/env python3
"""Select highD car-following long-tail contexts."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from process_highD.src.following_tail_generation import run_following_tail_generation
from tools.highd_longitudinal import load_highd_event_context_cache


FOLLOWING_TAIL_CONTEXT_CONFIG = {
    "event_context_cache_path": (
        ROOT / "results" / "highd_events" / "following_event_contexts.npz"
    ),
    "tail_context_path": (
        ROOT / "results" / "highd_following_tail" / "contexts" / "tail_contexts.npz"
    ),
    "condition_distribution_path": (
        ROOT
        / "results"
        / "highd_following_tail"
        / "contexts"
        / "scenario_condition_distribution.npz"
    ),
    "independent_tail_peaks_path": (
        ROOT
        / "results"
        / "highd_following_tail"
        / "exposure"
        / "highd_independent_tail_peaks.csv"
    ),
    "evt_model_path": (
        ROOT
        / "results"
        / "highd_following_tail"
        / "evt"
        / "longitudinal_peak_evt_model.json"
    ),
    "evt_summary_path": (
        ROOT
        / "results"
        / "highd_following_tail"
        / "exposure"
        / "highd_exposure_summary.json"
    ),
    "scenario": "following",
    "risk_value_key": "y_long",
    "context_key_dtypes": {
        "y_long": "float",
    },
    "context_loader": load_highd_event_context_cache,
    "fit_evt_hint": "process_highD/scripts/estimate_following_exposure.py",
    "estimate_exposure_hint": (
        "process_highD/scripts/estimate_following_exposure.py"
    ),
    "context_generation_method": "gaussian_copula",
    "include_empirical_contexts": True,
    "num_synthetic_contexts": 5000,
    "diffusion_dataset_dir": (
        ROOT / "results" / "diffusion_natural" / "following"
    ),
    "diffusion_checkpoint_path": "checkpoints/best_noise_mse_train_val_test.pt",
    "generated_scenarios_path": (
        ROOT
        / "results"
        / "highd_following_tail"
        / "generated"
        / "diffusion_generated_scenarios.npz"
    ),
    "generate_diffusion_rollouts": True,
    "num_diffusion_scenarios": 5000,
    "diffusion_batch_size": 512,
    "diffusion_inference_steps": 100,
    "diffusion_device": "auto",
    "diffusion_seed": 42,
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_following_tail_generation(FOLLOWING_TAIL_CONTEXT_CONFIG)


if __name__ == "__main__":
    main()
