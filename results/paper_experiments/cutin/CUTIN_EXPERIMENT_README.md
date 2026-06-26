# Cut-in Paper Experiments

This directory contains post-processed cut-in paper artifacts built from existing results only.
No cut-in diffusion training, EVT fitting, or subset simulation rerun was performed.

## Inputs

- `event_scores`: `results/highd_events/cutin_event_scores.csv`
- `event_cache_summary`: `results/highd_events/cutin_event_cache_summary.json`
- `subset_summary`: `IDM_subset/results/cutin/latent_subset_summary.json`
- `subset_level_stats`: `IDM_subset/results/cutin/latent_subset_level_stats.csv`
- `subset_samples`: `IDM_subset/results/cutin/latent_subset_samples.npz`
- `monte_carlo_summary`: `IDM_subset/results/monte_carlo_cutin/latent_monte_carlo_summary.json`
- `cutin_diffusion_dataset`: `results/diffusion_natural/cutin/dataset.npz`
- `evt_model`: `results/highd_cutin_tail/evt/cutin_peak_evt_model.json`
- `evt_summary`: `results/highd_cutin_tail/evt/cutin_peak_evt_summary.json`
- `exposure_summary`: `results/highd_cutin_tail/exposure/highd_cutin_exposure_summary.json`
- `tail_condition_distribution`: `results/highd_cutin_tail/contexts/scenario_condition_distribution.npz`
- `tail_contexts`: `results/highd_cutin_tail/contexts/tail_contexts.npz`
- `tail_generated_scenarios`: `results/highd_cutin_tail/generated/diffusion_generated_scenarios.npz`
- `tail_generated_summary`: `results/highd_cutin_tail/generated/diffusion_generated_scenarios_summary.json`
- `tail_distribution_similarity_summary`: `results/highd_cutin_tail/generated/figures/distribution_similarity_summary.json`

## Generated Artifacts

- `results/paper_experiments/cutin/cutin_safety_threshold_inverse_calibration.png`
- `results/paper_experiments/cutin/cutin_gpd_diagnostic_panel.png`
- `results/paper_experiments/cutin/cutin_tail_diffusion_generalization_panel.png`
- `results/paper_experiments/cutin/cutin_subset_level_score_histograms.png`

## Reused Existing Artifacts

- reused existing artifact: `cutin_tail_diffusion_generalization_panel: results/highd_cutin_tail/contexts/scenario_condition_distribution.npz`
- reused existing artifact: `cutin_tail_diffusion_generalization_panel: results/highd_cutin_tail/contexts/tail_contexts.npz`
- reused existing artifact: `cutin_tail_diffusion_generalization_panel: results/highd_cutin_tail/generated/diffusion_generated_scenarios.npz`
- reused existing artifact: `cutin_tail_diffusion_generalization_panel: results/diffusion_natural/cutin/dataset.npz`
- reused existing artifact: `cutin_subset_level_score_histograms: IDM_subset/results/cutin/latent_subset_samples.npz`

## Skipped Artifacts

- None

## Interpretation Notes

- All paper figures use the shared TREAD paper style: 300 dpi export, Times-compatible serif text, and STIX/LaTeX-style math rendering.
- Main exposure denominator is `all_vehicle_km`.
- ADS intensity is `conditional exceedance probability x highD tail peak exposure rate`.
- The probabilities are conditional on the highD cutin tail scenario-condition distribution, not unconditional road crash rates.
