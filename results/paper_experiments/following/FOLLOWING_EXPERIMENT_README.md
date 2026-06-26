# Following Paper Figures

This directory contains car-following paper figures built from existing highD following EVT, exposure, diffusion, Monte Carlo, and subset-simulation results.
No following diffusion training, EVT fitting, subset simulation, or tables were generated.

## Inputs

- `subset_samples`: `IDM_subset/results/following/latent_subset_samples.npz`
- `evt_model`: `results/highd_following_tail/evt/longitudinal_peak_evt_model.json`
- `exposure_summary`: `results/highd_following_tail/exposure/highd_exposure_summary.json`
- `tail_condition_distribution`: `results/highd_following_tail/contexts/scenario_condition_distribution.npz`
- `tail_contexts`: `results/highd_following_tail/contexts/tail_contexts.npz`
- `tail_generated_scenarios`: `results/highd_following_tail/generated/diffusion_generated_scenarios.npz`
- `following_segment_cache`: `results/highd_events/following_event_segments.npz`

## Generated Artifacts

- `results/paper_experiments/following/following_gpd_diagnostic_panel.png`
- `results/paper_experiments/following/following_safety_threshold_inverse_calibration.png`
- `results/paper_experiments/following/following_tail_diffusion_generalization_panel.png`
- `results/paper_experiments/following/following_tail_diffusion_acceleration_profiles.png`
- `results/paper_experiments/following/following_subset_level_score_histograms.png`

## Reused Existing Artifacts

- reused existing artifact: `following_gpd_diagnostic_panel: results/highd_following_tail/evt/longitudinal_peak_evt_model.json`
- reused existing artifact: `following_safety_threshold_inverse_calibration: results/highd_following_tail/evt/longitudinal_peak_evt_model.json`
- reused existing artifact: `following_safety_threshold_inverse_calibration: results/highd_following_tail/exposure/highd_exposure_summary.json`
- reused existing artifact: `following_tail_diffusion_generalization_panel: results/highd_following_tail/contexts/scenario_condition_distribution.npz`
- reused existing artifact: `following_tail_diffusion_generalization_panel: results/highd_following_tail/contexts/tail_contexts.npz`
- reused existing artifact: `following_tail_diffusion_generalization_panel: results/highd_following_tail/generated/diffusion_generated_scenarios.npz`
- reused existing artifact: `following_tail_diffusion_generalization_panel: results/highd_events/following_event_segments.npz`
- reused existing artifact: `following_tail_diffusion_acceleration_profiles: results/highd_following_tail/generated/diffusion_generated_scenarios.npz`
- reused existing artifact: `following_subset_level_score_histograms: IDM_subset/results/following/latent_subset_samples.npz`

## Skipped Artifacts

- None

## Interpretation Notes

- The following paper figures are generated directly in this directory; no `figures/` subdirectory is used.
- All paper figures use the shared TREAD paper style: 300 dpi export, Times-compatible serif text, and STIX/LaTeX-style math rendering.
- The panel shows the fitted POT/GPD tail diagnostics with the plotting range capped at `Y_long = 10`.
- The inverse calibration figure marks the selected 300 km all-vehicle return-level threshold from the exposure summary.
- The tail diffusion generalization panel compares empirical following EVT-tail contexts with generated lead trajectories; panel f uses the `lead_braking_duration` scenario-condition distribution used by `process_highD`.
- The acceleration-profile figure summarizes diffusion-generated long-tail lead-vehicle acceleration traces with a 5-95% envelope and representative braking modes.
- The subset level histogram shows how subset simulation concentrates mass toward the calibrated EVT risk threshold.
