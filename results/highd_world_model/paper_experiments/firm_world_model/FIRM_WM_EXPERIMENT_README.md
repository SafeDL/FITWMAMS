# FIRM-WM paper experiments

## Inputs

- `evaluation/evaluation_summary.json`
- `evaluation/calibration_metrics.json`
- `evaluation/heldout_rollouts.npz`
- `evaluation/counterfactual_rollouts.npz`
- optional: `evaluation/flow_firm_composition.json`, `evaluation/flow_firm_tail_scores.npz`, and `evaluation/baseline_comparison.json`

## Generated Artifacts

- `firm_world_model_closed_loop_reconstruction_panel.png`
- `firm_world_model_probabilistic_calibration_panel.png`
- `firm_world_model_interaction_response_panel.png`
- `firm_world_model_tail_distribution_panel.png`
- `firm_world_model_ablation_baseline_panel.png`
- `firm_world_model_physical_replay_audit_panel.png`

## Reused Existing Artifacts

- Frozen RAMP-WM, Semi-Markov, and CAT-TopK evaluation summaries when available.

## Skipped Artifacts

- FIRM architecture ablations: evaluation/ablation_summary.json is absent.

## Interpretation Notes

Figures are a deterministic post-process of saved evaluation arrays. The reconstruction panel uses the first saved nominal and EVT-tail examples in scan order; it never selects samples by error or visual quality. No figure script trains, resamples, fits EVT, or changes a model result. CAT-TopK remains information-asymmetric.

## Result Status

- FIRM-WM 5 s background-only held-out ADE/FDE: `0.192` / `0.695` m.
- FIRM-WM does not yet match matched frozen RAMP-WM FDE (0.695 vs 0.679 m).
- FIRM-WM matches or improves matched frozen Semi-Markov FDE (0.695 vs 0.833 m).
- Goal-document promotion gate: `not_promoted` (every required held-out and Flow gate must pass).
- Flow × FIRM-WM uses `10432` generated 5 s futures; risk q90 absolute error is `0.086`.
- **Not eligible for ADS testing:** Flow × FIRM-WM invalid-trajectory rate is `33.57%` (required < 1%) and overlap-point rate is `3.04%`.
