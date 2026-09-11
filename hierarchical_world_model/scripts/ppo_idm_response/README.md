# PPO--IDM response scripts

These are the only active response diagnostics. They load the frozen world
model and the selected PPO residual controller through
`config/ppo_idm_response.yaml`.

From the repository root in the `tread` environment:

```bash
python hierarchical_world_model/scripts/ppo_idm_response/evaluate_intervention_effects.py
python hierarchical_world_model/scripts/ppo_idm_response/render_playbacks.py --event-index 1266 --output results/hierarchical_world_model/ppo_idm_response/evaluation/playbacks
```

The first command evaluates every eligible held-out intervention with shared
prefix samples and exogenous noise. The second renders one diagnostic event;
neither command makes a safety claim.
