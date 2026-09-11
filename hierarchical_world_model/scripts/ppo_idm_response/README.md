# PPO--IDM response scripts

These are the maintained legacy-A2 response diagnostics. They load the frozen
world model and the selected PPO residual controller through
`config/ppo_idm_response.yaml`.

From the repository root in the `tread` environment:

```bash
python hierarchical_world_model/scripts/ppo_idm_response/evaluate_intervention_effects.py
python hierarchical_world_model/scripts/ppo_idm_response/render_playbacks.py --event-index 1321 --output results/hierarchical_world_model/ppo_idm_response/evaluation/intervention_effects/direct_probe_sensitivity
```

The first command evaluates every eligible held-out intervention with shared
prefix samples and exogenous noise. The second renders the selected direct
probe-sensitivity event and includes `a2_probe_sensitivity.gif`: frozen A2
with logged ego controls versus the exact same frozen A2 under the added ego
braking command. Neither command measures factual reconstruction or makes a
safety claim. The existing factual-reconstruction artifact has `controller =
A0_none`; it is a HiQR--HighwayEnv bridge check, not an A2 policy result.
