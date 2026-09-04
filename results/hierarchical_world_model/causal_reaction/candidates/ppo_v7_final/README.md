# Dynamic Residual PPO candidate (v7)

This directory is the reproducible candidate assembled from the frozen
Flow–Diffusion–HiQR world model, the one-global-parameter IDM/MOBIL calibration,
the selected highD BC/early-stop longitudinal prior, and three dynamic-scope
PPO controllers.  Every rollout in `evaluation/` and `visualization/` uses
`HighwayEnvClosedLoopWorld`; internal integration is used only for the bridge
audit.

## Artifacts

- `controllers/`: A1 pure residual PPO, A2 residual PPO + IDM reference, and
  A3 residual PPO + IDM + frozen human prior.  Each was trained for 700 PPO
  updates over 42,192 eligible dynamic training scenes; the actor/critic are
  independent of the frozen world-model checkpoint.
- `evaluation/four_arm_comparison_test.json`: full 10,151-sequence factual
  test plus 5,095 dynamic counterfactual test sequences.
- `evaluation/four_arm_comparison_validation.json`: 6,864 dynamic validation
  sequences from the 13,133-sequence validation source split.
- `evidence/`: PPO/GAIL/IDM training and distribution plots regenerated from
  the raw evaluation artifacts.
- `visualization/comparative_playbacks/`: closed-loop factual and
  counterfactual GIFs for rows 3456, 22312, and 67927.

## Gate status

The alignment and factual gates pass (A0 full-background ADE 0.03958 m; all
arms are exact A0 passthrough before event registration).  Dynamic influence
selection is local, causal, and persistent after the command window.  This is
not promoted to `formal/`: v7 validation has an A3/A2 −8 m/s² dose ratio of
0.862 for 15/25 frames but 0.841 for the 5-frame condition (required ≥0.85),
and A3 final-action KL remains above A2.  Conditional acceleration W1 improves
versus A2, while jerk W1 does not.  Keep those failures visible when
interpreting the candidate; the evidence does not support claiming full GAIL
success yet.
