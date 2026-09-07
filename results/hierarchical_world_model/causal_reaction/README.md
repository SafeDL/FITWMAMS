# NPC reaction artifacts

Active artifacts use semantic directories:

- `formal/idm_mobil/`: frozen highD rule calibration;
- `reaction_events/`: train-defined event support and split-isolated evidence;
- `calibrated_residual/`: current candidate caches, checkpoints, and reports;
- `archived_reaction_experiments/`: index of rejected GAIL/MLOO studies.

Historical candidate directories retain their original names because their paths
are part of already-recorded evidence. They are not imported by the current
configuration and must not be used as active training outputs.

The selected controller remains the frozen A2-transfer baseline until
`validate_reaction_policy.py` accepts the calibrated residual policy on
validation. Test is run once after policy, thresholds, and statistics are frozen.

`ppo_response_20260906/` is a self-contained, inconclusive calibration pilot.
Its root `decision.md` is authoritative; `evidence/` holds the immutable event
reference, `controllers/` the three calibration checkpoints, and `evaluation/`
the validation report and compact comparison tables.  Reusable plan caches are
kept next to the run rather than copied into its reports.

`ppo_response_round2_20260906/` closes the full validation diagnosis and the
supervised-first repair attempt.  The supervised-v2 candidate failed factual
non-inferiority and retained 189 strict causal OOD regressions, so PPO was not
started.  A2 remains the release controller and test remains untouched.
