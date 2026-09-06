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
