# Reaction policy round 2

This run uses validation only.  Test and release configuration were not changed.

- `diagnosis/` contains the full current-checkpoint diagnosis.
- `candidate_v2/` contains supervised-v2 training, validation, telemetry, and
  the authoritative `decision.md`.
- `cache/validation/` is the single reusable 6,872-sequence plan cache covering
  dynamic validation candidates plus every supported ego-leader event scene.

The validation audit is 2,021 total events → 2,017 supported → 266 ego-leader
→ 265 supported ego-leader → 265 mapped → 265 evaluated, across 216 unique
scenes and 9 recordings.  Supervised-v2 stopped before PPO: factual
non-inferiority failed and all 189 strict causal OOD regressions were attributed
to insufficient actor action.  The release selection remains frozen A2.
