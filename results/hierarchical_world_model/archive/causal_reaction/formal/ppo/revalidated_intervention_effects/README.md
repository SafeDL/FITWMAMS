# Archived A2 revalidation

This directory re-runs the archived `rl_residual_idm` (A2 PPO + IDM)
checkpoint against a frozen-HiQR, no-response baseline on the same 205 eligible
test interventions used for the active nominal-response audit. The fixed ego
probe adds -8 m/s² from 1.00 s to 2.00 s, and paired worlds share the prefix
sample and exogenous seed.

The A2 controller produces materially stronger same-rear braking and usually
larger minimum clearance on this diagnostic. It is an archived comparison
baseline, not the active method or a release claim. Its response can begin
before the added probe window when the already-observed causal state grants
authority; this must not be interpreted as anticipation of the future ego
command.

`maximum_extra_rear_braking/` is the single event with the strongest added
rear braking. It is a diagnostic extreme, not a representative example.
