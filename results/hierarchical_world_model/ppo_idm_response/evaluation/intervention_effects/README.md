# PPO--IDM held-out intervention audit

This directory evaluates the active `rl_residual_idm` checkpoint against a
frozen-HiQR, no-response baseline on 205 eligible highD test interventions.
The fixed ego probe adds -8 m/s² from 1.00 s to 2.00 s; paired worlds share
the prefix sample and exogenous seed.

The controller produces materially stronger same-rear braking and usually
larger minimum clearance. This is a response diagnostic, not a safety claim:
the paired audit records one removed collision and two added collisions. Its
response can begin before the added probe window when the already-observed
causal state grants authority; that is not anticipation of a future ego
command.

`maximum_extra_rear_braking/` is the single event with the strongest added
rear braking. It is a diagnostic extreme, not a representative example.
