# NPC causal-reaction artifact index

This directory separates the retained longitudinal NPC causal-reaction study
from a reversible quarantine of superseded experiments. All factual and
counterfactual metrics and GIFs use `HighwayEnvClosedLoopWorld`.

## Retained artifacts

| Location | Purpose | Status |
| --- | --- | --- |
| `formal/idm_mobil/` | highD IDM/MOBIL calibration and held-out diagnostics | frozen reference |
| `formal/ppo/` | verified persistent-following A0--A3 baseline | retained baseline |
| `candidates/ppo_v7_final/` | latest full dynamic-influence A0--A3 comparison; A2 is the best response controller | current response evidence |
| `candidates/gail_v4_temporal/` | current bounded-Gaussian highD human prior, BC comparison, and GAIL evidence | current prior |
| `candidates/a3_v4_balanced/` | latest A3 candidate and its explicit non-promotion evidence | rejected candidate |
| `FINAL_AUDIT.md` | HighwayEnv alignment, scope, and current decision summary | entry point |

`gail_v4_temporal` is retained because it is the configured frozen prior.
`a3_v4_balanced` is retained only because its acceptance report explains why
A3 was not promoted. Its checkpoint is not a release controller.

## Current decision

The released reaction choice remains A2 (`rl_residual_idm`). V4 GAIL improves
the standalone human-prior distribution relative to BC, but the A3
final-action constraint does not improve KL and jerk across every intervention
duration. See `candidates/a3_v4_balanced/evidence/a2_a3_v4_acceptance.json`.

## Reproducible entry points

- `calibrate_reaction_rules.py` and `evaluate_reaction_rules.py`: IDM/MOBIL.
- `train_human_prior.py` and `evaluate_human_driving_prior.py`: human prior.
- `train_naturalistic_reaction_ppo.py` and
  `evaluate_naturalistic_reaction_controllers.py`: A1--A3.
- `visualize_human_prior_evidence.py`, `visualize_a2_a3_fast_evidence.py`,
  and `visualize_reaction_naturalistic_evidence.py`: figures and metrics.
- `mine_reaction_ppo_cases.py` and
  `render_reaction_ppo_comparative_playbacks.py`: selected HighwayEnv GIFs.

New training must use an explicit `--output` or `--output-dir` under a new
candidate directory. Do not write into `formal/` or overwrite one of the
retained candidate directories.

## Quarantine

`quarantine/2026-09-superseded/` contains old smoke runs, fixed same-rear
variants, projection experiments, and superseded PPO/GAIL candidates. It is
not referenced by code or configuration and is excluded from the active result
index. It is kept temporarily as a recoverable deletion set.
