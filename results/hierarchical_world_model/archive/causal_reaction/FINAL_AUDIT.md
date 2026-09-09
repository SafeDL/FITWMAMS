# HighwayEnv alignment and dynamic NPC reaction audit

## Scope

All factual reconstruction, counterfactual evaluation, and playback GIFs use
`HighwayEnvClosedLoopWorld`. The rollout state contract is
`ANCHOR_INDEX + 1 : ANCHOR_INDEX + 150`: the state returned by one
HighwayEnv tick is written into HiQR history before the next tick.

The full factual test contains 10,151 source sequences. Dynamic
counterfactual evaluation uses the 5,095 test sequences for which the causal
influence graph has at least one possible affected NPC. Results use common
random numbers.

## Factual bridge

The same-action bridge passes: HighwayEnv versus offline diagnostic ADE is
`5.03e-05 m`. A0 full-background HighwayEnv factual ADE/FDE is
`0.03958/0.03668 m`; all six background slots are included. Before a causal
event is registered, A1--A3 are exact HiQR-action passthrough.

## Retained controller result

The selected response controller is A2 (`rl_residual_idm`) in
`candidates/ppo_v7_final/`. It is the current response baseline because it
adds local, delayed, persistent braking while preserving factual replay.

V4 GAIL is stored separately in `candidates/gail_v4_temporal/`. Its selected
pass was trained from the full highD train split and refined on 20,742 paired
two-second HighwayEnv relations. Its discriminator AUC is `0.505`; matched
closed-loop acceleration/jerk W1 is `0.119/2.277`, compared with BC
`1.231/9.766`.

## A3 decision

`candidates/a3_v4_balanced/` contains the latest GAIL-constrained A3. On the
complete dynamic test, its response-dose ratio to A2 is `1.008`, `0.917`, and
`0.917` for 0.2, 0.6, and 1.0 s `-8 m/s²` interventions. Final-action KL
improves by 33.0% and 23.8% in the latter two conditions, but is 3.2% worse
at 0.2 s; jerk W1 does not improve by 10% in every condition. A3 is therefore
not promoted. The machine-readable decision is
`candidates/a3_v4_balanced/evidence/a2_a3_v4_acceptance.json`.

## Retention rule

Only the formal baseline, current dynamic A2 evidence, V4 GAIL prior, and V4
A3 rejection evidence belong in the active index. Superseded runs are isolated
in `quarantine/2026-09-superseded/` and are not referenced by active code or
configuration.
