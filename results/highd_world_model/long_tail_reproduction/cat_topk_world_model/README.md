# CAT-TopK: long-tail reproduction analysis

## Reproduction and coverage

- Deterministic ADE/FDE: **0.197 / 0.779 m**.
- minADE@32/minFDE@32: **0.152 / 0.573 m**.
- Pairwise branch FDE distance: **0.273 m**; coverage at 1/2/5 m: **85.7% / 98.2% / 100.0%**.

## Risk-tail fidelity

The trajectory risk score uses TTC, DRAC, gap and longitudinal acceleration. Its all-tail empirical discrepancy is **W1=0.044**, **KS=0.032**. At the real q90/q95/q99 thresholds, generated exceedance probabilities are 10.9% / 5.1% / 1.6%.

The hard-braking subgroup has 17 scenarios. Its risk W1 is 0.290; the q99 generated/real exceedance is 2.4% / 5.9%. This small subgroup must be interpreted as a stress test rather than high-power population evidence.

## Interaction and dynamics

- Speed/acceleration/gap correlation-matrix MAE: **0.0081**.
- Rear-vs-front braking correlation: generated **0.246**, highD **0.302**.
- Acceleration ACF MAE: **0.021**.
- Braking/acceleration/lateral duration W1: **0.999 / 0.966 / 0.429 s**.

## Validity

Generated trajectories have invalid-rate **0.38%**, collision-overlap rate **0.00%**, and jerk-out-of-range rate **0.00%**.

See the four numbered PNG panels in this directory for the visual evidence. Values are evaluated on the fixed full held-out long-tail protocol, not per-event retraining.
