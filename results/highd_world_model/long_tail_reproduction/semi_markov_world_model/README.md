# Semi-Markov WM: long-tail reproduction analysis

## Reproduction and coverage

- Deterministic ADE/FDE: **0.235 / 0.928 m**.
- minADE@32/minFDE@32: **0.223 / 0.882 m**.
- Pairwise branch FDE distance: **0.027 m**; coverage at 1/2/5 m: **71.6% / 92.1% / 100.0%**.

## Risk-tail fidelity

The trajectory risk score uses TTC, DRAC, gap and longitudinal acceleration. Its all-tail empirical discrepancy is **W1=0.023**, **KS=0.015**. At the real q90/q95/q99 thresholds, generated exceedance probabilities are 10.1% / 4.7% / 1.5%.

The hard-braking subgroup has 17 scenarios. Its risk W1 is 0.174; the q99 generated/real exceedance is 0.0% / 5.9%. This small subgroup must be interpreted as a stress test rather than high-power population evidence.

## Interaction and dynamics

- Speed/acceleration/gap correlation-matrix MAE: **0.0006**.
- Rear-vs-front braking correlation: generated **0.327**, highD **0.302**.
- Acceleration ACF MAE: **0.017**.
- Braking/acceleration/lateral duration W1: **0.623 / 0.971 / 0.106 s**.

## Validity

Generated trajectories have invalid-rate **0.00%**, collision-overlap rate **0.00%**, and jerk-out-of-range rate **0.00%**.

See the four numbered PNG panels in this directory for the visual evidence. Values are evaluated on the fixed full held-out long-tail protocol, not per-event retraining.
