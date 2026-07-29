# RAMP-WM: long-tail reproduction analysis

## Reproduction and coverage

- Deterministic ADE/FDE: **0.190 / 0.752 m**.
- minADE@32/minFDE@32: **0.188 / 0.746 m**.
- Pairwise branch FDE distance: **0.004 m**; coverage at 1/2/5 m: **80.2% / 96.0% / 99.7%**.

## Risk-tail fidelity

The trajectory risk score uses TTC, DRAC, gap and longitudinal acceleration. Its all-tail empirical discrepancy is **W1=0.027**, **KS=0.015**. At the real q90/q95/q99 thresholds, generated exceedance probabilities are 10.4% / 4.3% / 1.2%.

The hard-braking subgroup has 17 scenarios. Its risk W1 is 0.162; the q99 generated/real exceedance is 0.0% / 5.9%. This small subgroup must be interpreted as a stress test rather than high-power population evidence.

## Interaction and dynamics

- Speed/acceleration/gap correlation-matrix MAE: **0.0088**.
- Rear-vs-front braking correlation: generated **0.324**, highD **0.302**.
- Acceleration ACF MAE: **0.009**.
- Braking/acceleration/lateral duration W1: **1.141 / 1.074 / 1.055 s**.

## Validity

Generated trajectories have invalid-rate **0.00%**, collision-overlap rate **0.00%**, and jerk-out-of-range rate **0.00%**.

See the four numbered PNG panels in this directory for the visual evidence. Values are evaluated on the fixed full held-out long-tail protocol, not per-event retraining.


## Flow×RAMP composed test distribution

This separate report samples **8** Flow C0/B0 starts and **4** RAMP candidate futures for each of **326** held-out replay conditions. It therefore contains **10,432** five-second futures. Two held-out replays are unsupported because their discrete event structures have no Flow-training support. It is a distribution-level replay-controlled test, so per-donor ADE/FDE and minFDE are intentionally not reported.

- Flow input mean C0/B0 W1: **1.291 / 0.084**.
- B0 execution MAE: **0.173**.
- Closed-loop risk W1/KS: **0.297 / 0.115**.
- Invalid trajectory / overlap rate: **32.28% / 2.98%**.
- Same-C0/B0 inner branch endpoint spread: **0.011 m**.

The frozen Flow does not model map geometry or the external ego policy. Both are supplied by a held-out replay matched by the cache-derived START event structure (same-front when present, otherwise the first active fixed slot); the result is consequently evidence for the composed test environment under that explicit policy, not a claim of unconditional five-second natural-traffic generation. **The current composition does not pass as a usable long-tail generator:** its 32.28% invalid-trajectory rate and 0.011 m same-start branch spread show that physical safety and stochastic diversity must be repaired before it is used for ADS testing.
