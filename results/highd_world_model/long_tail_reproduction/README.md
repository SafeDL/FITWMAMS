# highD world-model long-tail reproduction report

## Protocol

This report evaluates **328** held-out highD EVT-tail scenes for **5 s**. Every model receives the same initial traffic state, B0, road graph, ego history and observed ego replay; **32** futures are generated per scene. Empirical CDF metrics use a deterministic cap of 100,000 points while trajectory, tail probability and diversity metrics use all scenes and branches.

Physical event subsets are overlapping: high-risk following **183**, hard braking **17**, high-speed approach **162**, close interaction **215**, and strong within-slot relative-speed change **65**.

## How to read this directory

- `comparison/`: cross-model overview, the full risk CCDF, and the canonical full summary.
- `highd_real_tail/`: fixed protocol and event-selection definition.
- `ramp_world_model/`, `semi_markov_world_model/`, `cat_topk_world_model/`: model-only JSON, four publication-ready panels, and a compact interpretation.

## Quantitative comparison

| Model | FDE (m) ↓ | minFDE@32 (m) ↓ | Risk W1 / KS ↓ | Branch FDE (m) ↑ | Coverage ≤1 m ↑ | Interaction MAE ↓ | Invalid ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RAMP-WM | 0.752 | 0.746 | 0.027 / 0.015 | 0.004 | 80.2% | 0.0088 | 0.00% |
| Semi-Markov WM | 0.928 | 0.882 | 0.023 / 0.015 | 0.027 | 71.6% | 0.0006 | 0.00% |
| CAT-TopK | 0.779 | 0.573 | 0.044 / 0.032 | 0.273 | 85.7% | 0.0081 | 0.38% |

The numerical source of this table is `comparison/long_tail_reproduction_summary.json`; values in the per-model `metrics.json` files are identical subsets, included for independent use.

## Evidence-supported conclusions

1. **Logged-future reconstruction.** RAMP-WM is best on deterministic ADE/FDE (0.190/0.752 m), but its 32 branches are nearly identical (pairwise FDE 0.004 m). It reconstructs the conditional mean trajectory; it does not yet create a useful multi-modal test distribution.
2. **Test-distribution coverage.** CAT-TopK has the lowest minFDE@32 (0.573 m) and highest 1 m coverage (85.7%), with the only materially separated branches (0.273 m). Its archived START interface, however, receives a future-action summary; it remains a reproducibility reference rather than a same-information superiority claim.
3. **Risk and interaction fidelity.** Semi-Markov is closest on aggregate tail risk (W1/KS 0.023/0.015) and speed/acceleration/gap interaction (MAE 0.0006). Its generated braking-response correlation is 0.327 versus 0.302 in highD, but its deterministic FDE and 1 m coverage are weaker than the other two models.
4. **Rare physical dynamics remain the limiting evidence.** In the 17 hard-braking scenes, the q99 risk exceedance is under-reproduced by every model; several braking/acceleration-duration W1 values are near or above one second. These observations identify precisely which tail mechanisms require more data or targeted training.

## Scope and decision

The fixed-condition results establish conditional reconstruction and diagnostic capability under a logged C0/B0 and ego replay. They do **not** by themselves establish that the model can construct a long-tail test distribution: that claim requires the separate Flow×world-model composition test below. The hard-braking sample is small, event groups overlap, and CAT-TopK is information-asymmetric. Retain these boundaries in any performance claim.


## Flow×RAMP composed test distribution

This separate report samples **8** Flow C0/B0 starts and **4** RAMP candidate futures for each of **326** held-out replay conditions. It therefore contains **10,432** five-second futures. Two held-out replays are unsupported because their discrete event structures have no Flow-training support. It is a distribution-level replay-controlled test, so per-donor ADE/FDE and minFDE are intentionally not reported.

- Flow input mean C0/B0 W1: **1.291 / 0.084**.
- B0 execution MAE: **0.173**.
- Closed-loop risk W1/KS: **0.297 / 0.115**.
- Invalid trajectory / overlap rate: **32.28% / 2.98%**.
- Same-C0/B0 inner branch endpoint spread: **0.011 m**.

The frozen Flow does not model map geometry or the external ego policy. Both are supplied by a held-out replay matched by the cache-derived START event structure (same-front when present, otherwise the first active fixed slot); the result is consequently evidence for the composed test environment under that explicit policy, not a claim of unconditional five-second natural-traffic generation. **The current composition does not pass as a usable long-tail generator:** its 32.28% invalid-trajectory rate and 0.011 m same-start branch spread show that physical safety and stochastic diversity must be repaired before it is used for ADS testing.
