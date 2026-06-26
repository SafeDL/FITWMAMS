# A2C Evaluation Summary

The A2C following and cut-in evaluations use the shared highD tail context distributions, frozen diffusion priors, EVT thresholds, and IDM-matched lane configurations.

| Scenario | Subset probability | MC probability | Abs. gap | Subset evals | MC evals | Speedup vs MC | Stop reason |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| following | 0.179 | 0.18075 | 0.00175 | 3402 | 20000 | 5.879x | mc_ci_contains_subset_estimate |
| cut-in | 0.0297 | 0.03225 | 0.00255 | 3267 | 20000 | 6.122x | subset_ci_contains_mc_estimate |

Monte Carlo starts at 20000 samples, increments only if needed, and is capped at 200000 samples. Both A2C scenarios stopped at the 20000-sample start point because the MC estimates were close to the subset estimates while using substantially more closed-loop evaluations.
