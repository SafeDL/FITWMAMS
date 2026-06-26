# SAIRL Evaluation Summary

The SAIRL following and cut-in evaluations use the shared highD tail context distributions, frozen diffusion priors, EVT thresholds, and IDM ego configuration used by the comparable subset folders.

| Scenario | Subset probability | MC probability | Abs. gap | Subset evals | MC evals | Speedup vs MC | Stop reason |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| following | 0.17946667 | 0.16855 | 0.01091667 | 10585 | 20000 | 1.889x | absolute_probability_gap_within_0.015 |
| cut-in | 0.0353 | 0.0361 | 0.0008 | 3388 | 20000 | 5.903x | mc_ci_contains_subset_estimate |

Monte Carlo starts at 20000 samples, increments by 10000 if needed, and is capped at 200000 samples. Both selected runs stopped at the 20000-sample start point.
