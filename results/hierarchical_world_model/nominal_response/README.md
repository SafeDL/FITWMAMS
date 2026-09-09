# Nominal-preserving response

This directory contains the single selected research candidate from the `tread`
GPU run. `checkpoint.pt` is the supervised model selected before the fixed test
pass. The default controller and world-model configuration were not changed.

| Evidence | Validation | Test |
| --- | --- | --- |
| Factual compatibility | `evidence/validation/factual_summary.json` | `evidence/test/factual_summary.json` |
| Natural-response fit | `evidence/validation/natural_response_summary.json` | `evidence/test/natural_response_summary.json` |
| OOD physical execution | `evidence/validation/ood_summary.json` | `evidence/test/ood_summary.json` |

The candidate passed P1, P0, P2 and E4. It did not run S4 closed-loop tuning or
E5 risk calibration; see `acceptance.json` and `decision.md` for the resulting
claim boundary. The remaining root JSON files are the compact provenance and
protocol audit required to reproduce the selected chain.
