# A3 V4 balanced candidate

This A3 controller starts from the frozen dynamic A2 checkpoint and uses the
V4 GAIL prior for final-action KL regularization.  It completed the full 700
update dynamic training pass and a 100-update focused naturalness/response
tail.  The tail uses a stronger response-floor penalty and is kept as a
candidate; the release controller remains A2 until the strict all-condition
acceptance test passes.

Evaluation is entirely through `HighwayEnvClosedLoopWorld`: 10,151 factual
test records and 5,095 dynamic counterfactual records with common random
numbers.  A3 response dose is at least 90% of A2 in all three `-8 m/s²`
conditions.  Test KL improves 33.0% and 23.8% for 0.6/1.0 s interventions,
but is 3.2% worse at 0.2 s; jerk W1 is not improved by 10% in every
condition.  Therefore `evidence/a2_a3_v4_acceptance.json` selects A2 and this
checkpoint is not promoted.
