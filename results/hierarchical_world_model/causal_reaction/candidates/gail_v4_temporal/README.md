# GAIL V4 human prior

This candidate is a frozen longitudinal human-action prior trained from the
complete highD training split.  The generator is a two-layer 128-unit bounded
Gaussian actor with a separate critic; the discriminator scores each tick of
paired two-second HighwayEnv sequences.  The observation contains only the
realized parent/child history, gap, closing speed, TTC, role and the previous
realized child acceleration.

The selected pass is pass 2 of the adversarial refinement (20,742 dynamic
relations).  It has validation NLL -0.595, discriminator AUC 0.505,
acceleration W1 0.119 m/s² and jerk W1 2.277 m/s³; the BC baselines are
1.245 and 9.747.  Pass 3 is retained in `training_summary.json` as an
overfit diagnostic (AUC 0.842), not used by the checkpoint.

All formal evidence is in `evidence/`; held-out metrics are in
`heldout_validation_prior_metrics.json` and
`heldout_test_prior_metrics.json`.
