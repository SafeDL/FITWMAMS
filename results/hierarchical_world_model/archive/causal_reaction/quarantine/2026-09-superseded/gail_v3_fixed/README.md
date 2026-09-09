# HumanActionPriorV3

This candidate is the repaired longitudinal highD human prior.  It uses a
single bounded Gaussian actor, a separate critic, a per-tick MLP discriminator,
complete GAE/clipped PPO, paired two-second highD/HighwayEnv relations, and
role/TTC-stratified scene order.

The final checkpoint contains adversarial updates and is selected under a 5%
held-out BC-NLL trust region.  It visited 20,742 qualified two-second scenes per
complete pass.  The selected state has discriminator AUC 0.580 and validation
NLL 0.1667 versus BC 0.1603.  Its selected pass has discriminator AUC 0.620;
matched closed-loop acceleration/jerk W1 improve from 0.657/19.256 for BC to
0.427/16.244 for GAIL.

`evidence/gail_v3_four_panel_evidence.png` and the held-out JSON/NPZ files are
the authoritative diagnostics.  Legacy `gail_v2` and `gail_bc_v3` artifacts
remain comparison-only and are schema-incompatible with this checkpoint.
