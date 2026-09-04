# Persistent-following authority audit — retrained and re-evaluated

The former fixed-window authority results were superseded after the
persistent-following correction.  The current A1/A2/A3 checkpoints were each
retrained for 540 updates over the full 4,267 causally eligible highD training
reset scenes, with the frozen HiQR world model and the corrected state machine:

- an executed ADS intervention is a causal trigger, never a pending input;
- a same-rear relation remains armed for the episode and re-engages whenever
  its *realized* TTC re-enters the 4 s release horizon; and
- A1/A2/A3 share a non-IDM kinematic guard that prevents positive rear-NPC
  acceleration while an already-observed following risk remains unresolved.

The authoritative replacement reports are
`ppo/evaluation/four_arm_comparison_validation.json` and
`ppo/evaluation/four_arm_comparison_test.json`.  They use 64 fixed common-
random-number same-rear scenes per split (1,435 validation and 1,014 test
scenes are eligible in total), all 149 HighwayEnv ticks, and braking doses
2/4/6/8 m/s² with 0.2/0.6/1.0 s durations.

For the 1.0 s, -8 m/s² test condition, the rear-collision sequence rate is
0.5156 (A0), 0.0312 (A1), 0.0000 (A2), and 0.0000 (A3).  The post-command
unresolved-risk inactive and positive-acceleration-rebound rates are both zero
for A1/A2/A3.  This supports the persistent causal-response claim.  It does
not validate the GAIL naturalness claim: A3 has a larger conditional human KL
and substantially larger jerk p95 than A2 in this condition, so A3 must not be
presented as more natural.
