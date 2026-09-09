# A3 V3 focused retraining

This candidate initializes the dynamic A3 controller from the frozen validated
A2 checkpoint, attaches the frozen HumanActionPriorV3, automatically calibrates
the KL normalization from A2 validation states, and trains 700 PPO updates over
all 42,192 eligible highD training scenes.

It is **not accepted**.  Response dose is preserved, and final-action KL is
usually lower, but jerk W1 is worse for every validation/test brake duration;
the full-test 1.0 s condition also has a slightly higher controlled-NPC
collision rate.  A2 therefore remains selected, and no formal checkpoint or
GIF was overwritten.

See `evidence/a2_a3_v3_acceptance.json` for exact metrics and
`evidence/a2_a3_v3_focused_comparison.png` for the four-panel comparison.
