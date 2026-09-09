# Current checkpoint playbacks

`natural_replay.gif` runs the selected `checkpoint.pt` with the logged highD
ego command. The dotted trajectories are the highD reference; the solid
trajectories are the current simulated rollout.

`braking_probe.gif` restores the established response-playback layout: the
top row compares frozen common physics (left) with the selected checkpoint
(right) under the same fixed -8 m/s² ego-command offset from 1.00 s to 2.00
s. The lower row traces the affected same-rear NPC's acceleration and the
ego-to-rear gap. Both worlds use the same event-prefix sample and exogenous
seed.

The natural replay follows the maintained world-model reconstruction style:
one road view, red ego, blue simulated background, white/dotted highD
reference, 45-frame trajectory tails, and in-frame legend/interpretation
strips. The braking probe follows the established causal-response style,
including the paired worlds, affected-NPC highlighting, brake window, action,
and gap plots.

The event and exact rendering contract are recorded in `playback_manifest.json`.
This is a one-event diagnostic, not evidence of aggregate safety or risk
reduction.
