# PPO--IDM held-out intervention audit

This directory evaluates the active `rl_residual_idm` checkpoint against a
frozen-HiQR, no-response baseline on 205 eligible highD test interventions.
The fixed ego probe adds -8 m/s² from 1.00 s to 2.00 s; paired worlds share
the prefix sample and exogenous seed.

Relative to frozen HiQR, frozen legacy A2 produces materially stronger
same-rear braking (mean \(-1.688\) m/s²) and usually larger minimum clearance
(mean \(+2.998\) m). This is a response diagnostic, not a safety claim: the
paired audit records one removed collision and two added collisions.

The direct A2 sensitivity comparison holds the frozen A2 controller, initial
state, prefix sample, and all exogenous noise fixed, changing only the added
ego braking probe. It has exactly zero pre-probe action difference across all
205 events. During the 1–2 s probe window, the affected rear vehicle adds
mean \(-0.875\) m/s² braking (median \(-1.274\) m/s²). Its collision outcome is
unchanged in all 205 pairs. The post-probe final gap increases by mean
\(+1.811\) m (median \(+0.343\) m). The GIF uses event 1321, selected from
collision-free events in the strongest 10% direct-braking responses by largest
positive final-gap change; its direct figures are \(-1.625\) m/s² braking,
\(+1.683\) m post-probe minimum-gap change, and \(+8.469\) m final-gap change.
`direct_probe_sensitivity/a2_probe_sensitivity.gif` is the visual counterpart;
it is a deliberately selected diagnostic, not an aggregate performance or
safety result.

`direct_probe_sensitivity/` contains the event with the strongest direct A2
action response to the probe. It is a diagnostic extreme, not a representative
example.
