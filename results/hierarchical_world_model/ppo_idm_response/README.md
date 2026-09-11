# PPO--IDM response controller

This is the maintained legacy-A2 response baseline: a frozen
Flow--Diffusion--HiQR world model, a highD-calibrated IDM prior, and a PPO
residual longitudinal response controller. `checkpoint.pt` and
`rules/idm_mobil.json` are the exact pair used by the evaluation scripts.

The held-out, paired fixed-braking probe covers 205 eligible events from 8
recordings. Relative to frozen HiQR with the same prefix sample and exogenous
noise, the controller has mean rear braking increment −1.688 m/s² and mean
minimum-gap change +2.998 m. One collision is removed and two are introduced.
It therefore demonstrates a material response effect but is **not** a safety
or risk-release result. The historical factual file has `controller =
A0_none`, so it verifies the HiQR--HighwayEnv bridge rather than policy-on A2
factual fidelity.

A separate direct-sensitivity audit holds frozen A2 itself fixed and changes
only the documented ego braking probe. It has zero action difference before
the probe and mean extra rear braking −0.875 m/s² within it. Its 205 paired
collision outcomes are unchanged; post-probe final gap increases by mean
+1.811 m. The corresponding GIF is in
`evaluation/intervention_effects/direct_probe_sensitivity/`; it visualizes
local response, not a factual or safety claim.

`training/summary.json` records 540 actual updates, although the current YAML
lists a 700-update budget; it has no recorded checkpoint-selection or
early-stop rationale. Thus the artifact is a response baseline, not a fully
converged or fully accepted method. `acceptance.json` only establishes
response-effect and bridge evidence; it explicitly rejects risk-release and
safety-improvement claims. The missing policy-on factual protocol is specified
in [`FITWMAMS_A2_Factual_Protocol_Correction.md`](../../../doc/FITWMAMS_A2_Factual_Protocol_Correction.md).
