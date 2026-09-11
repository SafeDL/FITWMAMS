# PPO--IDM response controller

This is the sole active response method: a frozen Flow--Diffusion--HiQR world
model, a highD-calibrated IDM prior, and a PPO residual longitudinal response
controller. `checkpoint.pt` and `rules/idm_mobil.json` are the exact pair used
by the evaluation scripts.

The held-out, paired fixed-braking probe covers 205 eligible events from 8
recordings. Relative to frozen HiQR with the same prefix sample and exogenous
noise, the controller has mean rear braking increment −1.753 m/s² and mean
minimum-gap change +3.563 m. One collision is removed and two are introduced.
It therefore demonstrates a material response effect but is **not** a safety
or risk-release result.

`evaluation/intervention_effects/` contains the distribution figure, per-event
table, machine-readable summary, and a deliberately labelled diagnostic GIF.
`training/summary.json` records the full-train PPO run. `acceptance.json`
states the claim boundary.
