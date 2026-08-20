# Screened natural-event playback audit

These six GIFs are random ground-truth highD segments drawn from the current
96,055-row `natural_segments.csv` after background-slot and lateral-event
integrity screening (seed 20260813):

- two complete lane-change segments;
- two upstream strict cut-ins;
- two segments without a lane change.

They audit the data-selection semantics; they are not diffusion predictions.
The final diffusion reconstruction overview and selected rollout profiles are in
`results/background_diffusion/`.

The sibling `evt_tail/` directory contains the five highest-risk segments under
the current POT threshold `u=0.5346742868`. Without `--random-count`, the
playback script regenerates `natural_tail_contexts.csv` when needed and then
visualizes EVT-tail segments. With `--random-count`, it reads the full screened
natural-event table and applies the requested cohort filter.
