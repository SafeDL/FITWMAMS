# Result artifact index

The current result layout is documented in
[`causal_reaction/README.md`](causal_reaction/README.md).

Only the current A2 response evidence, the V4 human prior, and the V4 A3
rejection evidence remain under `causal_reaction/candidates/`. Earlier smoke
runs, fixed-scope controllers, projection experiments, and superseded
GAIL/IDM/PPO candidates are in `causal_reaction/quarantine/`; they are neither
loaded by the current configuration nor valid evidence for the study.
