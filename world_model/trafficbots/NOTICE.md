# TrafficBots V1.5 provenance

This directory adapts the core model from
https://github.com/zhejz/TrafficBotsV1.5 (snapshot commit
`9a379084adbefe9df005c4eae69e7a56c360a396`, mirrored in
`ref_code/TrafficBotsV1.5-main`). Copied upstream files retain their original
CC BY-NC 4.0 license headers.

The vendored model/utility files are unchanged except for:

- configurable dynamics `dt` (0.04 s for highD); and
- explicit posterior temporal indices spanning S0 through S149.

FITWMAMS supplies a separate cache-only adapter and training/evaluation
wrapper. It changes the highD schema and padding/KNN sizes, uses an S0-only
lane-destination surrogate, removes the unavailable warm start, applies
external ego controls, and disables WOSAC submission/collision-selection
code. The released HPTR/KNARPE, CVAE, destination predictor, Gaussian action
head and MultiPathPP background dynamics remain the model core.

This adapted baseline is for non-commercial research use only.
