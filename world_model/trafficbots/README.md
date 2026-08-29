# TrafficBotsV1.5-HighD

This is FITWMAMS's cache-only external TrafficBots V1.5 baseline.  It preserves
the upstream HPTR/KNARPE, CVAE, destination conditioning and MultiPathPP
background dynamics while adapting the data and evaluation protocol to highD.

Evaluation uses the shared `highd_follower_excluded_v1` population scope. The
trained checkpoint and all-slot training loss remain unchanged, while canonical
`same_rear` (agent index 2) is cleared and invalidated before TrafficBots sees a
test/IDM/AMS/Monte-Carlo scene. Metrics and playbacks inherit that same mask;
this is not a post-hoc collision filter.

Create the dedicated runtime from `requirements-trafficbots-highd.txt`; do not
alter the project `tread` environment. Then run:

```bash
python -m world_model.trafficbots.scripts.train --config world_model/trafficbots/config/highd.yaml
python -m world_model.trafficbots.scripts.evaluate --config world_model/trafficbots/config/highd.yaml --checkpoint PATH
```

The evaluator refuses to produce a main report when logged-control ego replay
fails the configured drift gate. It reports deterministic prior-mode, 16-sample
causal prior, Oracle diagnostic and paired-CRN brake/accelerate/left results.

Validate the completed baseline and run the matched causal comparison with:

```bash
python -m world_model.trafficbots.scripts.audit
python -m world_model.trafficbots.scripts.verify_full
```

The comparison command conditions both methods on the same logged S0 and test
ordering. It samples the hierarchical model's long-horizon constraints from
`p(K | C0, M)`; it never ranks the older held-out-future-K conditional
reconstruction against TrafficBots' S0-only rollout.

This is a method-faithful highD adaptation, not a bit-exact WOMD training
reproduction. `audit.json` records the mandatory runtime boundaries and the
small released-wrapper differences that remain explicit.
