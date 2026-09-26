# TrafficBotsV1.5-HighD

This is the canonical implementation directory indexed with the other
paper-derived baselines in [`external_model_baselines/`](../../README.md). Its only Python
package is `external_model_baselines.models.trafficbots`.

This is FITWMAMS's cache-only external TrafficBots V1.5 baseline.  It preserves
the upstream HPTR/KNARPE, CVAE, destination conditioning and MultiPathPP
background dynamics while adapting the data and evaluation protocol to highD.

Evaluation uses the shared `highd_all_background` population scope. All six
canonical background slots, including `same_rear`, remain valid for inference,
simulation, metrics, risk, and visualization. No background vehicle is masked
from the test scenes.

Create the dedicated runtime from `requirements-trafficbots-highd.txt`; do not
alter the project `tread` environment. Then run:

```bash
python -m external_model_baselines.models.trafficbots.scripts.train --config external_model_baselines/models/trafficbots/config/highd.yaml
python -m external_model_baselines.models.trafficbots.scripts.evaluate --config external_model_baselines/models/trafficbots/config/highd.yaml --checkpoint PATH
```

The evaluator refuses to produce a main report when logged-control ego replay
fails the configured drift gate. It reports deterministic prior-mode, 16-sample
causal prior, Oracle diagnostic and paired-CRN brake/accelerate/left results.

Validate the completed baseline and run the matched causal comparison with:

```bash
python -m external_model_baselines.models.trafficbots.scripts.audit
python -m external_model_baselines.models.trafficbots.scripts.verify_full
```

The comparison command conditions both methods on the same logged S0 and test
ordering. It samples the hierarchical model's long-horizon constraints from
`p(K | C0, M)`; it never ranks the older held-out-future-K conditional
reconstruction against TrafficBots' S0-only rollout.

This is a method-faithful highD adaptation, not a bit-exact WOMD training
external_model_baselines. `audit.json` records the mandatory runtime boundaries and the
small released-wrapper differences that remain explicit.

## Posterior scene-conditioned resimulation

The independent ADS diagnostic below infers TrafficBots' posterior latent and
reference destination once from the complete scene, freezes both values, and
then runs nominal and ADS-braking branches from simulated online state. The
paired branches never overwrite an online agent with its logged future state.
It is intentionally labelled scene-conditioned posterior resimulation and is
not interchangeable with the prior-only benchmark score.

```bash
conda run -n trafficbots-highd python -m \
  npc_behavior_benchmark.scripts.evaluate_trafficbots_posterior_resimulation \
  --limit 128
```

The retained report is
`results/baselines/trafficbots_highd/posterior_resimulation.json`.
The example writes `posterior_resimulation_run.json` in the same directory.
