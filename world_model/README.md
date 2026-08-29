# Shared World-Model Components

This directory contains only the components reused by the maintained traffic
world-model stack. The formal model, training stages, and release checks live
in [`hierarchical_world_model/`](../hierarchical_world_model/).

## Contents

- `src/hiqr/`: HiQR configuration, relational encoder, and observation filter.
- `src/core/`: shared highD data adapters, kinematic dynamics, Flow C0 helpers,
  sequence-cache loading, and utility functions.
- `src/traffic_graph/`: highD lane and agent-graph adaptation used while
  preparing the shared sequence cache.

The shared cache is located at:

```text
results/highd_shared_training_data/highd_sequence_cache/sequence_cache/
```

Rebuild it when source preprocessing changes:

```bash
conda run -n tread python process_highD/scripts/prepare_highd_sequences.py --rebuild
```

Do not add standalone model variants, checkpoints, or evaluation outputs here.
They belong in their owning module or in an explicitly maintained baseline.

`trafficbots/` is the explicitly maintained, CC BY-NC 4.0 external
TrafficBotsV1.5-HighD baseline.  It only consumes the canonical sequence cache
and writes outputs below `results/baselines/trafficbots_highd/`.
