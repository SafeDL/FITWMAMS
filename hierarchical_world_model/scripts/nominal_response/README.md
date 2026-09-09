# Nominal-preserving response scripts

The scripts here implement the one retained training and evaluation path. They
use the frozen Flow, Diffusion and HiQR components, build a prefix-only nominal
reference, then train or evaluate the selected response layer.

Run commands from the repository root in the `tread` environment:

```bash
python hierarchical_world_model/scripts/nominal_response/build_cache.py --output results/hierarchical_world_model/nominal_response/work/train_cache
python hierarchical_world_model/scripts/nominal_response/train.py --cache-dir results/hierarchical_world_model/nominal_response/work/train_cache --checkpoint results/hierarchical_world_model/nominal_response/checkpoint.pt
```

`evaluate_factual.py`, `evaluate_natural.py` and `evaluate_ood.py` respectively
implement P0 factual compatibility, P2 natural-response scoring and E4 OOD
execution checks. Their outputs should be written under a fresh `work/`
directory and aggregated into the semantic summaries under `evidence/` only
after the fixed validation gates pass.
