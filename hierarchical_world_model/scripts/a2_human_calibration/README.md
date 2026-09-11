# A2 human calibration

This paused candidate freezes the legacy A2 actor as a proposal prior and
would train only a stochastic calibration adapter. It is intentionally blocked
until policy-on factual replay is aligned with the historical bridge protocol.

Only the read-only support and protocol audit may run under `conda activate tread`:

```bash
python hierarchical_world_model/scripts/a2_human_calibration/prepare_support.py
python hierarchical_world_model/scripts/a2_human_calibration/audit.py
```

`train.py`, `supervised_gate.py`, and `evaluate.py` reject execution while
`method.protocol_status` is not `aligned`. The required alignment is specified
in [`FITWMAMS_A2_Factual_Protocol_Correction.md`](../../../doc/FITWMAMS_A2_Factual_Protocol_Correction.md).

Only after a future accepted test may `sensitivity.py --pairs 2000` run. It is
a paired diagnostic and explicitly never starts AMS.
