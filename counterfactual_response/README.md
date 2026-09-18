# Stochastic-driver matched response probes

This module audits B-IDM, MA-IDM, Dynamic-AR(5), and multi-regime B-IDM under
the same added leader-braking intervention. It follows the CIH-WM response-test
contract and is not another natural-trajectory accuracy benchmark.

The default probe is anchored to supported highD validation event row 55398
(recording 27, leader 2560, follower 2568). The row is selected only from its
pre-intervention state as the supported ego-led event nearest the models'
shared speed/gap calibration support; future response outcomes are not used.
The initial leader/follower speeds are 15.69/16.84 m/s and the bumper gap is
22.64 m.

For every posterior draw, the model first produces a no-intervention follower
action plan against the logged leader controls. At each brake dose, both
displayed branches then receive the identical logged-control-plus-brake leader:

- **frozen plan:** replays the no-intervention follower actions and cannot
  respond to the brake;
- **closed loop:** recomputes follower actions from the intervened state.

The two branches share parameters and innovations. Therefore their gap
difference measures the effect of follower action response and is not
contaminated by different leader trajectories. The plant runs at 25 Hz and the
retained drivers decide at 5 Hz. Doses are 1.5, 3.0, and 5.0 m/s² during 1–2 s,
with 256 paired futures per model.

B-IDM和MA-IDM从全部251个合格pair重拟合的部署后验抽样；recording-held-out后验只用于
自然驾驶OOF评测，不混入该仿真探针。

```bash
python -m counterfactual_response.scripts.run
pytest -q counterfactual_response/tests
```

Outputs in `artifacts/` include:

- `response_metrics.json`: protocol, anchor lineage, paired dose-response
  metrics, natural-rollout mismatch, and interpretation guards;
- `response_ensembles.npz`: natural, frozen-plan, and closed-loop trajectories;
- `counterfactual_braking_response_summary.png`: ensemble response surfaces;
- one 1500×800 GIF per model in the CIH-WM 2×2 playback style: matched road
  worlds, follower actions, and gaps; instantaneous speeds and closing speed
  are annotated inside the road panels.

## Result at the 3 m/s² dose

| Model | Response probability | Median latency | Peak extra braking | Terminal gap benefit |
|---|---:|---:|---:|---:|
| B-IDM | 96.1% | 0.20 s | 1.62 m/s² | 8.82 m |
| MA-IDM | 96.9% | 0.20 s | 1.49 m/s² | 8.36 m |
| Dynamic-AR(5) | 97.3% | 0.20 s | 1.30 m/s² | 7.86 m |
| Multi-regime B-IDM | 100% | 0.20 s | 1.68 m/s² | 8.59 m |

The natural no-intervention gap RMSE against the logged highD trajectory is
2.04, 2.56, 2.37, and 2.61 m respectively. This mismatch is reported rather
than hidden by choosing a convenient Monte Carlo sample. Under the shared-noise
contract, B-IDM, MA-IDM, and Dynamic-AR keep exactly the same stochastic
residual path across branches, so their response comes only from deterministic
IDM state feedback. Multi-regime changes its latent regime on 3.1% of native
frames on average and has the strongest median peak response in this anchor.

该canonical row没有完整5秒历史，因此multi-regime使用训练集众数style和当前单次观测冷启动；它不冒充prefix个体化结果。四个模型的响应滚动直接调用各自的在线driver实现。

These results demonstrate internal response mechanics, not real human
counterfactual correctness. The highD future supplies factual context only;
after adding leader braking there is no observed paired human outcome. Absolute
cross-model ranking is also limited because the retained posteriors were fitted
under different calibration protocols.

该四模型共同 probe 的机器可读汇总由 `python -m driver_reproduction.scripts.build_scorecard` 生成；共同能力与可比较性契约见 [`doc/00_Common_Protocol.md`](../doc/00_Common_Protocol.md)。
