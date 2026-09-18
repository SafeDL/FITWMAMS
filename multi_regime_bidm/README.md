# Multi-regime B-IDM：highD documented adaptation

本目录对应 `doc/03_Multi_Regime_B_IDM.md` 和 Wu et al. (2025) 的多机制随机跟驰框架。当前实现保留论文的三层结构：

1. 以平均时距、最大加速度和最大减速度进行三类驾驶风格聚类；
2. 以 `[gap, speed, closing speed]` 进行显式持续时间状态分段；
3. 在每类风格内拟合 regime-specific 层次 B-IDM，并用因果过滤状态进行随机滚动。

它是 **highD documented adaptation**，不是原文 Waymo/Lyft HDP-HSMM 的逐数值复现。当前有限 Gaussian HSMM、NUTS 及分布歧义见 `distribution_ambiguity.md` 和 `source_to_code_map.md`。

## 最终保留结果

最终协议固定 recording 25 为训练集，recording 26/36 为留出集；测试 style 只读取已完成的 5 s 前缀，未来 regime 只使用模拟状态更新。车辆以 25 Hz 推进，驾驶动作每 0.2 s 更新一次。

| 模型 | position RMSE (m) | speed RMSE (m/s) | acceleration CRPS (m/s²) | 90% position coverage |
|---|---:|---:|---:|---:|
| independently fitted pooled B-IDM | 0.66904 | 0.52847 | 0.19026 | 0.3696 |
| filtered multi-regime B-IDM | **0.62962** | **0.49455** | **0.18161** | 0.4044 |

在相同style数据、NUTS预算和配对创新下，多机制模型的点预测与CRPS优于独立拟合的pooled B-IDM，复现了论文的定性方向；但90%位置区间覆盖率仅40.44%，仍严重欠覆盖，因此当前结论保持 `rejected_not_well_calibrated`，不得接入世界模型。原文HDP-HSMM的分布表述歧义仍未解决，本实现只能称 documented adaptation。

## 证据目录

```text
evidence/
  dataset/       最终评测使用的 218-pair highD 队列及哈希清单
  segmentation/  recording 25 的风格聚类、有限 HSMM 和论文类比图
  posterior/     分style层次/pooled NUTS后验及采样诊断
  heldout/       recording 26/36 的完整因果评测及可视化
```

## 最小复跑入口

```bash
bayesian_ma_idm/.venv/bin/python -m multi_regime_bidm.scripts.fit_stage_a
bayesian_ma_idm/.venv/bin/python -m multi_regime_bidm.scripts.visualize_stage_a
bayesian_ma_idm/.venv/bin/python -m multi_regime_bidm.scripts.fit_stage_b_nuts
bayesian_ma_idm/.venv/bin/python -m multi_regime_bidm.scripts.visualize_stage_b_nuts
bayesian_ma_idm/.venv/bin/python -m multi_regime_bidm.scripts.evaluate_recording_heldout
bayesian_ma_idm/.venv/bin/python -m multi_regime_bidm.scripts.visualize_recording_heldout
```

`src/driver.py::MultiRegimeIDMDriver` 明确分开 `observe` 与 `decision`：prefix初始化后只用已实现状态更新过滤器，episode内保留联合regime参数。这个小队列留出诊断只与独立拟合且共享创新的pooled baseline比较；共同协议见 [`doc/00_Common_Protocol.md`](../doc/00_Common_Protocol.md)。

该driver已接入论文主项目，但由于欠覆盖默认被注册表拒绝。研究诊断必须显式执行：
`python -m hierarchical_world_model.scripts.stochastic_drivers rollout --model multi_regime --style-id 1 --allow-unaccepted`。
