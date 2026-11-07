# Multi-regime B-IDM 的 highD 适配

本目录实现 Wu et al. (2025) 的多机制随机跟驰框架。当前实现保留论文的三层结构：

1. 以平均时距、最大加速度和最大减速度进行三类驾驶风格聚类；
2. 以 `[gap, speed, closing speed]` 进行显式持续时间状态分段；
3. 在每类风格内拟合 regime-specific 层次 B-IDM，并用因果过滤状态进行随机滚动。

它是 **highD documented adaptation**，不是原文 Waymo/Lyft HDP-HSMM 的逐数值复现。当前有限 Gaussian HSMM、NUTS 及分布歧义见 `distribution_ambiguity.md` 和 `source_to_code_map.md`。

## 最终保留结果

最终协议固定 recording 25 为训练集，recording 26/36 为留出集；测试 style 只读取已完成的 5 s 前缀，未来 regime 只使用模拟状态更新。车辆以 25 Hz 推进，驾驶动作每 0.2 s 更新一次。

留出结果保存在 `evidence/heldout/recording_heldout_metrics.json`。该适配的区间覆盖不足，当前注册状态为 `rejected_for_deployment`，保留实现与拟合证据供诊断，不作为发布世界模型的驾驶人。

## 证据目录

```text
evidence/
  dataset/       最终评测使用的 218-pair highD 队列及结构清单
  segmentation/  recording 25 的风格聚类、有限 HSMM 和论文类比图
  posterior/     分style层次/pooled NUTS后验及采样诊断
  heldout/       recording 26/36 的完整因果评测及可视化
```

## 最小复跑入口

```bash
conda activate tread
python -m pip install -r external_model_baselines/models/bayesian_ma_idm/requirements-pymc.txt
python -m external_model_baselines.models.multi_regime_bidm.scripts.fit_stage_a
python -m external_model_baselines.models.multi_regime_bidm.scripts.visualize_stage_a
python -m external_model_baselines.models.multi_regime_bidm.scripts.fit_stage_b_nuts
python -m external_model_baselines.models.multi_regime_bidm.scripts.visualize_stage_b_nuts
python -m external_model_baselines.models.multi_regime_bidm.scripts.evaluate_recording_heldout
python -m external_model_baselines.models.multi_regime_bidm.scripts.visualize_recording_heldout
```

`src/driver.py::MultiRegimeIDMDriver` 明确分开 `observe` 与 `decision`：prefix初始化后只用已实现状态更新过滤器，episode内保留联合regime参数。这个小队列留出诊断只与独立拟合且共享创新的pooled baseline比较；共同的模型范围与证据边界见 [`external_model_baselines/README.md`](../../README.md) 和 [`registry.json`](../../registry.json)。

`MultiRegimeIDMDriver` 用于独立匹配评测和配对响应诊断；当前区间校准未通过，不替代已发布世界模型的 NPC。
