# 随机驾驶人统一评测

仓库级复现清单现统一位于 [`external_model_baselines/`](../../README.md)。本目录只承担
跨模型协议、匹配评测和证据汇总，不是另一个模型；论文机制、拟合代码与原生结果仍位于
各自模型目录，主项目仿真接入位于 `hierarchical_world_model/src/stochastic_drivers/`。

## 同协议highD结果

共享cohort包含218个长跟驰事件。recording 25的全部182个事件用于训练，recording
26/36的全部36个合格事件用于测试；每个事件使用5秒因果前缀、3秒闭环预测、64个future、
25 Hz plant与5 Hz driver。以下数字可在这一基准内比较：

结果位于 `results/baselines/driver_reproduction/matched_metrics.json`，模型复现状态见同目录 `driver_model_scorecard.json`。各模型共享场景与预测协议，但保留各自的推断方法和拟合预算。

## 入口

```bash
# 重建匹配评测及统一证据表
python -m external_model_baselines.evaluation.driver_reproduction.scripts.run_matched_evaluation
python -m external_model_baselines.evaluation.driver_reproduction.scripts.build_scorecard

# 回归测试
pytest -q external_model_baselines/evaluation/driver_reproduction/tests \
  external_model_baselines/models/active_inference_driver/tests external_model_baselines/models/bayesian_ma_idm/tests external_model_baselines/models/dynamic_ar_idm/tests \
  external_model_baselines/models/multi_regime_bidm/tests external_model_baselines/evaluation/counterfactual_response/tests
```

`artifacts/matched/` 仅保存匹配评测的train-only后验；`results/baselines/driver_reproduction/`保存测试
预测和指标。B/MA的全251-pair仿真后验位于
`external_model_baselines/models/bayesian_ma_idm/evidence/deployment/`，不得
拿部署后验重算留出指标。

## 版本控制

源码、配置、测试、JSON指标、图表、紧凑后验和共享评测队列都应跟随仓库。原始highD
数据、本地环境/缓存、Active-Inference逐实验pickle、Dynamic-AR长链和可由脚本重建的
全轨迹集合保持在Git之外。后三者在仓库中均有紧凑后验或JSON/图表摘要，不作为运行时
基线的唯一副本。
