# 随机驾驶人复现总索引

本目录只承担跨模型协议、匹配评测和证据汇总；论文机制、拟合代码与原生结果仍位于各自
模型目录，主项目仿真接入位于 `hierarchical_world_model/src/stochastic_drivers/`。

## 同协议highD结果

共享cohort包含218个长跟驰事件。recording 25的全部182个事件用于训练，recording
26/36的全部36个合格事件用于测试；每个事件使用5秒因果前缀、3秒闭环预测、64个future、
25 Hz plant与5 Hz driver。以下数字可在这一基准内比较：

| 模型 | position RMSE (m) | speed RMSE (m/s) | acceleration CRPS (m/s²) | 90% coverage |
|---|---:|---:|---:|---:|
| B-IDM | 0.76149 | 0.57096 | 0.20208 | 0.8637 |
| MA-IDM | **0.22203** | **0.26495** | **0.13321** | 0.6900 |
| Dynamic-AR(5) | 0.26524 | 0.30814 | 0.14851 | 0.7085 |
| pooled B-IDM（multi基线） | 0.66775 | 0.52990 | 0.19170 | 0.3759 |
| filtered multi-regime | 0.63131 | 0.49751 | 0.18336 | 0.4156 |

MA-IDM在点预测和CRPS上最好；B-IDM区间更宽、覆盖率最高，但仍低于名义90%。
Multi-regime优于同推断预算的pooled基线，却仍严重欠覆盖。测试协议完全相同，但论文特有
推断方法和Stage-B观测预算不同，不能把这张表解释为算法复杂度受控实验。

## 唯一入口

```bash
# 重建匹配评测及统一证据表
python -m driver_reproduction.scripts.run_matched_evaluation
python -m driver_reproduction.scripts.build_scorecard

# 查看主项目允许接入的driver及证据状态
python -m hierarchical_world_model.scripts.stochastic_drivers inventory

# 回归测试
pytest -q driver_reproduction/tests hierarchical_world_model/tests/test_stochastic_drivers.py \
  active_inference_driver/tests bayesian_ma_idm/tests dynamic_ar_idm/tests \
  multi_regime_bidm/tests counterfactual_response/tests
```

`artifacts/matched/` 仅保存匹配评测的train-only后验；`results/driver_reproduction/`保存测试
预测和指标。B/MA的全251-pair仿真后验位于`bayesian_ma_idm/evidence/deployment/`，不得
拿部署后验重算留出指标。
