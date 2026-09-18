# Multi-regime B-IDM 最终审计

## 保留协议

- 数据：固定保留的 218-pair highD 队列；队列结构记录在 `evidence/dataset/dataset_manifest.json`。
- 训练：highD recording 25，共 182 条连续跟驰事件。
- 留出：recording 26/36，共 36 条满足 5 s 前缀和 3 s 预测的事件。
- 时间桥接：5 Hz 驾驶决策、25 Hz ballistic plant。
- 测试 style：仅用已完成的 5 s 前缀匹配训练聚类中心。
- 测试 regime：显式持续时间因果 filter；预测阶段只读取自身模拟状态。
- 随机性：每条 future 固定一套 joint regime parameter draw，动作误差为 regime-specific IID Gaussian。

## Segmentation 边界

论文 §5.2 将 Poisson 写为 observation distribution、Gaussian 写为 duration distribution，与连续 `[s,v,Δv]` 和正整数 duration 的支持集不一致。当前采用固定三状态、对角 Gaussian emission、截断 Poisson duration 的 hard-EM finite HSMM。因此：

- `paper_exact_segmentation=blocked_ambiguity`；
- 离线 Viterbi 标签只用于训练 posterior；
- 在线评测不读取测试未来标签；
- 不把 highD 聚类强行命名为 neutral/timid/aggressive 或紧急状态。

## Posterior 诊断

最终层次NUTS与独立pooled NUTS使用每个 style/regime 最多 400 个离线训练点、4 chains、600 tune 和600 draws。六个拟合均为零 divergence，最大 R-hat 不超过1.0077，最小 bulk ESS 不低于606。采样诊断通过，但层次模型采用独立维度的稳定化先验，不是论文 LKJ 层次后验。

## Held-out 结论

在相同style训练数据、NUTS预算及共享测试创新下，filtered multi-regime B-IDM 的 acceleration CRPS 为0.18161，优于独立pooled的0.19026；position RMSE为0.62962 m，优于pooled的0.66904 m；90% position coverage仍仅为40.44%。因此最终验收为：

```text
accepted_calibrated_stochastic_driver = false
decision = rejected_not_well_calibrated
```

该实现对独立pooled B-IDM复现了论文的定性改善方向，但概率区间校准仍未达到部署门槛；当前不得接入 highway-env 或分层世界模型。unpooled基线和原文HDP-HSMM逐算法复现仍未完成。
