# 独立世界模型对比结果

本目录保留四种独立对比模型最后一次完整训练与测试的摘要：CAT-TopK、FIRM、RAMP 和
Semi-Markov。它们不是当前分层模型的旧版本，也没有被分层模型结果替代。

这些模型使用其结果配置中记录的冻结数据与 Flow 协议；协议早于当前
`C0(40)+M(6)+K(72)` 主线，因此数值只能作为对应实验协议下的对比证据，不能与当前
96,055 条数据上的扩散或分层模型指标直接横向比较。当前论文主结果位于：

```text
results/background_diffusion/
results/hierarchical_traffic_world_model/
results/highd_natural_driving_flow/
```

目录内不保留重复 checkpoint、开发期 readiness 分支或同一模型的多个版本结果。
