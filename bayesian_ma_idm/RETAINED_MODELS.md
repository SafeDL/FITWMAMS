# 保留模型与世界模型映射

## 对应目标文档

当前目录只对应 [`doc/01_Bayesian_MA_IDM.md`](../doc/01_Bayesian_MA_IDM.md)：

| 目标文档模型 ID | 此目录实现 | 是否用于世界模型 |
|---|---|---|
| `bayesian_idm` | 层次 B-IDM，IID 残差 | 仅作必要基线 |
| `ma_idm` | 层次 MA-IDM，联合参数抽样 + SE-GP 持续残差 | 是，普通跟驰随机驾驶人 |
| `dynamic_idm` | 方案02，未在此处实现 | 否 |
| `multi_regime_bidm` | 方案03，未在此处实现 | 否 |
| `active_inference_*` | 方案04，未在此处实现 | 否 |

## 最终保留的 MA-IDM 配置

`population` 是无历史新驾驶人的忠实论文配置；`class_conditioned` 与 `personalized` 是选择条件不同的运行方式，而不是额外论文模型。`calibrated_intervals` 是位置区间后处理层，明确不改变 MA-IDM 的动作过程。

| 配置 | 适用条件 | 3 s / 5 s position RMSE | 3 s / 5 s CRPS | 3 s / 5 s 90% coverage |
|---|---|---:|---:|---:|
| `population` | 新驾驶人，无个人历史 | 0.250 / 0.837 | 0.134 / 0.149 | 0.632 / 0.712 |
| `class_conditioned` | 已知 Car/Truck | 0.220 / 0.742 | 0.127 / 0.141 | 0.617 / 0.704 |
| `personalized` | 有5秒因果动作前缀 | 0.146 / 0.523 | 0.104 / 0.120 | 0.574 / 0.664 |
| `calibrated_intervals` | 无历史新驾驶人且需要可信位置带 | 0.280 / 0.911 | 0.142 / 0.154 | 0.942 / 0.960 |

覆盖率不足0.90表示原始人口/类别/前缀 MA-IDM 的不确定性仍欠分散，不能称“已概率校准”。最后一行使用 training-recording 的早期60%拟合、后期训练锚点按 gap 分组学习区间余量；其 OOF 位置覆盖通过0.90门槛，但带宽增至0.835 / 3.394 m。

## 最终证据布局

```text
evidence/
  paper_reference/         原论文20-pair NUTS、表II类比、图5--10类比
  highd_cohort/            251-pair、25 Hz 因果数据工件和选择审计
  population/              B-IDM 与原始人口 MA-IDM 的最终OOF结果
  class_conditioned/       最佳未知驾驶人点预测扩展
  personalized/            最佳有历史前缀的 MA-IDM 结果
  calibrated_intervals/    gap-regime 严格OOF位置区间层
  figures/                 上述最终结果的图表
```

作者20-pair NUTS MA-IDM 已达到 R-hat max 1.0021、0 divergence。论文没有做当前这种 recording-held-out 的未知驾驶人校准评测，因此其收敛不等同于 coverage 通过。
