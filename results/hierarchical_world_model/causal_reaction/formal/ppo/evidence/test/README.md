# 全量 highD test 图表阅读说明

统计母体为 record-isolated highD test 的 10,151 条序列中，1,014 条有效、同车道、接近中的 same-rear 场景。每一个点均在 `HighwayEnvClosedLoopWorld` 中，以相同初态、冻结 soft plan、随机流和 ADS 制动命令运行 A0--A3。

## `causal_response_evidence.png`

- 左上：横轴是已执行 ADS 制动剂量（2/4/6/8 m/s²），纵轴是受影响后车相对冻结 HiQR 的额外制动。A0 恒为零；A1/A2/A3 越高表示制动剂量越充分，并不单独表示更自然。
- 右上：同一条件下 ego 或 same-rear 任一碰撞的序列比例。曲线越低越好；A2/A3 在强制动条件下降至约 1.18%，但这不是“后车碰撞率”本身。
- 左下：首次有效后车制动相对于*已观测* ADS 制动的延迟，单位秒。A0 的负值表示没有有效响应；A1--A3 约 0.23--0.38 s，符合不提前响应的因果边界。
- 右下：无关车辆相对 HiQR 的最大修正幅度。四臂均为零，说明当前实现只允许 same-rear 槽位获得控制权。

## 碰撞率应如何解读

以 -8 m/s²、1.0 s 为例：A0 联合碰撞率 56.71%；A1 为 2.47%；A2/A3 为 1.18%。A2 的 1.18% 中，same-rear 碰撞为 0.296%，ego 前向碰撞为 0.887%，两类有少量重叠。后者不是当前“纵向后车修正器”能直接消除的失败。

## 自然性图

`conditional_human_distribution_evidence.png` 将反事实后车动作与 held-out highD same-rear 样本按已实现 TTC 分箱比较。它只能作为分布约束诊断：TTC 0--2 s 的 highD 匹配样本不足，不能作显著性结论；目前 A3 的 KL 也没有低于 A2。因此目前不能宣称 GAIL 已提高自然性。

## 事实重建边界

历史主干报告约 4 cm 的指标来自模型内部 rollout。`evaluation/highwayenv_factual_reconstruction_test.json` 则是同一类 1,014 条场景在真实 HighwayEnv 闭环中的单独审计：all-background ADE=1.091 m，same-rear ADE=1.135 m。两者不能混用；当前项目尚未证明 HighwayEnv 桥接下的“厘米级事实重建”。
