# 长尾场景条件 Flow 结果

最终模型使用全部 2,964 条 EVT 尾部事件拟合 `p(M) p(C0|M) p(K|C0,M)`。车辆尺寸 `D` 按车辆存在模式从经验分布采样，不参与 Flow 训练。

| 文件或目录 | 内容 |
| --- | --- |
| `checkpoints/scenario_flow.pt` | 全部尾部事件拟合的正式权重 |
| `training_summary.json`、`training_history.csv` | 训练设置与记录 |
| [evaluation_summary.json](evaluation_summary.json) | 目标分布拟合、对照方法与物理约束检查 |
| [cross_validation.json](cross_validation.json) | 按录像分组的五折评测，与最终全量拟合分开 |
| `samples/` | 固定种子生成的场景条件 |
| [figures/](figures/) | 拟合与评测 PNG |

当前样本分布指标优于 Gaussian 和 GMM-8，但五折密度似然并未领先；`D` 的统计衡量经验重采样，不是 Flow 拟合精度。具体数值以报告为准。

输入放在共享数据目录，运行入口见 [normalizing_flow](../../normalizing_flow/README.md)。
