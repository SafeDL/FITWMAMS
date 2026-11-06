# 场景条件 Flow 结果

`checkpoints/`、`reports/`、`figures/` 和 `samples/` 保存长尾分布结果，`natural/` 保存自然交通分布模型。两者使用相同 Flow 结构，但分别拟合、分别保存权重及统计。

长尾模型使用全部 2,964 条 EVT 尾部事件拟合 `p(M) p(C0|M) p(K|C0,M)`。车辆尺寸 `D` 按车辆存在模式从经验分布采样，不参与 Flow 训练。

| 文件或目录 | 内容 |
| --- | --- |
| `checkpoints/scenario_flow.pt` | 全部尾部事件拟合的正式权重 |
| `reports/training_summary.json`、`reports/training_history.csv` | 训练设置与记录 |
| [evaluation_summary.json](reports/evaluation_summary.json) | 目标分布拟合、对照方法与物理约束检查 |
| [cross_validation.json](reports/cross_validation.json) | 按录像分组的五折评测，与最终全量拟合分开 |
| `samples/` | 固定种子生成的场景条件 |
| [figures/](figures/) | 拟合与评测 PNG |

当前样本分布指标优于 Gaussian 和 GMM-8，但五折密度似然并未领先；`D` 的统计衡量经验重采样，不是 Flow 拟合精度。具体数值以报告为准。

输入放在共享数据目录，运行入口见 [normalizing_flow](../../normalizing_flow/README.md)。

`natural/` 使用 Train＋Val 的全部 85,904 个自然片段，不做 EVT 筛选，训练 20 轮；权重位于 `checkpoints/scenario_flow.pt`，数据 schema、训练摘要和历史记录集中在 `reports/`。它构成自然生成分布 $Q_{\mathrm{natural}}$ 的初态与结点分布，结合冻结 Diffusion 和世界模型用于自然生成评价及能力保持材料。Test 不参与这个 Flow 的拟合。

自然 Flow 入口为 `normalizing_flow/scripts/train_natural.py`，配置为 `normalizing_flow/natural_config.yaml`；与长尾检查点互不覆盖；小型 JSON／CSV 与 PNG 纳入 Git，权重通过链接访问 DATA 模型存储。
