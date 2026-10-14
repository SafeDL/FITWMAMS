# 数据与实验结果

主链：highD 数据处理 → EVT 长尾筛选 → Flow 场景条件采样 → Diffusion 软计划 → HiQR 与 NPC 响应 → highway-env 执行。

## 目录

| 目录 | 保存内容 |
| --- | --- |
| [highd_shared_training_data/](highd_shared_training_data/README.md) | 共享场景条件、真实时序及 NPC 标定输入 |
| [evt/](evt/README.md) | 风险统计、EVT 模型、尾部清单与原始轨迹回放 |
| [flow/](flow/README.md) | 尾部场景条件分布的权重、样本与拟合评测 |
| [diffusion/](diffusion/README.md) | 位置结点条件下的轨迹重建权重与 Test 评测 |
| [hierarchical_world_model/](hierarchical_world_model/README.md) | HiQR、NPC 正式参数、完整 Test 执行与自主交通评测 |
| [baselines/](baselines/README.md) | 外部对照模型的权重与评测 |
| [driver_reproduction/](driver_reproduction/README.md) | 驾驶行为对照模型的复核结果 |

## 数据口径

- highD 按录像划分为 42 / 9 / 9 段 Train / Val / Test，对应 72,771 / 13,133 / 10,151 个共享片段。
- EVT 在全部 96,055 个片段上拟合，Flow 使用筛出的全部 2,964 条尾部事件；Diffusion、HiQR 使用自然驾驶 Train 训练、Val 选择、Test 评估。
- 当前 NPC 仍为 Train 拟合、Val 选择，60 段录像均参与分组评测。

事实重建使用真实位置结点，NPC 自主交通评测不使用未来轨迹，ADS 场景生成则从 Flow 采样开始；三者分别查看对应报告。

各目录的报告保存完整指标，`figures/` 保存 PNG。世界模型 NPC 原速 GIF 位于 `npc_response/animations/`。历史试验与清理备份放在 [archive/](../archive/)，不作为当前版本成绩。
