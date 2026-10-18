# 数据与实验结果

保存当前主线需要的数据、正式权重、评测报告和回放。运行方法见各代码目录，具体指标以原始报告为准。

| 目录 | 内容 |
| --- | --- |
| [highd_shared_training_data/](highd_shared_training_data/README.md) | 场景条件、真实时序与 NPC 标定输入 |
| [evt/](evt/README.md) | 自然片段、风险统计、EVT 模型、尾部清单与原始轨迹 GIF |
| [flow/](flow/README.md) | 场景条件分布权重、固定种子样本与拟合评测 |
| [diffusion/](diffusion/README.md) | 软计划生成权重与真实结点条件下的 Test 重建评测 |
| [hierarchical_world_model/](hierarchical_world_model/README.md) | 发布参数、计划缓存、桥接 Test、干预 GIF 与 NPC 分项证据 |
| [baselines/](baselines/README.md) | 外部对照模型的权重与评测 |
| [driver_reproduction/](driver_reproduction/README.md) | 驾驶行为对照模型的复核报告 |

highD 按录像分为 42／9／9 段 Train／Val／Test，对应 72,771／13,133／10,151 个片段。EVT 在全部 96,055 个片段上拟合，Flow 使用 2,964 条尾部事件；Diffusion、NPC 与桥接按各自报告使用 Train 拟合、Val 选择。

自然回放衡量事实重建，人工干预检查响应机制，生成场景 ADS 评测检验被测算法。当前已完成自然 Test 和针对性干预，最新版生成场景 ADS 全量对照尚未完成。合理 NPC 响应后出现的事故仍有测试价值。

保留当前报告、复现所需输入与必要失败记录；临时缓存不作为结果归档。`figures/` 是各模块已有诊断图，桥接可视化位于 `hierarchical_world_model/bridge/playbacks/`，只生成 GIF。
