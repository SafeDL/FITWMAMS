# 数据与实验结果

按研究组件集中保存数据、模型和实验结果；方法设计见[文档索引](../doc/README.md)，运行入口见对应代码目录，指标以原始报告为准。

| 目录 | 内容 |
| --- | --- |
| [highd_shared_training_data/](highd_shared_training_data/README.md) | highD 场景条件、真实时序与 NPC 标定输入 |
| [evt/](evt/README.md) | 自然片段、安全评分、EVT 模型与 2,964 条尾部事件 |
| [flow/](flow/README.md) | 长尾 Flow；`natural/` 保存自然交通 Flow，分别使用独立权重和统计 |
| [diffusion/](diffusion/README.md) | 软计划生成权重与 Test 重建评测 |
| [hierarchical_world_model/](hierarchical_world_model/README.md) | 世界模型参数、桥接 Test、干预回放与 NPC 响应证据 |
| [baselines/](baselines/README.md) | TrafficBots 与驾驶人模型的对照结果 |
| [ads/](ads/README.md) | 当前 DP 安全增强版、搜索对照、全量修复与最终评价 |

highD 按录像分为 42／9／9 段 Train／Val／Test，对应 72,771／13,133／10,151 个片段。长尾 Flow 使用全部 2,964 条 EVT 尾部事件；自然 Flow 使用 Train＋Val 的 85,904 个自然片段，不做 EVT 筛选。两者供不同生成分布使用，不能互换检查点。

ADS 源码、配置、冻结源码、紧凑报告及原生交付元数据纳入 Git；大型逐例数据、训练日志、学习状态和模型权重保留在磁盘，不纳入 Git。正式运行通过目录链接访问 DATA 盘，模型实体位于 DATA 的 `FITWMAMS_models/`。实验版本与目录对应关系见 [ADS 结果说明](ads/README.md)。

当前交付为 Diffusion Planner Safety Enhanced，已淘汰候选不再续跑。研究状态见[修复安全分析](../doc/repair_safety_analysis.md)。
