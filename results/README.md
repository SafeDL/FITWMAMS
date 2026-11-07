# 数据、模型资产与实验结果

本目录按论文方法链组织材料。运行方法见对应源码目录，数值、实例身份和成本以各实验的原始报告为准。

| 目录 | 内容 |
| --- | --- |
| [highd_shared_training_data/](highd_shared_training_data/README.md) | 场景条件、真实时序与 NPC 标定输入。 |
| [evt/](evt/README.md) | 自然驾驶风险、EVT 拟合、完整尾部事件清单及诊断图。 |
| [flow/](flow/README.md) | 长尾 Flow 和独立保存的自然 Flow。 |
| [diffusion/](diffusion/README.md) | 软计划生成资产与 Test 重建评价。 |
| [hierarchical_world_model/](hierarchical_world_model/README.md) | 发布参数、桥接与 NPC 响应证据。 |
| [baselines/](baselines/README.md) | 外部交通世界和驾驶人模型的对照报告。 |
| [ads/](ads/README.md) | 当前 DP 安全增强版、搜索对照、修复和最终生成评价。 |

highD 按录像分为 Train／Val／Test。长尾 Flow 使用全部 2,964 个尾部事件，自然 Flow 使用 Train＋Val 的全部 85,904 个自然片段。共享数据、模型与正式运行的实例仍被执行器、训练材料或配对统计引用，应保留其内容和身份。

`ads/runs/` 通过目录链接访问 `/media/hp/DATA/nuplan_workspace/FITWMAMS_results/`；权重集中在同级 `FITWMAMS_models/`。这里的结果属于本工程，nuPlan 城市闭环结果另存目标工作区的 `exp/`。

源码、配置、必要图件和 `ads/reports/` 等紧凑报告纳入 Git；模型字节、大型数组和逐例运行材料按 `.gitignore` 保留在磁盘。Git 中的链接只记录目标路径。过时进度记录、已被最终报告替代的试跑检查和废弃协议输出不保留。
