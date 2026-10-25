# 数据与实验结果

保存当前主线需要的数据、正式权重、评测报告和回放。运行方法见各代码目录，具体指标以原始报告为准。

| 目录 | 内容 |
| --- | --- |
| [highd_shared_training_data/](highd_shared_training_data/README.md) | 场景条件、真实时序与 NPC 标定输入 |
| [evt/](evt/README.md) | 自然片段、风险统计、EVT 模型、尾部清单与原始轨迹 GIF |
| [flow/](flow/README.md) | 场景条件分布权重、固定种子样本与拟合评测 |
| `natural_flow/` | Train＋Val 自然分布的独立 Flow 权重、schema 与训练报告；配置和入口见 [normalizing_flow](../normalizing_flow/README.md) |
| [diffusion/](diffusion/README.md) | 软计划生成权重与真实结点条件下的 Test 重建评测 |
| [hierarchical_world_model/](hierarchical_world_model/README.md) | 发布参数、计划缓存、桥接 Test、干预 GIF 与 NPC 分项证据 |
| [baselines/](baselines/README.md) | 外部对照模型的权重与评测 |
| [driver_reproduction/](driver_reproduction/README.md) | 驾驶行为对照模型的复核报告 |
| [risk_stratified_repair_history/](risk_stratified_repair_history/README.md) | 已暂停的历史 ADS 实验、原始权重、路径迁移索引与可版本化报告快照 |
| [risk_stratified_repair/](risk_stratified_repair/README.md) | DP 三方法、两轮正式比较及三组留出均完成；CaRL 先导未接纳增强；说明和精选报告可版本化，完整运行分目录链接 DATA |
| `risk_stratified_repair_manual_nuplan/` | 人工评价场景、原生配置及交付权重；交接说明见 [manual_nuplan](risk_stratified_repair/reports/manual_nuplan/README.md) |

highD 按录像分为 42／9／9 段 Train／Val／Test，对应 72,771／13,133／10,151 个片段。EVT 在全部 96,055 个片段上拟合，Flow 使用 2,964 条尾部事件；Diffusion、NPC 与桥接按各自报告使用 Train 拟合、Val 选择。

自然回放衡量事实重建，人工干预检查响应机制，生成场景 ADS 评测检验被测算法。当前已完成世界模型自然 Test 与针对性干预，以及 DP 的生成场景 ADS 正式对照；其他算法的扩展状态见 ADS 方案。合理 NPC 响应后出现的事故仍有测试价值。

保留当前报告、复现所需输入与必要失败记录；临时缓存不作为结果归档。`figures/` 是各模块已有诊断图，桥接可视化位于 `hierarchical_world_model/bridge/playbacks/`，只生成 GIF。

新版风险分层测试与能力保持修复使用 `risk_stratified_repair/`，先导配置见 [diffusion_planner_pilot.yaml](../ads/configs/diffusion_planner_pilot.yaml)，正式单模型配置见 [diffusion_planner_experiment.yaml](../ads/configs/diffusion_planner_experiment.yaml)。三组最终留出的 24,000 次配对执行已完成；DP 原始／修复权重已交付，nuPlan 人工评价另存于 `risk_stratified_repair_manual_nuplan/`。

ADS 结果的说明、路径与哈希索引、精选报告快照纳入 Git；完整轨迹、权重、日志和指向 DATA 盘的链接继续留在本地。历史报告快照位于 [reports/](risk_stratified_repair_history/reports/)，保持原始字节和指标；旧路径对应的新位置见 [index.json](risk_stratified_repair_history/index.json)。
