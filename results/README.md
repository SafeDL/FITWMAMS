# 结果目录

本目录保存当前方法的运行资产和评测证据，以及紧凑的外部对照摘要。
本目录及其每个一级子目录各有一个 `README.md`，说明结果的用途和来源；更深层目录按需说明，不逐层强制设置。

| 目录 | 内容 | 状态 |
| --- | --- | --- |
| [highd_natural_driving_evt/](highd_natural_driving_evt/README.md) | 96,055 条清洗后自然驾驶序列与全槽位 SEI-EVT | 当前数据来源；风险指标须注明口径 |
| [highd_natural_driving_flow/](highd_natural_driving_flow/README.md) | 全量场景条件 Flow：`p(M)p(C0\|M)p(K\|C0,M)` | 上游生成模型证据；未完成主方案端到端验收 |
| [highd_shared_training_data/](highd_shared_training_data/README.md) | 150 状态点、149 转移的共享序列缓存及确定性 cohort | 当前共享数据 |
| [background_diffusion/](background_diffusion/README.md) | 118 维条件长时程扩散模型 | 上游生成模型证据；未完成主方案端到端验收 |
| [hierarchical_world_model/](hierarchical_world_model/README.md) | 冻结 HiQR 运行资产与单次在线 MA-IDM 闭环评测 | 当前唯一维护的世界模型结果；仅保留同物理条件消融和必要审计 |
| [baselines/](baselines/README.md) | TrafficBots highD 外部基线的权重和验收记录 | 仍被基线评测引用 |
| [driver_reproduction/](driver_reproduction/README.md) | 随机驾驶人共享 cohort 匹配评测和模型 scorecard | 支持 MA-IDM 参数选择 |

全槽位 `highd_natural_driving_evt/` 是当前统一多车评测和 AMS 配置的数据来源。
当前多车评估保留全部六个背景槽位，包括主车同车道后方车辆。
`highd_natural_driving_evt/` 中的 GIF、Flow 采样 NPZ、共享序列数组及
checkpoint 都可由对应脚本再生成；仓库只把必要的契约、指标、图表和 manifest 视为论文
证据。任何结果引用都应先运行相应模块的 `verify_*` 入口，不应把旧数据协议的数字混入当前
主模型结论。

目录容量主要来自 `highd_shared_training_data/` 中约 3.6 GB 的 highD 序列数组。
当前 Flow、Diffusion 和 HiQR 评测按原路径读取这些数组；它们是运行输入，不能当作
无用的历史结果删除。

当前方法的评测以 [`hierarchical_world_model/evaluation/`](hierarchical_world_model/evaluation/README.md)
中的原始报告为准；跨方法的协议边界见
[`doc/HighD_Behavior_World_Model_Evaluation.md`](../doc/HighD_Behavior_World_Model_Evaluation.md)。
条件事实重建、历史 prefix-only 多车模型、场景 Flow 和单车跟驰模型不得组成一个数值排行榜。

随机驾驶人 scorecard 位于
`driver_reproduction/driver_model_scorecard.json`；根目录不再单独放置
跨模型驾驶人比较工件。与当前方法无关的旧 prefix-only 聚合实验已移除；TrafficBots
权重和诊断保存在 `baselines/trafficbots_highd/`。
