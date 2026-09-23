# 结果目录

本目录只维护当前数据协议的正式结果，以及仍用于论文对比的独立世界模型结果。

| 目录 | 内容 | 状态 |
| --- | --- | --- |
| `highd_natural_driving_evt/` | 96,055 条清洗后自然驾驶序列与全槽位 SEI-EVT | 训练数据来源；旧评估口径 |
| `highd_natural_driving_evt_same_rear_excluded/` | 相同自然片段排除 `same_rear` 后重新计算的 SEI-EVT | 历史世界模型与现有 AMS 风险标尺 |
| `highd_natural_driving_flow/` | 全量场景条件 Flow：`p(M)p(C0\|M)p(K\|C0,M)` | 当前正式模型与结果 |
| `highd_shared_training_data/` | 150 状态点、149 转移的共享序列缓存及确定性 cohort | 当前共享数据 |
| `background_diffusion/` | 118 维条件长时程扩散模型 | 当前正式模型与结果 |
| `hierarchical_world_model/` | 全槽位事实 HiQR 与未通过验收的 CIH 响应候选 | 事实锚点有效；交互候选已拒绝 |
| `interactive_behavior_world_model/` | 严格因果多车模型的自然、事件、干预和 ADS 闭环测试 | 当前统一交互评测 |
| `driver_reproduction/` | 随机驾驶人共享cohort匹配评测和完整预测样本 | 当前复现模型横向比较 |
| `comparisons/` | 文献约束下的全量 highD 分协议汇总 | 当前跨家族能力总表 |
| `legacy/world_model/` | 重命名前的分层模型工件 | 仅作 provenance/regression 参考，禁止进入正式 acceptance |

全槽位 `highd_natural_driving_evt/` 是当前统一多车评测的数据来源；现有 AMS 仍使用
`highd_natural_driving_evt_same_rear_excluded/` 的历史风险标尺，两种结果不得混合。AMS
切换到全背景车之前，不得与当前多车主表直接比较。
`highd_natural_driving_evt/` 中的 tail-context CSV 和 GIF、Flow 采样 NPZ、共享序列数组及
checkpoint 都可由对应脚本再生成；仓库只把必要的契约、指标、图表和 manifest 视为论文
证据。任何结果引用都应先运行相应模块的 `verify_*` 入口，不应把旧数据协议的数字混入当前
主模型结论。

跨模型引用应优先使用
[`doc/HighD_Behavior_World_Model_Evaluation.md`](../doc/HighD_Behavior_World_Model_Evaluation.md)
及 `comparisons/highd_behavior_world_model_summary.json`。条件事实重建、prefix-only 多车模型、
场景 Flow 和单车跟驰模型不得组成一个数值排行榜。
