# 结果目录

本目录只维护当前数据协议的正式结果，以及仍用于论文对比的独立世界模型结果。

| 目录 | 内容 | 状态 |
| --- | --- | --- |
| `highd_natural_driving_evt/` | 96,055 条清洗后自然驾驶序列与 SEI-EVT 标定 | 当前正式数据与风险标尺 |
| `highd_natural_driving_flow/` | 全量场景条件 Flow：`p(M)p(C0\|M)p(K\|C0,M)` | 当前正式模型与结果 |
| `highd_shared_training_data/` | 150 状态点、149 转移的共享序列缓存及确定性 cohort | 当前共享数据 |
| `background_diffusion/` | 118 维条件长时程扩散模型 | 当前正式模型与结果 |
| `hierarchical_world_model/` | 扩散引导 HiQR 闭环世界模型的三目标评价 | 当前主模型；干预门槛为 partial |
| `legacy/world_model/` | 重命名前的分层模型工件 | 仅作 provenance/regression 参考，禁止进入正式 acceptance |
| `highd_world_model/` | CAT-TopK、FIRM、RAMP、Semi-Markov 独立对比模型 | 保留的冻结协议对比结果 |

`highd_natural_driving_evt/` 中的 tail-context CSV 和 GIF、Flow 采样 NPZ、共享序列数组及
checkpoint 都可由对应脚本再生成；仓库只把必要的契约、指标、图表和 manifest 视为论文
证据。任何结果引用都应先运行相应模块的 `verify_*` 入口，不应把旧数据协议的数字混入当前
主模型结论。
