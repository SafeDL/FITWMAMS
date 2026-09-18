# 25 Hz Active-Inference 随机驾驶员

这是对 Schumann et al. (2026) *Active inference as a model of collision avoidance behavior in human drivers* 的独立、纵向适配实现。它直接使用项目内 `highD` 25 Hz 车辆对，并不复制作者以非商业许可发布的代码。

模型保留论文的关键机制：75 个 belief particles、30 个 0.2 s 预测步、CEM（100 plans × 10 iterations）、基于 looming 的观测限制、non-reactive target prediction、式 (12)–(13) 的确定性 surprise 累积、阈值 1 时完整重规划、以及 a0 = -0.1 m/s² 的单脚踏板过渡约束。外部车辆 plant 固定为 25 Hz，决策动作严格保持 5 帧。

由于 highD 跟驰任务没有横向道路场景，本目录实施的是 `active_inference_longitudinal_adapted`：`no_steering_adaptation=true`。它不是论文的三场景原生模型。论文自身并未以 highD 作为该人因拟合数据；本项目把 highD 作为外部域的 25 Hz 闭环评测。

## 已复现与边界

`artifacts/native_reproduction/full_torch220/` 是外部 v1.0.0 官方源码的原生后追尾复现：28 个条件 × 32 随机复现 = 896 条轨迹。作者的原始分段线性分析得到 RT MAE `0.02 s`、减速度 MAE `0.48 m/s²`；论文 Source Data 的 full-model 值分别为 `0.02 s`、`0.50 m/s²`。数值门槛审计在 `native_human_fit_audit.json`。

同 seed、同后端的重复轨迹逐数组一致（`same_seed_repeat_audit.json`）。与 OSF 历史 CUDA 归档并非逐点位级相同，见 `historical_osf_trajectory_comparison.json`；不得把这一项表述为通过。

highD 的 304/304 个领导车制动事件均已完成。完整适配模型为 0 次碰撞，关闭证据积累为 146 次；详见 `artifacts/full_highd/highd_protocol_audit.json`。原生小预算消融在 `native_ablation_summary.json`。

独立纵向 adapter 的旧桥接在五个原生制动条件中只有 3/5 满足一个原生 tick（0.2 s）的时序门槛，见 `artifacts/full_highd/original_vs_adapter.json`，故仍不具有人因时序继承声明。

新增 `official_wrapper.py` 以外部方式逐步调用锁定的官方 v1.0.0 POMDP，不复制受限源码。它在一个原生条件的 32 条随机轨迹、60 个决策步中与官方控制、belief、权重最大差均为 0，见 `native_reproduction/full_torch220/official_step_wrapper_audit.json`。在五个原生后追尾条件上，官方动作以 0.2 s 请求、5 个 25 Hz plant tick 保持执行，响应时间差均为一个原生 tick（0.2 s），见 `official_25hz_bridge.json` 和 `official_25hz_native_bridge.png`。

该官方 wrapper 的原生桥接通过不等同于 highD 人因重现：当前 full highD 闭环结果仍是明确标注的 `active_inference_longitudinal_adapted` 评测；highD 缺少完整横向道路/几何语义，不能把它称为官方 2D 模型验证。

从仓库根目录运行：

```bash
python -m active_inference_driver.run --max-events 304 --output active_inference_driver/artifacts/full_highd
pytest -q active_inference_driver/tests
```

正式 highD 输出在 `artifacts/full_highd/`；`artifacts/native_reproduction/` 只保留原生论文复现和审计：

- `highd_evaluation.json`：25 Hz 时钟、CEM 预算、future-action firewall、事件指标与消融；
- `representative_highd_response.png`：论文 Fig. 3 风格的速度、加速度、间距、证据/重规划轨迹；
- `ablation_metrics.png`：full / no-evidence / no-pedal 的 highD 比较。
- `stochastic_response_band.png`：同一 highD 事件的 12 个随机流、10–90% 区间和实测轨迹。
- `original_vs_adapter.json`、`human_metric_reproduction.json`、`planning_budget.json`、`future_action_firewall.json`：25 Hz 适配边界和必要的可审计证据。

评测以真实 highD 领车作为外部 plant（每一 25 Hz 帧只提供当下观测），模型预测只使用当前 belief 与内部采样的非反应式未来；评测器绝不把领车真实未来输入 planner。

共同能力与证据边界见 [`doc/00_Common_Protocol.md`](../doc/00_Common_Protocol.md)。通过原生人因与官方wrapper的能力属于外部锁定源码；独立纵向适配桥接仍失败，不得替代官方模型。

主项目注册表默认只允许官方wrapper；运行时必须提供锁定源码和`Results_following`路径：
`python -m hierarchical_world_model.scripts.stochastic_drivers rollout --model active_inference_official --official-source-dir PATH --official-following-dir PATH`。
