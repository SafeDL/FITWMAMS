# 方案04：Active Inference 碰撞避免——首先复现实证紧急响应，而不是再造避碰奖励

**依据**：Julian F. Schumann et al., *Active inference as a model of collision avoidance behavior in human drivers*, Nature Communications 17, 5009 (2026)，DOI `10.1038/s41467-026-73345-0`，附件25页。  
**模型ID**：`active_inference_original`、`active_inference_longitudinal_adapted`。先读[共同协议](00_Common_Protocol.md)。  
**优先级**：高；五篇附件中与“极端条件下何时响应、选什么动作、怎样执行”直接对应的文献。它仍不是任意极端场景的真实性保证。

## 1. 必须纠正此前的简化解释 [P]

1. PDF13与17式(12)–(13)：证据积累是**确定性**的，无accumulation noise、无decay；随机性主要在信念和候选策略采样。不能默认加OU证据噪声或泄漏因子。
2. 积累的是当前策略的pragmatic value相对最优偏好的差，不只是looming或TTC阈值。
3. 达到阈值1时触发**完整重规划**，不是直接切换到固定最大制动。未达阈值时仍扩展当前policy尾端。
4. 一套手调参数用于全部driver；文中明确因此未覆盖个体差异，反应时间方差偏小。不能声称已从highD学到人口后验。
5. 其agent是真正反馈重规划，但内部预测假设其他车不响应自身动作（PDF14–16）。不是递归多车博弈。
6. 模型使用finite CEM规划能力；增加候选数会改变行为，不能一边大幅缩减计算预算一边称忠实复现。

## 2. 原生机制与参数 [P]

### 感知/预测

视觉角与looming形成观测，含检测阈值；以KDE/GMM Bayesian update更新particle belief。预测其他车时使用噪声自行车模型和norm-conditioned particle filter。检测到对方违背规范后放松规范约束，避免一直相信对方会回到安全状态。

### 决策

\[
G(\pi)=\sum_{\tau=t+1}^{t+H}[-g_{pragm}(\tau)-g_{epist}(\tau)].
\]

\[
\epsilon_t=H\max_o\log p(o)-\sum_{\tau=t+1}^{t+H}g_{pragm}(\tau),
\quad E_t=E_{t-1}+\lambda\epsilon_t.
\]

阈值为1。不可用一个额外TTC shield替代此过程。重规划后的E reset、原policy延展和第一帧执行顺序以官方实现为准，需记录逐tick轨迹。

### 执行

踩踏板切换在`a0=-0.1 m/s²`保持0.2s，限制jerk。CEM选最后轮最优policy，不是抽其平均。动作变量是纵向加速度和**转向角速度**；不是FITWMAMS的yaw rate。

原文表1（PDF18）必须锁定的量：
```yaml
native_dt_s: 0.2
H: 30
belief_particles: 75
cem_iterations: 10
cem_plans: 100
cem_elite_fraction: 0.1
looming_threshold_per_s: 0.00215
norm_prediction_horizon: 20
accumulation_drift: 1.122018454e-6  # 10^(-5.95)
pedal_hold_s: 0.2
coast_acceleration_mps2: -0.1
```

其余感知/偏好/动作限幅参数从原表和官方配置逐项导出，不把FITWMAMS `[-8,4]`当成原论文限制。

## 3. 代码、许可和数据 [C/P]

官方：`https://github.com/tud-hri/Active-Inference-Collision-Avoidance`，复现锁定 `v1.0.0`，对应提交 `56655de845644c45f01ab2898544e55316a3279b`。固定归档：`https://doi.org/10.5281/zenodo.20049511`。数据页：`https://osf.io/gs4bu`；论文另提供Source Data和Supplementary Information。

本次实际读取：
```text
README.md
simulation_rear_end.py
src/utils/simulation.py
```
README确认原生入口：
```text
simulation_rear_end.py / visualization_rear_end.py / Analysis_rear_end.py
simulation_oncoming.py / Analysis_oncoming.py
simulation_side.py / Analysis_side.py
```
核心模块由代码import确认：
```text
src/common/agent.py::POMDPAgent
src/common/encoder.py::Encoder
src/common/decoder.py::Decoder
src/common/dynamics.py::Dynamics/BeliefDynamics
src/common/mpc_discrete.py::CEM
src/common/belief_reward.py::IGEstimator/BeliefReward
```

[P] PDF21写明非商业许可，允许研究/教学/论文benchmark；不是MIT/Apache。复用前读取LICENSE，不将其代码换标许可证后并入FITWMAMS。优先外部依赖包装。

[C] `src/utils/simulation.py`初始化存在 `lf=config['dynamics']['lr']`写法；原生实验前后轴长度若相同可能无影响。迁移到不同车辆几何必须检查。原生复现保留、修正版另标，不静默改变作者结果。

## 4. 复现步骤：先原生，后迁移

### Stage A：官方后碰实验原样运行

先用4个原生条件×8重复作smoke，仅检查可执行。正式采用原文28个条件×32重复=896条，不削减N/M/K/H后宣称复现。场景网格、前车制动时刻与jerk直接从官方脚本导出；源码`t_brake=.6`与论文示例可见动作`.8`的区别要通过采样边界解释，不能手动平移结果迎合图。

保存belief、surprise、E、replan事件、candidate数、选中计划、pedal状态、真实动作。所有内部预测不得读取模拟器未来scheduled ADS动作。

### Stage B：论文指标复核

按PDF17原算法拟合分段线性速度确定brake response time及制动强度。不能拿本项目“连续3帧低于阈值”直接复刻论文指标。

原文front-to-rear拟合域：
- 时间gap `0.9–3.6s`；
- 刹车开始时inverse TTC `0–1.0 s^-1`。

figure3/6之外极小timegap的结果只作为模型预测，原文明确0.5s条件缺相应人类数据（PDF6）。其余两场景用类别Jensen-Shannon divergence及反应时间Wasserstein距离；保留作者提取规则。

### Stage C：FITWMAMS纵向适配

先从官方agent抽出observe/update/predict/plan/execute接口，用一辆实际ADS前车作为target，NPC作为原文ego。官方“ego”与本文ADS不能混淆。

**仅纵向版本是新适配**：禁止转向会改变候选集合；它可能在原模型可转向避碰时发生碰撞，这是限制的结果，不是对原模型的否证。报告 `no_steering_adaptation=true`。

原生0.2s决策、25Hz外部plant保持5个tick；踏板0.2s仍是5个tick，不能缩成1个25Hz tick。保持6s内部H物理时间。增加观测频率或重规划频率需要重新做人因验证。

### Stage D：再考虑与HiQR组合

不直接把完整Active-Inference输出做差后假定性质保留。先将其作为`absolute_driver`评测，再测试已有事实意图作为初始policy/偏好能否使用；**这也是适配，不是原论文**。

其预测需要的是当前belief和可行未来，而不是follower真实未来。HiQR计划可以作候选/意图先验，不能优先级高到阻止必要重规划。差分版只作探索，必须检查共同协议“紧急动作被抵消”的反例。

## 5. 代码任务

```text
active_inference_driver/model.py                  # 独立纵向适配（未通过bridge）
active_inference_driver/official_wrapper.py       # 锁定官方v1.0.0的外部wrapper
active_inference_driver/scripts/                  # 原生复现和25 Hz审计
hierarchical_world_model/src/stochastic_drivers/  # 主项目注册与默认拒绝门
```

状态至少包括：belief粒子与权重、current policy、E、踏板切换计时、前一真实执行动作、steering angle/native clock、CEM与感知随机流。`CEM count`和time budget不能随simulator负载自动改变，它们属于驾驶员行为参数。

关键边界：
```python
obs = perception_model(realized_state, perception_noise)
belief = update_belief(previous_belief, obs, previous_executed_action)
future_target_belief = predict_other(belief, prediction_noise) # 无真实未来
plan, evidence = planner_step(belief, current_plan, evidence, cem_noise)
request = execute_first_with_pedal_state(plan, driver_state)
```

随包`surprise_step()`只对应原文12–13式，用来测试无额外噪声/无decay；不是完整agent。

## 6. 验收：保持人类特征，不追求最强避碰

### 原生复现门槛 [A]

- 官方同seed重复结果一致；包装前后在同后端同精度轨迹最大差≤`1e-5`。
- 小网格单测证明完整重规划仅在阈值到达时发生；未触发仍延展policy；踏板hold符合0.2s。
- 用Source Data/官方归档重算figure3d/e与figure6 full model指标，在原生条件下复核。原图近似`I_delay≈0.03s, I_a≈0.50m/s²`仅作转录核验参考；最终以数值Source Data为准。
- [A]复现实验与官方原始模型指标差的bootstrap95%区间应落在预注册容差：反应时间拟合误差增量≤0.10s、制动拟合误差增量≤0.20m/s²。此容差是工程复现判断，不是普适人类误差标准。
- Source Data/原始实验输出未下载成功时，`quantitative_human_replication=blocked_missing_data`；可完成代码行为复现，但不能宣称人类实证重现。

### 必要机制检查

只做两个小预算消融：`no evidence accumulation`与`no pedal constraint`。前者应明显影响响应时机，后者约移除0.2s执行延迟；与原figure6定性比较。无需全七项大规模重跑。

### 迁移门槛

同原生可比的后碰条件上比较原生agent与25Hz适配版；反应时间差≤一个原生tick（0.2s），制动曲线、峰值和pedal trace不应存在系统性桥接偏差。超过则停止，不去调human参数补偿bridge bug。

真实highD自然事件另报75帧ES、误触发、速度/gap、恢复等；原生人因通过不能代替常态跟驰有效性，反之亦然。迁移后的强制动超出论文人因覆盖范围仍属外推。

### 不能作为成功条件

零碰撞、最大制动力更大、反应更快都不是单独成功标准。原论文有意保留受感知、规范期待和有限规划导致的失败；计算预算提高可能反而更不像人。

## 7. 运行与成本边界

原生已有命令（在外部官方目录）：
```bash
conda activate tread
# 先审计requirement.txt；不直接覆盖已有Torch
python simulation_rear_end.py
python Analysis_rear_end.py
```
完整scripts可能含全部消融/循环，先查看底部实验选择器；新增wrapper必须只选择声明的full model，不能偷偷改CEM参数。

主项目接入：
```bash
python -m hierarchical_world_model.scripts.stochastic_drivers inventory
python -m hierarchical_world_model.scripts.stochastic_drivers rollout \
  --model active_inference_official \
  --official-source-dir PATH --official-following-dir PATH
```

[A]首轮单卡GPU预算8h：先profile 32条，估算896条总成本；若超预算先保存阻塞报告，不降低模型推理预算冒充完成。可通过batching提高速度，但要验证随机流不随batch变化。横向和路口完整复现是可选后续，不作为当前纵向接入前置条件。

最终必须保留 `original_vs_adapter.json`、`human_metric_reproduction.json`、`planning_budget.json`、`future_action_firewall.json`。只有源模型、适配模型分别过关，才能说“该适用域有紧急人因证据支持”。
