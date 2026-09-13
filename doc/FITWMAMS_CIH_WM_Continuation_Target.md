# FITWMAMS / CIH-WM：基于当前实现的续建与验收目标

> **文档用途**：交给 Codex 执行的定向修复与单候选训练任务书。  
> **核对仓库**：`https://github.com/SafeDL/FITWMAMS`。  
> **核对提交**：`82e71e7ceabc993db307179301b296be019e6187`，提交时间 2026-09-13 09:49:35 UTC。  
> **执行环境**：先执行 `conda activate tread`；现有设备按 RTX 4090 D 单卡预算考虑，实际设备须重新检测。  
> **继承对象**：当前 CIH-WM，而不是旧的 `a2_human_calibration` 脚本链。  
> **保留内容**：冻结事实模型、Flow/Diffusion 权重、由 A2 继承的完整响应先验、当前运动学执行器、直接跟驰作用域。  
> **实施性质**：本文件给出待实现修改，不表示这些修改已经运行、已经有效或已经通过验证。

---

## 0. 结论与范围

**可以在当前代码基础上继续完成既定目标，而且应当继续使用当前 CIH-WM，不应回退或另建世界模型。**

但不能直接执行上一份《FITWMAMS_Hierarchical_Behavior_World_Model_Causal_Response_Target.md》。当前代码已经更换了正式执行入口、修正了部分参考轨迹来源、继承了完整响应先验，并明确关闭了尚无证据的横向及二级传播。旧文档中的部分路径、缺陷判断和实现建议已经失效。

本轮保留三个研究重点：

1. **事实锚定**：在统一信息预算和全背景车口径下，尽量保持冻结事实层的重建能力；不能通过关闭被评测车辆或切换控制器获得虚假非劣性。
2. **完整响应序列的概率校准**：真实事件监督必须对应真实历史；闭环分布优化必须评价从响应启动到持续与恢复的序列，而不是只评价平均加速度。
3. **数据外交通变化的合理响应**：在没有人类标签的工况中，使用有适用范围的机制约束与闭环成对检查；不能把规则动作当作人类真值，也不能用“零碰撞”代替真实性。

本轮只开发一个修复后的 `causal_influence_response` 候选。新方法名称继续使用 **CIH-WM**。不新增 personalized-IDM 网络、GAIL、另一套 diffusion policy、凸优化控制器或大型自博弈。MC/AMS 的扩量与新型搜索算法不属于本轮任务。

### 0.1 证据等级

全文采用以下区分：

- **已核对事实**：在指定提交的源代码或结果文件中可以直接查到，使用 `[Sxx]` 标注，源码索引见附录 A。
- **机制分析**：由已核对实现推导的可能影响，不冒充已完成消融实验的因果结论。
- **拟修改设计**：本轮应实现或验证的技术方案。
- **尚无证据**：不作通过或成功判断；不能用注释、命名、训练轮数代替实验。

本次已阅读关键训练、控制器、评测、规划、影响图、配置及诊断工件，并执行了四项独立的公式/状态递推检查。**未加载用户本机训练权重和完整 highD 缓存，未运行仓库全量测试、GPU 训练或正式评测。** 四项检查不是仓库测试通过证明，见附录 C。

---

## 1. 当前实现应如何继承

### 1.1 已经具备，应保留的能力

| 能力 | 当前实现 | 续建处理 |
|---|---|---|
| 明确的分层方法入口 | `cih_model.py` 组装冻结事实层、影响路由和响应策略 | 保留，不另建 A2 项目目录 |
| 同一训练/评测执行后端 | `cih_training.py` 调用 `evaluation.rollout`，使用 `KinematicTrafficDynamics` | 保留，不强制迁回 HighwayEnv |
| 完整的冻结响应先验 | `FrozenResponsePrior` 严格加载 prior 的整个 `state_dict`，参数 `requires_grad=False` | 保留全部 actor、输出头及噪声参数，不只复制 hidden layers |
| 当前状态的父子跟驰关系 | `_dynamic_idm_reference()` 使用 `influence_parent` 与 `idm_pair_reference()` | 保留 parent-aware 机制，不回退为固定 ego/槽位错配 |
| 取消直接 highD 未来补全 | `complete_endogenous_response_plans()` 对空计划只读取 anchor 状态外推 | 保留；不要把已经删除的旧函数再次列作当前 bug |
| 明确限制研究范围 | `enable_secondary=false`，lateral response disabled | 保留，不用本轮顺手拓展多车链式或横向策略 |
| 部分有效的闭环响应 | 已提交诊断中探针前动作差为 0，规则方向一致率约 88.5% | 作为未通过候选的诊断证据，不改写成已接受结果 |
| 成对对比只用于训练/评测 | 当前主策略不使用 nominal shadow action 覆盖输出 | 保留，不把影子世界重新塞回在线策略 |

来源：`README.md`、`cih_model.py`、`cih_training.py`、`reaction_controller.py`、`planner.py`。[S01–S08]

响应先验工件的 SHA-256 在已提交 manifest 中为：

```text
ffa9f934b6b2236c2ab75d007c84958076b2bfea1851644bfef681d40e42a4d5
```

事实模型 SHA-256 为：

```text
8f49d5f203f0a0f2ebf26890daf3ba5b9aa33d65864aaf58a6a223e6673f5cb9
```

这是保留既有权重的可核查依据。运行时仍须重新计算本机文件哈希；路径相同不代表字节相同。[S10]

### 1.2 最新诊断不能混用成“当前全槽位正式结果”

已提交的 `candidate_unaccepted/diagnostic_policy_validation_256.json` 使用 **256 条 validation 序列**，不是完整 validation，也不是 test。[S09]

| 该诊断中的口径 | 冻结事实层 | CIH-WM / PPO 后 |
|---|---:|---:|
| 历史排除 same_rear 的 ADE | 0.036800 m | 0.036906 m |
| 全背景车 ADE | 0.119025 m | 0.130994 m |
| same_rear ADE | 1.156804 m | 1.318879 m |
| same_rear FDE | 3.867151 m | 3.712783 m |
| same_rear P95 位移误差 | 5.056538 m | 4.956659 m |
| 成对探针执行动作方向一致率 | 不适用 | 0.885250 |
| 探针前最大动作差 | 不适用 | 0 |

对应 supervised 候选的方向一致率约为 **0.887862**，全背景车 ADE 为 **0.134141 m**。这些点估计不支持“PPO 已经明显增强因果方向”的结论，也没有提供人类序列概率校准的完整验证。[S09–S11]

**重要限制**：当前 `world_model.yaml` 和 CIH 配置已经改成全背景车口径，但上述结果仍带历史 `same_rear` 掩码字段和旧 schema。README 明确说明全背景车最新口径尚未重新完成评测。[S01,S03,S09]

因此，本轮不能：

- 将历史 0.040227 m 的全测试 ADE 写成全槽位响应模型精度；
- 将 same_rear 1.16 m 的误差简单归因于新响应器——冻结事实层在该旧诊断下本身就存在这一误差；
- 将“当前 YAML 已经不屏蔽后车”解释为“现有权重在新口径下已经验证”；
- 直接将旧 schema 字符串改名后把旧工件视作新配置的评测结果。

### 1.3 当前最可能的瓶颈

本轮首先检查下面三类问题，而不是增加 PPO 更新数：

1. **底座口径与行为保护对象不明确**：后车的事实能力未在最新全槽位协议下确认；当前关闭校准 gate 保护的是 A2 proposal，不是 HiQR。
2. **训练对象不一致**：监督缓存是模型闭环状态，却使用原日志动作作逐帧标签；监督映射与实际执行参数不同。
3. **统计和执行证据不闭合**：训练有 Energy Score，但正式验收未包含该指标；PPO 的随机动作、确定性 gate、release 掩码及噪声契约尚未严格统一。

这些是由代码支持的优先排查方向，不是声称已经量化证明了每个问题对误差的贡献比例。

---

## 2. 对上一版目标的明确修订

| 上一版要求/判断 | 本版处理 | 原因 |
|---|---|---|
| HighwayEnv 必须是唯一正式执行器 | **撤销**；正式后端沿用当前 `evaluation.rollout + model.dynamics` | 当前训练和评测已经统一，不能再次制造后端差异 |
| 当前 same_rear 直接用 highD 未来补全 | **不再作为当前事实** | 已改为 anchor 外推；但 K 条件来源仍需另外审核 |
| 再实现 FrozenA2ResponsePrior | **不重复建设** | 当前 `FrozenResponsePrior` 已完整继承响应权重 |
| 在响应触发时才切换为恒速计划，防止看未来 | **撤销** | 触发前未来计划可能已进入 hidden state；在触发时换参考也会混入新的分布偏移 |
| 新增名义影子动作/成对规则覆盖 | **禁止恢复** | 成对分支是离线证据，不应成为在线动作真值 |
| `5/10/25` 加权差分 reward 一定更正确 | **撤销该无条件建议** | `.25S5 + .35(S10−S5) + .40(S25−S10)` 等于 `−.10S5−.05S10+.40S25`，会给早期 score 负权重 |
| 25 帧已经代表完整制动及恢复 | **修正** | 25 Hz 下只覆盖 1 s；应使用现有可用恢复窗口并明确截尾 |
| 只保留命令校准 jerk 即证明最终动作连续 | **修正** | 先验动作和最终裁剪仍可引入执行 jerk，必须分别报告 |
| 不惩罚碰撞即可保证风险无偏 | **撤销** | 这是训练目标选择，不是风险无偏定理 |
| 任何 NPC–NPC 碰撞都是仿真错误 | **撤销** | 交通冲击可能真实传播；必须结合局部性和轨迹诊断，而非按碰撞对象自动定性 |

**执行优先级**：本文件覆盖旧目标中冲突的工程要求。历史文档保留为过程记录，不由 Codex 自动删除。

---

## 3. 必须先修的实现问题与证据

### D01｜监督数据不是 teacher forcing：逐帧标签条件错位（P0）

**已核对代码**：`cih_training.response_teacher_cache()` 先用当前控制器完成 `response_policy_trace()`，然后将该 rollout 的 `features / nominal / previous calibration` 与 highD 的同一时间索引加速度配对。函数没有把每步状态和控制器上下文重建为真实历史。[S06]

因此实际数据是：

\[
(H_t^{model}, a_t^{log}),
\]

而不是：

\[
(H_t^{log}, a_t^{log}).
\]

日志动作在模型已偏离的状态下不再是已观察到的人类动作。这可以作为明示的序列追踪代理目标，但不能当成真实条件动作监督。

**必须修改**：新增纯 logged-context 缓存，按第 7 节重建真实历史上的冻结事实层/先验特征。模型自生成状态只用于闭环分布 score 或机制辅助，不再配原日志逐帧 human target。

### D02｜监督动作映射与在线执行参数不同（P0）

`response_supervised_loss()` 调用 `map_calibration_tensors()` 时，没有传入 controller 的 `correction_jerk_limit_mps3`、scale/residual 参数。映射函数默认 jerk 为 12，而当前 CIH 配置和在线 `forward()` 使用 24。因此同一个 raw action 与 previous command 在监督阶段最多变化 0.48 m/s²，在线可以变化 0.96 m/s²。[S03,S04,S06]

**必须修改**：将所有参数封装为不可变 `ExecutionSpec`，监督、PPO 辅助重算和在线执行使用同一个入口。禁止复制默认常数。单测需要比较的不只是 loss 值，而是两条路径的逐元素最终动作。

### D03｜“关闭校准 gate”不等于事实锚定（P0）

当前映射的定义是：

\[
 a^{final}\approx a^{prior}+g\{0.75\tanh(u_s)\delta^{prior}+2\tanh(u_r)\}.
\]

当 `event_gate=0`，输出保留 `nominal=prior proposal`，不返回事实基座。源码注释明确说明了这一语义。[S04]

因此，把 pre-event/non-event 的 gate 训练成 0，不会自动消除 A2 在普通跟驰中已施加的额外制动。`frame >= logged onset` 的二分类标签，也不等价于“该时刻应该校准多少”。

**必须修改**：保留两维 actor 和完整 A2 prior，使用第 5 节的可表达“保留 prior / 抑制 prior / 返回事实基座”的动作映射。不再用事件标签 gate 掩盖 prior 自身偏差。

### D04｜确定性 gate 与 PPO 概率不在同一计算图（P0）

`event_gate` 影响最终动作，却不参与 `distribution_and_value()` 与 `evaluate_raw_action()` 返回的 Normal 密度。当前 PPO 主损失无法直接训练这组 gate 权重；若以后在 PPO epochs 中用额外损失更新 gate，动作映射改变也不在 raw policy ratio/KL 中体现。[S04,S06]

**必须修改**：本轮不引入第三个随机头。将独立 `event_gate` 退出新候选的运行映射，权重保留在历史 checkpoint；控制选择由已有两维随机 actor 完成。其他确定性执行参数在一次 PPO rollout/epochs 内完全冻结。

这不是说 raw-action PPO 本身错误：在给定状态下使用固定、可多对一的动作映射时，raw policy ratio 仍然合法。错误在于不加说明地将额外可训练的映射变量遗漏在策略定义之外。

### D05｜恢复状态机可能无法进入 recovery（P0）

`influence_graph.update()` 中：

- `safe_frames` 仅在 `old.phase == 1` 且冲突已消失时递增；
- 第一个无冲突帧若未达到 `stable_release_frames=13`，`phase` 立即变为 0；
- 下一帧因 `old.phase != 1`，`safe_frames` 又归零。

在持续开距、冲突刚解除的递推中，这会使 13 帧稳定释放条件无法累计，随后只能依赖校准命令的外部 release，而非预期的恢复状态。[S05]

**必须修改**：区分 `engaged / clearing / recovering / idle`，或者在 clearing 期间保持旧 phase 和父子关系，直到累计达到阈值。不能第一帧清零。恢复时依然只看已实现状态，不能看日志事件结束标签。

### D06｜policy-active 与 deterministic release 掩码混用（P0）

输出 `active=active | release`，但存储的 `log_prob` 乘的是 `proposal.active`。PPO 用 `buffer.mask & buffer.active` 选择样本，可能把没有随机动作作用的 release 帧纳入，old log-prob 与重算概率不一致。[S04,S06]

**必须修改**：输出 `policy_active`、`execution_active`、`release_active` 三个不同字段。纯确定性 release 不进入 PPO ratio。正常 recovery 若由 actor 决定动作，则应明确属于 `policy_active`。

### D07｜当前执行接口仍携带未执行 ego 命令（P0 接口风险）

`evaluation.rollout()` 在物理更新前，把 `ego_action=ego_block[:,0]`、由本帧命令变化计算的 `reaction_enabled` 传入 controller context。[S07]

当前 `CausalInfluenceResponsePolicy` 的主要 authority 路径依赖 influence graph，未发现它使用这些字段直接产生当前动作。因此，**不能据此断言现有候选已经通过该字段作弊**。但接口与“只使用已实现历史”的声明不一致，容易在后续改动中泄漏。

**必须修改**：CIH 在线上下文移除 pending ego、nominal ego 和 nominal action 字段；本帧 NPC 动作计算完成后，ego 与 NPC 动作共同进入动力学。历史 ego 命令只能在下一决策边界成为观测。

### D08｜随机世界契约尚未贯通正式 rollout（P0）

`evaluation.rollout()` 的 `motion_seed` 只控制部分 scene/agent noise；控制器未收到完整的显式 prior/calibrator noise，因此会调用全局采样。`response_policy_trace()` 传 `motion_seed=None`，使事实层走 deterministic 路径，但 controller 可以随机。[S06,S07]

这能进行某一种条件策略训练，却不能因此宣称完整随机世界已经实现：

```text
same world variables + same ADS => same trajectory
```

**必须修改**：将现有 `WorldExogenousState` 的所有被消费块接入正式 rollout；记录哪些块固定、哪些块采样。按 `(scene key, future key, agent key, time, block)` 固定索引，不能因 batch/chunk 改变而重排随机样本。第 6 节规定分叉测试。

### D09｜当前 ES 是有限样本经验评分，不能无条件声称优化总体 proper score（P1）

`event_energy_score()` 使用 `K²` 分母的经验 V-statistic；当前 centered LOO 又使用删去一个样本后的 score。它不是简单的“符号写反”，但其期望目标、有限 K 偏差和归一化都需要说明。[S12]

**必须修改**：训练使用第 8 节的 fair/U-statistic Energy Score 和可检查的 score-function 估计器；历史 raw V-score保留作兼容报告，不覆盖历史数值。不得声称 PPO clipping、GAE 或标准化后的整体算法是无偏梯度。

### D10｜训练了分布目标，正式验收却没有验收分布目标（P0）

`evaluate_cih_world_model.py` 当前 acceptance 主要由位置非劣、规则方向、jerk 与 dose monotonic 决定。它没有在正式 report 中计算 natural-event Energy Score，也未对 final 与 supervised 做人类序列分布的统计比较。[S13]

**必须修改**：正式 validation 必须包含多次随机未来的 25 帧兼容 ES、75 帧主 ES、响应诊断以及 recording-cluster 配对区间。没有这些结果不能称为“人类响应校准通过”。

### D11｜校准命令平滑与执行动作平滑被混为一谈（P1）

当前保留 jerk-limited `calibration` 命令，而最终动作再次 clamp；先验 nominal 也随时间变化。仅对命令 `calibration` 差分通过，不能证明实际 `final` 或 `final−factual_base` 的 jerk 通过。[S04,S13]

**必须修改**：分别存储命令、执行后的校准、总响应修正、最终动作及每一级差分。约束和 report 必须使用同一个定义，不能换日志字段让 gate 通过。

### D12｜成对方向/单调性分母有条件选择偏差（P1）

当前方向检查只统计已激活且动作差不为零的槽位帧；不同剂量的均值采用不同样本集合。增加制动剂量后“平均 effect 变大”，不等于同一批事件的剂量响应单调。`causal_contrast_cache()` 默认取排序靠前的有限 rows，并要求所有剂量都有有效方向后才留下帧，覆盖范围也受限制。[S06,S13]

**必须修改**：以探针前几何条件冻结基本事件集；no-response 单独列为 miss，不能自动从主要分母消失；剂量单调性在同一事件、同一窗口、同一父子关系上比较。固定子集按 recording 分层，不反复只取头部 rows。

### D13｜正式验收与工件一致性仍有缺口（P1）

- 当前 test 绑定的是 checkpoint 路径，不是 checkpoint 内容哈希。
- 训练来源校验主要看 manifest 声明的轮数，不等于验证实际执行轮数或配置一致性。
- 旧结果是带 `_v2` 的 schema，新代码只接受新字符串；需要显式读取/迁移规则，而不是悄悄改历史工件。
- 配置中的相对 5% 门槛未在当前 same_rear 检查中完整落实。
- “non-event”使用 `is_evt_tail` 的补集；非尾部风险样本不等于没有制动交互。
- event key 仅由 recording 与 onset frame 构造，多个车辆对同一帧起始时存在碰撞可能。[S06,S13,S14]

**必须修改**：见第 10、11 节；这些是封版质量问题，不是额外模型创新。

---
## 4. 本轮目标架构：不换 CIH-WM，只统一语义

```text
明确来源的场景条件 / 长期计划
                │
                ▼
冻结 FactualDynamicsModel（现有 HiQR 权重）
                │ base action
                ▼
冻结 MechanismGuidedResponsePrior（现有 A2 权重）
                │ prior proposal + frozen features
                ▼
现有两维 HumanResponseCalibrator（唯一可训练主体）
                │ calibration raw sample
                ▼
统一动作映射与执行约束
                │
                ▼
现有 KinematicTrafficDynamics
                │
                └── 新的已实现状态 → 下一帧观测/路由/响应
```

**不新增“事实模式关策略、干预模式开策略”的实验开关。** 同一策略对相同历史、参考、控制器状态和外生随机量应输出同一分布。协议模式只能决定参考的来源与评分方式，不得进入 actor 或改变在线激活规则。

原 A2 保留为完整冻结的先验和独立历史对照。保留权重不等于保证它在新参考、新 mask 和新执行约束下仍然产生历史报告中的同样轨迹；这一点必须用回归检查和同协议比较确认。

### 4.1 事实锚定的两层含义

- **结构层**：动作表示中确实包含“返回事实动作”的选项；没有局部作用权限且没有未释放修正时，不额外修改 base。
- **实证层**：在全槽位、相同参考和相同随机条件下，完整策略通过 factual 非劣性与绝对精度检查。

两者不能相互替代。一个选择了零修正的网络能轻易通过第一层，却未必会响应；一个很会制动的网络也未必通过第二层。

---

## 5. 控制器最小修改方案

### 5.1 使用现有两维随机 actor，不新增网络族

保留完整冻结先验，令：

\[
 b_t=a_t^{factual},\qquad p_t=a_t^{prior},\qquad d_t=p_t-b_t.
\]

由当前小型 actor 输出二维 raw sample：

\[
(u^w_t,u^r_t)\sim q_\phi(\cdot\mid h_t^{prior},d_t,b_t,\alpha_t,c_{t-1}).
\]

建议本轮固定使用可有限取到端点的映射：

\[
 w_t=\operatorname{clip}(1+u^w_t,0,2),\qquad
 r_t=R\tanh(u^r_t),
\]

\[
 a_t^{request}=b_t+w_t d_t+\alpha_t r_t.
\]

这里 `d_t` 已经包含先验自己的作用域与 authority，不能无意再乘一次同一 authority。`alpha_t` 用于额外 signed residual 的局部权限。

这样同一表示中存在：

- `w=1,r=0`：保留先验 proposal；
- `w=0,r=0`：回到事实动作；
- `r>0`：纠正过度制动或提供合理恢复；
- `r<0`：追加有限制动。

该映射是本轮拟修改设计，不是当前代码已有性质。它保留二维 Normal raw policy，允许 clipping 引起多对一执行映射；PPO 仍对 **raw sample** 计算概率，而非假装最终动作是高斯分布。

`event_gate` 独立网络不再参与新候选的在线输出。可在 checkpoint 兼容读取中保留其历史参数，但不计入训练参数集合、不再做 onset 二分类监督。若保留历史候选评测，必须走显式历史映射，不能改变旧结果的含义。

### 5.2 初始化与权重保留

- `FrozenResponsePrior` 全部参数严格冻结；训练前后哈希或 tensor checksum 不变。
- 事实层和 Flow/Diffusion 参数严格冻结。
- 新校准 actor 初始化为 `mean=0`，小方差，例如 `log_std=-3`；这是预算内的预设，不是已验证最优值。
- **deterministic mean** 在执行投影前满足 `w=1,r=0`。随机运行有非零方差，不能称为逐样本精确复现 A2。
- 当前未接受 CIH calibrator 的 checkpoint 保留用于诊断。输出映射语义改变后，不能直接加载它并声称行为保持；需要记录 `action_map_hash`，使用兼容性审计或重新进行本轮小型 warm start。
- 禁止为了继承“旧能力”而把先验的硬安全覆盖伪装成学到的人类行为。proposal、legacy mapping、legacy guard 分别记录。

### 5.3 执行约束使用统一定义

定义相对事实基座的总响应修正：

\[
 c_t=a_t^{exec}-b_t.
\]

存储四项，不再互换名称：

```text
requested_total_correction
projected_total_correction
executed_total_correction
calibration_relative_to_prior
```

在每一步，以实际执行后的上一修正为参考，求动作范围与响应增量范围的交集：

\[
\mathcal C_t=
[a_{min}-b_t,a_{max}-b_t]
\cap[c_{t-1}-J_c\Delta t,c_{t-1}+J_c\Delta t].
\]

将请求修正投影到该区间，并执行 `b_t+c_t`。这是标量裁剪，不是新增 MPC/QP。

若交集为空，说明 base 自身变化、上一动作与新约束不兼容：

1. 优先保持硬物理动作界；
2. 记录 `correction_constraint_infeasible` 和实际执行 jerk；
3. 不得通过只记录未执行命令，宣称所有约束都已满足；
4. 在正式非劣/执行资格判断中单独处理，不静默放宽限值。

冷启动无 authority、无待释放修正时，若 base 自身处于物理界内，应精确返回 base。已有残余修正退出时需要有限释放；不能同时强求“退出第一帧立刻回 base”和“修正严格连续”。

`J_c` 首轮保持当前配置的声明值，先做兼容性审计；不要为了重现历史数值任意提高。必须同时报告 raw final jerk、base jerk、总响应 jerk、相对 prior 的校准 jerk。

### 5.4 影响关系与恢复

继续仅支持直接纵向跟驰。明确检查 `parent` 位于 follower 前方、车辆有效且关系满足当前几何定义。不要把半径内所有同车道车辆默认视为同一直接跟驰关系；是否存在中间遮挡车辆应由当前状态判断。

修复恢复递推后，保留原有参数值作为起点：

```text
engaged → clearing（持续累计稳定帧）→ recovering → idle
```

- clearing 期间保持 episode ID、parent、role 和累积计数；新风险出现则返回 engaged。
- recovering 可继续由策略输出，但权限按既定恢复过程衰减。
- authority 真正归零后仅进行必要的确定性释放，该时段不进入 PPO。
- 不根据人工 intervention start/stop 或日志未来的事件结束时刻控制在线 phase。

---

## 6. 一个执行器，两种信息预算；不引入第二套策略

### 6.1 参考来源要从运行开始就声明

新增/完善 `ReferenceContract`，但它是数据来源契约，不是 actor 的输入。

| 契约 | 允许的信息 | 可以支撑的主张 |
|---|---|---|
| `conditioned_reconstruction` | 真实宏观 K、已声明的场景条件 | 给定宏观条件的轨迹重建、同条件受控分叉 |
| `prefix_sampled` | 起始历史、地图、C0 与从冻结模型采样的 K/计划 | 不读取 NPC 真实未来的生成与响应验证 |

当前 `frozen_diffusion_plans()` 使用 `BackgroundTrajectoryDataset`，后者从真实轨迹提取 2 s/4 s/5.96 s 的 NPC 状态结点。其 `ego_future_in_condition=False` 只排除了 ego future，不表示 NPC future 条件也消失了。[S08,S15]

因此：

- 当前 anchor-only fallback 的修复是真实进展，不应否定；
- 同时不能把整条计划链称为 prefix-only；
- 给定 K 的因果探针可以明确作为“条件世界内部的响应检查”，不自动等于未知未来的人类反事实；
- 对外宣称自主可采样环境时，必须完成 `prefix_sampled` 路径的独立验证。

### 6.2 首选复用现有 Flow → diffusion 输入转换

现有 `diffusion/src/data.py` 提供 `prepare_flow_condition()` / `prepare_external_condition()`。将冻结 Flow 生成的 K 经过这些接口传给 diffusion；不要为了去掉 K_GT，直接丢弃现有生成器而把所有车辆换成恒速模型。[S15]

Codex 应核对当前 Flow sampler 的实际入口后连接；本文不假装旧版本已删除的 `composition.py` 或 `prefix_reference.py` 仍是当前有效接口。

若某槽位计划真正缺失，当前 anchor 外推只能作为显式 fallback：报告槽位、频率、持续时间和质量，不能默默代替该槽位的生成能力。尤其不能通过 highD future completion 补回被检验的响应答案。

### 6.3 不能到事件触发时才“清洗未来”

在 `prefix_sampled` 模式下，自运行起点起，所有会影响策略的 NPC 未来均不可进入参考、过滤状态或 encoder。只在 onset 后移除目标后车轨迹，无法清除之前已经进入 hidden state 的信息，也无法阻断其他车辆参考通过关系编码传递答案。

未来扰动测试应为：固定合法 prefix、地图、C0、采样随机量和 ego 的外生动作序列，改变不可用的 NPC 日志未来，**整个生成 rollout 均保持不变**。离线目标指标可以变化，策略输出不能变化。

### 6.4 明确主车日志执行的边界

自然事件验证可让主车按日志动作逐帧执行；未来命令只存于执行器，NPC 不可读取尚未执行的命令。这检验的是在真实主车刺激下的条件响应，而不是主车也会对新 NPC 行为自主调整的全人类世界。

所有最终结论必须声明对应条件。不得将 `conditioned_reconstruction` 的厘米级误差与 `prefix_sampled` 的行为指标拼接成一项无条件精度保证。

### 6.5 正式随机性、快照和前缀分叉

当前 `environment.py` 的 snapshot 环境是测试 fixture，不是 CIH 正式执行器；其状态也没有覆盖当前完整校准/路由状态。[S16]

建议把 `evaluation.rollout` 的局部运行状态抽为同文件或相邻文件中的 `RolloutState`，**保持原来的 model 调用、动力学、更新顺序不变**。外层原 `rollout()` 成为循环包装器。

快照至少包括：

```text
states / valid / realized history / map identity
reference tensor, reference origin and index
filter global/agent state
scene/agent latent and style state
committed ego actions
intervention memories
influence state: parent/role/authority/phase/age/safe counter/recovery counter
previous executed action / previous total correction / release state
完整外生噪声块的标识、索引和未消费后缀
```

三项先决 Gate：

1. **同世界同策略重放**：完整随机输入一致，轨迹/动作可复现；
2. **snapshot/restore**：不中断 rollout 与恢复续跑一致；
3. **shared-prefix identity**：分叉前缀一致，改变未来 intervention 或合法 response-noise suffix 不影响过去。

任一失败，停止后续模型训练。不得重采已经实现的场景 latent 或重新生成已经使用过的计划。

为控制范围，不实现新的 TTS 或 AMS 内核。这里仅补齐当前世界的可采样、可重放与分叉契约。

---

## 7. 人类响应监督：先把有真实标签的部分用对

### 7.1 严格对齐时间

冻结如下定义，并写入数组契约：

```text
decision t: policy observes H_t and x_t
a_t: action applied over [t,t+dt)
x_(t+1): post-step state
jerk_t = (a_t - a_(t-1))/dt
```

自然动作可以从速度前向差分形成监督标签，但 `x_(t+1)` 只能用于 target/下一步 teacher forcing，不可混入本步 actor feature。首个响应 jerk 必须使用真实前一执行动作，不能人为设为 0。

### 7.2 缓存真实物理上下文，不缓存未训练策略造成的混合历史

新增真正的 logged-context builder，重用正式 executor 的纯观测与特征构造函数：

1. 从真实 prefix 初始化，按时间依次加入真实已发生状态。
2. 冻结事实模型和 A2 prior 在该真实上下文上前向，构建 base、prior features、作用关系。
3. 主缓存使用 deterministic base/prior 诊断值，记录随机模式；分布学习阶段另用独立随机 rollout。
4. target 是对应时刻真实 follower 动作。
5. previous physical action 使用真实历史；learner 的内部校准命令若影响输出，不应永久缓存为旧 checkpoint 的输出。

训练时按连续片段重放当前 adapter 的内部修正状态，物理历史保持 teacher-forced。也就是说，缓存保存可复用的物理证据和冻结特征，**每轮可训练记忆由当前 adapter 重新计算**，不把任意旧校准轨迹当成不可更改的输入。

缓存必须带 `reference_contract_hash / data_hash / factual_hash / prior_hash / feature_spec_hash / timing_spec_hash`。改变任一项后，旧 cache 失效。

### 7.3 使用已有的完整事件窗口

当前 `reaction_evidence.py` 定义 `PRE_EVENT_FRAMES=25`、`EVALUATION_FRAMES=25`、`RECOVERY_FRAMES=75`。[S18] 正式读取事件工件实际长度并核对与代码一致后，优先使用：

```text
pre-event: 25 frames
post-onset: 75 frames（最多 3 s，含原 25 帧响应窗口）
```

当前事件范围通常允许在 149 帧 rollout 内取得这段数据；必须逐事件检查，禁止越界填零。75 帧也不意味着所有驾驶员都已恢复，未恢复者按右截尾报告。

保留 `25-frame raw ES` 作为历史兼容指标，但不得称其独自覆盖了完整恢复。

### 7.4 监督目标

自然事件的真实状态上：

\[
 L_{human}=\operatorname{Huber}(a^{exec}_\phi(H_t^{log}),a_t^{log}).
\]

普通跟驰片段上，在真实动作监督之外，加入小权重的事实修正约束：

\[
 L_{anchor}=\operatorname{Huber}(a^{exec}_\phi,a^{log})
 +\lambda_0|a^{exec}_\phi-a^{factual}|.
\]

这里 anchor 指向事实基座，不是无条件要求“保持 A2 的每一处偏差”。事件与非事件权重必须实际进入 loss。

**从该监督目标移除以下两项**：

- 对 `frame >= event_onset` 的 event-gate BCE；
- 对真实人类事件同时要求 `final_action == IDM rule_target` 的绝对规则动作回归。

真实标签与规则冲突时，应优先检查可执行性、时间对齐和模型不确定性，而不是让规则覆盖人类证据。

### 7.5 非事件必须真的是非交互事件

不能用 `~is_evt_tail` 代替无制动响应的普通跟驰。建立与事件起始速度、间距、closing 等匹配的训练片段，并用离线事件扫描排除对应时间窗内的目标刺激。这个标签仅用于选择训练样本，不输入策略。

真实但稀有的训练事件仍拥有真实标签，不应仅因 kNN 邻居不足就丢弃其监督。`empirical support` 描述证据覆盖和泛化可信度，不等同于“有没有发生过这次真实行为”。synthetic 才是没有人类 response target 的另一类来源。

---

## 8. 概率校准：使用可核查的完整序列目标

### 8.1 目标与独立性

给定同一事件起始历史、规定参考和真实主车刺激，生成 K 条独立未来响应：

\[
Y_i=\operatorname{vec}([a_{1:T},|j|_{1:T}]),\quad i=1,\ldots,K.
\]

`T=75` 为主要目标；`T=25` 只作兼容报告。距离按 train-only channel IQR 标准化，并可除以固定的 `sqrt(T)` 以避免不同窗口长度单纯改变尺度。所有模型使用同一尺度。

在同一 ensemble 内，定义为随机的预测因素应独立采样；跨模型、跨 intervention 分支才使用索引一一对应的 CRN。将 K 条完全相同噪声的轨迹复制 K 次不构成概率预测。

若 K/长期计划固定，评分对应的是该条件下响应分布；若评估完整 `prefix_sampled` 分布，应按声明的预测测度同时采样长期因素。二者分别报告，不混用。

### 8.2 训练采用 fair Energy Score

现有 V-statistic 保留为 `es_legacy_v`。新训练目标采用：

\[
U_K=\frac1K\sum_i d(Y_i,y)
-\frac1{2K(K-1)}\sum_{i\ne j}d(Y_i,Y_j),\qquad K\ge2.
\]

对给定条件下的独立样本，它是总体 Energy Score 的无偏估计。相较之下，原 V-statistic 的期望为：

\[
\mathbb E[V_K]=ES(P,y)+\frac1{2K}\mathbb E d(Y,Y').
\]

因此 K=8 的训练与 K=32 的验证，若不区分经验有限集合评分与底层分布评分，包含不同的有限样本项。

这属于现有 proper-scoring/fair-ensemble 原理的应用，不宣称为本文首创（《Strictly Proper Scoring Rules, Prediction, and Estimation》；《Fair scores for ensemble forecasts》）。它保证的是目标的统计含义，不保证有限数据下学得的模型等于真实人类，更不保证未见反事实已被识别。

### 8.3 使用有明确 score-function 含义的 LOO 基线

对每条 rollout i，计算删除它后的 `U_(−i)`，定义：

\[
R_i=K[-U_K+U_{-i}],\qquad K\ge3.
\]

令 `ell_i` 为该 rollout 所有相关随机校准动作的联合 log probability。在未 clipping 的 on-policy score-function 层：

\[
\frac1K\sum_i R_i\nabla_\phi\ell_i
\]

是 `∇ E[−U_K]` 的一个估计器。理由是 `U_(−i)` 不依赖第 i 条独立 rollout，故其与第 i 条 score function 的乘积期望为 0。

**限定**：这是独立 rollout、固定数据条件、固定动作映射下的原始梯度恒等式。PPO clipping、多 epoch、GAE、batch 标准化和辅助损失会改变估计性质，不能对整个 PPO 算法宣称无偏。

原来的 centered LOO 并非简单“符号错误”。本轮替换它，是为了避免隐藏的 ensemble-size 目标及尺度歧义，并给出能用有限状态枚举检查的实现。

### 8.4 首轮不使用加权 prefix 差分塑形

首轮把 `R_i` 放在事件 75 帧评分结束点，使用有限时域 Monte Carlo return：`gamma=1, gae_lambda=1`，critic 作状态基线。保留 5/10/25/75 帧诊断图，但不加入多个相互干扰的塑形目标。

不能只保留 onset 后 25 帧的 PPO mask，而将此前已影响事件状态的可训练动作全部丢掉。起点到 score 结束之间所有真正有策略作用的相关动作必须进入 credit；或者从已声明的固定真实事件 prefix 开始分叉，并保证 prefix 不包含本轮可训练行为。

若多个活动 NPC 的动作共同影响事件 follower，联合 log probability 要包含这些动作；不能一边让它们由同一个可训练策略变化，一边断言仅 follower 的分量就是完整目标梯度。首版通过直接、最近有效 follower 路由控制规模，不新增复杂多智能体 credit 算法。

损失归一化按“事件组 → rollout → 有效决策步”明确实现：总体目标对事件等权，单条 rollout 的 log probability 在相关步骤上求和，再对 K 条未来和事件组取平均。不能将全部 active frames 直接混成一个平均数，从而隐式提高长响应或多车事件的权重。若为数值稳定而除以固定时域常数，应对所有事件使用相同常数并记录。

后续若确有需要使用前缀塑形，必须先明确新的目标或采用严格势函数差分。禁止继续使用上一版带负系数的加权差分，并声称它与完整 score 等价。

### 8.5 PPO mask 与执行映射

```text
policy_active = 本时刻随机 raw action 可以影响执行动作
execution_active = 有任何尚未结束的响应或释放
release_active = 本时刻只是确定性释放
```

仅 `policy_active` 进入 PPO。物理 clipping 是固定映射的一部分，即使局部导数为零也不必自动排除概率样本；但完全忽略 raw action 的 deterministic release 必须排除。

同一 rollout 的 `old_log_prob` 与训练前重新计算结果必须逐元素一致。否则先修接口，不允许开始多 epoch 更新。

### 8.6 PPO 期间继续维护事实约束

现有 trainer 在 supervised 后的 PPO loop 没有继续执行人类动作 anchor 或机制辅助步骤。[S14] 本轮增加固定比例的辅助 minibatch：

\[
L=L_{PPO}^{sequence}+\lambda_A L_{teacher\ anchor}+\lambda_M L_{mechanism}.
\]

辅助 teacher 仍只来自真实状态。机制辅助来自新采样的闭环状态，不拥有日志动作标签。

PPO clipping 不能约束单独辅助更新造成的漂移。每个完整更新后，在固定小批已采特征上测总 policy KL；超出预登记上限则回滚该次更新并记录原因，不静默继续。首轮无需引入复杂梯度投影或新的优化器。

---
## 9. 数据外机制学习：使用同一事件的有限窗口，不用答案覆盖

### 9.1 复用已有成对训练，不扩张新模型

保留 `causal_contrast_cache / causal_contrast_loss` 的基本思路，但修正采样、选择条件与目标：

- 固定 seed 按 recording 分层采样，轮换训练池，记录实际唯一 scene/event 数；不始终只用前 256 条。
- 保留当前三档 `1.5/2.25/3.0 m/s²` 作为兼容探针；较强 step/ramp/pulse 用作明确的压力补充。
- 不增加每轮总分支数。不同剂量/形状在更新之间轮换；训练与验证按场景和参数范围区分。
- 每次 intervention 记录请求偏移和实际执行后的偏移；发生加速度饱和时，不把相同的实际刺激当成更大的 dose。

“数据外”按照实际交通条件的训练覆盖判定，不按 `synthetic=True` 或制动幅值标签自动决定。合成工况即使邻域覆盖较高，也没有这条合成干预对应的真实人类 response 标签。

### 9.2 固定公共前缀和比较总体

同一个事件的 baseline/intervention 共享初始历史、K/计划及已实现外生因素。只改变规定的 ego 后缀操作；既有 response noise 使用对应 CRN。

基本 eligibility 由探针前的有效父子几何、数据质量和物理状态定义，不能依赖策略是否已经做出正确反应。之后父子关系改变、饱和、碰撞、无响应分别记录，不静默筛掉困难案例。

### 9.3 窗口内响应变化

记录同一事件、同一父子关系下：

\[
\Delta a_t=a_t^{intervention}-a_t^{baseline},\qquad
\Delta f_t=f_{IDM}(H_t^{intervention})-f_{IDM}(H_t^{baseline}).
\]

窗口累计量：

\[
D_a=\sum_{t\in W}\Delta a_t\Delta t,\qquad
D_f=\sum_{t\in W}\Delta f_t\Delta t.
\]

只在局部跟驰假设有效、机制变化有辨识度、且比较窗口未被碰撞后动力学破坏时，使用软方向和宽幅度约束。

不要求“更强主车制动后每一帧都必须更强后车制动”。因为两条轨迹之后的 gap、closing 和恢复阶段已经不同，后段差异可能合理反转。瞬时机制检查可以保留为历史诊断，但不能当成所有轨迹都应严格满足的交通定律。

### 9.4 防止无响应与过早响应两种退化

- 不能用 `sign(delta_model) * sign(delta_rule) >= 0` 将 `delta_model=0` 算成功。
- 也不能从主车命令下发同一帧就要求 NPC 已经制动。
- direction success、no-response、saturation、关系切换分别报告。
- 反应时延由已实现运动变化起算；自然事件延迟统计用于合理性诊断，不把普通制动的固定分位数当成紧急条件下不可改变的硬延迟。
- 剂量单调性只在同一组可比事件、同一预登记窗口中判断；饱和允许并列，不奖励人为追加制动。

### 9.5 梯度解释

继续允许在闭环访问到的状态上，detach 物理状态、重算两条分支的当前 actor 动作，然后对两边更新小型校准器。这是 **visited-state auxiliary / semi-gradient**，不是对整段交通的精确 policy gradient，也不是已识别的真实因果效应。

若声称优化了累计响应全过程，必须能区分“当前状态上的局部重算损失”和“实际闭环累计结果”。本轮用前者训练、后者验收即可，不引入端到端可微物理或另一个 RL 算法。

### 9.6 碰撞、恢复和风险

不以消灭所有 ADS–NPC 碰撞作为唯一奖励。保持有限响应、动作界、观测时序，并保存碰撞/间距/TTC 等结果。

同时不能将任何新碰撞都解释为“保留真实风险”。对碰撞发生前是否存在明显数值错误、错误关系、异常瞬时动作进行诊断。NPC–NPC 碰撞也可能是合法冲击传播，不能仅按对象类型自动处罚或自动判为仿真缺陷。

若发生碰撞，之后的运动学重叠/穿透不作为人类响应标签。不能直接删掉碰撞 rollout 或缩短其负损失暴露从而让“早撞”获得更好 reward。必须预登记统一终止/吸收约定，报告丢失的有效时间与碰撞概率；该约定未完成前，风险结果仅为几何冲突诊断。

**不惩罚碰撞只是目标设计；它不证明风险概率无偏。**

---

## 10. 正式评测：把三个创新都纳入验收

### 10.1 评测矩阵

保留少量必要对照，不为每个修复再训一个网络。

| 对照 | 是否需要训练 | 作用 |
|---|---|---|
| 冻结事实转移层 `NoReactionController` | 否 | 相同 base 调用路径的事实基准 |
| 冻结完整响应先验/历史 A2 | 否 | 已学会的响应能力及旧 guard 贡献 |
| 当前未接受 CIH candidate | 否 | 旧训练结果的诊断对照，可只用固定小子集 |
| 本轮 supervised calibrator | 同一训练链中间产物 | 检验真实状态监督是否有效 |
| 本轮 PPO calibrator | 单一候选 | 检验闭环概率校准增量 |

**基准路径要统一**：当前 evaluator 的 `controller=None` 会令 `apply_intervention_adapter=True`，而启用 CIH 时该标志为 false。历史复现可以保留原路径；隔离 CIH 响应贡献时，应使用显式无响应 hook 或统一标志，使 base 计算不因比较 arm 不同而变化。[S07]

只保持冻结事实层中的已训练模块不变；不要在本轮悄悄关闭/重训它的显式 ego 响应模块。报告哪部分属于 base、哪部分属于新响应。

### 10.2 Gate E0：协议与全槽位基线就绪

先测冻结事实层，覆盖最新全槽位协议和单独 same_rear。核对：

```text
实际样本数、recording 数、槽位有效数
所有 condition / plan / mask 的来源
per-slot、全背景及 same_rear 的 ADE/FDE/P95
fallback 比例及其槽位分布
关键噪声流、快照、共享前缀一致性
```

当前仓库绝对事实阈值为 ADE 0.06 m、FDE 0.06 m、P95 0.12 m。[S17] 保留这些阈值的原始定义与适用口径，先报告新全槽位底座是否达到，不得事后改阈值让它通过。

如果底座本身不合格，优先修复 mask/condition/cache/时间对齐等确定性问题。不得默认要求小型响应器一边学习人类行为，一边修复一个未验证后车的生成底座。仍无法满足时，输出 `factual_base_not_ready`，保留诊断，停止长期 PPO；是否开展底座专项再训练另作决定，不自动扩大本轮。

**“高精度”主张还必须看目标后车的绝对误差。** 不能通过其余五个槽位平均掩盖后车米级误差。未达到目标对象的既定精度要求时，只能称为相对非劣或改进候选，不能称整个世界均为高精度。

### 10.3 Gate E1：监督 warm start

使用完整真实事件 teacher forcing 后，在固定 validation 子集上检查：

- 相同输入预算下，均值动作错误与执行映射无系统错位；
- factual 没有明显退化；
- 序列得分不比本轮同映射的初始化更差；
- 对既有冻结 A2 的差距单独报告。

**不再要求同时“事实严格非劣”且“必须达到一个可能事实不可行的 A2 的 1.10 倍 ES”才允许研究。** A2 是响应参考，不是天然合格的联合基线。也不能因为达不到该复合门槛就认定代码一定有 bug。

若监督信号与目标动作明显不一致、输出饱和或有效梯度接近零，则先停止检查，不依靠 PPO 从零补救。

### 10.4 Gate E2：完整 validation 的 factual 保持

同协议、同计划、同噪声比较 final 和冻结事实层：

| 指标 | 允许绝对退化 | 同时允许相对退化 |
|---|---:|---:|
| ADE | 0.02 m | 5% |
| FDE | 0.06 m | 5% |
| P95 位移误差 | 0.10 m | 5% |

同时报告全槽位与 same_rear；P95 差异应从配对重采样后的聚合分布计算，不能把“各行 P95 的均值”冒充全局 P95。

主表包含 deterministic mean 和固定 CRN 下的 stochastic drift。随机轨迹逐样本不必等于唯一日志 realization；但若部署使用随机策略，就必须报告其运动分布、漂移和有效性，不能只验收 deterministic 模式。

### 10.5 Gate E3：人类序列分布与 PPO 增量

在独立自然事件上，使用相同 K=32 和相同预登记随机变量：

```text
75-frame normalized fair Energy Score：主要指标
25-frame normalized fair Energy Score：短时响应
25-frame legacy raw V-score：历史兼容，不改原指标
```

必须比较 final 与自己的 supervised checkpoint：

\[
\Delta ES=ES_{supervised}-ES_{final}.
\]

配对 recording-cluster bootstrap 的 95% 区间下界 > 0，才能主张本次 PPO 的概率校准增量。没有显著增量时保留 supervised 结果，不能通过增加训练轮数或更换 test 子集寻找显著性。

与冻结 A2、冻结事实层同时报告。如果 A2 也满足同一 factual 约束，要求新模型在该可行集内至少非劣；如果 A2 不可行，则报告受约束的改进和剩余差距，不能隐藏 A2 更好的某一单项指标。

稀疏事件下 bootstrap 区间不稳定时必须报告 recording 数和有效事件数；“未拒绝相同”不是“证明分布相同”。

### 10.6 Gate E4：响应时序和恢复

统一计算观测与生成序列：

```text
sustained latency
peak deceleration
P95 / peak absolute jerk
braking dose
minimum net gap / maximum closing / minimum TTC
sustained recovery time / censored fraction
```

latency 使用连续数帧满足条件，而不是单帧噪声过阈值。恢复必须在真实响应开始、峰值制动之后寻找；未恢复标记为截尾，不能把窗口末端强行填成恢复时间。

诊断退化容忍使用冻结 train 尺度和数据噪声下限，预登记后不改；沿用 0.1×train IQR 时要处理 IQR≈0 与 censored 变量，不能机械除以极小数。

该 Gate 不以碰撞更少替代响应更像人。

### 10.7 Gate E5：因果信息顺序与机制外推

必须通过：

- 固定 `H_t / reference / controller state / noise`，改变当前尚未执行的 ego 命令，当前 NPC 动作不变；
- 改变未来 probe 或未来 noise suffix，分叉前动作和状态不变；
- prefix-only 模式中改变不可用 NPC 日志未来，生成结果不变；
- 冷态无局部权限时不修改 base；
- 不使用 shadow action 或规则覆盖来直接制造“方向正确”；
- 对同一可比事件集，窗口方向、no-response、剂量排序及饱和情况可解释。

旧的逐帧 95% 方向门槛保留为历史指标；新增窗口指标的定义在训练前冻结，不能在失败后换一个更容易的分母宣布成功。当前同一 commit 中未重跑的诊断也不能直接套到新定义。

可以将 95% 用作机制一致性工程门槛，但它不是来自 highD 的“人类正确率真值”。达不到时报告错误类型；达到时仅说明相应受控探针通过。

### 10.8 Gate E6：可采样与安全评价就绪

同一个 `CIH-WM` 运行内核应能消费冻结 world variables，执行指定 ADS，并保存完整 trace。风险总体、主车策略、NPC 参数/噪声分布和参考来源必须显式声明。

前述 Gate 未通过前不启动新 MC/AMS。通过后允许最多 2,000 组 CRN 配对世界作风险敏感性诊断，分别报告：

```text
真实人类记录的条件风险基准（同场景总体、同时间窗）
人类日志主车 + 自由 NPC 响应的模型偏差
固定 ADS 下不同背景模型的风险差异
```

这里的人类日志主车不会自适应新 NPC 行为，不能称为全人类自主仿真。低频事件计数不足时报告宽区间和指标分布，不声称精确真实事故概率。

---

## 11. 配置、工件与评测程序修复

### 11.1 必需的哈希身份

每个运行至少绑定：

```text
code_commit + working_tree_diff_hash
resolved_config_hash
factual_checkpoint_hash
response_prior_checkpoint_hash
calibrator_checkpoint_hash
reference_contract_hash + plan_cache_hash
dataset_split_hash + event_schema_hash
action_map_hash + observation_timing_hash
randomness_spec_hash
```

正式 test 验证这些哈希，不只检查 checkpoint 文件路径。替换同一路径下权重后不得继续沿用旧 validation acceptance。

### 11.2 schema 迁移不是重写历史

保留旧 `*_v2` 工件原样。新增显式兼容读取规则，输出“原 schema / 读取器版本 / 与当前配置差异”。不能批量替换字符串再宣称旧结果由当前代码生成。

用户希望方法和目录名称简洁，与工件必须有技术 schema 身份不冲突。公开方法名继续 CIH-WM；版本可用内容哈希和内部 schema 元数据，不需要增加新的“第几代方法”名称。

### 11.3 event identity

统一使用完整物理事件键：

```text
(recording_id, leader_id, follower_id, absolute_onset_frame)
```

数组可用结构化字段或无歧义序列化后的 SHA-256。不得只使用 `recording*constant + frame`。车辆 ID 在不同 recording 内独立，不因数字 ID 相同而排除另一个 recording 的合法样本。

旧 cache 存在 key 合并风险时须重建，禁止在读取时用字典静默覆盖。

### 11.4 验收不等于“训练够多少轮”

manifest 要记录计划更新上限和实际完成的 optimizer steps、有效事件数、有效 frame 数、终止原因。5000 supervised /160 PPO 不是科学资格本身；协议修正后，允许预登记的较小固定预算，不由验收程序强制刷满。

已用历史 test 分析过的事实须保留。新结果称为“候选冻结后的单次确认性评估”，不声称整个 test split 从未被查看。

### 11.5 只维护一个正式执行链

在现有文件中定向重构，不新增四个互相复制的模拟器：

| 当前文件 | 修改重点 |
|---|---|
| `src/cih_model.py` | 继续组装同一方法；绑定配置与 checkpoint 哈希；清除与 `enable_secondary=false` 冲突的说明 |
| `src/reaction_controller.py` | 两维 retention/residual 映射、固定 PPO 映射语义、分离 active mask、执行 traces |
| `src/influence_graph.py` | 修复 clearing/recovery 递推、保持 parent/role、检查直接跟驰关系 |
| `src/evaluation.py` | 抽出同一状态转移内核、显式噪声、快照/分叉、去除 pending ego 接口、统一 base hook |
| `src/cih_training.py` | 真实历史缓存、shared execution mapping、75 帧评分、合法 PPO mask、机制辅助与 teacher anchor |
| `src/human_response_training.py` | fair ES、LOO 梯度契约和单元测试；历史 score 保留独立名称 |
| `src/human_response_prior.py` | 完整 event key、train-only coverage、25/75 两种目标字段、禁止 query 库污染 |
| `src/planner.py` | 显式 K 来源、fallback 审计、缓存哈希；不恢复 highD future completion |
| `scripts/train_cih_world_model.py` | 单候选 staged training、有限预算、监督检查、resampling 统计、有效步数及可恢复日志 |
| `scripts/evaluate_cih_world_model.py` | 增加 natural response score/PPO 增量；固定事件分母、真正的非事件、相对非劣、哈希绑定 |
| `scripts/promote_cih_world_model.py` | 全部 Gate 与已冻结 validation 哈希通过后才能执行一次 test；不得只看训练轮数 |
| `src/protocol.py` | 将历史口径和新全槽位口径明确区分；阈值/证据随 report 绑定 |

`ReferenceContract` 与正式 `RolloutState` 可作为两个小型新模块或放入相应现有模块；它们是契约抽取，不是新学习模型。

---

## 12. 固定预算与执行顺序

### Phase A：只读预检与历史复现

先不训练。

```bash
# 非交互 shell 需先初始化 conda。
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tread

python - <<'PY'
import sys, torch
print('python', sys.version)
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('cuda_runtime', torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit('GPU training preflight failed; do not silently use CPU')
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
nvidia-smi
git rev-parse HEAD
git status --short
```

本地 HEAD 不同时记录差异，不 reset、不覆盖用户改动。收集现有 tests 并记录实际数量，不继续假设旧版本“86 项测试”仍然有效。

预检先确认最新全槽位参考/权重能力及当前未接受候选身份。使用实际入口的 `--help` 核对参数。本文新增的功能需先实现，不能把不存在的脚本当成可运行命令。

### Phase B：语义修复和低成本 Gate

完成 D01–D08，先跑纯函数、上下文、随机性和状态机测试。再用固定少量训练/validation 场景做 base 与 prior 对照，不根据其结果改测试集。

必须先通过 replay / snapshot / prefix identity。未通过时禁止算法训练。修复不改变数值语义的部分应回归到现有轨迹；改变动作映射或参考协议的部分单独标记，不伪称完全等价重构。

### Phase C：建立同协议基线

复用 cache 完成最新全槽位冻结事实层评测，单独核对 same_rear。旧 masked 指标只用于历史复现，不参与新候选选择。每个基线只运行/缓存一次相应协议。

若 Phase C 暴露底座未就绪，输出原因与最小修复建议，停止长训练。不能自动切换到重新训练整个 Flow/Diffusion/HiQR。

### Phase D：一次 supervised warm start

建议上限：

```yaml
training_seeds: 1
supervised_updates: 400
supervised_minibatch_size: 2048
pre_event_frames: 25
response_frames: 75
```

这些是本轮预算设计，不是已证明最佳的超参数。缓存构建后允许按片段小批次训练及梯度累积，实际 optimizer step 与样本重复次数全部记录。

只训练小型 calibrator。A2/事实权重不变。Gate E1 失败时不开始 PPO；先报告动作可达性、标签对齐、饱和和梯度状况。

### Phase E：一次小步闭环校准

建议配置：

```yaml
policy_optimization:
  event_groups_per_update: 8
  futures_per_event: 8
  maximum_updates: 160
  epochs_per_update: 2
  learning_rate_start: 3.0e-5
  learning_rate_end: 1.0e-5
  gamma: 1.0
  gae_lambda: 1.0
  clip_ratio: 0.2
  inner_stop_kl: 0.02
  full_update_kl_limit: 0.02
  quick_validation_interval: 10
  early_stopping_patience: 4
human_response:
  primary_horizon_frames: 75
  score: fair_event_energy
  compatibility_horizon_frames: 25
  compatibility_score: legacy_raw_energy
  prefix_reward_shaping: false
```

每 10 次仅跑固定 recording 分层的 32 个事件、每事件 8 futures 的 quick validation，以及固定小 factual 子集；完整 validation 不每 10 轮重跑。

每次更新含 64 个自然 rollout；以 149 帧上限计，160 次约为 1,525,760 个 scene-step，尚未计机制分支、cache 和验证。不得以“只有 160 updates”隐瞒真实仿真成本。

显存不足可以降低物理 batch 并梯度累积；不得改变噪声配对和有效样本量后不记录。模型型轨迹执行通常同时受 CPU、GPU 与数据传输影响，不承诺未经测量的小时数。

早停按预登记规则和事实可行性执行。保留 supervised、选中 candidate 和有报告结果的 rejected checkpoint，不扩大更新数补救。

### Phase F：完整 validation 与确认性 test

只对选定候选完成 Gate E2–E5。不同参考来源分别报告；任何关键 Gate 缺失不能写 `accepted=true`。

validation 通过后绑定全部哈希，执行一次新候选 test。不能根据 test 重选 checkpoint 或调整门槛。

### Phase G：可选的后验风险诊断

只有 Gate E6 就绪才执行不超过 2,000 组配对世界。不自动启动 AMS，不引入新的危险搜索模型。

---

## 13. 最小测试清单

以下为**待新增/增强测试契约**，不是声称仓库已存在相应函数。

### 13.1 基座与 A2 保留

```text
test_frozen_fact_and_prior_parameters_unchanged
test_legacy_prior_checkpoint_loaded_strictly
test_historical_mapping_reproduction_is_separate_from_new_mapping
test_base_hook_identical_between_no_response_and_cih_arms
```

### 13.2 动作与状态机

```text
test_retention_mapping_contains_base_and_prior_endpoints
test_identity_mean_equals_prior_before_execution_projection
test_cold_inactive_action_is_exact_base_when_feasible
test_pending_release_is_not_falsely_exact_passthrough
test_execution_projection_uses_same_spec_in_training_and_runtime
test_empty_constraint_intersection_is_reported
test_command_and_executed_correction_are_not_conflated
test_clearance_counter_reaches_recovery_after_required_frames
test_recovery_preserves_parent_and_role_until_release
test_deterministic_release_excluded_from_policy_ratio
```

### 13.3 数据与时间

```text
test_teacher_features_come_from_logged_history_not_free_rollout
test_action_label_maps_state_t_to_state_t_plus_one
test_first_response_jerk_uses_previous_action
test_75_frame_response_and_censoring_are_valid
test_event_key_contains_recording_pair_and_onset
test_same_numeric_vehicle_id_in_different_recordings_is_not_same_vehicle
test_non_event_is_not_defined_by_evt_tail_complement
test_train_only_scaling_and_coverage
```

### 13.4 因果与随机性

```text
test_current_pending_ego_command_cannot_change_current_npc_action
test_prefix_sampled_reference_is_invariant_to_logged_npc_future
test_reference_mode_does_not_enter_actor_features
test_same_random_world_and_ads_replays_exactly
test_snapshot_restore_matches_uninterrupted_rollout
test_fork_changes_suffix_only
test_batch_and_chunk_order_do_not_change_world_identity
test_future_rollouts_are_independent_within_an_event_ensemble
```

### 13.5 概率目标与优化

```text
test_fair_energy_matches_naive_pairwise_formula
test_fair_energy_expectation_matches_population_score_on_finite_support
test_loo_baseline_policy_gradient_matches_exact_finite_support_gradient
test_ppo_old_and_recomputed_logprob_match_before_updates
test_every_runtime_trainable_policy_parameter_has_a_defined_gradient_path
test_auxiliary_update_is_included_in_total_kl_check
test_zero_response_is_not_counted_as_direction_success
```

### 13.6 验收与复现

```text
test_full_factual_includes_same_rear_and_all_declared_rows
test_human_sequence_score_and_ppo_increment_are_required_for_acceptance
test_test_gate_rejects_replaced_checkpoint_at_same_path
test_old_schema_is_read_as_historical_not_relabelled_as_current
test_actual_optimizer_steps_and_stop_reason_are_recorded
```

---

## 14. 工件与停止规则

建议新输出仍在当前方法目录下：

```text
results/hierarchical_world_model/cih_wm/continuation/
  execution_plan.md
  resolved_config.yaml
  provenance.json
  baseline_audit.json
  source_contract_audit.json
  replay_snapshot_prefix_tests.json
  teacher_cache_manifest.json
  supervised_summary.json
  training_history.jsonl
  checkpoints/
  validation/
  test/
  risk_diagnostic/
  decision.json
```

目录不使用日期/round 作为方法身份；用 manifest 哈希区分运行。已有目录非空时拒绝覆盖，或使用用户明确指定的新目录。不可生成一串自动试验分支。

`decision.json` 至少区分：

```text
implementation_blocked
factual_base_not_ready
supervised_not_ready
candidate_rejected
response_validated_risk_not_ready
candidate_accepted_for_declared_protocol
```

每个状态包括失败 Gate、来源工件、缺失证据和下一步允许操作。不能只写“passed=false”而无法解释原因。

允许删除可再生 cache，但保留所有用于报告的 checkpoint、配置、哈希、日志、指标和样本键。永久删除历史代码/权重必须另获用户确认。

---

## 15. 方法论与论文措辞

### 15.1 本轮的学术动机

面向自动驾驶安全评价的交通世界模型，需要同时刻画给定场景条件下的事实交通运动，以及主车行为改变后背景车辆的响应过程。高精度条件重建不能独自证明交互有效性；另一方面，将背景策略训练成尽可能避免碰撞的控制器，也不能独自证明其代表真实人类驾驶行为。因此，本文在保留冻结事实转移能力与既有响应先验的基础上，将研究重点放在局部响应的条件分布校准及其执行一致性，而不是重新训练整个交通生成系统。

自然驾驶事件提供了真实历史与真实行为的配对证据，可用于学习响应启动、制动力建立、持续与恢复的统计规律；闭环展开则使策略接触其自身动作导致的状态分布。两类信息需要在训练中明确分工：真实状态对应真实动作监督，自生成状态对应序列概率评价或机制约束。对观察数据缺少覆盖的极端工况，模型应公开其机制假设与外推范围，而不能将合成轨迹当成人类反事实标签。

### 15.2 三项候选方法贡献

**事实锚定**：冻结事实层与完整已学响应先验，使用可明确退回事实行为的小型校准映射，并以相同执行路径及全槽位协议检验非劣性。仅冻结权重或关闭一个 gate 并不等于已经保证事实精度。

**完整响应序列概率校准**：对真实事件的完整加速度—jerk 序列使用 finite-ensemble-aware scoring，并通过含义明确的 rollout 级 score-function/LOO 信号进行闭环微调。Energy Score、LOO 与 PPO 均不是本文首创；创新需要体现在问题对象、执行一致性和相对既有响应先验的实证改进上。

**数据外合理响应**：以统一实际历史和公共随机前缀构造干预分支，在局部跟驰假设内约束响应变化，区分无响应、延迟、饱和与关系切换。该证据支持模型内部的因果时序及机制一致性，不证明真实人类个体反事实已被唯一识别。

### 15.3 不允许的主张

即使本轮全部通过，也不能直接声称：

- 已恢复任意 ADS 干预下真实人类的唯一正确行为；
- 75 帧窗口内所有驾驶员都已完成恢复；
- K_GT 条件重建的厘米级精度就是仅凭历史预测的精度；
- 所有碰撞都是模型错误，或所有未受惩罚的碰撞都是真实风险；
- 规则方向一致率等于人类真实性概率；
- 有 snapshot fixture 就代表正式随机世界已完成可重放；
- 只要训练满 160 轮就获得更好的策略；
- MC/AMS 采样区间小就说明交通模型偏差小。

**更合适的最终表述**：在明确场景总体与信息预算下，构建事实行为保持、自然事件响应经统计验证、数据外响应具有机制约束的 CIH-WM，并报告其用于 ADS 条件安全比较时的模型偏差与适用边界。

---

## 16. 可直接交给 Codex 的启动说明

> 请以当前 SafeDL/FITWMAMS 的 CIH-WM 为唯一主线执行本任务书，不回退到旧 `a2_human_calibration` 架构，不迁回 HighwayEnv，不重新训练 Flow、Diffusion、事实层和既有响应先验。先在 `conda activate tread` 环境记录 HEAD、工作区 diff 与权重哈希。核对本任务书引用的提交与本机差异，不 reset 用户改动。
>
> 先修复真实状态监督、共享动作映射、恢复状态机、PPO active/log-prob、参考来源及正式随机重放契约。保留当前 parent-aware IDM、直接作用域、横向关闭及禁止 nominal shadow action 的设计。replay/snapshot/prefix identity 任一失败时，不得开始长期训练。
>
> 完成最新全槽位底座审计后，仅训练一个小型 calibrator；自然事件用完整序列 fair Energy Score，synthetic 用有范围的机制辅助，不使用伪人类动作标签，不以所有碰撞消失作为唯一目标。正式 validation 必须同时包括 factual、same_rear、自然序列得分、PPO 对自身 supervised 的增量及因果探针。test 必须绑定 checkpoint 内容哈希和已接受 validation，一次执行后不调参。
>
> 新增函数和测试按本文契约实现，不能只修改注释和 schema 名称。每个阶段产出 machine-readable evidence；失败保留 checkpoint 与 decision，不自动扩大预算、不新增模型族、不自动运行 AMS。

---

## 附录 A. 本文核对的源代码与结果索引

下列链接全部固定到本次核对提交，不随 `main` 后续更新而变化。代码事实来自这些源文件；拟新增设计不是源文件已有功能。正文中的组合编号或区间表示相应的一组文件。

### [S01] `hierarchical_world_model/README.md`

当前 CIH-WM 定位、正式执行链、历史掩码指标与全槽位尚未完成评测的说明。

[查看固定提交中的文件](https://github.com/SafeDL/FITWMAMS/blob/82e71e7ceabc993db307179301b296be019e6187/hierarchical_world_model/README.md)

### [S02] `hierarchical_world_model/src/cih_model.py`

`CausalInfluenceHierarchicalWorldModel`、`build_response_policy`、冻结层与 scope 校验。

[查看固定提交中的文件](https://github.com/SafeDL/FITWMAMS/blob/82e71e7ceabc993db307179301b296be019e6187/hierarchical_world_model/src/cih_model.py)

### [S03a] `hierarchical_world_model/config/cih_world_model.yaml`

唯一 CIH 方法配置：直接作用域、jerk=24、calibrator、监督/策略预算和验收字段。

[查看固定提交中的文件](https://github.com/SafeDL/FITWMAMS/blob/82e71e7ceabc993db307179301b296be019e6187/hierarchical_world_model/config/cih_world_model.yaml)

### [S03b] `hierarchical_world_model/config/world_model.yaml`

冻结事实 checkpoint、全背景车 scope、非劣性阈值及生成依赖。正文 [S03] 同时指向 S03a/S03b。

[查看固定提交中的文件](https://github.com/SafeDL/FITWMAMS/blob/82e71e7ceabc993db307179301b296be019e6187/hierarchical_world_model/config/world_model.yaml)

### [S04] `hierarchical_world_model/src/reaction_controller.py`

`MechanismGuidedResponsePrior`、`FrozenResponsePrior`、`HumanResponseCalibrator`、`map_calibration_tensors`、`forward` 与概率评估。

[查看固定提交中的文件](https://github.com/SafeDL/FITWMAMS/blob/82e71e7ceabc993db307179301b296be019e6187/hierarchical_world_model/src/reaction_controller.py)

### [S05] `hierarchical_world_model/src/influence_graph.py`

`CausalInfluenceGraph.update`：直接父子关系、clearing/recovery 递推与权限。

[查看固定提交中的文件](https://github.com/SafeDL/FITWMAMS/blob/82e71e7ceabc993db307179301b296be019e6187/hierarchical_world_model/src/influence_graph.py)

### [S06] `hierarchical_world_model/src/cih_training.py`

监督缓存、监督 loss、策略 trace/buffer/update、成对机制缓存与 loss。

[查看固定提交中的文件](https://github.com/SafeDL/FITWMAMS/blob/82e71e7ceabc993db307179301b296be019e6187/hierarchical_world_model/src/cih_training.py)

### [S07] `hierarchical_world_model/src/evaluation.py`

当前正式 `rollout`；base hook、输入时序、噪声来源、状态更新和 diagnostics。

[查看固定提交中的文件](https://github.com/SafeDL/FITWMAMS/blob/82e71e7ceabc993db307179301b296be019e6187/hierarchical_world_model/src/evaluation.py)

### [S08] `hierarchical_world_model/src/planner.py`

anchor-only fallback；`frozen_diffusion_plans` 的 Dataset 条件及缓存。

[查看固定提交中的文件](https://github.com/SafeDL/FITWMAMS/blob/82e71e7ceabc993db307179301b296be019e6187/hierarchical_world_model/src/planner.py)

### [S09] `results/hierarchical_world_model/cih_wm/candidate_unaccepted/diagnostic_policy_validation_256.json`

256 序列未接受 PPO 候选诊断；历史 masking、same_rear 误差及因果探针数据。

[查看固定提交中的文件](https://github.com/SafeDL/FITWMAMS/blob/82e71e7ceabc993db307179301b296be019e6187/results/hierarchical_world_model/cih_wm/candidate_unaccepted/diagnostic_policy_validation_256.json)

### [S10] `results/hierarchical_world_model/cih_wm/candidate_unaccepted/manifest.json`

冻结事实/A2 权重哈希、训练事件数量、执行器和已报告训练预算。

[查看固定提交中的文件](https://github.com/SafeDL/FITWMAMS/blob/82e71e7ceabc993db307179301b296be019e6187/results/hierarchical_world_model/cih_wm/candidate_unaccepted/manifest.json)

### [S11] `results/hierarchical_world_model/cih_wm/candidate_unaccepted/diagnostic_supervised_validation_256.json`

相同历史诊断链的 supervised 候选指标。

[查看固定提交中的文件](https://github.com/SafeDL/FITWMAMS/blob/82e71e7ceabc993db307179301b296be019e6187/results/hierarchical_world_model/cih_wm/candidate_unaccepted/diagnostic_supervised_validation_256.json)

### [S12] `hierarchical_world_model/src/human_response_training.py`

现有 V-statistic Energy Score、centered LOO、加权前缀和机制辅助。

[查看固定提交中的文件](https://github.com/SafeDL/FITWMAMS/blob/82e71e7ceabc993db307179301b296be019e6187/hierarchical_world_model/src/human_response_training.py)

### [S13] `hierarchical_world_model/scripts/evaluate_cih_world_model.py`

当前正式评测与 acceptance；缺少自然响应 ES、方向分母、non-event 及哈希绑定问题。

[查看固定提交中的文件](https://github.com/SafeDL/FITWMAMS/blob/82e71e7ceabc993db307179301b296be019e6187/hierarchical_world_model/scripts/evaluate_cih_world_model.py)

### [S14] `hierarchical_world_model/scripts/train_cih_world_model.py`

训练阶段、cache、contrast refresh、PPO loop、manifest 与事件键。

[查看固定提交中的文件](https://github.com/SafeDL/FITWMAMS/blob/82e71e7ceabc993db307179301b296be019e6187/hierarchical_world_model/scripts/train_cih_world_model.py)

### [S15] `diffusion/src/data.py`

`BackgroundTrajectoryDataset.__getitem__` 从真实 NPC 未来提取 K；`prepare_flow_condition` 和外部条件转换。

[查看固定提交中的文件](https://github.com/SafeDL/FITWMAMS/blob/82e71e7ceabc993db307179301b296be019e6187/diffusion/src/data.py)

### [S16] `hierarchical_world_model/src/environment.py`

供测试 fixture 使用的 `ClosedLoopWorld`/snapshot；非当前完整 CIH 正式执行器。

[查看固定提交中的文件](https://github.com/SafeDL/FITWMAMS/blob/82e71e7ceabc993db307179301b296be019e6187/hierarchical_world_model/src/environment.py)

### [S17] `hierarchical_world_model/src/protocol.py`

现有 factual 绝对阈值、协议与工件校验逻辑。

[查看固定提交中的文件](https://github.com/SafeDL/FITWMAMS/blob/82e71e7ceabc993db307179301b296be019e6187/hierarchical_world_model/src/protocol.py)

### [S18] `hierarchical_world_model/src/reaction_evidence.py`

事件定义、25 帧历史/短响应、75 帧恢复窗口等常量；执行时须核对实际缓存。

[查看固定提交中的文件](https://github.com/SafeDL/FITWMAMS/blob/82e71e7ceabc993db307179301b296be019e6187/hierarchical_world_model/src/reaction_evidence.py)

---

## 附录 B. 评分方法的外部依据及适用边界

本次以仓库为主要依据，没有重新宣称逐篇阅读前面会话的全部交通仿真 PDF。为核对第 8 节的评分统计含义，使用下列原始研究的公开出版信息与评分框架；具体 finite-K 公式及梯度关系在本文中单独给出，并通过附录 C 的独立例子核验。

1. Gneiting, T.; Raftery, A. E. **Strictly Proper Scoring Rules, Prediction, and Estimation**. *Journal of the American Statistical Association*, 2007, 102(477): 359–378. DOI: `10.1198/016214506000001437`。
2. Ferro, C. A. T. **Fair scores for ensemble forecasts**. *Quarterly Journal of the Royal Meteorological Society*, 2014, 140(683): 1917–1923. DOI: `10.1002/qj.2270`。在线首次发表为 2013 年，卷期年份为 2014 年。

两项来源支撑的是 proper scoring 与有限 ensemble 评价的统计原则，**不支撑“本文已学会真实人类反事实”“PPO 整体无偏”“新目标一定更有效”等更强主张**。对当前 CIH-WM 的有效性仍必须使用第 10 节的独立实验。

不同评分空间的解释需要保留：对 `[acceleration, abs_jerk]` 的序列校准，不等于对完整多车联合行为分布的证明；其下游 gap、TTC、恢复及风险仍应独立检查。存在未观测驾驶员状态时，得到的是观察条件下的行为预测，不自动具有反事实可识别性。

---

## 附录 C. 本次实际执行的四项低成本检查

**范围声明**：以下是在隔离的小脚本中复写公式、枚举有限状态或递推状态机的检查。没有加载 FITWMAMS 的训练数据、checkpoint 或 GPU 环境，不等于仓库完整单测通过，也不等于修复后模型通过验收。

### C1. Fair Energy Score 的 score-function 验证

使用可枚举的二元行为分布，真实概率 `p=0.7`，模型参数 `theta=0.3`，`K=4`。枚举全部观测值与模型未来样本组合，比较总体负 Energy Score 的解析梯度与第 8.3 节的估计器期望：

```text
解析期望梯度：0.8000000000000000
枚举估计结果：0.7999999999999998
结论：该例中的数值一致性通过。
```

这验证了本文给出的系数和符号在该最小例子中自洽，不是高维交通 PPO 收敛证明。

同一个例子中，当前 centered LOO + V-statistic 的“按 rollout score-function 求和”结果为约 `0.6666666667`。不能仅据这个不同数值将原算法称为符号错误：它针对不同 finite-ensemble 目标与归一化。本轮要求将其目标写清楚，而不是靠 loss 名称判断正确性。

### C2. 上一版加权前缀增量的代数检查

旧建议：

\[
0.25S_5+0.35(S_{10}-S_5)+0.40(S_{25}-S_{10})
\]

实际展开：

\[
-0.10S_5-0.05S_{10}+0.40S_{25}.
\]

因此本轮撤销“此构造只是更细的等价 credit assignment”的说法。它已经改变目标，而且早期分量带负权重。首版采用单一完整序列目标，先排除这种目标混淆。

### C3. 当前恢复递推检查

从 `old.phase=1` 开始，令冲突连续消失、距离持续增大，按当前 `influence_graph.update` 中的 phase/safe-counter 公式递推 20 帧：

```text
phase：0, 0, 0, 0, ...（20 帧均为 0）
未进入 phase=2。
```

这定位的是 clearing 阶段计数被切断的问题，不是关于整套交通模型稳定性的判断。修复后的测试应检查第 13 个连续安全帧能按约定进入恢复，风险重现时计数与父子关系合理更新。

### C4. 关闭 gate 的行为检查

设置 `factual_base=-0.2`、`prior_proposal=-2.0`、无前序校准、`event_gate=0`。按当前动作映射得到：

```text
最终输出：-2.0
不是：-0.2
```

这与当前源码注释一致，说明“关闭校准”保护先验 proposal，而不是自动保护事实动作。它不是计算错误，而是与事实锚定目标之间的职责错位。

---

## 附录 D. 可直接实现的评分纯函数契约

下面代码用于说明新评分层的最小接口与系数，属于**待并入项目并补齐测试的参考实现**。它不负责 rollout、碰撞截尾、数据筛选或 PPO；这些必须遵守正文契约。历史 score 保留原函数，不覆盖。

```python
from __future__ import annotations

import math
import torch


def fair_event_energy_with_loo(
    futures: torch.Tensor,       # [K, T, C]
    observed: torch.Tensor,      # [T, C]
    channel_scale: torch.Tensor, # [C], train-defined and strictly positive
    *,
    normalize_horizon: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fair ES and per-future score-function rewards.

    Sampling assumptions: independent futures conditional on the declared
    event/context; no identical copies used as separate futures.
    Reward convention: maximize -ES, event loss averages over futures.
    """
    if futures.ndim != 3 or observed.shape != futures.shape[1:]:
        raise ValueError("Expected futures [K,T,C] and observed [T,C]")
    k, t, c = futures.shape
    if k < 3 or t < 1 or channel_scale.shape != (c,):
        raise ValueError("LOO requires K>=3, T>=1 and scale [C]")
    scale = channel_scale.to(futures)
    if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
        raise ValueError("Channel scale must be finite and positive")
    if not bool(torch.isfinite(futures).all()) or not bool(torch.isfinite(observed).all()):
        raise ValueError("Invalid response: use the declared failure protocol")

    horizon_scale = math.sqrt(t) if normalize_horizon else 1.0
    x = (futures / scale[None, None, :] / horizon_scale).flatten(1)
    y = (observed.to(futures) / scale[None, :] / horizon_scale).flatten()
    target_distance = torch.linalg.vector_norm(x - y[None, :], dim=1)
    pair_distance = torch.cdist(x, x)
    pair_distance = pair_distance.masked_fill(
        torch.eye(k, device=x.device, dtype=torch.bool), 0.0
    )

    target_sum = target_distance.sum()
    pair_sum = pair_distance.sum()
    score = target_sum / k - pair_sum / (2.0 * k * (k - 1))

    # Removing row i removes its distances to all other samples twice.
    leave_one_out = (
        (target_sum - target_distance) / (k - 1)
        - (pair_sum - 2.0 * pair_distance.sum(dim=1))
        / (2.0 * (k - 1) * (k - 2))
    )
    rewards = k * (-score + leave_one_out)
    return score, rewards.detach()
```

必须验证：与逐条删除样本的朴素实现数值一致；K=3 的边界正确；尺度来源只来自 train；全部相同样本不产生 NaN；移除某条样本的方向解释符合定义；loss 归一化和第 8.3 节公式一致。

本次已在独立随机张量上运行该参考函数：K=3/4/8/32 时与逐条删除的朴素实现一致，最大 reward 绝对差约 `2.14e-14`；全部相同的零序列测试也通过。此检查不涉及项目数据或训练，不能替代集成后的 policy-gradient、mask 与 rollout 验证。

**最后的执行原则**：本轮不是通过修改名称、阈值或选择范围使旧结果“变为通过”，而是让当前 CIH-WM 的训练证据、在线信息、动作概率、执行后果和验收指标真正描述同一个系统。保留已有能力，先修正不一致，再用一次有限预算的候选实验检验改进。
