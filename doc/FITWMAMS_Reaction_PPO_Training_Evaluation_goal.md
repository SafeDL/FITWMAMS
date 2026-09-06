# Goal：分层行为世界模型的 PPO 响应训练与 A2 对照验收

> 任务性质：在现有实现上补齐必要正确性检查，完成候选响应策略的训练、评测和决策，不重新设计世界模型。
>
> 审阅基线：SafeDL/FITWMAMS，提交 `3ce752e0576ccbd4a8c77d527e3f8aff5e0055b1`，2026-09-06。
>
> 本文件区分“当前代码已有事实”和“本次执行要求”。它不是已经完成训练或通过验收的结果报告。执行时以本地实际 HEAD 为准核查差异，不回滚或覆盖用户改动。

## 1. 唯一目标

在冻结的 Diffusion–HiQR 高保真交通参考之上，训练一个由 PPO 优化的纵向响应策略，使其在自然事件中产生与日志证据一致的最终行为，在外部 ego 改变已实现交互条件后产生必要、局部、物理可行的响应。

主要比较对象是：**相同新版运行时、相同参考协议下重新评测的 A2-transfer**。同时保留冻结 HiQR、IDM-only 和候选的监督预训练版本，用来回答“是否保持重建精度”“是否优于规则控制”和“PPO 是否提供额外价值”。

本轮交付不是更多架构建议，而是可复现的训练工件、统一口径的比较结果，以及 `accept / reject / inconclusive / blocked` 之一的明确决定。失败时保留证据，不为通过门槛临时更改目标、权重或测试集。

## 2. 首先澄清 PPO 和 IDM 的分工

### 2.1 PPO 没有被删除

当前 `src/reaction_training.py` 包含监督初始化、GAE、PPO clipped objective、critic 更新和闭环 rollout。PPO 是**训练响应 actor 的算法**，不是运行时额外调用的驾驶模型。

训练阶段：

`自然事件/普通跟驰/合成压力 → HighwayEnv rollout → PPO 更新响应 actor/critic`

执行阶段：

`冻结 HiQR 基础动作 → 已训练响应 actor → 执行约束 → HighwayEnv`

正式仿真执行冻结 actor；critic 不决定动作；不允许接入 ADS 后在线更新 PPO。[S2]

### 2.2 IDM 仍然计算参考动作，但不同控制器用法不同

| 控制器 | HiQR 的作用 | IDM 的作用 | PPO 的作用 |
|---|---|---|---|
| `frozen_hiqr` | 直接产生背景动作 | 不使用 | 不使用 |
| `idm_only` | 作用域外维持基础行为 | 作用域内给出规则动作 | 不使用 |
| `a2_transfer` | 基础动作 | 进入旧动作融合/约束映射 | 执行冻结旧 PPO actor |
| `calibrated_residual` | 唯一的残差基点 | 计算参考加速度，作为 actor 输入 | 学习有符号修正及修正权重 |

当前候选的设计为：

\[
a^{\rm base}_{i,t}=\pi_{\rm HiQR}(H_t,K_{\rm ref},\xi_t),
\qquad a^{\rm IDM}_{i,t}=f_{\rm IDM}(H_t;\hat\eta),
\]

\[
a^{\rm desired}_{i,t}=a^{\rm base}_{i,t}
+\Delta a_\theta(H_t,a^{\rm base}_{i,t},a^{\rm IDM}_{i,t},K_{\rm ref},\mathcal G_t,\xi^R_t),
\]

\[
a^{\rm exec}_{i,t}=\mathcal P(a^{\rm desired}_{i,t},a^{\rm exec}_{i,t-1},H_t).
\]

`calibrated_residual` 中 IDM 不是强制制动下限，不再直接替换 HiQR 的残差基点。A2 的旧 IDM 融合模式单独保留为对照，不在本轮重新切换候选的动作语义。[S3]

注意：零残差保证的是 **desired action 等于 base action**；若执行层 guard/jerk limiter 被触发，executed action 仍可能变化。只有作用域关闭，或执行约束不改变该动作时，才要求最终动作精确透传。不得将三者混淆。

### 2.3 本轮明确不做

不恢复 GAIL/MLOO，不新增 MOBIL 在线横向控制，不重新训练 Flow、Diffusion、HiQR 或 IDM 参数，不引入其他 RL 算法，不添加对抗奖励、EVT 奖励或促使 ADS 失效的奖励，不自动修改正式 `release.yaml`，不启动新的 MC/AMS 概率主实验。

候选本轮验证重点是同车道纵向跟随响应；次级传播、切入和其他角色只按实际已实现范围报告，不把几何图支持的角色全部说成人类行为已经校准。

## 3. 冻结实验条件与比较臂

### 3.1 冻结内容

运行前生成 `run_manifest.json`，记录：源提交及工作树 diff、环境、数据 split/事件表、世界模型/Diffusion/IDM/A2 checkpoint 哈希、参考生成方式、控制器映射、执行约束、随机 schema、训练配置、评测配置和验收阈值。

固定 25 Hz、149 个状态转移、当前 fixed-K 条件重仿真协议，使用 `HighwayEnvClosedLoopWorld` 作为唯一正式执行后端。保留有效 `same_rear`，不得继续用旧 follower-excluded 数据屏蔽后车。

所有比较臂使用相同初态、车辆有效性、车道几何、外层物理后端、参考协议和共同外生随机数组。不能只写“seed 相同”而不核查实际消费的随机块。A2 的内部历史动作映射及其自带 guard 属于基线机制，应保留并披露；不得静默改写后仍称为冻结 A2，也不得把不同内部约束造成的收益全部归于网络权重。

参考逐槽位记录来源：`diffusion_from_K_GT` 或 `logged_future_completion`。当前 `complete_missing_background_plans()` 会使用完整日志未来补齐缺失有效槽位，尤其是历史上被屏蔽的后车；这必须披露，不能统一称为“仅根据稀疏 K 生成全部未来”。[S7]

### 3.2 必要比较臂

| 名称 | 目的 | 是否新增训练 |
|---|---|---|
| `frozen_hiqr` | 事实精度锚点，观察未加响应策略的行为 | 否 |
| `idm_only` | 分离规则先验的贡献 | 否 |
| `a2_transfer` | 当前主比较基线 | 否，旧权重冻结，在新协议下重评 |
| `calibrated_supervised` | 检验仅监督初始化的效果 | 是，保存 PPO 前的候选 checkpoint |
| `calibrated_residual` | 检验监督初始化后 PPO 的增量 | 是，从同一个 supervised checkpoint 继续 |

`calibrated_supervised` 只是同一候选网络的预训练工件别名，不新增控制器架构。已有 `a1_transfer` 可作补充，但不是本轮必需的大规模训练臂。

**A2-transfer 是效果对照，不是人类真值。**其旧人工触发协议的历史指标不能直接放进新版结果表。迁移了运行时/参考协议后必须重新生成指标。

## 4. 训练前只修复必要的一致性问题

对以下项目先检查现有实现；若已修复，仅补测试或记录，不重复改造。遵循仓库 `doc/style.md`（若存在）及现有目录规范。

### 4.1 自主作用域与恢复状态机

输入只允许已实现历史、参考条件、先前动作、内部状态与显式随机变量；事件类别及未来 ego 控制不得进入策略或作用域开关。

检查 `CausalInfluenceGraph`：冲突解除后，在累计安全帧达到门槛之前保留等待/交互状态，之后进入 recovery 并平滑释放。不得在第一个安全帧立即清零 phase，从而使累计计数永远达不到门槛。

增加“冲突 → 连续安全等待 → 恢复 → 释放”和“恢复途中再发生冲突”的合成测试。[S8]

### 4.2 不把 A2 权重加载成功当作行为初始化等价

当前训练入口将 A2 完整 `state_dict` 传给候选，但 A2 和候选的 raw-action 映射不同；维度兼容不代表相同 raw output 有相同动作含义。[S3][S4]

本轮默认：A2 checkpoint 只用于冻结对照。候选保留其零均值输出头初始化；若复用 A2 特征层，只复制语义与尺寸一致的 actor 隐藏层，重新初始化动作头、方差和 critic，并记录复制清单。不得无说明加载旧动作头覆盖零修正初始化，也不将其称作 A2 策略的等价微调。

必须检查初始化后的 desired/exec 动作、随机样本幅度和约束触发率。零均值并不代表随机策略是零残差策略。

### 4.3 训练监督与实际执行动作对齐

同一控制器的预训练与评测必须使用一致的动作映射、历史定义、有效 mask、上一帧实际动作及执行约束。当前预训练主要拟合 guard 前的 desired action，不能直接称为最终执行动作监督。[S2][S3]

对有自然标签的状态，比较最终执行动作与标签；复用统一可执行映射，或显式说明被 guard 覆盖的样本无法通过 actor 任意拟合，并单独统计。不要通过改变 guard 来追逐精度，也不要声称对执行层反向传播就是对完整 HighwayEnv 求导。

既允许必要制动，也允许零修正与恢复性动作。不得恢复 residual-dose floor，不要求后车残差与 ego 剂量同比，不要求无条件追上原日志时序。

### 4.4 统一时间、评分与碰撞口径

- 状态、执行加速度和 human target 使用一致的帧定义。jerk 首帧必须由事件前最后一个实际动作计算，内部选模和最终评测一致，不设成零。
- acceleration 和 jerk 拼接评分前，用训练集冻结的尺度标准化；所有臂、所有 split 共用尺度。保留原始单位分量指标。
- 时延统一为秒；未在观察窗内产生可检测响应时标记为删失/未响应，不伪造一个精确时延。阈值及持续帧数只由训练协议确定。
- 恢复误差分别评价速度、间距、相对速度等分量，不直接把不同量纲拼成未归一化的状态向量范数。
- `rear_collision` 必须来自碰撞参与车辆及实际前后关系；“任意 NPC crashed”只能称为 NPC-involved collision，不能误标为后车追尾。[S5]
- guard 激活条件比例和实际改写动作比例分别报告。仅统计 `TTC < 2` 不是实际 guard 干预率。

## 5. 数据与证据准备

复用 `prepare_reaction_evidence.py` 和 `reaction_evidence.py`，不另建一套数据工程。[S6]

当前来源是 canonical 七车片段及源 metadata；真实车辆 ID 用于回溯和去重，但尚不能称为已扫描全部原始 recordings。按实际来源写审计报告。

训练、validation、test 按 recording 隔离。事件唯一标识使用 recording、leader、follower 和绝对 onset；不能把重叠窗口、32 个随机未来或同一事件的多帧当成独立人类样本。

沿用 train 定义的支持门槛：每个 cell 至少 100 个去重事件，来自至少 5 个 recording。分别报告全部事件、可重仿真事件、`leader_slot == 0` 事件及各 cell 的计数。若某个工况不满足要求，标记支持不足；不降低门槛以获得通过结果。

自然事件保留实际 onset 和完整行为过程。onset 强度与事件后续峰值分别记录；后者仅用于离线统计，不作为当前时刻输入。匹配非事件需要确认目标窗口内确实没有所研究的制动事件，不能仅因不属于 supported event pool 就认为没有事件。

合成压力工况只用于闭环物理训练；其 `human_target_gate` 必须为零，不用原日志后车未来作为强 ego 干预的人类真值。

## 6. 分阶段执行

### Stage 0：预检和基线冻结

1. 解析现有 A2、HiQR、Diffusion、规则模型工件，检查 schema/hash 和数据可用性。不猜测不存在的路径，不覆盖旧实验。
2. 修复第 4 节的必要契约，执行相关单元和最小集成测试；保存 `preflight.json`。
3. 选定训练内的小型固定事件集合，包含事件和匹配非事件；此集合仅用于目标正确性检查，不用于最终有效性结论。
4. 在相同新版运行时得到 frozen HiQR、IDM-only、A2-transfer 和初始化候选的基线。保存 epoch/update 0 的指标，不能等训练完成才发现已有基线更好。

若预检失败，输出具体原因和证据，决策为 `blocked`。不得因此推导“RL 不必要”。

### Stage 1：自然事件监督初始化

仅使用 train 中的事件与匹配非事件，以最终执行动作及连续行为为监督目标；不更新主干和规则参数。

保存 `supervised.pt`，作为独立的 `calibrated_supervised` 对照。记录优化步数、唯一事件覆盖率和实际样本数。原配置 `pretrain_epochs=3` 对应的实现不能未经核对就解释为遍历全数据三轮。

预训练使用当前配置作为起点。只有在同一固定场景、同一随机数组下观察到真实目标下降，且随机执行没有显著异常，才继续 PPO。4 个事件/8 个未来可作为软件 smoke 检查，不能作为统计准入证据；正式 pilot 的事件应覆盖可用 recording 和支持 cell。

随机尺度必须声明：若此阶段只优化均值、保持方差固定，就如实记录。不得将均值拟合改善自动解释为人类随机分布已经校准。

### Stage 2：PPO 闭环训练

从同一个 `supervised.pt` 继续训练 actor/critic，监督版本保持冻结供对照。

沿用当前三流 batch：32 个真实响应事件、16 个匹配普通跟驰事件、16 个合成压力案例；保持 25 Hz、149 步。自然标签用于最终行为一致性；synthetic 只使用当前已声明的碰撞、TTC/间距和动作/jerk 可行性目标。

保留现有 PPO clipped objective、GAE、critic 和显式策略随机块。不添加 GAIL、MLOO 或新奖励头。当前做法是“监督初始化 → 含自然行为误差奖励的 PPO”；除非代码实际加入持续监督损失，不称为每个 PPO minibatch 都执行监督联合优化。

每次更新记录自然流与 synthetic 流各自的 reward 分量、有效策略动作数、desired/exec 差异、均值/方差、critic loss、entropy、clip fraction、数值异常及耗时。避免仅报告一个总 reward。

先完成小预算 pilot，默认最多 50 个 PPO updates；保存 checkpoint 0、supervised 和所有 validation checkpoints。通过 pilot 后，再在同一冻结配置和随机状态下继续，单次正式预算最多沿用 700 updates，不自动扩为多轮权重搜索。继续训练的依据是人类事件指标或独立 OOD 响应诊断有改善，并且不存在事实明显退化/数值异常；总 reward 上升不是唯一依据。

每 10 updates 使用固定 validation event IDs 和固定随机未来评测。没有改善时按既定 patience 早停；必须保留 PPO 前的 supervised 模型为候选，不强制选某个已被 PPO 更新的模型。

保存 sampler、优化器、随机状态、checkpoint/config/data hash。resume 时核查协议一致性，不将不同配置接成一条训练曲线。

### Stage 3：冻结后统一评测

先在 validation 做选择和验收。只有策略、超参数、尺度、作用域、统计量、碰撞规则和随机未来数量全部冻结后，才对 test 做一次最终比较；test 失败不得返回去针对同一个 test 调参再宣称独立测试。

三套协议必须分开：

1. **完整 factual replay**：日志 ego 控制，响应器自主正常工作；不能因“这是日志”而强制关闭。按真实完整 split 报告样本数、总体及角色分组 ADE/FDE/P95。
2. **真实响应事件**：每个 held-out 事件固定相同初态和参考，生成 32 个不同的随机未来；所有臂使用相同外生数组，评价事件级分布及最长 3 秒恢复。
3. **合成压力**：沿用 -2/-4/-6/-8 m/s² 制动，以及未用于训练的 ramp/pulse。它们是物理压力测试，不是人类反事实真值。明确实际使用的 OOD context 数量；若只用 64 条，写明为固定小样本诊断，不称全量验证。

本轮默认按现有一个训练 seed 完成主流程，32 个 rollout seeds 不能替代多个训练 seeds。若只完成一个训练 seed，结论限定为本次可复现实验；不要声称稳定的跨种子普遍改善，也不自动启动额外大预算训练。

## 7. 指标与比较规则

### 7.1 主要效果和对照分别是什么

- **候选是否比现有方案更好？** 主要与 `a2_transfer` 比。
- **是否保持高精度重建？** 对比新版同协议的 `frozen_hiqr`，同时报告 A2 的事实误差。
- **PPO 是否必要？** 对比 `calibrated_supervised` 与 `calibrated_residual`。
- **是否只是规则模型在起作用？** 对比 `idm_only`。

不使用旧 A2 的历史低碰撞率直接作为新协议中的真值。

### 7.2 事件级自然性主指标

设第 e 个事件观测到的 acceleration＋absolute jerk 序列经训练尺度标准化后为 y_e，同一条件下生成 N=32 个未来 X_e^(n)。沿用当前经验 Energy Score 形式：

\[
ES_e=\frac1N\sum_{n=1}^{N}\|X_e^{(n)}-y_e\|_2
-\frac{1}{2N^2}\sum_{n=1}^{N}\sum_{m=1}^{N}\|X_e^{(n)}-X_e^{(m)}\|_2.
\]

ES 越低越好。固定公式、N、时间窗和尺度，不混用不同定义的历史数值。同步报告 acceleration、jerk 的分量误差、分布距离和样本多样性，不以单一 ES 证明全部交互真实性。[S6]

主比较：

\[
\Delta ES_e=ES_e^{A2}-ES_e^{candidate}.
\]

按 recording 做配对簇 bootstrap；以单侧 95% 下界大于 0 作为人类事件主指标的改善依据。按 cell/recording 报告结果，披露独立 recording 数。如果样本不足或区间过宽，写 `inconclusive`，而不是“已等价/没有必要”。

### 7.3 事实保持

沿用现有数值作为**本轮预登记工程阈值**，不是领域通行安全标准：[S5]

| 指标 | 相对 A2-transfer 最大允许绝对增加 | 同时满足的相对增加上限 |
|---|---:|---:|
| ADE | 0.02 m | 5% |
| FDE | 0.06 m | 5% |
| P95 位移误差 | 0.10 m | 5% |

必须同时列出冻结 HiQR 的实际数值和候选相对它的误差变化。若 A2 本身已明显退化，候选仅优于 A2不能被总结为“保住了原 HiQR 精度”；这种情况下只能判定局部响应改善，不能作完整模型的发布推荐。

确定性事实指标用于保持与现有协议一致；随机策略的自然性使用随机 rollout 单独评价，不把二者混成一个统计量。

### 7.4 响应诊断

评价响应时延、峰值加速度、jerk、gap、closing speed 和恢复过程。保持当前按 train IQR 设置允许退化量的原则；现有阈值为 0.1×train IQR，评估前固定。[S1][S5]

响应强度不是越大越好，残差不是越负越好。人类事件中比较最终行为与实际观测；合成压力中只评价物理与闭环合理性。

### 7.5 物理与碰撞

硬契约：有限数值、合法动作、正确 mask、不偷看待执行 ego 命令、时序一致、执行层 jerk 契约、相同随机状态可重放。物理限制与 guard 是模型假设，不是人类真值。

分别报告实际 guard 改写率、改写幅度、碰撞参与者类型、near/far 局部性、恢复阶段持续制动及异常追赶。不允许用“低 TTC 条件满足率”冒充实际 guard 修改率。

当前验收器未把碰撞差异纳入独立统计门槛；本轮必须补报告候选相对 A2 的成对碰撞率差及区间。[S5][S9] 只要出现明确恶化证据，不推荐发布；区间很宽则标记证据不足。若已有预登记碰撞非劣界，按该值使用；若没有，不得看到 test 后才设置一个有利阈值，也不能把“数值与动作有限”概括为“碰撞已通过”。

零碰撞或零成对差异时，不得将普通退化 bootstrap 的 [0,0] 当作总体风险无不确定性；应披露样本量并使用适合稀疏事件的区间/上界。

## 8. 决策与论文结论

### 可推荐候选的必要证据

在人类事件主指标改善、诊断不明显退化、事实精度保持、物理契约有效、碰撞没有明确恶化且证据足够的情况下，才给出替换 A2-transfer 的**候选建议**。本文件不授权自动覆盖正式发布配置。

### 必须单独回答 PPO 的贡献

- 候选优于 A2，但 supervised-only 已达到同等效果：记录候选有效，PPO 增量未证实。
- PPO 比 supervised-only 改善未见工况处理，同时自然性和事实保持：可主张 PPO 有闭环增量。
- PPO 只降低碰撞，但自然响应明显失真：不能称为更高保真人类模型。
- 未达门槛：保留 A2-transfer 对照，保存负结果，不自动恢复 GAIL/MLOO。

所有结论限定为当前参考条件和验证范围。不能把具有人类日志参考的条件重仿真描述为任意 ADS 下真实反事实过程已被识别，也不能将 OOD 物理压力结果描述为真实人类事故率。

只有响应模型独立验收通过后，才另立 all-slot EVT 标定、同一冻结世界下 MC/AMS 一致性和背景模型敏感性实验。本轮不输出新的现实失效率。

## 9. 工程交付与命令入口

只复用或最小扩展以下入口，不创建第二套世界模型：

- `hierarchical_world_model/scripts/prepare_reaction_evidence.py`
- `hierarchical_world_model/scripts/train_reaction_policy.py`
- `hierarchical_world_model/scripts/evaluate_reaction_policy.py`
- `hierarchical_world_model/scripts/validate_reaction_policy.py`
- `hierarchical_world_model/config/reaction_policy.yaml`

增加 supervised checkpoint 的保存和评测支持即可，不引入新的网络目录。训练入口中 A2 初始化的语义按 4.2 节纠正；参数可保留作兼容/基线定位，但帮助文本不能继续暗示动作等价。

执行前自动解析并记录以下变量：`CONFIG`、`A2_CHECKPOINT`、`EVENTS_DIR`、`RUN_DIR`、`CANDIDATE_CHECKPOINT`、`SUPERVISED_CHECKPOINT`。路径来自现有配置与 manifest，不让用户手工猜测。先检查 `--help` 和本地环境，再执行；如 `tread` 环境不存在，报告环境问题而非自动新建大型环境。

现有命令入口示意（由 Codex 填入实际已存在变量，修改参数时同步更新帮助与记录）：

```bash
conda run -n tread python hierarchical_world_model/scripts/prepare_reaction_evidence.py \
  --config "$CONFIG" --output-dir "$EVENTS_DIR"

conda run -n tread python hierarchical_world_model/scripts/train_reaction_policy.py \
  --config "$CONFIG" --events-dir "$EVENTS_DIR" \
  --a2-transfer-checkpoint "$A2_CHECKPOINT" --output-dir "$RUN_DIR" --updates 50

conda run -n tread python hierarchical_world_model/scripts/evaluate_reaction_policy.py \
  --config "$CONFIG" --events-dir "$EVENTS_DIR" --split validation \
  --a2-checkpoint "$A2_CHECKPOINT" --candidate-checkpoint "$CANDIDATE_CHECKPOINT" \
  --output "$RUN_DIR/evaluation/validation.json"

conda run -n tread python hierarchical_world_model/scripts/validate_reaction_policy.py \
  "$RUN_DIR/evaluation/validation.json"
```

Pilot 后是否继续到 700 updates，按第 6 节决策。`--updates 700` 表示总更新上限，不是额外追加 700 次。训练前完成验证所需的函数/工件准备；不要在 test 阶段临时补定义。

必要工件目录：

```text
results/hierarchical_world_model/causal_reaction/<unique_run>/
  run_manifest.json
  preflight.json
  data_audit.json
  controllers/calibrated_residual/
    initial.pt
    supervised.pt
    training_progress.pt
    reaction_policy.pt
    training_summary.json
  evaluation/
    validation.json
    test.json                 # 仅在冻结后执行
    comparison.csv
    acceptance.json
    ppo_increment_report.json
  figures/
  playbacks/
  decision.md
```

图表只保留训练/validation 趋势、成对 ES 差、事件响应时序、事实误差与物理诊断四类；选少量语义分层案例显示 base/IDM/A2/supervised/PPO，不按最有利结果筛 GIF。每个案例绑定场景、模型哈希、参考来源和完整随机状态。无需每个阶段制作大量动画。

禁止删除旧 A2 或历史失败工件。运行结束如实输出：完成到哪一步、实际算力与更新预算、哪些门槛通过、哪些失败、PPO 是否有增量、是否具备后续概率实验条件。不得以单元测试通过代替行为有效性。

## 10. 代码依据与本文件新增要求的边界

以下来源均固定于审阅提交，描述当前实现，不代表效果已经通过验证。本文件第 4 节的契约修复、保存 supervised-only 对照、统一指标及分阶段执行，是基于审阅提出的执行要求，而非伪装成仓库已有功能。

[S1] 架构、比较臂与现行协议：
https://github.com/SafeDL/FITWMAMS/blob/3ce752e0576ccbd4a8c77d527e3f8aff5e0055b1/hierarchical_world_model/README.md

[S2] 监督初始化、三流环境、GAE 与 PPO：
https://github.com/SafeDL/FITWMAMS/blob/3ce752e0576ccbd4a8c77d527e3f8aff5e0055b1/hierarchical_world_model/src/reaction_training.py

[S3] IDM-only、旧 A2 映射、有符号候选与执行 guard：
https://github.com/SafeDL/FITWMAMS/blob/3ce752e0576ccbd4a8c77d527e3f8aff5e0055b1/hierarchical_world_model/src/reaction_controller.py

[S4] 候选训练命令与 A2 权重加载：
https://github.com/SafeDL/FITWMAMS/blob/3ce752e0576ccbd4a8c77d527e3f8aff5e0055b1/hierarchical_world_model/scripts/train_reaction_policy.py

[S5] factual、事件、OOD 评测和现有容忍界：
https://github.com/SafeDL/FITWMAMS/blob/3ce752e0576ccbd4a8c77d527e3f8aff5e0055b1/hierarchical_world_model/scripts/evaluate_reaction_policy.py

[S6] 事件证据、Energy Score 和 recording bootstrap：
https://github.com/SafeDL/FITWMAMS/blob/3ce752e0576ccbd4a8c77d527e3f8aff5e0055b1/hierarchical_world_model/src/reaction_evidence.py

[S7] Diffusion 参考及完整日志未来补齐：
https://github.com/SafeDL/FITWMAMS/blob/3ce752e0576ccbd4a8c77d527e3f8aff5e0055b1/hierarchical_world_model/src/planner.py

[S8] 自主作用图与恢复状态机：
https://github.com/SafeDL/FITWMAMS/blob/3ce752e0576ccbd4a8c77d527e3f8aff5e0055b1/hierarchical_world_model/src/influence_graph.py

[S9] 当前自动验收逻辑：
https://github.com/SafeDL/FITWMAMS/blob/3ce752e0576ccbd4a8c77d527e3f8aff5e0055b1/hierarchical_world_model/scripts/validate_reaction_policy.py

[S10] 当前默认训练预算与事件门槛：
https://github.com/SafeDL/FITWMAMS/blob/3ce752e0576ccbd4a8c77d527e3f8aff5e0055b1/hierarchical_world_model/config/reaction_policy.yaml
