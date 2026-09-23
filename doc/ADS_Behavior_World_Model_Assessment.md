# 面向自动驾驶测评的交互行为世界模型盘点

参考文献推导出的完整评价维度、全量 highD 覆盖定义和分协议数值总表见
[`HighD_Behavior_World_Model_Evaluation.md`](HighD_Behavior_World_Model_Evaluation.md)。本文件保留
架构取舍，数值比较以该总表为准。

## 结论

自动驾驶测评需要的不是“轨迹预测最准”或“NPC 会动”二选一，而是同一个可执行模型同时
满足：在 ADS 没有改变场景时高精度复现真实长尾轨迹；在 ADS 改变动作后，相关 NPC 能从
当前状态自主、因果一致地反应；长时间闭环不靠大量碰撞、越界制造所谓压力。

按仓库中的完整 test 证据，**当前还没有候选同时满足这三点**：

- 冻结事实 CIH/HiQR 是最好的轨迹锚点，测试 ADE/FDE 为 0.0400/0.0363 m，但现有自主
  响应候选未通过机制验收，不能宣称为已完成的交互世界模型。
- B2-RL/B2-RLP-e3 是当前最完整的自研闭环 NPC 候选，但自然测试 ADE 约 0.624 m、
  FDE 约 2.16 m；横向响应率有所提高，真正的主动横向避让率仍约 0.1%。
- TrafficBots 是当前最有价值的外部多智能体基线，但 highD 适配后的自然测试 ADE/FDE
  为 0.672/1.955 m，不能保持 CIH 级别的事实复现。
- B-IDM、MA-IDM、Dynamic-AR IDM、Multi-regime B-IDM 和 Active Inference 是局部驾驶人
  机制或纵向基线，不是完整的六车交互世界模型。

因此论文主线应收敛为一条：**以冻结高精度事实轨迹为名义运动，用因果、动作条件的局部
残差策略处理 ADS 引起的偏离**。不再把每个组件或实验候选写成一条平行“世界模型分支”。

## 现有方法的角色与证据

不同方法的条件信息和测试空间不同，下表只用于判断角色，不能把所有误差直接排成一个
排行榜。

| 方法 | 已验证的优势 | 主要不足 | 在统一方案中的角色 |
|---|---|---|---|
| 冻结事实 CIH/HiQR | 全部 10,151 条 test 序列 ADE 0.0400 m、FDE 0.0363 m、P95 0.0910 m | 响应候选的固定窗口方向一致率 0.758、剂量排序率 0.512，均未过预注册 0.95 门槛；未进入确认性 test | 唯一事实轨迹锚点 |
| CIH 响应层 | 保留事实精度；具有明确因果路由和机制约束 | validation 已拒绝，Flow→Diffusion prefix sampler 尚未接入维护中的 rollout，不能用于风险发布 | 重做自主响应层，而非继续追加训练轮次 |
| 约束条件 Diffusion | oracle 状态结点条件下 ensemble ADE/FDE 0.0264/0.0046 m，物理平滑 | 显式使用 2/4/5.96 s 未来结点；低 FDE 不是 C0-only 自主预测，多样性较弱 | 名义轨迹生成器，不是闭环 NPC 策略 |
| 场景条件 Flow | 学习 `p(M)p(C0|M)p(K|C0,M)`；联合 NLL 92.589，物理投影合法 | 少数边际 KS 最高约 0.414；只生成场景和稀疏约束，不响应 ADS | 场景先验和可重放随机状态 |
| B2-RL | prefix-only、动作闭环；纵向配对响应概率 0.348；刺激后新增碰撞率 0.00554 | 自然 ADE/FDE 0.625/2.158 m；cruise-PNC 下 NPC 碰撞 0.00875、越界 0.0878 | 当前自研交互策略基线 |
| B2-RLP-e3 | 纵向响应 0.352，横向响应由 0.134 提高到 0.161 | 自然 ADE/FDE 0.624/2.155 m；主动横向避让 0.00091，提升主要不是有效避让 | 横向机制诊断候选，不晋升主模型 |
| TrafficBots V1.5-highD | 完整多智能体随机策略，闭环自主反应，天然适合作为外部基线 | 自然 ADE/FDE 0.672/1.955 m；真实事件 FES 0.222，明显弱于 B2-RL 的 0.097；当前只是 highD 方法适配 | 必留的外部世界模型基线 |
| IDM lane keep | 简单、稳定、可解释 | 自然 ADE/FDE 1.794/4.559 m，无横向学习和多模态能力 | 下界和控制诊断 |
| MA-IDM | 匹配 36 对 highD 上位置/速度 RMSE 0.222/0.265，四个随机跟驰模型中最好 | 单跟驰纵向机制，没有道路级多车横向交互 | 因果响应 teacher 或消融基线 |
| Dynamic-AR(5) IDM | 同 cohort 位置/速度 RMSE 0.265/0.308，能表达时间相关残差 | 当前部署是稳定 MAP/Laplace 工程近似，不是完整收敛的论文后验 | 纵向随机残差消融 |
| B-IDM | 参数可解释、后验随机性清楚 | 位置/速度 RMSE 0.762/0.571，精度较弱且仅跟驰 | 经典概率基线 |
| Multi-regime B-IDM | 显式驾驶风格和状态切换 | 位置/速度 RMSE 0.631/0.498，部署候选已拒绝 | 只保留复现证据 |
| Active Inference | 对碰撞规避机制和人类反应时间有解释性 | 独立驾驶人/特定规避任务，不构成完整交通世界 | 特定碰撞规避基线 |

事实 CIH 证据来自
`results/hierarchical_world_model/cih_wm/evaluation_index.json` 和
`results/hierarchical_world_model/cih_wm/continuation/decision.json`；交互候选和 TrafficBots
来自 `results/interactive_behavior_world_model/benchmark_v1/aggregate/test_seed_summary.csv`；
随机驾驶人同 cohort 结果来自
`results/driver_reproduction/matched_highd/matched_metrics.json`。Diffusion 和 Flow 的条件边界
分别记录在 `diffusion/README.md` 与 `normalizing_flow/README.md`。

## 推荐的单一世界模型

推荐保留三个职责清晰的层，而不是三个并列模型：

```text
场景先验 Flow ──> 名义事实轨迹/约束 ──> 因果交互残差策略 ──> 统一 25 Hz 交通执行器
                       │                         ▲
                       └──── 当前闭环状态 + ADS 动作 ────┘
```

名义层提供“如果 ADS 沿真实行为运行，场景应如何发展”的高精度参考。交互层只在 ADS 或
邻车使局部状态偏离参考后，预测受影响 NPC 的纵向加速度和横向角速度残差；每 0.2 s 用
真实闭环状态重算，不读取部署时不可获得的日志未来。因果路由限制受影响车辆，未受影响
车辆继续贴近名义轨迹，从结构上兼顾事实精度和交互自由度。

同一执行器支持两种合法场景来源：

- 历史长尾重放：日志未来只定义名义场景；ADS 一旦偏离，NPC 由交互残差策略接管。这用于
  可重复的案例测评，不能把日志未来称为模型预测。
- 生成式测评：Flow 采样初态和约束，Diffusion 生成名义轨迹，交互策略闭环执行。这用于
  新场景与概率风险估计，不允许使用真实未来结点。

这一路线可以直接复用 CIH 事实层、Flow/Diffusion 的可重放随机状态、B2-RL 的 rolling
action 接口和 CIH 的因果路由；随机 IDM 家族只作为 teacher、先验或消融，不再各自扩展成
道路级世界模型。

## 单一晋升门

后续候选必须在同一 release decision 中同时通过以下证据，不能只凭某一张榜单晋升：

1. 事实保持：完整自然 test、真实事件和长尾分层均报告 ADE/FDE、Energy Score、动作与
   物理量误差；相对冻结事实层的退化必须预先限定。
2. 因果交互：纵向和横向配对干预同时检查方向、剂量排序、延迟、局部性与未受影响车辆
   漂移；保留现有 0.95 机制门，不能只统计“动作发生过”。
3. 闭环可用性：在 cruise、IDM、MPC 和 learned ADS 下报告新的碰撞、越界、进度和数值
   稳定性。压力测试结果必须明确是条件分布，不能写成自然交通碰撞率。
4. 随机性：相同随机状态逐点可重放，不同随机状态具有经过校准的多样性；不能用单一
   确定轨迹掩盖不确定性，也不能靠噪声扩大覆盖率。
5. ADS 长尾：前四项通过后，才进入 `idm_ams/`。比较模型必须使用相同背景车
   范围、EVT、ADS、失败事件和概率空间；固定真实 `K` 的诊断结果不得与 full-prior 概率
   直接排名。

“NPC 碰撞率 19.4%”若来自筛选后的压力场景、AMS 最终条件尾部或旧的搜索实验，只能解释
为该条件集合中的诊断比例，不是自然碰撞概率；即便如此，它仍说明该候选不适合作为默认
仿真交通。正式维护应优先降低无干预时的碰撞和越界，再讨论增加压力覆盖。

## 代码与命名边界

- `hierarchical_world_model/`：项目原创 CIH-WM；当前已发布能力是冻结事实层，响应候选仍
  为 rejected。
- `interactive_behavior_world_model/`：项目原创闭环 NPC 候选和统一交互基准；原
  `npc_interaction/` 名称会误解为一个指标，现已更正。
- `reproduction/models/`：五个论文来源的可执行复现；随机 IDM 和 Active Inference 不再
  散落在仓库顶层。
- `reproduction/evaluation/`：复现模型的匹配评测与反事实探针。
- `idm_ams/`：恢复后的 IDM 子集模拟/AMS ADS 评测；原 `IDM_subset/` 名称会误解
  为 IDM 模型子集，现按用途更名。
- AMS 中的 `factual_hiqr` 表示 Flow→Diffusion→冻结事实 HiQR 执行链。它没有冒充尚未
  通过验收的 `cih_world_model`；真正的 CIH 适配器只能在自主响应层通过统一晋升门后加入。

历史结果内的 `npc_interaction_*` schema/version 保留用于追溯，不作为新包名或兼容入口。
VBD、CATK 等 `ref_code/` 快照没有项目适配、训练和完整评测，不计为已复现模型。
