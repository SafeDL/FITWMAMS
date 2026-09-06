# 扩散引导的分层交通世界模型

本模块实现一条保持 25 Hz 因果时序的条件重仿真链：

\[
p(M)p(C_0\mid M)p(K\mid C_0,M)
\rightarrow p_\theta(\tau_{soft}\mid C_0,M,K,z)
\rightarrow \pi(a_{bg}\mid H_t,K_{ref},z).
\]

Flow 生成场景约束，冻结 Diffusion 生成六个背景槽位的 soft plan，冻结 HiQR
每帧只根据已实现历史提交下一帧动作，HighwayEnv 负责闭环物理执行。这里的
`K_GT` 是宏观参考条件，因此方法是“给定真实宏观参考的条件重仿真”，不是无条件未来预测。

## 代码组织

```text
hierarchical_world_model/
├── config/
│   ├── release.yaml              # 冻结主世界模型协议
│   └── reaction_policy.yaml      # 当前反应策略协议
├── src/
│   ├── composition.py            # Flow、Diffusion、HiQR 世界组合
│   ├── highway.py                # HighwayEnv 闭环与 snapshot/restore
│   ├── influence_graph.py        # 从已实现交通历史建立自主作用域
│   ├── reaction_controller.py    # 反应控制器及统一动作约束
│   ├── reaction_evidence.py      # highD 事件证据与统计量
│   ├── reaction_training.py      # 三流预训练与 PPO
│   ├── reference.py              # K_GT 预览协议
│   └── execution.py              # ADS rollout 与 all-slot EVT 风险接口
├── scripts/
│   ├── prepare_reaction_evidence.py
│   ├── train_reaction_policy.py
│   ├── evaluate_reaction_policy.py
│   └── validate_reaction_policy.py
└── tests/
```

正式世界模型发布仍使用 `release.yaml`。反应策略只有上述一个语义配置；旧 GAIL、
MLOO 和按实验标签触发的训练入口已经退出主干。

## 自主反应契约

`CausalInfluenceGraph` 只接收当前与历史真实状态、上一帧实际背景动作和内部图状态。
logged、synthetic、ADS、文件来源等标签不会进入 world、影响图或 controller context。
硬契约为：

\[
(H_t,K_{ref},z,controller\ state)\ 相同
\Rightarrow p(a_{NPC,t})\ 相同.
\]

传入当前 `ego_action` 时，NPC 动作已经由上一时刻历史计算完毕；当前 ego 命令先由
HighwayEnv 实现，最早只能在下一次 25 Hz 边界改变 NPC 分布。作用域表示“允许局部策略
处理的车辆”，不表示必须制动。风险解除并持续开放后，authority 平滑释放。

当前比较臂为：

- `frozen_hiqr`：冻结 HiQR，不施加残差；
- `idm_only`：在同一自主作用域和物理约束下使用规则动作；
- `a1_transfer`：冻结旧纯残差权重，仅迁移到自主作用域；
- `a2_transfer`：冻结旧 IDM-referenced 残差权重与旧映射，仅迁移运行时；
- `calibrated_residual`：当前候选，IDM 只作输入特征，输出有符号 bounded residual。

候选的确定性残差中心为零时精确透传 HiQR。它允许必要制动与恢复性加速；IDM 不再
构成动作上下界、人类反应下限或奖励目标。所有反应臂共享动作边界、jerk layer 和
`TTC < 2 s` 的物理 guard，guard 触发率独立报告。

## K_GT 参考协议

交互引起的纵向位置偏差会完整平移到后续 preview，不随参考时钟推进而消除；局部参考
速度仍提供恢复方向，横向参考仍可回到 K_GT 的车道结构。训练没有“追上原时刻位置”、
“残差必须回零”或异常加速追回 schedule error 的奖励。所有主比较臂使用同一
`reference_rebase_weights=(1, 0)`，旧结果只按其历史协议解释。

## 人类响应证据

事件从原始 highD recording 和真实车辆 ID 提取。父车加速度从非制动状态跨过
`-0.5 m/s²` 且至少持续三帧才构成 onset；同一 recording、leader、follower 在一秒内的
片段合并。唯一键为：

```text
(recording_id, leader_id, follower_id, absolute_onset_frame)
```

支持 cell 由 train split 定义：父车制动强度 × 事件前 TTC，每个 cell 至少 100 个独立
事件且来自至少 5 个 recording。validation/test 只继承 train 支持表，不重新定义支持域。
事件工件保存前 1 秒、后 3 秒的 acceleration、abs jerk、speed、gap、closing 和 TTC；
recording、车辆对与事件键均接受 split 隔离检查。

构建证据：

```bash
conda run -n tread python hierarchical_world_model/scripts/prepare_reaction_evidence.py \
  --output-dir results/hierarchical_world_model/causal_reaction/reaction_events
```

## 训练

训练固定为每个 update 32 个支持自然事件、16 个匹配非事件和 16 个 synthetic 物理压力
案例。自然事件与非事件监督最终执行动作；基础动作正确时目标自然是零修正。synthetic
案例只使用碰撞、间距、可执行动作和 jerk 等物理目标，`human_target_gate` 恒为零。

正式 PPO 前先在固定支持场景上比较预训练前后的实际执行 acceleration＋jerk Energy
Score。若没有改善则停止，不扩大预算。HiQR、Diffusion 和 IDM 参数始终冻结；恢复训练
同时恢复事件 sampler 与随机流。

```bash
conda run -n tread python hierarchical_world_model/scripts/train_reaction_policy.py \
  --a2-transfer-checkpoint <frozen-a2-transfer.pt>
```

## 评测与验收

评测入口直接运行相同世界和参考协议，不依赖预先拼装的 NPZ：

```bash
conda run -n tread python hierarchical_world_model/scripts/evaluate_reaction_policy.py \
  --a2-checkpoint <frozen-a2-transfer.pt> \
  --candidate-checkpoint <calibrated-residual.pt> \
  --output <evaluation.json>

conda run -n tread python hierarchical_world_model/scripts/validate_reaction_policy.py \
  <evaluation.json>
```

报告包含：控制器正常运行的完整 factual reconstruction；held-out 自然事件每事件 32 个
共同随机未来；最长三秒恢复诊断；`-2` 至 `-8 m/s²`、未见 ramp/pulse 的纯物理 OOD。
自然性主指标是 acceleration＋jerk 序列的事件级 Energy Score，按 recording 做配对簇
bootstrap。帧、随机未来和同 recording 内事件不会被当作独立样本。

候选只有在以下条件同时满足时才可替换 A2-transfer：人类响应 Energy Score 的单侧 95%
改善下界大于零；factual ADE/FDE/P95 同时满足绝对与 5% 相对非劣界；时延、峰值加速度、
jerk、gap、closing 和恢复误差的配对上界不超过 train IQR 的 10%；所有 OOD 动作、数值和
jerk 约束有效。碰撞是独立安全门槛，不计入人类真实性分数。

## ADS 风险边界

新风险接口默认包含 `same_rear`，并接受冻结 controller 与统一 K_GT 协议。旧的
follower-excluded EVT/AMS 结果属于旧测度，不与新结果合并。只有候选通过独立人类响应
验收后，才能加入背景模型敏感性集合并进入 MC/AMS。

需要分别报告 sampling interval 与背景模型 sensitivity。MC/AMS 一致性只验证
`\hat p-p_{\theta,FC}`，不能消除 `p_{\theta,FC}-p_{real}`。增加 AMS 样本量不应被表述为
缩小现实概率的模型误差。

历史 GAIL/MLOO 工件索引位于
`results/hierarchical_world_model/causal_reaction/archived_reaction_experiments/`；这些工件
只保留为拒绝证据，不参与当前导入、训练、选择或风险结论。
