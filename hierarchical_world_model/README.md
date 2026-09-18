# CIH-WM 分层交通世界模型

本目录维护一个完整方法和一个可独立审计的事实层：

1. **冻结事实转移层**：一个扩散计划条件的交通状态转移模型，负责事实重建；它是 CIH-WM 的基础能力，不作为本方法的新增算法主张。
2. **CIH 响应层**：以实现状态因果影响路由确定可响应车辆，再以机制引导、约束保持的人类响应校准生成纵向动作修正。这是当前方法的核心。

事实层内部的关系编码、状态过滤、jerk 解码和运动学积分是实现选择；PPO、DDIM 采样步数及 jerk 限制是训练或执行约束，不单列为算法创新。

```text
config/
├── world_model.yaml          # 冻结事实层及其发布评测协议
└── cih_world_model.yaml      # CIH-WM 的唯一方法配置

scripts/
├── evaluate.py                 # 冻结事实层评测
├── train_cih_world_model.py    # 响应校准与策略优化
├── evaluate_cih_world_model.py # 事实重建、同车道后车和因果探针
└── promote_cih_world_model.py  # 验证集选择与一次性测试验收

results/hierarchical_world_model/
├── factual_hiqr/              # 历史掩码协议的冻结检查点和发布评测
├── factual_all_slots/         # 后续全背景车评测输出（首次运行时创建）
└── cih_wm/                    # 先验、固定证据与当前 CIH-WM 候选
```

冻结事实层现已在完整测试集（10,151 个序列）按全背景车口径重新评测，ADE
**0.040020 m**、FDE **0.036345 m**、P95 位移误差 **0.091024 m**。完整 validation
包含 13,133 条序列；全槽位 ADE/FDE/P95 为 **0.037140/0.029709/0.091529 m**，
`same_rear` 为 **0.037557/0.028678/0.098626 m**，没有槽位计划 fallback。工件见
`results/hierarchical_world_model/factual_all_slots/evaluation.json` 和
`results/hierarchical_world_model/cih_wm/continuation/baseline_audit.json`。

当前续建链使用真实 logged context 的 75 帧响应监督、二维 prior-retention/residual
映射、fair Energy Score、明确的 policy/execution/release mask 和可快照的显式随机世界。
`cih_wm/candidate_unaccepted/` 中的旧结果仍只用于历史诊断；当前保留的有效响应校准器及验收证据位于
`cih_wm/continuation/`。最新完整重训位于 `cih_wm/training/`，但因闭环因果指标退化而未晋升。

本轮完整 validation 对 13,133 条序列和全部 13,133 个因果探针行评测了监督与
PPO 检查点。两者均通过全槽位及 `same_rear` 事实保持；监督/PPO 的固定窗口方向
一致率分别为 **75.93%/75.76%**，剂量排序率分别为 **51.20%/51.25%**，未达到
预注册的 95% 机制门槛。PPO 的 75 帧 fair Energy Score 相对监督检查点改善
**0.0000718**，recording-cluster bootstrap 95% 区间为
**[-0.0000127, 0.0001501]**，不显著。依照停止规则保留监督检查点、不增加 PPO
轮数、不进入确认性 test；机器可读结论见 `cih_wm/continuation/decision.json`。
该次完整重训的机器可读结论见 `cih_wm/training/decision.json`。

横向通道目前只做事实 yaw-rate 重建；在拥有变道/转向干预证据和独立验收指标前，不启用横向因果响应。

## 随机驾驶人复现接入

`src/stochastic_drivers/` 是论文复现模型进入本项目的唯一纵向边界。它把 CIH-WM
的 `[x,y,vx,vy,ax,ay]` 状态转换为跟驰观测，在25 Hz plant内保持5 Hz动作，并支持
NumPy driver的随机状态快照/恢复。限幅不反写GP、AR或regime状态。

```bash
python -m hierarchical_world_model.scripts.stochastic_drivers inventory
python -m hierarchical_world_model.scripts.stochastic_drivers rollout --model ma_idm
```

默认只允许证据状态可用的B-IDM、MA-IDM、稳定MAP Dynamic-AR和需要外部固定版本源码的
官方Active Inference wrapper。Multi-regime和独立纵向Active适配仍可用于研究诊断，但必须
显式传入 `--allow-unaccepted`，不能静默进入自动驾驶测试。

同数据比较使用218-pair共享cohort：recording 25的全部182个事件训练，recording 26/36
的全部36个合格事件评测，结果位于
`results/driver_reproduction/matched_highd/matched_metrics.json`。各模型的论文原生复现仍
单独保留；共同评测不会覆盖论文特有的推断结论。

## 单一方法链

CIH-WM 把已验证的纵向响应能力合并为冻结的 `mechanism_guided_response_prior`，再由因果影响路由和受约束校准器扩展它；这不是并列的旧模型或兼容层。该先验的来源证据、当前检查点和角色位于 `cih_wm/response_prior_lineage.json`。

`cih_wm/evaluation_index.json` 汇总这一个方法链的冻结事实层、响应先验来源证据、当前 CIH 监督/PPO 评测和验收状态。索引明确标注不同协议的范围，不将历史固定制动探针改写为当前全槽位因果验收率。
