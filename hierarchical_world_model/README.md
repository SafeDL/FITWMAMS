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
├── factual_all_slots_v2/      # 后续全背景车评测输出（首次运行时创建）
└── cih_wm/                    # 先验、固定证据与当前 CIH-WM 候选
```

冻结事实层曾在完整测试集（10,151 个序列、历史 `same_rear` 掩码口径）上得到
ADE **0.040227 m**、FDE **0.036870 m**、P95 位移误差 **0.090438 m**。当前代码已切换到
包含 `same_rear` 的全背景车口径，尚未重新训练或评测；因此这些历史数值不能与后续全量结果直接比较，也不应写成 CIH-WM 因果响应指标。

当前 CIH-WM v2 已移除名义影子动作、成对硬门和规则差值覆盖，并加入不读取记录未来的名义/三档 ADS 制动对比训练。全量训练候选在 256 序列诊断中仍未通过因果方向和 `same_rear` 非劣门槛，因此
`cih_wm/candidate_unaccepted/` 中的结果只能用于诊断，不能称为正式最佳模型。方法边界、组件证据和下一步验收协议见
[`FITWMAMS_A2_Preserving_Human_Response_Revision_Plan.md`](../doc/FITWMAMS_A2_Preserving_Human_Response_Revision_Plan.md)。

横向通道目前只做事实 yaw-rate 重建；在拥有变道/转向干预证据和独立验收指标前，不启用横向因果响应。
