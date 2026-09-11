# 分层交通世界模型

项目维护的响应基线是冻结的 Flow--Diffusion--HiQR 世界模型，后接 **IDM 规则先验
+ PPO 残差响应控制器**（legacy A2）。A2 human calibration 是保留的实现候选，当前
因 policy-on factual 协议未对齐而暂停，不能视为已验证方法。

```text
config/
├── world_model.yaml          # 冻结世界模型的权重和数据依赖
├── ppo_idm_response.yaml     # 维护中的 legacy A2 基线
└── a2_human_calibration.yaml # 暂停的 calibration 候选

scripts/ppo_idm_response/
├── evaluate_intervention_effects.py
└── render_playbacks.py
```

`world_model.yaml` 是已验证的生成与物理执行基座，不代表控制器“发布”。
`ppo_idm_response.yaml` 指向 A2 checkpoint、IDM 规则和共享 highD 事件。关于
bridge factual、A2 response test 与 policy-on stress 的边界见
[`FITWMAMS_A2_Factual_Protocol_Correction.md`](../doc/FITWMAMS_A2_Factual_Protocol_Correction.md)。

```text
results/hierarchical_world_model/
├── world_model/          # 冻结基座的必要权重与评测
├── reaction_events/      # train / validation / test 的共享 highD 事件
└── ppo_idm_response/     # PPO--IDM checkpoint、规则和最终审计
```

全测试固定制动探针显示 A2 相对冻结 HiQR 会增加后车制动（均值 −1.688 m/s²）且通常增加
最小间距（均值 +2.998 m）。在保持 A2 本身、初始状态与全部随机量不变、只新增 ego 制动的
直接成对审计中，后车额外制动均值为 −0.875 m/s²，且探针前动作差严格为零。同一 A2--HiQR
审计亦出现 2 个新增碰撞，因此它是已验证的**响应基线**，不是安全提升结论。完整证据见
[`results/hierarchical_world_model/ppo_idm_response/README.md`](../results/hierarchical_world_model/ppo_idm_response/README.md)。
