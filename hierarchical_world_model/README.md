# 分层交通世界模型

项目维护一条可执行的响应方法链：冻结的 Flow--Diffusion--HiQR 世界模型，后接
**IDM 规则先验 + PPO 残差响应控制器**。它只在同车道后车关系具备因果权限时修改纵向
动作；世界模型、训练事件和响应控制器分别存放，避免把历史候选混入运行入口。

```text
config/
├── world_model.yaml          # 冻结世界模型的权重和数据依赖
└── ppo_idm_response.yaml     # 唯一活跃的响应控制器与训练规格

scripts/ppo_idm_response/
├── evaluate_intervention_effects.py
└── render_playbacks.py
```

`world_model.yaml` 是已验证的生成与物理执行基座，不代表“发布”控制器。
`ppo_idm_response.yaml` 指向唯一的 checkpoint、IDM 规则和共享 highD 事件；所有活跃
响应脚本都只读取这份配置。

```text
results/hierarchical_world_model/
├── world_model/          # 冻结基座的必要权重与评测
├── reaction_events/      # train / validation / test 的共享 highD 事件
└── ppo_idm_response/     # PPO--IDM checkpoint、规则和最终审计
```

全测试固定制动探针显示该控制器会增加后车制动（均值 −1.753 m/s²）且通常增加最小
间距（均值 +3.563 m）。同一审计亦出现 2 个新增碰撞，因此它是已验证的**响应研究
候选**，不是风险发布或安全提升结论。完整证据见
[`results/hierarchical_world_model/ppo_idm_response/README.md`](../results/hierarchical_world_model/ppo_idm_response/README.md)。
