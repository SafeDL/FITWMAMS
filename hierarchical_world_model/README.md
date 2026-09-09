# 分层交通世界模型

当前维护的是一条研究候选链：冻结的 Flow–Diffusion–HiQR 世界模型，加上
`nominal_preserving_response` 短时纵向响应层。候选在既定 highD 划分上通过了
性质、事实兼容、自然响应与 OOD 执行门槛；它仍是研究候选，不是已发布控制器。

## 活跃配置与入口

```text
config/
├── world_model.yaml          # 冻结的世界模型依赖与权重路径
└── nominal_response.yaml     # 唯一活跃的响应候选、训练与验收规格

scripts/nominal_response/
├── build_cache.py
├── train.py
├── evaluate_factual.py
├── evaluate_natural.py
└── evaluate_ood.py
```

`world_model.yaml` 只描述被本轮冻结复用的生成与执行基座；它不表示本轮方法已
release。`nominal_response.yaml` 显式引用该基座和共享的事件证据，所有活跃响应
脚本从该配置解析路径。

## 当前结果

```text
results/hierarchical_world_model/
├── world_model/              # 冻结基座的必要权重与评测
├── reaction_events/          # train/validation/test 的共享 highD 事件证据
├── nominal_response/         # 唯一选中的 checkpoint 与最终证据
└── archive/                  # 已拒绝或历史方法，绝不作为活跃输入
```

选中权重和最终结论见
[`results/hierarchical_world_model/nominal_response/README.md`](../results/hierarchical_world_model/nominal_response/README.md)。
历史 calibrated-residual、PPO、GAIL 与早期诊断入口被移入 `config/archive/`、
`scripts/archive/` 和 `results/.../archive/`，不参与当前训练、选择或风险结论。
