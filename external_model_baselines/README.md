# 外部论文模型基线

这里集中存放从已发表方法构建的外部模型基线，共 5 个可执行模型家族，另有配套评测工具。
它们用于与项目原创世界模型比较；“基线”不代表每个实现都是论文结果的严格复现，
具体适配程度和可部署性见下表。

## 模型基线

| 家族 | 类型 | 统一入口 | 当前证据状态 |
|---|---|---|---|
| B-IDM / MA-IDM | 随机跟驰驾驶员 | `models/bayesian_ma_idm/` | 保留为普通跟驰随机基线 |
| Dynamic-AR IDM | 随机跟驰驾驶员 | `models/dynamic_ar_idm/` | 工程近似；论文参考 NUTS 未收敛 |
| Multi-regime B-IDM | 随机跟驰驾驶员 | `models/multi_regime_bidm/` | 保留复现证据，拒绝部署 |
| Active Inference | 随机碰撞规避驾驶员 | `models/active_inference_driver/` | 保留官方 v1.0.0 wrapper；独立适配不冒充原模型 |
| TrafficBots V1.5 | 多智能体行为世界模型 | `models/trafficbots/` | highD 方法适配基线，不是 WOMD bit-exact 复现 |

上述目录保存模型的**唯一实体源码、历史后验和验证证据**。项目不保留旧包别名；源码、
测试、配置和命令统一使用 `external_model_baselines.models` 命名空间。例如：

```bash
python -m external_model_baselines.models.bayesian_ma_idm.scripts.run_full_cv --model ma_idm
python -m external_model_baselines.models.trafficbots.scripts.verify_full
```

## 不是新模型的评测工具

| 统一入口 | 角色 |
|---|---|
| `evaluation/driver_reproduction/` | 同 cohort 匹配评测、证据总表和跨模型契约 |
| `evaluation/counterfactual_response/` | 随机驾驶人的配对响应探针，不是第五个驾驶人模型 |

`hierarchical_world_model/`、`npc_behavior_benchmark/`、`diffusion/` 和
`normalizing_flow/` 属于项目原创方法，不放进外部基线集合。`ref_code/` 只是上游源码快照；
其中 VBD、CATK 等在没有项目适配、训练和验证之前，不能记为“已复现模型”。

## 防止目录再次失控

[`registry.json`](registry.json) 是机器可读的唯一清单，记录实现位置、论文、证据状态、
评测工具和仓库顶层分类。每次增加外部基线或顶层方法目录都必须先更新它，并运行：

```bash
python -m external_model_baselines.audit
pytest -q external_model_baselines/tests
```

审计会检查论文、实体实现、重复登记，以及未分类的顶层目录。实验结果仍统一写入
`results/`；不要额外复制 checkpoint、数据集或结果目录。
