# 随机驾驶人复现的共同协议

所有模型共享 highD 的原始 25 Hz 时钟；车辆以 25 Hz ballistic plant 推进，
随机驾驶人每 0.2 s 决策一次，动作保持五个原生帧。模型的随机状态由自身生成并
因果递推，plant 限幅不得反写 GP/AR 状态，评测器不得把真实未来输入驾驶人。

## 能力对齐

“对齐”要求每个方法分别具备四层证据：论文机制和推断、可持久化的在线随机状态、
25 Hz 闭环滚动、与其适用域相符的验证。不能用一个统一分数掩盖某层失败。

| 方法 | 论文机制/推断 | 在线驾驶人 | 项目评测 | 当前结论 |
|---|---|---|---|---|
| B-IDM / MA-IDM | 作者20-pair、5 Hz、层次NUTS通过诊断；MA保留SE-GP | `BayesianIDMDriver`，参数联合抽样，因果GP历史 | 251-pair recording-held-out | 可作为普通跟驰随机基线；没有紧急人因依据 |
| Dynamic-AR IDM | AR作用于IDM残差；论文参考NUTS未收敛，MAP/Laplace另标 | `DynamicIDMDriver`，最新残差在前，平稳MAP部署 | 251-pair同驾驶人诊断、ring stress | 工程近似复现论文AR优于IID的方向；不是已收敛论文后验 |
| Multi-regime B-IDM | 三style、有限Gaussian HSMM适配、分style层次NUTS及独立pooled NUTS | `MultiRegimeIDMDriver`，prefix style与因果显式持续时间过滤 | recording 26/36留出、共享创新 | 点预测优于独立pooled基线，但90%覆盖率仅约41%，拒绝部署；原文HDP-HSMM分布歧义仍阻塞逐算法复现 |
| Active Inference | 锁定官方v1.0.0的896轨原生人因复现通过 | 官方外部 `OfficialPOMDP25Hz` wrapper；独立NumPy纵向适配另存 | 官方25 Hz桥接通过；独立适配桥接失败；304个highD事件仅作外部域诊断 | 未来仿真应使用受许可约束的官方wrapper；不得把独立纵向适配称为原模型 |

`counterfactual_response` 是上述随机跟驰模型的共同响应探针，不是第五个驾驶人模型。
Active Inference 的随机规划流与动作空间不同，当前保留独立官方桥接证据，不能为了
形成排行榜而降低CEM预算或把它硬塞进四模型探针。

## 可比较性

只有数据过滤、训练/测试划分、anchor、观测前缀、预测步长、future 数量、时钟、
积分语义和指标定义完全一致的结果才可直接比较。`benchmark_id` 不同的行是证据，
不是排行榜。MA-IDM 的 out-of-fold 数字、Dynamic-AR 的同驾驶人 posterior 数字和
Multi-regime 的36-pair留出结果不可互相排名。

另设一个真正匹配的highD迁移基准：同一218-pair因果cohort，仅用recording 25的全部
182个事件训练，并在recording 26/36的全部36个合格事件上，以相同5秒前缀、3秒预测、
64 futures、25/5 Hz时钟和指标同时评测 B、MA、Dynamic、pooled与multi-regime。
测试协议可以直接比较；论文特有推断和Stage-B观测预算仍按模型保留并在报告中披露。

统一证据汇总可由下列命令重建；它不会替代模型各自的拟合、driver和评测：

```bash
python -m driver_reproduction.scripts.build_scorecard
python -m driver_reproduction.scripts.run_matched_evaluation
python -m hierarchical_world_model.scripts.stochastic_drivers inventory
pytest -q driver_reproduction/tests active_inference_driver/tests \
  bayesian_ma_idm/tests dynamic_ar_idm/tests multi_regime_bidm/tests \
  counterfactual_response/tests
```

新结果必须写明数据 lineage、split、后验用途和状态初始化。不要保留 smoke、旧修复前
曲线、分析工具产生的整目录副本或可由最终后验直接再生的临时输出。
