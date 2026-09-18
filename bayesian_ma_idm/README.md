# Bayesian MA-IDM

这是方案01（`doc/01_Bayesian_MA_IDM.md`）的独立实现，仅包含两个论文模型：

- `bayesian_idm`：层次 B-IDM，作为独立高斯残差基线；
- `ma_idm`：层次 MA-IDM，使用联合 log-normal 驾驶人参数和零均值平方指数（SE）GP 残差。

它不是 Dynamic-AR IDM（方案02）、multi-regime B-IDM（方案03）或 Active Inference（方案04）的实现。

## 保留的部署模式

1. **人口 MA-IDM**：新驾驶人没有个人历史时，按人口联合后验抽样。
2. **类别条件 MA-IDM**：按可观测的 Car/Truck 类别选择人口分布；这是 highD 中最好的未知驾驶人点预测配置，但属于工程扩展，不是原论文层次。
3. **前缀个体化 MA-IDM**：有至少5秒已观测动作时，仅用预测起点之前的已完成0.2秒动作更新驾驶人参数。
4. **gap-regime 区间层**：训练记录上按当前 gap 学习位置区间余量；它只校准位置区间，不能称为原始 MA-IDM 后验。

所有仿真均为 25 Hz ballistic plant，驾驶人每 0.2 s（5 Hz）采样并保持动作。`evidence/` 只保存可复核的最终数据、后验、评测和图；不保留历史消融或 smoke 结果。

## 主要结果

所有数字均为 highD 的 251 个 source-loader pair、leave-one-recording-out、64 futures 的 3 s / 5 s 结果：

| 配置 | position RMSE (m) | action CRPS (m/s²) | 90% position coverage |
|---|---:|---:|---:|
| B-IDM 基线 | 0.870 / 1.773 | 0.214 / 0.195 | 0.746 / 0.766 |
| 原始人口 MA-IDM | 0.250 / 0.837 | 0.134 / 0.149 | 0.632 / 0.712 |
| 类别条件 MA-IDM | **0.220 / 0.742** | **0.127 / 0.141** | 0.617 / 0.704 |
| 前缀个体化 MA-IDM | **0.146 / 0.523** | **0.104 / 0.120** | 0.574 / 0.664 |
| 人口 MA-IDM + gap-regime 区间层 | 0.280 / 0.911 | 0.142 / 0.154 | **0.942 / 0.960** |

前两项是模型本体比较；最后一项使用较早的训练前缀，因此点预测不可与完整训练人口模型直接比较。完整边界、来源与结果索引见 [RETAINED_MODELS.md](RETAINED_MODELS.md)。

## 最小复跑入口

```bash
# 原论文 20-pair、5 Hz、NUTS 方法复现
bayesian_ma_idm/.venv/bin/python -m bayesian_ma_idm.scripts.fit_author_reference

# source-loader highD 的 B-IDM / MA-IDM 人口评测
python -m bayesian_ma_idm.scripts.run_full_cv --model b_idm
python -m bayesian_ma_idm.scripts.run_full_cv --model ma_idm

# 已有前缀时的个体化评测
python -m bayesian_ma_idm.scripts.run_full_cv --model ma_idm --personalize-from-prefix

# 论文 Fig.10 的 25 Hz highway-env 类比图
bayesian_ma_idm/.venv/bin/python -m bayesian_ma_idm.scripts.run_highway_env_ring

# 通过论文主项目的25 Hz仿真边界运行全251-pair部署后验
python -m hierarchical_world_model.scripts.stochastic_drivers rollout --model ma_idm
```

`src/driver.py::BayesianIDMDriver` 是在线仿真边界：每个episode固定一组联合参数，B-IDM递推IID残差，MA-IDM递推因果SE-GP历史，外部plant负责五帧动作保持和限幅。共同能力规则见 [`doc/00_Common_Protocol.md`](../doc/00_Common_Protocol.md)。
交叉验证后验只用于评测；`evidence/deployment/` 中使用全部251个合格pair重拟合的后验才是主项目仿真默认资产。
