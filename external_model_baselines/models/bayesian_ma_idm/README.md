# Bayesian MA-IDM

这是 Bayesian MA-IDM 的独立实现，仅包含两个论文模型：

- `bayesian_idm`：层次 B-IDM，作为独立高斯残差基线；
- `ma_idm`：层次 MA-IDM，使用联合 log-normal 驾驶人参数和零均值平方指数（SE）GP 残差。

## 保留的部署模式

1. **人口 MA-IDM**：新驾驶人没有个人历史时，按人口联合后验抽样。
2. **类别条件 MA-IDM**：按可观测的 Car/Truck 类别选择人口分布；这是 highD 中最好的未知驾驶人点预测配置，但属于工程扩展，不是原论文层次。
3. **前缀个体化 MA-IDM**：有至少5秒已观测动作时，仅用预测起点之前的已完成0.2秒动作更新驾驶人参数。
4. **gap-regime 区间层**：训练记录上按当前 gap 学习位置区间余量；它只校准位置区间，不能称为原始 MA-IDM 后验。

所有仿真均为 25 Hz ballistic plant，驾驶人每 0.2 s（5 Hz）采样并保持动作。`evidence/` 只保存可复核的最终数据、后验、评测和图；不保留历史消融或 smoke 结果。

## 数据与证据

`evidence/` 保存原论文参考队列、录像留出后验、全量部署后验和对应指标。人口、类别条件、前缀个体化及区间校准的来源和结果索引见 [RETAINED_MODELS.md](RETAINED_MODELS.md)。留出评测使用训练录像后验；`evidence/deployment/` 的全量后验用于仿真，不用于重算留出成绩。

## 最小复跑入口

```bash
conda activate tread

# NUTS 入口首次运行时安装固定的 CPU 推断依赖
python -m pip install -r external_model_baselines/models/bayesian_ma_idm/requirements-pymc.txt

# 原论文 20-pair、5 Hz、NUTS 方法复现
python -m external_model_baselines.models.bayesian_ma_idm.scripts.fit_author_reference

# source-loader highD 的 B-IDM / MA-IDM 人口评测
python -m external_model_baselines.models.bayesian_ma_idm.scripts.run_full_cv --model b_idm
python -m external_model_baselines.models.bayesian_ma_idm.scripts.run_full_cv --model ma_idm

# 已有前缀时的个体化评测
python -m external_model_baselines.models.bayesian_ma_idm.scripts.run_full_cv --model ma_idm --personalize-from-prefix

# 论文 Fig.10 的 25 Hz highway-env 类比图
python -m external_model_baselines.models.bayesian_ma_idm.scripts.run_highway_env_ring
```

`src/driver.py::BayesianIDMDriver` 是在线仿真边界：每个episode固定一组联合参数，B-IDM递推IID残差，MA-IDM递推因果SE-GP历史，外部plant负责五帧动作保持和限幅。共同的模型范围与证据边界见 [`external_model_baselines/README.md`](../../README.md) 和 [`registry.json`](../../registry.json)。
交叉验证后验只用于评测；`evidence/deployment/` 中使用全部251个合格pair重拟合的后验用于本模型的配对响应检查。
