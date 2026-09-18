# Dynamic-AR IDM（highD，25 Hz）

本目录对应 `doc/02_Dynamic_AR_IDM.md`，实现 Zhang、Wang 与 Sun（2024）的随机驾驶人模型：

```text
a_t = IDM(h_t; theta) + e_t
e_t = rho_1 e_(t-1) + ... + rho_p e_(t-p) + eta_t
```

AR 项作用于 IDM 残差，不作用于原始历史加速度。校准与动作决策保持论文的 5 Hz 网格；车辆用 ballistic plant 以 25 Hz 推进，每个动作保持 5 个原生帧。plant 的速度投影不反写 AR 残差状态。

## 最终保留结果

项目评测使用 highD 全量合格队列：251 对车辆、76,865 个 5 Hz 决策，来自 recording 25、26 和 36。最终部署诊断采用联合 Dynamic-AR empirical-Bayes MAP/Laplace 后验，并固定平稳的 MAP `rho`；这是可稳定复跑的工程近似，不冒充 NUTS 后验。

5 s、64 futures、2,708 个 rolling origins 的结果如下：

| 模型 | acceleration RMSE | speed RMSE | gap RMSE | acceleration CRPS | speed CRPS | gap CRPS |
|---|---:|---:|---:|---:|---:|---:|
| AR(5) | 0.161 | 0.273 | 0.444 | 0.099 | 0.158 | 0.228 |

在相同全量队列和协议下，AR(5) 相对 AR(0) 的三项 RMSE 分别改善 20.0%、16.0%、33.5%，三项 CRPS 分别改善 24.8%、28.5%、45.2%。该方向复现了论文“AR 随机记忆优于 IID 残差”的关键定性结论；绝对数值不能与论文 20-pair NUTS 结果直接等同。

论文参考轨仍保留 20-pair 数据与 AR(0)/AR(5) 四链长程 NUTS 审计。两次长程拟合均无 divergence 且 BFMI 合格，但最大 rank-R-hat 为 2.840、最小 bulk ESS 为 4.59，未通过收敛门槛。其 AR(5) 均值 `rho=(.851, .594, -.087, -.316, -.081)` 接近论文表 1，但只能作为实现交叉核对，不能作为已收敛后验或部署结果。

## 证据目录

```text
artifacts/
  full_data/        251-pair highD 数据与队列审计
  full_posterior/   最终 AR(0)/AR(5) MAP/Laplace 后验及各自独立的平稳性报告
  full_evaluation/  全量指标、AR(0) 对照、ring stress 与最终论文类比图
  data/             论文 20-pair 参考队列，仅供 NUTS 交叉核对
  nuts/             最终四链长程 NUTS 诊断与逐链预测审计
```

`artifacts/full_evaluation/lag_order_test_full.json` 是最终模型选择结论；`natural_metrics_full_stable_map_rho.json` 是部署评测主结果。论文参考 NUTS 未收敛，明确排除在 RMSE/CRPS 与可视化主结论之外。

## 最小复跑入口

从仓库根目录运行；项目环境规范见 `doc/style.md`。PyMC 当前由 `bayesian_ma_idm/.venv` 提供。

```bash
# 全量队列准备、拟合、评测和可视化
python -m dynamic_ar_idm.scripts.prepare_full_highd
python -m dynamic_ar_idm.scripts.fit
python -m dynamic_ar_idm.scripts.evaluate
python -m dynamic_ar_idm.scripts.summarize
python -m dynamic_ar_idm.scripts.visualize
python -m dynamic_ar_idm.scripts.ring

# 论文参考 NUTS；输出路径显式隔离于全量工程结果
bayesian_ma_idm/.venv/bin/python -m dynamic_ar_idm.scripts.fit \
  --dataset dynamic_ar_idm/artifacts/data/paper_highd_20pairs_25hz.npz \
  --backend pymc --ar-order 5 --tune 3000 --draws 5000 \
  --chains 4 --cores 4 --sampler-init adapt_full \
  --output dynamic_ar_idm/artifacts/nuts/dynamic_ar5_paper_nuts_long.npz

pytest -q dynamic_ar_idm/tests
```

## 论文到代码

| 论文内容 | 实现 |
|---|---|
| 式 8–10：IDM 残差 AR 过程 | `model.py`、`evaluate.py:rollout_25hz` |
| 式 11：层次 log-IDM 参数与 AR 先验 | `fit.py:fit_dynamic_map`、`fit.py:fit_pymc` |
| 式 12：速度似然 | `fit.py:fit_pymc` |
| 表 2 与图 5、7、9 | `evaluate.py`、`visualize.py` |
| 图 10 | `ring.py` |

当前拟合采用作者 highD speed likelihood，不把论文联合位置/速度似然误称为已实现。作者 notebook 使用共享 `rho`，本实现保持这一约定。NUTS 的 R-hat、ESS、divergence、BFMI 与 AR 平稳性均显式写入报告。

`model.py::DynamicIDMDriver` 保留episode参数、最新在前的残差历史和外生创新；plant限幅不反写AR状态。AR(0)/AR(5) 重跑分别写入 `dynamic_ar0_full_posterior_stationarity.json` 和 `dynamic_ar5_full_posterior_stationarity.json`，不会再互相覆盖。本模块的同驾驶人评测不可与 out-of-fold MA-IDM 排名；共同协议见 [`doc/00_Common_Protocol.md`](../doc/00_Common_Protocol.md)。

论文主项目接入：`python -m hierarchical_world_model.scripts.stochastic_drivers rollout --model dynamic_ar5`。接入层只部署平稳MAP rho；未收敛NUTS和非平稳Laplace抽样仍保留为审计证据。
