# 方案01：Bayesian IDM / MA-IDM——人口异质性与时间相关随机跟驰

**依据**：Chengyuan Zhang, Lijun Sun, *Bayesian Calibration of the Intelligent Driver Model*, IEEE T-ITS 25(8), 2024，DOI `10.1109/TITS.2024.3354102`。附件 `Zhang-2024-Bayesian-calibration-of-the-intelli.pdf`，13页。  
**先读**：[共同协议](00_Common_Protocol.md)。模型ID：`bayesian_idm`、`ma_idm`；同一方案内先B-IDM后MA-IDM。  
**预期贡献**：可复现的人口参数联合分布和持续性随机跟驰基模。**不是紧急驾驶模型，也不直接证明ADS极端反事实真实性。**

## 1. 原文实际做了什么 [P]

| 页/式 | 内容 | 实施含义 |
|---|---|---|
| PDF3，式(1)–(3) | IDM五参数、ballistic积分，`Δv=v_f-v_l`，δ固定4，s1=0 | 不增加第六个自由指数参数 |
| PDF4–5，式(5)–(12) | pooled、hierarchical、unpooled；正参数log-normal，LKJ相关；MA-IDM加零均值SE-GP | 参数应联合采样，不逐个独立抽边缘 |
| PDF6，式(13)–(22) | 速度/位置联合似然，过程噪声和观测噪声分开 | 不把观测误差作为驱动加速度 |
| PDF6–7 | highD降采样5Hz，跟驰>50s，20对（10轿车+10卡车），4s/20点分段；NUTS warmup5000 | 此为原文小样本方法验证，不是全数据新司机预测 |
| PDF9，V-C | 数据完整性和可识别性：无自由流时v0难辨識 | 后验近先验要报告，不解释为识别成功 |
| PDF10–11，式(23) | 先抽参数，逐时刻条件GP采样；缓存线性代数 | 可低成本接入在线仿真 |
| PDF10表II | 3s RMSE/CRPS；表格数值乘10展示 | 转录时须除10；不原样套到新split |
| PDF12 | 输入相关GP、动态风格属于future work | 不能称原模型已经具备风险自适应噪声 |

**范围修正**：论文使用SE核。Matérn/OU是可选拓展，不应默认替换后还叫忠实MA-IDM复现。长度尺度约1.4–1.6s描述其数据下的残差相关，不等于驾驶员神经反应时间。

## 2. 官方代码与复现条件 [C]

仓库：`https://github.com/Chengyuan-Zhang/IDM_Bayesian_Calibration`，本次核对main提交 `7520b15a8b163f52532cd3e5c7f7d4db05fc43f0`。执行时锁定该版本，另存实际HEAD。

本次已读取README、`PGM_highD`目录和`Simulator/simulation_ring.py`。确认存在：
```text
PGM_highD/Bayesian_IDM_hierarchical.ipynb
PGM_highD/MA_IDM_hierarchical.ipynb
PGM_highD/Bayesian_IDM_pool_car.ipynb
PGM_highD/Bayesian_IDM_unpool_car.ipynb
PGM_highD/MA_IDM_pool_car.ipynb
PGM_highD/MA_IDM_unpool_car.ipynb
Simulator/simulation_ring.py
```
其他truck及stochastic notebooks依README定位，执行前列目录核实。README要求PyMC4。GitHub元数据的license字段为null，不能因公开可读就假定MIT许可；公开合并代码前检查LICENSE及作者许可。可在本地外部目录运行，自己按公式实现并给出来源，不直接把整仓库复制进FITWMAMS。

**作者代码需要审计，不应照搬**：
- ring脚本IDM含`max(dynamic_gap,0)`，原文展示式无该截断；记录 `equation_idm` 与 `donor_ring_idm` 两种语义。
- GP示例对长时域一次性建立协方差，不能拿它直接跑大批量25Hz世界。
- posterior参数有标准化再乘参考尺度的处理；必须建立单位映射表，避免重复反标准化。
- 缓存pickle需真实存在；没有缓存不能伪造后验或默默用推荐值代替。

## 3. 拟合与模拟的数学对象 [P]

\[
a_{i,t}=f_{IDM}(s_{i,t},v_{i,t},\Delta v_{i,t};\theta_i)+e_{i,t},
\quad \theta_i=(v_0,s_0,T,\alpha,\beta).
\]

B-IDM：`e_t`独立高斯；MA-IDM：
\[
e_i\sim GP(0,k_i),\quad
k_i(t,t')=\sigma_{k,i}^2\exp[-(t-t')^2/(2\ell_i^2)].
\]

采用PDF6式(19)速度似然作为首版主拟合轨；联合似然式(16)为对照审计。保持向量排序 `[v_all,x_all]` 与 `C=[dt,0.5dt²]` 一致。若使用净间距代替位置，过程噪声交叉项符号要随`gap=x_l-x_f-length`改变，不能直接拷贝位置协方差。

原文强先验包括`λ=100, η=2, λϵ=λk=5000, λx=1000, λv=1e5, σσ=.05`等。**必须先核对论文与notebook中参数是在原单位还是归一化单位、Exp参数为rate还是scale。** 不应在原始m/s单位直接机械应用这些数值。产出 `prior_unit_audit.json`，逐项标明来源和换算。

## 4. 本文可行的分阶段实现 [A]

### A. 原算法小规模复现

在外部目录运行或导出作者notebook的纯代码部分；移除绘图不改变概率模型。保留`source_protocol.yaml`，复现20对/5Hz/4s窗口。若无法获得作者相同事件ID，仅声称方法复现。

### B. FITWMAMS迁移

从既有train split中抽取20条满足条件的不同leader–follower序列，尽量覆盖多个recording；轿车与卡车分层。先B-IDM、再MA-IDM，不先跑六种层次的全因子搜索。层次MA是主候选，pooled B-IDM是基线；unpooled只用于少量参考司机检查。

新的validation driver只能使用前缀个体化或人口预测，不能读其未来来拟合参数。数据不足50s时单独报告短轨迹实验，不暗改原文资格。

### C. 在线GP

实现两个清楚命名的选项：
1. `full_gp_small_horizon`：短序列完整GP，作为数值金标准。
2. `finite_memory_gp`：仅用过去固定窗口条件采样，预计算Cholesky/条件系数；这是计算近似，单独验证。

时刻t的GP状态只保留过去残差及时间。预测后的残差进入自己历史，不能从未来真实动作重算。前缀残差有测量噪声时，通过相应后验抽样或滤波初始化，不能当成无噪声真实过程。

## 5. 代码任务

当前实现：
```text
bayesian_ma_idm/src/driver.py                         # B/MA在线随机状态
bayesian_ma_idm/src/reference_kernels.py              # IDM/SE-GP参考核
bayesian_ma_idm/scripts/                              # 论文拟合及OOF评测
hierarchical_world_model/src/stochastic_drivers/      # 主项目25 Hz接入边界
```

必备函数：
```python
build_idm_likelihood(data, prior_spec, likelihood_kind)
fit_population_idm(data, pooled: bool, memory: str, budget)
sample_driver_joint(population_draw, driver_uniforms)
condition_gp_prefix(times, residual_posterior, gp_parameters)
gp_step(history, timestamp, standard_normal)
```

随包 `reference_kernels.py::idm/se_kernel/gp_next` 是可执行参考核。核心条件采样为：
\[
\mu_*=K_{*p}K_{pp}^{-1}e_p,\quad
\sigma_*^2=K_{**}-K_{*p}K_{pp}^{-1}K_{p*}.
\]
代码用Cholesky求解，不显式求逆。小于数值容差的负方差可归零；显著负方差必须抛异常，不默默修复。

25Hz适配第一版每0.2s更新driver，每0.04s推进plant；禁止把每一步重复当独立GP噪声。

## 6. 验收标准

### 数学/原文复现

- 在20点人工时间网格，SE协方差、条件均值/方差与直接多元正态求解最大误差≤`1e-8`（float64）。
- 10000次条件抽样的均值/协方差在各自Monte Carlo误差范围内；这是小型数学测试，不是10000次完整交通仿真。
- 参数联合采样保留后验相关；不得只匹配5个边缘直方图。
- 用同数据/设置对照作者notebook，报告全部不一致；原表只能在同样数据和评分下设定复现容差。
- 原表II按×10还原；例：hierarchical car MA-IDM的`CRPS(a)=0.58×0.1=0.058 m/s²`，是那个示例的参考值，不是全highD门槛。

### 迁移价值

采用共同协议的cluster ES/CRPS标准；特别报告3s与5s是否性能退化、预测区间是否只是变宽、ACF是否比IID接近数据。若只在已拟合司机上改善而对新司机未改善，标记 `retrospective_only`。

### 紧急能力与接入

- 首先报告绝对MA-IDM在强制动条件下何时达到plant饱和、有无长时正误差阻碍制动。
- `β`仅为舒适减速度，不能当紧急能力上限。
- 无需通过零碰撞门槛；本论文没有紧急人因证据，`emergency_human_evidence=none`。
- 差分接入必须跑共同协议的噪声抵消测试。若实际/参考使用同自治GP，随机误差必然抵消；这时只称“参数异质性机制差分”，不能称“完整MA-IDM随机响应”。

## 7. 固定预算与命令

[A] smoke：4个driver、2链、300warmup+300draw，目的仅检查代码。正式：20个train driver、4链、5000warmup+1000draw；先计算一次预计CPU成本，达到预注册wall-clock上限24h则保存为预算不足，不自动追加。PyMC是CPU工作负载，4090并不保证加速。

建议clone主环境做离线旧依赖隔离，需用户确认安装；统一执行与评测仍`conda activate tread`，只读取导出的数值后验，不跨环境加载任意pickle。

已实现CLI：
```bash
bayesian_ma_idm/.venv/bin/python -m bayesian_ma_idm.scripts.fit_author_reference
python -m bayesian_ma_idm.scripts.run_full_cv --model ma_idm
python -m driver_reproduction.scripts.run_matched_evaluation
python -m hierarchical_world_model.scripts.stochastic_drivers rollout --model ma_idm
```

## 8. 完成判定

保留三份独立报告：原文模型复现、highD前缀迁移、世界接入。若GP只是扩大方差或计算成本过高，不继续新增神经残差；保留其作为普通跟驰基线，优先转02。论文中能写“层次参数和持续误差改善普通跟驰概率拟合”，不能写“恢复了真实人类极限避险机制”。
