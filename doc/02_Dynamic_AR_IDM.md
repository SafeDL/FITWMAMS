# 方案02：Dynamic IDM——贝叶斯动态回归与AR随机记忆

**依据**：Chengyuan Zhang, Wenshuo Wang, Lijun Sun, *Calibrating car-following models via Bayesian dynamic regression*, Transportation Research Part C 168 (2024) 104719，DOI `10.1016/j.trc.2024.104719`，附件15页。  
**先读**：[共同协议](00_Common_Protocol.md)。模型ID `dynamic_idm`。本方案是优先普通驾驶候选，不是神经PPO变体。

## 1. 原文与前次讨论的关键区别 [P]

PDF6式(8)–(11)将误差而不是加速度本身设为AR过程：
\[
a_t=f_{IDM}(h_t;\theta)+e_t,\qquad
e_t=\sum_{k=1}^{p}\rho_ke_{t-k}+\eta_t,
\quad \eta_t\sim N(0,\sigma_\eta^2).
\]

等价形式：
\[
a_t=f_t+\sum_{k=1}^{p}\rho_k(a_{t-k}-f_{t-k})+\eta_t.
\]

不能误写成`a_t=f_t+Σρ_k a_{t-k}+noise`；每个历史动作都必须减去**同一历史时刻**的IDM值。

| 原文位置 | 事实 |
|---|---|
| PDF6 | p=0退化为B-IDM；p=1与OU/Matérn1/2的联系是特殊情形 |
| PDF6–7式(12)–(16) | 速度、位置及联合似然；过程和观测噪声分开 |
| PDF7–8 | highD和OpenACC均5Hz；highD>50s，每类车辆20对；OpenACC后四辆是ACC，不全是真人 |
| PDF8表1 | 比较AR阶数0–10；参数和噪声变化；未只用AR(1) |
| PDF9图5–6 | 该数据存在短时正相关和较长时负相关，AR(5)可表达SE不足之处 |
| PDF10–11 | 1000组后验，1–10s展开；表2为5s，表值×10 |
| PDF11–12 | ring和platoon用于长期动力学，不能当紧急人因验证 |

论文讨论的约10s关联范围，不表示AR(5)直接存了10s观测：在5Hz下直接lag只有1s，递推可产生更长相关。

## 2. 可复用代码 [C]

与01同官方仓库，核对提交 `7520b15a8b163f52532cd3e5c7f7d4db05fc43f0`。已确认 `PGM_highD/AR_IDM_hierarchical.ipynb`；README还有`PGM_highD_joint`及stochastic AR notebook路径说明。`Simulator/simulation_ring.py`实际使用AR4缓存、4项buffer、`-10 m/s²`裁剪以及固定索引；**不能直接视为论文AR5的正式实现**。

复现前生成 `source_to_code_map.md`：式(8)/(9)/(10)/(14) -> notebook cell -> 新函数 -> 单测。至少核对：
1. `rho`与buffer是最新在前还是最旧在前。
2. 参数的归一化尺度是否[33,…]还是[33.3,…]。
3. `sigma_eta`是标准差、方差还是precision的变换。
4. 作者结果使用加速度、速度还是联合似然。

## 3. 为什么它适合本项目 [A/H]

其在线计算仅一个IDM与长度p的内积，易于批量运行和保存状态。它适合作为“数据校准的随机普通跟驰模型”，用于替代纯IID动作噪声。**是否足以响应ADS极端制动是另外的试验问题，不能由低CRPS推出。**

## 4. 拟合规格

### 忠实复现轨 [P]

维持论文的层次log参数、NUTS以及Normal AR系数先验。PDF8给出warmup3000；先验示例`λ0=100, η=2, λη=2e6, λv=1e6, λx=1e7`。由于先验强且可能有标准化，先核对作者代码后执行，不擅自使用数值或改成“更合理”的值。

原文明确讨论IDM/AR成分重叠导致可识别性问题。必须联合估计，不能先点估IDM、再fit AR后宣称重现层次贝叶斯模型；该两阶段方法可以做初始化但要标清。

### 迁移轨 [A]

为节约成本，先与01用同20个train driver；默认p=5，p=0复用B-IDM，p=1只做必要低阶比较。仅当p=5收敛且有明确残差证据，才追加p=4或6之一；不扫0–10全表。

前缀个体化仅使用起点前数据。若可用prefix不到5–10s，记录长度并报告冷启动，不拿起点后的数据补齐历史。

### 平稳性不能被忽视

原文对ρ使用Normal先验，没有显式声明保证每个后验抽样都平稳。本项目部署前计算AR companion matrix谱半径。不能逐项限制`|ρ_k|<1`代替稳定性；例如AR(2)单个系数>1仍可能整体平稳。

- `paper_faithful`：保留原分布，报告非平稳后验比例，不隐藏发散样本。
- `deployable_stable`：如需要稳定参数化，采用偏自相关映射并**重新拟合**，标为适配版。
- 禁止丢弃不稳定样本后仍声称采样自原后验；这改变了分布。

## 5. 状态更新与裁剪语义

新增：
```python
@dataclass
class DynamicIDMState:
    driver_parameters: object
    rho: object
    sigma_process: float
    residual_history: object   # newest-first
    last_native_decision_time: float
    held_acceleration: float
```

核心代码由随包`ar_innovation()`提供：
```python
error, next_history = ar_innovation(error_history, rho, sigma_eta, z_t)
requested = idm(gap, speed, closing, params) + error
```

**两个不同实现不可混称**：
- 生成式误差状态：推进所抽样的`error`，plant clipping不反写该过程。
- 执行动作偏差状态：以`executed-f_IDM`反写历史，属于执行耦合适配。

原文公式以实际动作写等价递推，在无额外clamp时两者相同；加了FITWMAMS限幅后不再自动等价。保留原生无约束core与部署plant，记录被裁剪比例和二者差异，不能静默切换。

AR在0.2s更新；25Hz中间4个tick不重复采样，不把ρ按dt幂次随意变换。需保持滤波状态/native clock和随机消费索引一并snapshot。

## 6. 当前代码组织

```text
dynamic_ar_idm/model.py                                  # AR核与在线driver
dynamic_ar_idm/fit.py                                    # MAP/Laplace及可选NUTS
dynamic_ar_idm/scripts/                                  # 数据、拟合、评测和ring
hierarchical_world_model/src/stochastic_drivers/         # 主项目25 Hz接入边界
```

额外函数：
```python
build_dynamic_likelihood(kind='speed'|'position'|'joint')
check_ar_stationarity(posterior_samples)
initialize_error_history(prefix, parameter_draw, measurement_model)
step_native(observation, driver_state, innovation)
```

联合似然式(14)以 `[position,speed]` 排列，过程协方差为：
\[
\sigma_\eta^2\begin{bmatrix}\tfrac14\Delta t^4&\tfrac12\Delta t^3\\
\tfrac12\Delta t^3&\Delta t^2\end{bmatrix}
+\operatorname{diag}(\sigma_x^2,\sigma_v^2).
\]
不要与01的`[speed,position]`顺序混用；转gap时交叉项换号。无未来数据时只能据生成状态递推。

## 7. 验收

### 概率模型正确

- p=0与相同参数B-IDM同随机流动作一致；p=1与解析AR(1)方差/ACF一致。
- ρ lag顺序单测、谱半径单测、native clock单测通过。
- 作者同设置小批轨迹的模拟均值/协方差与本实现误差落在预设MC容差内。
- 后验诊断按共同协议；先验与似然的标准化逐项可追踪。

### 原文效果复核

PDF11表2 AR5真实单位参考：`RMSE(a)=0.166, RMSE(v)=0.265, RMSE(s)=0.429`，`CRPS(a)=0.095, CRPS(v)=0.149, CRPS(s)=0.217`（由×10表还原）。这些是原文样本与协议，不是新split硬验收。

在原文轨尽量复核“较高阶AR在5s概率仿真优于IID/MA-IDM”的方向，不要求所有指标或所有driver都严格最优。不同筛选得到不同绝对值须解释，不能当bug重训直到匹配。

### 项目价值

共同协议75帧ES/CRPS达到改善门槛；ACF在0–5s和5–10s分别报告。不能强迫任何新highD子集必须出现负相关；有数据证据才要求捕获。长时稳定性报告20s/60s不发散、合理车速/间距及饱和比例，不能用“无波动”要求去消除真实stop-and-go。

### 极端能力

固定driver、不同制动刺激，对比延迟、制动量、残差在紧急段贡献。若AR误差长期保持正值使紧急制动异常变弱，记录限制，不自动归零误差或调用TTC guard。任何风险相关噪声衰减都是新模型，需要单独验收。

### 事实差分接入

自治AR在双分支共用创新时也会抵消；必须显示 `Var[delta | fixed_theta]`。需要人口异质性可采样，不等于保留了逐时刻动作噪声。仅在绝对driver合格后测试锚定版；不通过即保持`world_integration=failed`，不否定AR原算法复现。

## 8. 预算与目标命令

[A] 正式最多20个train driver、4链、3000warmup+1000draw、CPU wall-clock 18h；只做p=0/5主要比较，p=1使用较小诊断预算。达到上限停止并留工件。模型拟合不需要从零PPO；GPU主要用于现有世界批量评测。

```bash
python -m dynamic_ar_idm.scripts.fit --ar-order 5
python -m dynamic_ar_idm.scripts.evaluate
python -m driver_reproduction.scripts.run_matched_evaluation
python -m hierarchical_world_model.scripts.stochastic_drivers rollout --model dynamic_ar5
```

最终分别保留AR(0)/AR(5)的 `*_stationarity.json`，并保留`lag_order_test.json`、`clock_contract.json`、`natural_metrics.json`、`stress_metrics.json`和三等级decision。优先候选的理由是可在线递推和可解释，不是预设它一定能取代紧急人因模型。
