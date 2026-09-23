# highD 交通行为世界模型多维评价与全量汇总

## 结论

面向 ADS 测评，一个合格的交通行为世界模型必须同时回答三个问题：

1. ADS 不改变场景时，能否复现真实轨迹和长尾事件；
2. ADS 改变动作后，NPC 是否会及时、方向正确、强度合理且局部地响应；
3. 多车长时间闭环后，是否仍保持安全、道路约束、交通统计和数值稳定。

现有结果中，没有一个模型同时满足三项。带真实未来稀疏结点的事实 HiQR 重建精度最高，
但它不是仅凭历史自主预测；B2-RL 和 B2-RLP-e3 能读取闭环状态自主响应，但 5.96 s 自然
测试的样本均值 ADE/FDE 仍约为 0.624/2.16 m，NPC 越界率约为 8.3%--8.8%；
TrafficBots V1.5-highD 的自然 FDE 和 Energy Score 较好，但当前全背景车协议下尚无干预和
ADS 闭环结果。因此，当前最合理的论文主线仍是“高精度名义轨迹 + 因果交互残差策略”，
而不是从已有候选中直接宣布一个完整世界模型。

机器可读汇总由 `tools/summarize_highd_behavior_world_models.py` 从正式结果工件生成，输出在
`results/comparisons/`。不同协议的结果禁止放入同一排行榜。

## “全量 highD”的统一含义

本报告中的全量不是把训练集也拿来测试，而是：扫描 highD 全部 60 个 recording，按
recording 隔离训练、验证和测试，再使用所有符合协议的测试样本。

| 数据层 | 训练 | 验证 | 测试 | 说明 |
|---|---:|---:|---:|---|
| 标准场景缓存 | 72,771 | 13,133 | 10,151 | 共 96,055 条，60 个 recording 无跨 split 泄漏 |
| 严格因果固定车群 | 71,198 | 12,744 | 9,688 | 只保留完整 1 s 历史、5.96 s 未来且有背景车的场景 |
| 真实交互事件 | — | — | 913 | 测试集内全部合格事件 |
| 纵向配对干预 | — | — | 11,784/seed | 共同随机数、多个刺激剂量 |
| 横向配对干预 | — | — | 18,342/seed | 相邻车并入和前车切出 |

全量多车结果使用包含 `same_rear` 的 `highd_all_background` 口径。历史
`highd_follower_excluded_v1` 结果不进入主表。随机 IDM 的筛选虽然扫描了 60 个 recording，
但“至少连续跟驰 50 s、无换道”等条件最终只在 recording 25、26、36 中得到合格车对；
因此它们是全数据扫描后的跟驰子集，不是 60 个 recording 上的道路级世界模型结果。

## 从参考文献提炼的评价维度

### 1. 事实轨迹与事件重建

应同时报告样本均值 ADE/FDE、联合 best-of-K ADE/FDE、速度/间距/加速度误差以及真实事件
分层结果。ADE/FDE 衡量几何误差，RMSE 衡量跟驰状态误差，但二者都不能证明模型会交互。
VBD 明确定义了 ADE/FDE、minADE/minFDE，并与碰撞、越界、逆行和运动学不可行共同报告
（[VBD，第 7--8 页](<../references/Versatile_Behavior_Diffusion_for_Generalized_T.pdf>)）；
Dynamic-AR IDM 和 MA-IDM 使用加速度、速度、间距的 RMSE
（[Dynamic-AR IDM，第 10--12 页](<../references/Calibrating Car-Following Models via Bayesian Dynamic Regression.pdf>)；
[MA-IDM，第 11--12 页](<../references/Zhang-2024-Bayesian-calibration-of-the-intelli.pdf>)）。

### 2. 概率质量、校准与多样性

随机模型不能只报最优样本。应报告 proper score（Energy Score、CRPS 或 NLL）、置信区间
覆盖率和宽度、样本间距离，并将样本均值误差与 minADE 分开。MA-IDM、Dynamic-AR IDM
和 Multi-regime B-IDM 都使用 CRPS 检查随机模拟分布，而非只检查均值轨迹
（[Multi-regime B-IDM，第 15--16 页](<../references/Modeling car-following behaviors considering driver heterogeneity A multi-regime stochastic framework.pdf>)）。
TrafficBots 的论文也提醒，按碰撞筛选 128 个样本再保留 32 个会改变安全指标，minADE 不能
代表整体分布质量（[TrafficBots V1.5，第 3--4 页](<../references/TrafficBots V1.5 Traffic Simulation via Conditional VAEs and Transformers with Relative Pose Encodi.pdf>)）。

### 3. 运动学、道路与舒适性

应比较速度、纵横向加速度、角速度、角加速度和 jerk 的分布，并报告越界、逆行和运动学
不可行率。SceneDiffuser/WOSAC 将线速度、线加速度、角速度、角加速度、道路边缘距离、
越界等组合为分布真实性指标
（[SceneDiffuser，第 7--9 页](<../references/Scenediffuser_Efficient_and_controllable_driving_simulation_initialization_and_rollout.pdf>)）。
自然对抗场景论文进一步表明，高碰撞率可能只是过大的加速度和转向范围造成的非自然行为，
因此碰撞、换道、加速度/转向分布及碰撞类型必须共同解释
（[自然驾驶先验对抗生成，第 10--12 页](<../references/Adversarial safety-critical scenario generation using naturalistic human driving priors.pdf>)）。

### 4. 多车交互真实性

WOSAC 将车间距离、碰撞和 TTC 分布归入 interactive metrics；仅有单车 ADE 不能衡量场景
一致性。更关键的是，ADS 测评需要配对反事实干预：保持场景和随机数不变，只改变 ADS
动作，检查 NPC 响应方向、剂量单调性、响应延迟、响应幅度、无关车辆漂移和新增碰撞。
Active Inference 论文用制动/转向选择、响应时间、减速度强度和碰撞结果与人类数据对齐，
并对类别结果使用 Jensen--Shannon divergence、对响应时间使用 Wasserstein distance
（[Active Inference，第 4--12、17--18 页](<../references/Active inference as a model of collision avoidance behavior in human drivers.pdf>)）。

### 5. 闭环安全，而不是碰撞率单指标

应在多个固定 ADS 控制器下报告 ego/NPC 新增碰撞、NPC--NPC 碰撞、越界、进度和失败前
暴露时间。安全严重度还应加入最小 TTC、TET（危险 TTC 的暴露时长）和 TIT（危险程度的
时间积分）。两篇 RSS 文献都指出最小 TTC 只反映瞬时最危险点，TET/TIT 才能描述整个冲突
过程（[RSS cut-in，第 6、12--13 页](<../references/Calibration and evaluation of responsibility-s.pdf>)；
[RSS car-following，第 2--5 页](<../references/Safety Evaluation of Responsibility-Sensitive.pdf>)）。

### 6. 长时程稳定性、泛化与效率

应按预测时域、长尾类型、车辆类别、recording 和随机种子分层，并检查长时间 platoon/ring
稳定性、闭环误差累积、推理延迟与重规划频率。Dynamic-AR IDM 用 1--10 s 误差及 ring/
platoon 验证长时程；SceneDiffuser 表明普通自回归模型随重规划频率增加会累积误差，因而
离线一次性预测精度不能替代闭环评测。至少三个训练种子和场景级 bootstrap 置信区间应成为
神经模型正式结果的一部分。

### 7. 可控性与长尾覆盖

场景生成器还要报告约束满足率、生成分布与真实分布距离、长尾覆盖和生成效率；这些指标
评价的是场景先验，不应与 NPC 策略的 ADE 混排。SceneDiffuser 的硬约束、VBD 的引导生成
和自然驾驶先验对抗生成都同时强调可控性与自然性。压力场景中的碰撞率是条件分布结果，
不能写成自然交通碰撞概率。

## 全量严格因果测试：可直接比较的多车模型

下表模型均使用 1 s 历史、5.96 s 闭环未来、全部背景车和 9,688 条严格因果 test 场景。
神经模型使用 3 个训练种子；TrafficBots 当前为 1 个适配训练种子。数值越低越好，只有
多样性不是单调指标。ADE/FDE 为 16 个未来样本的样本均值，不是挑最好结果。

| 模型 | Energy Score | ADE (m) | FDE (m) | 多样性 (m) | 新增碰撞率 |
|---|---:|---:|---:|---:|---:|
| IDM lane keep | 1.1273 | 1.7943 | 4.5589 | 0.0000 | 0.134% |
| Response A2 | 0.6432 | 1.1187 | 3.5361 | 1.0695 | 5.468% |
| B2 | 0.5332 | 0.6797 | 2.3551 | 0.4324 | 0.671% |
| B2-R | 0.4456 | 0.6322 | 2.1791 | 0.4559 | 0.366% |
| B2-RL | 0.4381 | 0.6249 | 2.1580 | 0.4529 | 0.344% |
| B2-RLP-e3 | 0.4404 | **0.6240** | 2.1552 | 0.4519 | **0.340%** |
| Semantic C2 | 0.7785 | 0.7895 | 2.7811 | 0.3082 | 1.586% |
| TrafficBots V1.5-highD | **0.4210** | 0.6719 | **1.9545** | 0.1556 | 0.384% |

解释：TrafficBots 的整体分布分数和 FDE 最好，但纵向 90% 覆盖率只有 6.82%，多样性也
明显小于 B2 系列；它不是无条件胜者。B2-RLP-e3 相对 B2-RL 的自然误差变化很小，说明
横向响应训练没有明显破坏事实轨迹。Response A2 的多样性最大却有 5.47% 新增碰撞，说明
“更随机”并不等于“分布更真实”。IDM 碰撞率最低但轨迹误差最大，也说明碰撞率不能单独
用于评判行为世界模型。

### 真实事件重建

913 个测试事件上，B2-RL/B2-RLP-e3 明显优于 TrafficBots 和早期候选。

| 模型 | Energy Score | ADE (m) | FDE (m) | 换道 Brier | 制动 Brier |
|---|---:|---:|---:|---:|---:|
| B2-R | 0.0999 | 0.2042 | 0.6703 | 0.2286 | 0.0455 |
| B2-RL | **0.0972** | 0.2011 | 0.6605 | **0.2257** | 0.0449 |
| B2-RLP-e3 | 0.0973 | **0.2011** | **0.6602** | 0.2263 | **0.0443** |
| TrafficBots V1.5-highD | 0.2222 | 0.3989 | 0.9474 | 0.2934 | 0.0552 |

### 配对因果干预

| 模型 | 纵向响应率 | 纵向条件延迟 (s) | 横向响应率 | 主动横向避让率 | 无关车漂移 (m) |
|---|---:|---:|---:|---:|---:|
| B2-RL | 34.75% | 0.859 | 13.45% | **0.103%** | **0.0440** |
| B2-RLP-e3 | **35.19%** | **0.848** | **16.07%** | 0.091% | 0.0447 |

B2-RLP-e3 的“横向响应率”提高约 19.5%，但主动横向避让反而没有提高，峰值横向响应也几乎
不变；增加的响应主要仍是纵向制动。因此不能把它写成已经学会横向博弈。TrafficBots 在
当前全背景车协议下没有 T2b 配对干预结果，不能仅凭模型结构宣称其交互更真实。

### 固定 ADS 闭环

| NPC 模型 | ADS | ego 碰撞率 | NPC 碰撞率 | NPC 越界率 | ego 进度 (m) |
|---|---|---:|---:|---:|---:|
| B2-RL | cruise | 1.841% | 0.875% | 8.781% | 185.33 |
| B2-RLP-e3 | cruise | 1.841% | 0.867% | 8.785% | 185.33 |
| B2-RL | trajectory MPC | **0.209%** | 0.636% | 8.338% | 183.01 |
| B2-RLP-e3 | trajectory MPC | 0.217% | **0.633%** | **8.317%** | 183.00 |

MPC 显著降低 ego 碰撞，但两种 NPC 模型仍有约 8.3% 越界。这个越界率比碰撞率更直接地
阻止其成为正式仿真背景交通。当前还缺 TrafficBots 在相同全背景车、相同 ADS、相同终止
规则下的闭环结果，因此不能完成外部基线闭环排名。

## 不可混排但必须保留的组件结果

### 条件事实重建

全槽位事实 HiQR 在全部 10,151 条标准 test 序列上的 ADE/FDE 为
0.0400/0.0363 m，P95 位移误差 0.0910 m，速度 MAE 0.0324 m/s。它依赖 Flow/Diffusion
提供的真实未来稀疏状态约束，属于“事实重建上限/名义轨迹锚点”，不能与 prefix-only
模型的自主未来预测直接排名。

对应 CIH 自主响应候选已在 validation 拒绝：固定窗口方向一致率 0.7576、剂量单调率
0.5125，均未达到预注册的 0.95 门槛；没有运行确认性 test。当前证据只支持事实锚点，
不支持“CIH 已同时实现高精度与自主交互”的结论。

### 条件 Diffusion 和场景 Flow

- 条件 Diffusion 在 10,151 条 test 上的 4 样本均值 ADE/FDE 为 0.0309/0.0091 m，
  Energy Score 为 0.02146；但显式使用 2、4、5.96 s 的真实未来背景车结点，终点还是条件，
  因此只能评价插值/重建能力。
- 场景 Flow 的 test joint NLL 为 92.589，初态和未来结点平均 KS 为 0.0842/0.1243，物理
  投影后合法率为 100%；最差单变量 KS 为 0.4208，出现在右后车纵向位置。它适合做场景
  先验，但不响应 ADS。

### 随机跟驰驾驶人

相同 36 对 held-out 跟驰车对、3 s、64 个未来样本的直接比较为：

| 模型 | 位置 RMSE (m) | 速度 RMSE (m/s) | 加速度 CRPS | 位置 90% 覆盖率 |
|---|---:|---:|---:|---:|
| B-IDM | 0.7615 | 0.5710 | 0.2021 | **86.37%** |
| MA-IDM | **0.2220** | **0.2649** | **0.1332** | 69.00% |
| Dynamic-AR(5) IDM | 0.2652 | 0.3081 | 0.1485 | 70.85% |
| Pooled B-IDM | 0.6678 | 0.5299 | 0.1917 | 37.59% |
| Multi-regime B-IDM | 0.6313 | 0.4975 | 0.1834 | 41.56% |

MA-IDM 是当前最好的纵向跟驰 teacher，但 69% 的“90%区间”覆盖率显示不确定性仍欠校准。
B-IDM 的覆盖率较高是以过宽区间为代价，其平均位置区间宽 2.13 m，而 MA-IDM 为
0.55 m。Active Inference 的 highD 纵向适配在 304 个制动事件上的速度/间距 RMSE 为
0.759 m/s 和 1.866 m，且没有横向转向适配；它适合作为响应机制参考，不是多车世界模型。

## 当前模型取舍

| 方法 | 事实精度 | 自主交互 | 闭环道路稳定 | 当前角色 |
|---|---|---|---|---|
| 事实 HiQR | 很高，但使用未来约束 | 响应候选未通过 | 未完成可发布闭环 | 名义事实轨迹锚点 |
| B2-RL | 中等 | 有纵向响应，横向弱 | 越界率过高 | 当前自研交互基线 |
| B2-RLP-e3 | 与 B2-RL 相当 | 响应次数增加，主动避让未增加 | 未改善越界 | 机制消融，不晋升主模型 |
| TrafficBots V1.5-highD | 中等，FDE/FES 有优势 | 结构上可交互，当前探针缺失 | 当前全背景闭环缺失 | 必留外部基线 |
| Flow + Diffusion | 条件重建很高 | 不响应 ADS | 非独立闭环策略 | 场景先验和名义轨迹生成 |
| 随机 IDM 家族 | 仅纵向跟驰 | 对前车状态有因果响应 | 无道路级横向能力 | teacher、下界和消融 |
| Active Inference | 特定制动事件一般 | 规避机制可解释 | 非完整多车环境 | 特定碰撞规避参考 |

## 尚缺的确认性证据

1. 将 TrafficBots 在 `highd_all_background` 上完成 T2b 和 T3，才可与 B2-RL 做完整外部比较。
2. 为所有闭环模型补充最小 TTC、TET、TIT、逆行率、运动学不可行率和 jerk 分布；当前
   仅有碰撞/越界不足以解释失败严重度和舒适性。
3. 将 paired response 从“会不会响应”升级为“是否像人”：需要真实相似事件的响应时间、
   操作选择和强度分布作为校准目标。
4. 新交互残差策略必须在同一次 release decision 中同时通过事实保持、方向/剂量/局部性、
   多 ADS 闭环和随机校准，之后才能进入 `idm_ams/` 做长尾风险估计。

## 结果来源

- `results/comparisons/highd_behavior_world_model_summary.json`
- `results/comparisons/highd_natural_test.csv`
- `results/comparisons/highd_logged_event_test.csv`
- `results/comparisons/highd_matched_car_following.csv`
- `results/interactive_behavior_world_model/benchmark_v1/aggregate/test_seed_summary.csv`
- `results/hierarchical_world_model/factual_all_slots/evaluation.json`
- `results/background_diffusion/evaluation_summary.json`
- `results/highd_natural_driving_flow/evaluation_summary.json`
- `results/driver_reproduction/matched_highd/matched_metrics.json`
