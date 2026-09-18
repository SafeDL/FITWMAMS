# 方案03：多机制层次B-IDM——驾驶风格与速度状态异质性

**依据**：Shubo Wu, Dong Ngoduy, Zhengbing He, Yajie Zou, Jian Sun, *Modeling car-following behaviors considering driver heterogeneity: A multi-regime stochastic framework*, TRC 179 (2025) 105282，DOI `10.1016/j.trc.2025.105282`，附件25页。  
**模型ID**：`multi_regime_bidm`。先读[共同协议](00_Common_Protocol.md)。这不是普通/紧急状态切换器。

## 1. 严格按原文定义两个层次 [P]

论文流程：
```text
整条驾驶事件特征 -> K-means三种style
(s,v,Δv)序列 -> HDP-HSMM分段
在style内 -> pooled / hierarchical / unpooled Bayesian IDM
```

重要区别：
- style为neutral/timid/aggressive，依据平均时距、最大加速度、最大减速度聚类（PDF9–10）。
- 虽然PDF10提到跟驰研究常用F/A/D/S，本文最后按平均速度把识别结果命名为LS/HS/MS（PDF11）。**不能用“紧急制动状态”替代其结果。**
- 层次B-IDM的individual层是同style内的regime，不是给每个驾驶员单独一个参数集（PDF6–7、式19–25）。
- 随机性是联合后验参数加IID高斯动作误差；并没有引入01/02的GP/AR。

原数据：Waymo提取459个HV–HV、71个HV–AV事件；Lyft每类1000个，原频率10Hz，Waymo持续≥15s、Lyft≥20s，gap<85m、follower speed>3m/s、车辆长3–6m。原文仿真主要选每种style/mode一个代表事件，共12个，N=1000参数组（PDF9、15–16）。因此不能将表4/5解释为完整held-out新司机验证。

## 2. 原文中必须记录的未决问题

### 2.1 观测/持续时间分布存在文字疑点 [P→A]

PDF10 §5.2明确写：Poisson和Gaussian分别用于observation和state duration。与此同时其观测为连续`s,v,Δv`（Δv可负），duration需为正整数。这一文字与通常的数据支持集不相容，论文没有在此处给出转换细节。

**禁止静默调换并声称完全照原文实现。** 建立`deviations.md`记录：
- 忠实复现需要作者代码/补充说明澄清；当前 `paper_exact_segmentation=blocked_ambiguity`。
- 为可执行迁移，[A]采用Gaussian多维观测+正整数Poisson duration（明确0长度平移或截断规则），称为 `documented_adaptation`。
- 在获得作者说明前，不以该适配结果复刻原表绝对数值。

### 2.2 离线分段不等于在线状态转移 [P→A]

论文的HDP-HSMM对既有序列分段；未清楚给出可直接用于任意ADS干预的在线regime选择实现。不能在部署时读取整条测试轨迹的分段结果。

### 2.3 风格分类也可能偷看未来

最大加减速度及整段平均时距可以用于train风格发现；测试开始时不能用测试未来计算style。必须单独实现prefix分类或从train style频率抽样。

## 3. 代码来源和选择 [C]

本次未确认作者公开完整实现，不编造GitHub地址。论文给出的Lyft加工数据地址 `https://github.com/RomainLITUD/Car-Following-Dataset-HV-vs-AV` 是**数据来源**，不是该模型代码。

可参考基础推断库 `https://github.com/mattjj/pyhsmm`：本次读取README，明确已停止维护、最后测试Python3.7、主要使用weak-limit HDP-HSMM近似。因此：
- 不直接把旧库装进tread并降级主环境。
- 可单独隔离环境运行推断，仅导出数值参数；许可证在复用前核查。
- 或依据Johnson/Willsky框架实现小型截断HDP-HSMM，但必须标明截断和推断算法，不把有限HMM当HDP-HSMM。

## 4. 第一版实施策略 [A]

**先验证这个额外层次是否有价值，不默认扩散到所有模型。**

### Stage A：离线分段复核

train内使用K-means K=3；所有标准化只在train拟合。HDP-HSMM最大3状态、α=γ=4按照原文；观测使用`s,v,Δv`而非加速度标签。style解释只能在train做，使用明确记录的簇中心排序，不预设某一cluster一定“激进”。

输出每个train recording、style、regime的样本数、持续时间、速度/gap/closing分布。若3状态在新highD中塌成1状态，这不是自动实现失败，但削弱复杂模型必要性；不继续强迫得到三种有意义状态。

### Stage B：层次B-IDM

对每个style拟合一个population，regime参数部分池化，比较`pooled B-IDM`和`hierarchical B-IDM`。unpooled只作小型诊断。采用速度似然式(27)，原文NUTS5000warmup+2500posterior samples；迁移小预算与完整论文设置分别命名。

### Stage C：因果在线滤波

保存后验质量`P(regime,剩余duration | H_≤t)`。当前观测只来自已执行状态；先预测持续时间/转移，再用当前`s,v,Δv`更新观测似然。决策用过滤分布而非后验平滑/Viterbi全轨迹标签。

可参考随包`hsmm_filter_step()`。它仅实现**给定参数的有限显式duration滤波**，不是HDP训练器。对剩余duration=1的概率质量，使用排除自转移的矩阵转移并抽新duration；其他质量只减小剩余时间。

同一driver的style场景内固定；regime可以变化。参数表按一个联合posterior draw在场景开始抽取，不在每个tick独立重抽五参数。改变状态的随机性来自在线过滤与duration模型。

## 5. 代码设计

```text
multi_regime_bidm/src/hsmm.py              # documented finite-HSMM适配
multi_regime_bidm/src/online_filter.py     # 因果显式持续时间过滤
multi_regime_bidm/src/driver.py            # episode级联合参数与在线动作
multi_regime_bidm/src/pymc_stage_b.py      # hierarchical/pooled NUTS
hierarchical_world_model/src/stochastic_drivers/  # 主项目接入与拒绝门
```

状态字段：
```text
style_id or style_posterior
regime_duration_mass [R,Dmax]
sampled_regime
regime_parameter_table [R,5]
last_native_time
held_acceleration
```

Dmax依据train持续时间设定，截断尾部质量必须报告。超出训练支持的观测对全部regime都很低似然时，输出`low_observation_likelihood`；禁止突然跳到手写emergency标签。

## 6. 验收标准

### 算法正确

- 持续时间概率、转移行和、观测归一化均合法。
- 小型R=2、Dmax=3合成例，用穷举路径对比online filter最大误差≤`1e-8`。
- 修改t后的观测不改变t时刻过滤结果；离线平滑可变化但不得流入controller。
- style未来峰值泄漏测试、duration离散单位测试、regime参数联合采样测试通过。
- 原文分布疑点仍未澄清时，严格复现状态不得写passed；适配可单独通过。

### 迁移增量

用同原生dt、数据、样本数比较：
1. pooled B-IDM；
2. offline oracle regime B-IDM（仅回顾性上界）；
3. **online filtered hierarchical B-IDM**（可部署候选）。

主验收只看3相对1的CRPS/序列ES与时序diagnostics，采用共同协议标准。2显著好于3不能算成功，说明未来分段信息被拿走后优势消失。

报告持续时间分布、切换频率、过滤置信度与失败事件，不要求“同一个人任何时刻style都变化”。高频regime震荡和jerk峰值需明确诊断。

### 极端边界

本论文没有专门验证紧急避碰。仅在`regime OOD`、动作边界、响应方向/延迟上做压力报告。LS/HS/MS绝不能被重命名成安全/危险/紧急；无碰撞不能作为人类真实性证明。

### 世界接入

绝对driver先通过；可探索差分但必须保存两个独立regime滤波器状态以及共同随机量。两个分支可能在不同regime，输出跳变需报告，不用隐藏jerk滤波掩盖。模型原生10Hz与25Hz桥接按共同协议处理。

## 7. 预算与停止

[A] 第一轮只做同一批train样本，1次style/HSMM拟合、1次层次B-IDM拟合；HSMM最多1000 Gibbs sweeps，观察记录后500sweeps是否稳定；Bayesian先2链smoke，再4链正式。正式总CPU时间上限24h，超时blocked而不是更换为HMM后冒充原算法。

默认不迁移WOMD全量；highD迁移不等于原文数据复现。若在线regime版不能超过单一B-IDM，或优势只来自oracle segmentation，停止这一支，保留02为普通随机模型候选。

已实现CLI：
```bash
python -m multi_regime_bidm.scripts.fit_stage_a
python -m multi_regime_bidm.scripts.fit_stage_b_nuts
python -m driver_reproduction.scripts.run_matched_evaluation
python -m hierarchical_world_model.scripts.stochastic_drivers rollout --model multi_regime --style-id 1 --allow-unaccepted
```

最终工件必须包含 `distribution_ambiguity.md`、`style_leakage_audit.json`、`filtered_vs_oracle.json`。本方案可说明多状态建模的经验价值，不能支持“自动发现了人类紧急策略”。
