# 被测 ADS：风险分层测试与能力保持修复

本目录实现 ADS 策略、官方模型适配、风险搜索和学习修复。长尾分布使用全部 **2,964 个 highD 尾部事件**的冻结资产；NPC 由现有世界模型闭环演化。使用现有 **`conda activate tread`**，单实验种子 **`20260926`**。

目标是为每个 ADS 得到一个兼顾长尾应对与原有驾驶能力的增强模型，同一份部署参数用于 highway-env 与 nuPlan。方案、学习目标、准入规则和实测结果统一见 [技术方案](../doc/risk_stratified_repair.md)，实验概览见[第 8.5 节](../doc/risk_stratified_repair.md#85-当前实验设置与预期证据)。

## 当前范围与入口

截至 2026-10-25，DP 已完成三方法、两轮正式搜索修复，以及三组各 2,000 个实例的 24,000 次最终评价。三个最终分支均保留首轮接纳模型。本文模型在长尾留出上的碰撞减少 34.4%；两类自然集合的变化及局部回退分别报告，见[第 8.5 节](../doc/risk_stratified_repair.md#85-当前实验设置与预期证据)和[精选结果](../results/risk_stratified_repair/README.md)。CaRL 两轮先导未接纳增强，保留官方参数，其已验收实现已合入主线。DP 正式原生权重已交付，nuPlan 城市闭环由用户手动执行。

| 配置／命令 | 用途 |
| --- | --- |
| [diffusion_planner_pilot.yaml](configs/diffusion_planner_pilot.yaml)；`python -m ads.diffusion_planner.pilot` | 已完成的 DP 单方法两轮先导；结果在 `development/diffusion_planner_functional/` |
| [diffusion_planner_experiment.yaml](configs/diffusion_planner_experiment.yaml)；`python -m ads.run_campaign --config ads/configs/diffusion_planner_experiment.yaml` | 已完成的 DP 三方法正式比较；结果在 `formal/diffusion_planner/` |
| [risk_stratified_repair.yaml](configs/risk_stratified_repair.yaml) | 五模型扩展模板，供新学习器验收后使用 |

DP 与 CaRL 的学习器和实测结果分别见[第 2.4 节](../doc/risk_stratified_repair.md#24-dp-开发结果与当前修复问题)、[第 2.5 节](../doc/risk_stratified_repair.md#25-carl-当前基线与强化学习先导)。DTPP、Flow Planner、STR2 待专属能力保持学习器验收后扩展；DQN、PPO、PlanT 2.0 保留历史实现与权重。PlanTF、PLUTO 已移除。

## 代码组织

| 位置 | 职责 |
| --- | --- |
| `common/world.py`、`instances.py` | 冻结世界、完整实例、历史、谱系与用途隔离 |
| `common/environment.py`、`control.py`、`observations.py` | 因果输入、5 Hz 决策／25 Hz 物理执行、动作与轨迹转换 |
| `search/` | 条件采样、内部搜索分支、失效档案与反馈 |
| `common/expert.py`、`demonstrations.py` | 闭环专家、成功资格、候选额度与执行缓存 |
| `diffusion_planner/` | 专用修复学习器、混合材料、候选接纳、先导及正式比较 |
| `common/repair.py`、`carl/train.py`、`nuplan/train.py` | CaRL 能力保持 PPO 与其他算法的现有学习入口 |
| `common/cohorts.py`、`evaluation_sampling.py`、`reporting.py` | 三组最终集合、配对身份与统计 |
| `risk_stratified_repair.py`、`worker.py`、`run_campaign.py` | 实验流程、单任务和依赖调度 |
| `nuplan/` | 原生特征、算法损失、数据准备、导出及人工评价交接 |
| 各策略目录的 `vendor/`、`artifacts/` | 官方源码、配置、权重与来源哈希 |
| `dqn/`、`idm/` | 独立固定事件工具与世界模型评测，见各自 README |
| `tests/`、`idm/tests/` | 本工程接口与行为测试 |

共享专家执行产生物理轨迹，各算法单独构造原生学习张量。DP 已采用专用能力保持学习器；CaRL 专用学习器已从独立先导副本合入，使用同一 `tread`。其余模型的通用训练入口需要接入相应保持机制后再用于新版正式实验。

## 实验协议

- **固定生成链：**EVT 尾部事件 → 冻结 Flow 初态与结点分布 → Diffusion 软计划 → 世界模型闭环响应。搜索只改变已有外生随机变量。策略输入不含未来计划、NPC 潜变量或搜索信息。
- **等预算测试：**MC、标准全局子集搜索、guided 从同一官方模型及初始先验开始，每方法每轮实际执行 **5,000 次**，共两轮。路径结束后从新先验重启；拒绝提议、分支检查和档案重测均计费。主要发现比较使用第一轮共同策略。
- **学习预算：**DP 每轮 1,000 次梯度更新、有效 batch 64、学习率 `1e-5`；500／1,000 更新各保存候选。监督批次为原任务／highway 保持／修正＝32／16／16。CaRL 先导每轮 20,480 个新决策步，8,192／20,480 步保存候选。
- **保持能力：**原任务人类轨迹和合格专家修正使用原生监督；成功实例约束原生模型输出。固定原始教师、轮首接纳模型与局部参数更新分别保持原有能力和已学会的成功行为。
- **开发准入：**固定 500 个长尾、500 个自然生成实例及最多 64 个档案端点。对照官方模型与上一接纳模型，碰撞等安全指标不增加，平均进度至少保持 98%；不通过则保留上一策略。200 个 nuPlan 开发场景由用户手动核查。
- **安全统计：**风险评分引导搜索，真实失效为自车碰撞或驶出道路。主要报告碰撞；停车、进度和 NPC 间碰撞分列。发现实例与限额档案分别计数，搜索命中率不当作原分布事故率。

| 冻结后最终集合 | 数量 | 作用 |
| --- | ---: | --- |
| 长尾生成留出 | 2,000 | `P_tail` 全新 IID 实例，检验未见长尾交互 |
| 经验自然 Test | 2,000 | 现有 highD Test 按录像／方向比例、不放回抽样 |
| 自然生成留出 | 2,000 | `Q_natural` 全新 IID 实例，检验非定向交通能力 |

初始模型与三种最终模型使用相同实例及随机流，报告配对碰撞率差和 95% 区间。最终集合不用于模型选择。两类自然集合是 ADS 阶段留出；全量尾部拟合涉及部分 Test 源事件，重用信息随实例记录。nuPlan 人工结果单独保存，不加入 highD 分母。

## 策略与原生交付

| 策略 | 初始化 | 新版学习路线 |
| --- | --- | --- |
| Diffusion Planner | 官方 `model.pth`、`args.json` | 最后一个 DiT block 与输出层；原生去噪监督与功能保持 |
| CaRL | 官方 `nuplan_51892_1B/model_best.pth` | 新 on-policy PPO、局部 actor／value、Beta KL 与候选准入；先导完成、未接纳增强，实现已集成 |
| DTPP | 官方 `base_model.pth` | 原生候选选择、条件预测与局部代价学习，待接通 |
| Flow Planner | 官方 `model.pth`、`model_config.yaml` | 原生 Flow Matching 与解码器保持，待接通 |
| STR2 | 官方 `checkpoint-297000` | 原生规划监督、输出头更新与后处理，待接通 |

CaRL 本地 `carl/artifacts/official.pth` 与指定的 nuPlan 权重一致，保留已核对的 BEV、动作缩放及执行滤波。DP／Flow 的未知未来掩码已经修正，旧实验权重保留历史身份。

大型被测模型权重及配套学习状态集中保存在 `/media/hp/DATA/nuplan_workspace/FITWMAMS_models/`，工程中的原文件路径通过符号链接访问。模型配置、轨迹词表和来源记录在工程中保留，并在 DATA 中保存副本，支持从两处加载。已纳入 Git 的三份小型 DQN 策略保留原件。此次恢复的 PlanT 第二、第三份官方权重也已存入 DATA；逐文件路径和校验结果见[模型存储记录](../results/model_storage_20261025.json)。必要模型按此方式保留，清理前先判断其是否仍承担测试或复现用途。

原生导出入口为 `ads.nuplan.export`、`ads.carl.export`、`ads.nuplan.handoff`，保留官方配置、归一化及必要 EMA，不覆盖官方初始文件。DP 配置见 [planner/diffusion_planner.yaml](configs/planner/diffusion_planner.yaml)，人工共同场景见 [scenario_filter/manual_evaluation.yaml](configs/scenario_filter/manual_evaluation.yaml)。当前已完成官方加载器回读、输出一致性及 CLI 配置解析，尚未执行城市仿真；手动命令见[方案第 8.2 节](../doc/risk_stratified_repair.md#82-nuplan-独立人工评价与单模型交付)。

## 数据与结果

| 路径 | 内容 |
| --- | --- |
| `results/risk_stratified_repair/` | 当前说明及精选报告在工程内，完整运行按分目录链接到 DATA 盘 |
| [risk_stratified_repair_history/](../results/risk_stratified_repair_history/README.md) | 历史实验、仍被引用的初始权重、运行参考与归档 |
| [历史索引](../results/risk_stratified_repair_history/index.json) | 路径映射、报告／权重 SHA256 与维护归档清单 |
| `results/risk_stratified_repair_manual_nuplan/` | 人工场景清单、原生配置、加载验收及已交付 DP 权重 |
| `results/natural_flow/` | 85,904 个 Train＋Val 自然片段拟合的独立自然 Flow |

nuPlan 根目录为 `/media/hp/DATA/nuplan_workspace`。现有 Val 留作人工评价；新增官方 Train 源池共 758 DB／49.66 GiB。DP 已准备 2,000 个保持和 200 个开发场景的原生特征，来自隔离的 585／145 个日志；清单在 `preparation/retention/diffusion_planner/materials.json`，数据验收与预算见技术方案。正常 highway 保持库有 491 个成功实例。

各轮 `search.jsonl` 记录全部执行，`archive/` 保存限额代表，`teacher_selection.json` 记录修复输入，`admission.json` 记录候选接纳；这些数量分别解释，具体口径见技术方案。`unit_status.json`／`status.json` 为运行状态，`pilot_summary.json`／最终报告为完成结果。

Git 保存源码、当前配置、说明、索引与精选报告；完整权重、轨迹、日志和 DATA 链接留在本地。历史验收与运行日志已合并到可恢复归档，位置见历史索引；历史报告的名称、schema、采样命名空间及原始指标保持原身份。

## 环境与验收

```bash
conda activate tread
export USE_TF=0
python -m pytest ads/tests -q
python -m ruff check ads --exclude '**/vendor/**' --target-version py310
```

依赖见 [requirements.txt](requirements.txt)。使用现有 Torch／NumPy 和本地 nuPlan devkit，不创建虚拟环境。原生启动禁用用户包目录时也需使用 `tread` 中已补齐的依赖。官方资产准备入口为 `ads.assets` 与 `ads.nuplan.assets`；现用模型已齐备。确需重新获取大文件时，先保存并校验到 DATA 模型目录，再恢复工程链接；STR2 安装入口会获取完整分发包，日常加载现用权重无需重跑该入口。

正式实验以 `source_snapshot.json`、`recipe_freeze.json` 和完整源码包登记实际实现。已完成的 DP 保留 424 个 Python 文件的源码归档；其结束后，CaRL 的 17 文件实现已合入，合入时核对 434 个 Python 文件与已验收副本一致，101 项回归检查及 Ruff 通过。CaRL 正式续接因没有接纳增强而停止。

当前代码中，初始策略只通过 `Campaign.baseline()` 导入，`Campaign.train()` 只负责修复；专家标签与生成成本只统计修复轮次。重复的初始训练分支、空 `baseline` 配置和旧专家缓存路径推断已删除。已封存实验的复现使用对应源码包和冻结配置；后续代码整理不改写这些快照、权重或最终结果。

离散策略修复直接加载已有 checkpoint 并累计训练步数；随机状态统一使用 `mutate()`，回放按归档声明的 schema 加载对应状态类型。
