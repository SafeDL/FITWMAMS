# 学习型 ADS：危险交互发现与安全修复

本目录承接全部 2,964 个 highD 尾部事件建立的冻结长尾分布和分层世界模型，完成事故搜索、专家纠正、能力保持修复与原生权重交付。使用现有 `conda activate tread`，单种子 `20260926`。

当前交付为 **[Diffusion Planner Safety Enhanced](../results/ads/runs/safety_enhanced/manual_nuplan/enhanced/model.pth)**。权重身份见[模型清单](../results/ads/reports/final/model.json)，执行参数见[配置](configs/safety_enhanced.yaml)和运行内 `training_protocol.json`，数值见[结果](../results/ads/README.md)。已淘汰修复分支及上一版交互搜索的测试记录已移除，不再续跑候选。

## 策略范围

| 策略 | 初始化与接口 | 实验状态 |
| --- | --- | --- |
| Diffusion Planner | 官方 `model.pth`、`args.json`；扩散规划、周车预测及原生导出 | 论文主要 ADS，已完成搜索、修复和生成评价 |
| CaRL | 官方 `nuplan_51892_1B/model_best.pth`；BEV、Beta 连续动作、PPO 与原生导出 | 先导未接纳增强，保留官方模型 |

当前仅注册上述 **两种策略**，与 `/media/hp/DATA/nuplan_workspace/run_closed_loop.sh` 可直接运行的策略完全一致。官方版本与 SHA256 位于各算法 `artifacts/official.json` 和 `vendor/` 来源记录；不能用随机参数代替加载失败的官方权重。`idm/` 是世界模型的规则驾驶诊断工具，不计入被测学习型 ADS。

## 代码结构

| 位置 | 内容 |
| --- | --- |
| `common/world.py`、`instances.py`、`setup.py` | 冻结资产、完整实例、历史、随机实现与用途隔离 |
| `common/environment.py`、`control.py`、`observations.py` | 因果观测、5 Hz 决策／25 Hz 执行和轨迹跟踪 |
| `search/` | 交互反馈提议、覆盖加权子集搜索、档案与离线比较 |
| `common/expert.py`、`demonstrations.py` | 闭环专家、成功执行和原生监督标签 |
| `diffusion_planner/` | 当前全量学习、偏移状态纠正、保持约束、评价及导出 |
| `carl/` | CaRL 原生策略、PPO 与 highway 适配 |
| `nuplan/` | 原生特征、训练材料、监督目标、导出与人工评价交接 |
| `common/cohorts.py`、`evaluation_sampling.py`、`reporting.py` | 实例隔离、配对统计与安全结果 |
| `risk_stratified_repair.py`、`worker.py`、`run_campaign.py` | 既有对照与其他注册策略的通用实验调度 |
| `tests/`、`idm/tests/` | 当前接口与行为验收 |

当前 DP 协议入口是 `python -u -m ads.diffusion_planner.capability`，配置为 [safety_enhanced.yaml](configs/safety_enhanced.yaml)。已完成运行位于 `results/ads/runs/safety_enhanced/`；冻结执行源码在运行内保存。该入口不表示需要再次执行实验。

`python -m ads.search.allocation` 只分析已有当前搜索日志，不新增世界执行。随机采样、标准子集保留为已有对照，上一版交互方法不再参与比较。共同先验和被当前模型实际使用的旧成功执行，分别保存在 `runs/comparison/common_prior/` 和 `shared_successes/`；它们是当前材料依赖。

## 数据与原生交付

使用全部 12,685 个有效 nuPlan 原任务场景、23,271 个 highway 成功实例及全部合格事故纠正。开发与最终长尾／自然生成集合各 5,000 个，最终实例、开发实例和保护谱系不进入学习。详细访问与训练记录随已完成运行保存。

原生权重位于 `/media/hp/DATA/nuplan_workspace/FITWMAMS_models/ads/diffusion_planner/safety_enhanced/native/enhanced/model.pth`，工程通过结果目录链接访问。名称为安全增强版，原生文件名保持官方加载器要求的 `model.pth`；历史报告中的 `guided` 是原实验方法标识。

highway-env 与 nuPlan 使用相同部署参数。原生加载回读已核对，nuPlan 离线及城市闭环由用户手动执行；命令与共同场景见[人工评价说明](../results/ads/runs/safety_enhanced/manual_nuplan/README.md)。安全指标优先，ADE／FDE、进度和个例低速用于诊断。

```bash
conda activate tread
python -m ads.nuplan.export diffusion_planner TRAINING_CHECKPOINT OUTPUT/model.pth
```

CaRL 使用相同导出入口及算法名。导出保存配置、来源、SHA256 和必要状态，不覆盖官方初始模型。

源码、配置、紧凑报告和交付元数据纳入 Git；大型实例、逐例数据、模型与训练状态按 `.gitignore` 保留在存储中。当前与对照运行保存各自实际执行的冻结源码。

## 验收

```bash
conda activate tread
export USE_TF=0
PYTHONDONTWRITEBYTECODE=1 python -m pytest ads/tests ads/idm/tests -q -p no:cacheprovider
python -m ruff check ads --exclude '**/vendor/**' --target-version py310 --no-cache
```

依赖见 [requirements.txt](requirements.txt)。接口测试覆盖严格加载、坐标与历史、控制调度、快照重放、用途隔离、配对统计与原生导出；不替代 nuPlan 城市闭环评价。
