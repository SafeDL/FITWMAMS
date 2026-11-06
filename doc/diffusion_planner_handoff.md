# Diffusion Planner Safety Enhanced：跨工程交接说明

交接日期：2026-11-06。当前交付为已经接纳并冻结的 **Diffusion Planner Safety Enhanced（安全增强版）**，用于和官方原始策略进行 nuPlan 配对评价。highway-env 与 nuPlan 使用同一份部署参数，各自保留必要的观测和执行适配。

本说明随权重存放于 `/media/hp/DATA/nuplan_workspace/FITWMAMS_models/ads/diffusion_planner/safety_enhanced/README.md`；工程内版本为 `/home/hp/FITWMAMS/doc/diffusion_planner_handoff.md`。本次交接只补充说明和文件校验，不更新模型、不重新训练或运行场景评价。

## 1. 交付文件与模型身份

权重实体目录：

```text
/media/hp/DATA/nuplan_workspace/FITWMAMS_models/ads/diffusion_planner/safety_enhanced/
├── README.md                         # 本交接说明
├── handoff.json                      # 实际文件哈希、来源和既有加载核验
├── native/
│   ├── initial/model.pth             # 修复前：官方原始策略的原生导出
│   ├── initial/args.json             # 原始策略的配套配置与归一化
│   ├── enhanced/model.pth            # 修复后：已接纳安全增强版，部署 EMA
│   └── enhanced/args.json            # 增强策略的配套配置与归一化
└── checkpoints/guided/phase_2/policy.pth  # 工程训练检查点，不直接传给 nuPlan
```

| 文件 | SHA256 |
| --- | --- |
| `native/initial/model.pth` | `8d4f8a7d41cb207f7d110e4c1117864a184a6431d987372f0ec901fd98a5a39f` |
| `native/enhanced/model.pth` | `84af836087a6bd92d3df83729c5ec5adaa37ee621ad503daea9a15332c5342a2` |
| 两份 `args.json`，内容相同 | `1acd6861b108dda24be4e06828eecc1bc281160e9b045f97f8b556f11b6bcceb` |
| `checkpoints/guided/phase_2/policy.pth` | `3a10d924e97fc830ff3147f28e5a6693a2d9f9eed74ec38043892f052bdb85ed` |

官方下载原文件为 `/home/hp/FITWMAMS/ads/diffusion_planner/artifacts/model.pth`，SHA256 为 `7a441df91ebe1c912d8262010c40486da24f425f757e2b4228072e251ab67d45`。`initial/model.pth` 是其部署参数的原生导出，文件组织和序列化不同，因此文件哈希不同；原始参数及输出一致性已有核验记录。增强版导出与 highway 部署模型也已通过官方加载器回读，参数一致、固定输入与随机流下最大输出差为 0。

历史路径和报告中的 `guided` 是该次实验方法标识；人工评价统一使用 `initial`、`enhanced` 两个角色。本次名称整理未改变权重参数。

## 2. 运行依赖与加载方式

使用现有 **`tread`** 环境。当前 nuPlan 启动脚本仍依赖 `/home/hp/FITWMAMS` 中的源码；这份权重目录本身不包含完整规划器实现。

- 工作区：`/media/hp/DATA/nuplan_workspace`，启动脚本为 `run_closed_loop.sh`，环境路径由 `env.sh` 设置。
- 规划器配置：`/home/hp/FITWMAMS/ads/configs/planner/diffusion_planner.yaml`。
- 原生类：`ads.diffusion_planner.vendor.diffusion_planner.planner.planner.DiffusionPlanner`。
- 保留的官方源码版本：`a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df`，来源记录为 `/home/hp/FITWMAMS/ads/diffusion_planner/vendor/SOURCE.json`。
- 传入原生 `model.pth` 和同目录 `args.json`，保持 `enable_ema=true`；不把训练状态 `policy.pth` 作为原生权重加载。
- nuPlan 配置使用 2 秒、20 点历史和 8 秒、80 点未来，保留原生特征、归一化与轨迹后处理。highway 的 5 Hz 决策／25 Hz 执行属于另一执行适配。

两种角色分别运行，场景、随机种子、仿真配置与设备保持一致：

```bash
conda activate tread
cd /media/hp/DATA/nuplan_workspace
source env.sh

ADS_MODEL_ROOT="$NUPLAN_WORKSPACE/FITWMAMS_models/ads/diffusion_planner/safety_enhanced/native"
ADS_LIMIT=2
for ADS_ROLE in initial enhanced; do
  PLANNER=diffusion_planner PLANNER_DEVICE=cuda \
    SCENARIO_SPLIT=manual_evaluation SIMULATION=closed_loop_reactive_agents \
    bash run_closed_loop.sh "dp_${ADS_ROLE}_${ADS_LIMIT}_20261106" \
      "$ADS_LIMIT" "$ADS_MODEL_ROOT/$ADS_ROLE/model.pth"
done
```

`ADS_LIMIT=2` 用于少量场景加载检查；手动完整评价时改为 `ADS_LIMIT=all`，两份权重使用相同设置。启动脚本固定 `seed=20260926`，自动从权重同目录读取 `args.json`，并记录模型和配置的 SHA256。CPU 运行可将两次命令的 `PLANNER_DEVICE` 都改为 `cpu`。

工程迁移时还需同步上述源码、规划器 YAML、场景配置及相关 nuPlan 数据路径，并修改启动脚本的 `FITWMAMS_ROOT`；仅复制 `model.pth` 不足以复现当前加载环境。

## 3. 共同场景与 nuPlan 结果归档

共同清单和配置位于：

```text
/media/hp/DATA/nuplan_workspace/FITWMAMS_results/safety_enhanced/manual_nuplan/
├── scenarios.json
├── common_scenarios.json
└── preparation/config/scenario_filter/manual_evaluation.yaml
```

清单包含 1,118 个候选 token、14 类场景，`scenarios.json` 的 SHA256 为 `87b184d569c85830f9b46d8001c59d8e4fb77bed4d254901c82d1a32b026a3a0`。实际构建、成功完成和失败数量以 nuPlan 运行记录为准，候选数不直接视为已完成样本数。

上述命令默认分别输出到 `$NUPLAN_EXP_ROOT/dp_initial_<数量>_20261106/` 和 `$NUPLAN_EXP_ROOT/dp_enhanced_<数量>_20261106/`；`env.sh` 默认将 `NUPLAN_EXP_ROOT` 设为工作区 `exp/`，也允许调用者预先指定其他目录。

交接后的评价保留：两份模型及配置哈希、共同场景清单、实际完成与失败场景、启动参数和随机种子、`terminal.log`、nuPlan 原生逐场景指标与汇总。以共同实际完成场景做配对比较，另外报告未完成场景及原因。主要观察碰撞和安全表现，同时检查停车、进度及原生驾驶指标；ADE/FDE 上升不直接否决安全改善。

nuPlan 结果独立保存，不并入 highD 生成集合的事故率分母，也不用于重新挑选这份已经冻结的权重。**目前尚未据此声明完成 nuPlan 城市闭环安全验证；该项由用户手动执行。**

## 4. 本工程已经完成的修复与证据

策略从官方权重开始，基于全部 2,964 个 EVT 尾部事件对应的冻结长尾生成链发现事故。修复使用 12,685 个有效 nuPlan 原任务场景、23,271 个 highway 成功实例、682 个初始事故示范及 372 个成功纠正后缀。32 个保护祖先及其后代不进入训练。

修复只更新最后一个 DiT block 与输出层，采用原生监督、原模型功能保持及参数更新约束；两阶段共 32,726 次实际更新。全部有效材料完成访问，关键纠正帧至少访问四次。原任务材料并非官方最初训练全集，不将其描述为已获得原始训练全集。

权重冻结后的主最终评价为长尾生成、自然生成各 5,000 个新实例，修复前后使用相同实例和策略随机流：

| 集合 | 原始碰撞 | 增强版碰撞 | 配对碰撞率差及 95% 区间 |
| --- | ---: | ---: | --- |
| 长尾生成 | 177／5,000（3.54%） | 149／5,000（2.98%） | −0.56 个百分点；[−0.952, −0.162] |
| 自然生成 | 91／5,000（1.82%） | 76／5,000（1.52%） | −0.30 个百分点；[−0.710, 0.113] |

长尾修好 34 个碰撞实例、新增 6 个；自然修好 27 个、新增 12 个。长尾驶出道路为 1→3，自然为 1→1；两组无停车，平均进度略有增加。三个既有追加批次每类共 15,000 个实例，长尾碰撞合计 522→404，自然 288→273，其中一个自然批次为 90→92。当前证据支持未见长尾净安全收益，自然收益较小且存在批次波动与局部新增事故。

生成评价中的“未见”指 ADS 搜索与修复阶段的实例隔离；全部尾部源事件已经参与生成模型拟合。经验自然 Test 暂停新增评价，nuPlan 城市结果另行报告。

## 5. 原始证据入口

| 内容 | 当前绝对路径 |
| --- | --- |
| 模型发布身份 | `/home/hp/FITWMAMS/results/ads/reports/final/model.json` |
| 已完成方法协议 | `/home/hp/FITWMAMS/doc/ads_search_repair.md` |
| 当前修复配置 | `/home/hp/FITWMAMS/ads/configs/safety_enhanced.yaml` |
| 完整运行与学习记录 | `/media/hp/DATA/nuplan_workspace/FITWMAMS_results/safety_enhanced/` |
| 全量学习验收 | 上述运行目录的 `verification/full_training_acceptance.json` |
| 主最终配对及原生导出核验 | 上述运行目录的 `verification/final_pairing.json` |
| 追加评价 | 上述运行目录的 `additional_evaluation/` |
| 结果解释 | `/home/hp/FITWMAMS/results/ads/README.md` |

已有加载回读记录分别保存在人工评价目录的 `initial/handoff.json` 和 `enhanced/handoff.json`。本权重目录的 `handoff.json` 记录本次文件核验与这些既有证据的来源，不表示新增执行过城市评价。当前不再续跑已淘汰的修复分支。
