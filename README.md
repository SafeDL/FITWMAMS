# FITWMAMS

从 highD 真实交通数据构建长尾测试分布和可交互的交通世界，用于学习型自动驾驶系统（ADS）的失效发现与安全增强。

## 方法框架

**自然驾驶数据 → EVT 长尾事件 → Flow 场景条件与 Diffusion 软计划 → 分层交通世界 → 危险交互搜索 → 关键状态纠正 → 未见实例安全评价。**

1. **构建长尾分布。** 对自然驾驶片段计算安全包络风险，以 EVT 识别尾部事件。保留全部 2,964 个尾部事件拟合长尾 Flow，采样车辆占位、初始状态和稀疏运动结点；Diffusion 据此生成 NPC 软计划。自然 Flow 使用 Train＋Val 的合格自然片段单独拟合。
2. **闭环演化交通。** 跟踪器执行软计划，计划适用性机制判断是否交权，接管后的 NPC 根据实际交通自主响应。道路、车辆类别、动力学及发布的生成资产固定；ADS 只获得当前状态与已发生的历史。
3. **发现事故并修复策略。** 交通交互反馈引导既有外生随机变量的提议，风险分层搜索兼顾事故产出与执行状态覆盖。搜索改变测试资源的分配，原始生成分布保持冻结。将事故转成经过执行验证的纠正材料，结合原任务及成功驾驶材料更新 ADS。

计划适用性决定何时交权，NPC 决定如何响应。合理交互中暴露的 ADS 事故属于测试目标；世界模型的事实重建、交互响应与 ADS 安全性分别评价。

## 交付与评价原则

被测策略统一放在 `ads/`，当前选取与 nuPlan 工作区闭环入口一致的 **CaRL、Diffusion Planner**。论文主实验以 Diffusion Planner 为代表。每个 ADS 交付一份增强权重，highway-env 与 nuPlan 使用各自必要的输入、执行适配，并共享模型参数。

**闭环安全性和事故率改善是首要目标。** 分别记录碰撞、驶出道路、修好事故与新增事故。ADE／FDE、进度和低速行为用于诊断；允许避险所需的轨迹偏移与减速，不以 ADE／FDE 上升或单个实例低速否决安全收益。

长尾生成评价检验未见危险交互，自然生成评价检查驾驶能力保持。开发、修复与最终实例按用途隔离，修复前后配对使用相同实例及策略随机实现。全部尾部事件已参与长尾建模，因此这里的留出是 ADS 开发阶段的实例隔离。nuPlan 评价另行保存。

## 工程结构

| 目录 | 职责 |
| --- | --- |
| [process_highD/](process_highD/README.md) | 自然片段、共享时序、安全评分与 EVT 事件 |
| [normalizing_flow/](normalizing_flow/README.md) | 长尾和自然场景条件分布 |
| [diffusion/](diffusion/README.md) | 初态与稀疏结点条件下的多车软计划 |
| [hierarchical_world_model/](hierarchical_world_model/README.md) | 跟踪、计划适用性、NPC 接管和交通执行 |
| [ads/](ads/README.md) | 策略接入、危险搜索、纠正学习与原生交付 |
| [external_model_baselines/](external_model_baselines/README.md) | 外部驾驶人和交通世界模型对照 |
| [tools/](tools/README.md) | 共用 EVT、IDM 配置与绘图工具 |
| [HighwayEnv/](HighwayEnv/README.md) | 仿真环境依赖 |
| [highD_dataset/](highD_dataset/README.md) | 原始数据和配套读取、可视化工具 |
| [external_resources/](external_resources/README.md) | 参考论文和实际保留的上游依赖 |
| [doc/](doc/README.md) | 代码规范和按需维护的设计文档 |
| [results/](results/README.md) | 共享数据、发布资产及实验报告 |
| `archive/` | 本地历史模型及来源，不纳入 Git |

各目录 README 说明本模块的必要实现、入口和输出；实验数值以原始报告为准。源码、配置、紧凑报告及必要的图件纳入 Git。大型数据、权重和学习状态保留在磁盘，通过资产清单与 SHA256 核对；目录链接不代表目标文件已被 Git 保存。

## 长期约定

- 使用现有 `conda activate tread`，不另建环境；ADS 实验使用单种子 `20260926`。
- 根 README 只记录宏观设计、工程结构与长期约定；具体方案放在 `doc/`，实现和结果说明放在所属目录。
- 当前接纳交付名为 **Diffusion Planner Safety Enhanced（安全增强版）**。保留其学习材料、评价证据与原生权重；不自行追加实验或修复分支。
- 复用已有随机采样和标准子集结果，比较时说明预算与协议差异，不为严格对齐自动重跑对照。
- nuPlan 原生离线与城市闭环由用户手动执行，highD 结果与 nuPlan 结果分别保存。
- 图件使用 PNG，数值使用 CSV／JSON；回放使用 GIF，不生成 PDF 图件。
- 清理遵循 [doc/style.md](doc/style.md)：先核查依赖，保留复现所需参数和来源，删除已失去用途的旧代码、重复材料与中间记录。
