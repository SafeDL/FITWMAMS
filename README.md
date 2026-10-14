# FITWMAMS

以 highD 自然驾驶数据构建用于自动驾驶系统（ADS）测试的交通环境。

主链：自然驾驶数据 → EVT 尾部事件 → Flow 场景条件 → PCHIP + 条件残差 Diffusion 软计划 → HiQR 动作、NPC 响应及计划进度协调 → highway-env 执行。Flow 建模占位、初始条件和位置结点；车辆尺寸单独经验采样。

世界模型分别验证完整 Test 的事实重建，以及 NPC 自主运行的交通分布。ADS 接入使用同一执行链，不能用事实重建成绩代替 ADS 测试结果。当前报告见 [results](results/README.md)。

## 一级目录

| 目录 | 用途 |
| --- | --- |
| [`highD_dataset/`](highD_dataset/README.md) | highD 配套的 Matlab/Python 数据处理与可视化工具。 |
| [`process_highD/`](process_highD/README.md) | 清洗、重采样和抽取自然驾驶片段，构建共享序列缓存及 SEI-EVT 风险标签。 |
| [`normalizing_flow/`](normalizing_flow/README.md) | 连续建模尾部场景的 M、C0 与位置结点 K，并按占位模式经验采样尺寸 D。 |
| [`diffusion/`](diffusion/README.md) | 根据初始场景与稀疏位置结点重建多车软轨迹，尺寸不输入模型。 |
| [`hierarchical_world_model/`](hierarchical_world_model/README.md) | 当前 HiQR、统一几何与动力学、NPC 在线响应和闭环执行。 |
| [`HighwayEnv/`](HighwayEnv/README.md) | 项目使用的 HighwayEnv 道路仿真环境源码。 |
| [`ads_learning/`](ads_learning/README.md) | 学习型 ADS 的环境、DQN/PPO 策略训练与评估；基础续训用于测试策略适应性。 |
| [`idm_ams/`](idm_ams/README.md) | 固定 IDM ADS 的长尾风险评测工具；不是主世界模型。 |
| [`external_model_baselines/`](external_model_baselines/README.md) | 外部论文模型的项目适配实现、评测和模型登记。 |
| [`tools/`](tools/README.md) | EVT、固定 IDM ADS 和绘图样式等跨模块共享工具。 |
| [`doc/style.md`](doc/style.md) | 代码、文件组织与清理规范。 |
| [`external_resources/`](external_resources/README.md) | 参考论文与外部模型的上游源码。 |
| [`results/`](results/README.md) | 当前主线的数据、权重、报告与论文对照；EVT、Flow、Diffusion 分别放在简短的同名目录。 |
| [`archive/`](archive/README.md) | 已退出主线的代码、候选和历史结果，保留以便恢复，不是另一套运行入口。 |

运行环境为 `conda activate tread`；入口和数据口径见各模块 README。根目录运行 `python -m pytest` 检查项目测试；`pytest.ini` 限定收集范围，不包含上游源码及归档测试。
