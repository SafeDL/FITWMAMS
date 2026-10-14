# FITWMAMS

本工程以 highD 自然驾驶数据构建交通行为仿真世界，用于自动驾驶系统（ADS）的事件测试。只维护一条主方法链：共享自然驾驶数据 → EVT 尾部事件 → Flow 采样 M、C0、K、D → PCHIP 参考轨迹 + 条件残差 Diffusion → HiQR + MA-IDM + 校准 MOBIL 换道决策 + 学习型换道轨迹规划与 Frenet 跟踪 → HighwayEnv 闭环测试。GT 评估固定观察到的 K_GT、D_GT，进入同一套轨迹重建和世界模型，保留精度与响应检查。学习型 ADS 与固定 IDM 都是被测控制器；策略特异失效区域搜索不属于当前主线。研究目标见 [`doc/00_总索引与实施顺序.md`](doc/00_总索引与实施顺序.md)，实际指标以 [`results/`](results/README.md) 中的报告为准。

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
| [`doc/`](doc/README.md) | 论文研究方案、T01–T12 工程目标与编码约定。 |
| [`external_resources/`](external_resources/README.md) | 外部论文 PDF 与上游源码快照；只有本地 nflows 源码被 Flow 实际调用。 |
| [`results/`](results/README.md) | 当前主线的数据、权重、报告与论文对照；EVT、Flow、Diffusion 分别放在简短的同名目录。 |
| [`archive/`](archive/README.md) | 已退出主线的代码、候选和历史结果，保留以便恢复，不是另一套运行入口。 |

运行 GPU 实验使用系统 Conda 环境 `tread`。各模块的具体入口、数据契约和结果边界以其目录 README 为准。隐藏的 `.git/`、`.idea/`、`.vscode/` 与 `.pytest_cache/` 是版本控制、编辑器或测试缓存目录，不属于工程模块。
