# 被测 ADS

这里按算法组织被测自车，背景车辆由[交通世界模型](../hierarchical_world_model/README.md)控制。
自车 IDM 与 NPC 内部的 IDM 参数和控制权各自独立。

| 目录 | 用途 |
| --- | --- |
| [idm/](idm/README.md) | HighwayEnv IDM 自车、Monte Carlo 与子集模拟评测。 |
| [dqn/](dqn/README.md) | DQN 自车、训练、固定事件评测和定向续训。 |

从仓库根目录运行：

```bash
conda activate tread
python -m ads.dqn.train --help
python -m ads.dqn.evaluate --help
python -m ads.idm.scripts.run_world_models_idm --help
```

桥接判断原计划是否需要调整，NPC 负责合理响应；响应后暴露的 ADS 失败仍有测试价值。
评测分别分析接管理由、NPC 行为和 ADS 决策，不以减少事故数选择桥接参数。
权重和配对 replay 保存在 `dqn/artifacts/policies/`，评测结果写入 `results/`。
当前更新后的生成场景 ADS 全量对照尚未完成，已有结果的范围以对应报告为准。
