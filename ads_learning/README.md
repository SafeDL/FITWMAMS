# 学习型 ADS 控制器

本目录提供与固定 IDM ADS 对应的学习型被测控制器及其训练接口：

- `ADSHighwayEnv`：49 维因果观测、五个语义动作、5 Hz 决策与 25 Hz 世界执行；
- `train.py`：DQN 训练与 replay-buffer 续训；
- `evaluate.py`：固定事件上的碰撞、越界、停车和进度评估；
- `repair.py`、`compare.py`：给定事件集的基础续训与配对回归检查。

ADS 不读取未来计划、驾驶员潜变量、事件风险或随机种子。失效搜索不是本目录的职责；
依赖 S1/S4 搜索的旧 T10 正式编排已退役，基础续训不依赖它。

Example:

```bash
conda activate tread
python -m ads_learning.train \
  --events results/evt/manifests/mixed_normal_evt_train.csv \
  --sequence-cache results/highd_shared_training_data/highd_sequence_cache/sequence_cache \
  --world-checkpoint results/hierarchical_world_model/hiqr_training/checkpoints/stochastic_heads_best.pt \
  --output-dir ads_learning/artifacts/dqn_run --steps 10000 --seed 20260926
```

局部 `--response-gate` 接口仍供已有 ADS 实验使用，但不是当前世界模型主线的响应选择器。
标准世界模型使用标定 IDM、校准 MOBIL、学习型换道规划和 Frenet 跟踪。
旧 ADS 结果没有因目录整理重新训练或评测；其旧世界权重保存在
`archive/results/hierarchical_world_model/factual_hiqr/`，不能当作当前主线结果。

当前唯一维护的学习型 ADS 是三种子 DQN 10,000 步策略，权重和配对 replay 见
[`artifacts/`](artifacts/README.md)。旧选策与评测已随其专用清单整体归档到
[`archive/results/paper_benchmark/`](../archive/results/paper_benchmark/README.md)；它们没有在
当前世界权重上重跑，因此不属于当前结果主链。
训练步数和种子号是实验条件，不是版本号；旧策略、阶段权重与逐文件冻结清单
已退出活动树。后续训练权重放在本模块，评测输出放在 `results/`。
