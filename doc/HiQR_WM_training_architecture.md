# HiQR-WM：层次化交互查询—续接交通世界模型

HiQR-WM 位于 `world_model/src/hiqr/`，与 QR-WM 并存。它使用相同的 highD 150 状态只读缓存和 76 维 Flow 坐标，但拥有独立 checkpoint、结果目录和 START 元数据 sidecar。

## 数据前提

正式训练只接受 QR 的 canonical raw-150 cache：`cache_format=qr_start_roll_raw150`，包含 149 个真实状态转移，即 `1.00 s START + 4.96 s ROLL`。不得以旧的 `semi_markov_sequence_cache` 或其他重建/截断 cache 替代。

默认路径 `results/highd_world_model/training_data/qr_sequence_cache/sequence_cache/` 不存在时，先运行 `prepare_qr_sequence.py` 构建该只读共享 cache；这一步不训练 QR，也不写入 QR checkpoint。随后 `prepare_hiqr_sequence.py` 只会在 HiQR 结果目录写入内容校验的 B0/审计 sidecar，绝不改写 QR cache。

## 模型

- `UnifiedRelationalQueryEncoder` 在一个 `forward(mode=...)` 路径内处理 START token 与真实 ROLL 历史；每辆车先匹配最近 polyline，再以 polyline index 差编码同车道/相邻车道关系，绝不比较各自横向偏移。TTC/DRAC 不会进入编码器，也不加载 lane graph edges。
- `HierarchicalStochasticInteractionState` 仅用 Flow 的 B0 与 t0 slot mask 初始化 `h0`；primary-risk slot 来自事件峰值风险，因而只保留为 Flow 审计/统计元数据，绝不进入 HiQR 训练批次或世界模型。随后模型按响应采样场景变量 `g` 与车辆残差 `z`，并更新持久交互状态。
- `AdaptiveJointPlanContinuationDecoder` 在一次 agent-time 联合解码中生成 START 的完整计划。ROLL 的前 20 帧 token 编码对应 carry action，并以标量 gate 混合 carry 与 proposal；gate bias 初始化为 `-2.5`，训练以 L1 gate loss 偏向保留旧计划，`revised` 只在实际动作变化超过阈值时标记。末尾 5 帧使用独立 tail token 生成新动作。零 logit 对应零加速度（范围仍为 `[-8, 4] m/s²`）。没有动作锚点、固定 carry mix 或重复 refiner。

## 运行

```bash
# 仅在 canonical QR raw-150 cache 尚不存在时执行
python world_model/scripts/prepare_qr_sequence.py
# 仅写入 HiQR 的 B0/事件审计 sidecar
python world_model/scripts/prepare_hiqr_sequence.py
# 从零训练：8 + 12 + 20 = 40 epochs
python world_model/scripts/train_hiqr_world_model.py
# 从 checkpoints/last_training_state.pt 恢复
python world_model/scripts/train_hiqr_world_model.py --resume
python world_model/scripts/test_hiqr_world_model.py
python world_model/scripts/evaluate_hiqr_long_tail.py
```

HiQR 的随机世界须传入 `HiQRWorldRandomness`。在已完成 START 的响应边界，`level="scene"` 替换所有未执行响应的 g/z 随机流；`level="residual"` 保留父世界的 g 流、替换所有未执行响应的 z 流；两者都保留相同的已发生前缀。

训练每个 epoch 保存 `checkpoints/last_training_state.pt`，其中包含模型、AdamW、学习率调度器、课程阶段、全局步数和随机数状态；课程阶段只调整学习率，不会重建优化器。

离线评测报告轨迹误差及跟驰 gap、TTC、DRAC 和生成碰撞率；Flow×HiQR 评测还比较闭环回放的风险变量分布、跟驰误差与碰撞率。每个 Flow START 的 event structure、三项对数概率、Flow checkpoint/schema hash、采样种子/温度与 rejection 配置均为 audit-only 元数据，随 reset、快照、恢复、分支和故障报告保留，并写入 `flow_start_audit.npz`。
