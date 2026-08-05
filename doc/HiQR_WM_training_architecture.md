# HiQR-WM：层次化交互查询—续接交通世界模型

HiQR-WM 位于 `world_model/src/hiqr/`，与 QR-WM 并存。它使用相同的 highD 150 状态只读缓存和 76 维 Flow 坐标，但拥有独立 checkpoint、结果目录和 START 元数据 sidecar。

## 模型

- `UnifiedRelationalQueryEncoder` 在一个 `forward(mode=...)` 路径内处理 START token 与真实 ROLL 历史；TTC/DRAC 不会进入编码器，也不加载 lane graph edges。
- `HierarchicalStochasticInteractionState` 仅用 Flow 的 B0 与 t0 slot mask 初始化 `h0`；primary-risk slot 来自事件峰值风险，因而只保留为 Flow 审计/统计元数据，绝不进入 HiQR 训练批次或世界模型。随后模型按响应采样场景变量 `g` 与车辆残差 `z`，并更新持久交互状态。
- `AdaptiveJointPlanContinuationDecoder` 在一次 agent-time 联合解码中生成 START 的完整计划。ROLL 的前 20 帧 token 编码对应 carry action，再预测 gate 与修正；末尾 5 帧使用独立 tail token 生成新动作。没有动作锚点、固定 carry mix 或重复 refiner。

## 运行

```bash
python world_model/scripts/prepare_hiqr_sequence.py
python world_model/scripts/train_hiqr_world_model.py
# 从 checkpoints/last_training_state.pt 恢复
python world_model/scripts/train_hiqr_world_model.py --resume
python world_model/scripts/test_hiqr_world_model.py
python world_model/scripts/evaluate_hiqr_long_tail.py
```

HiQR 的随机世界须传入 `HiQRWorldRandomness`。在已完成 START 的响应边界，`level="scene"` 替换所有未执行响应的 g/z 随机流；`level="residual"` 保留父世界的 g 流、替换所有未执行响应的 z 流；两者都保留相同的已发生前缀。

训练每个 epoch 保存 `checkpoints/last_training_state.pt`，其中包含模型、AdamW、学习率调度器、课程阶段、全局步数和随机数状态；课程阶段只调整学习率，不会重建优化器。

离线评测报告轨迹误差及跟驰 gap、TTC、DRAC 和生成碰撞率；Flow×HiQR 评测还比较闭环回放的风险变量分布、跟驰误差与碰撞率。
