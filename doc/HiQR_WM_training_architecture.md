# HiQR-WM：层次化交互查询—续接交通世界模型

HiQR-WM 位于 `world_model/src/hiqr/`，与 QR-WM 并存。它使用相同的 highD 150 状态只读缓存和 76 维 Flow 坐标，但拥有独立 checkpoint、结果目录和 START 元数据 sidecar。

## 模型

- `UnifiedRelationalQueryEncoder` 在一个 `forward(mode=...)` 路径内处理 START token 与真实 ROLL 历史；TTC/DRAC 不会进入编码器，也不加载 lane graph edges。
- `HierarchicalStochasticInteractionState` 用 Flow 的 B0、slot mask 和 primary-risk slot 初始化 `h0`，随后按响应采样场景变量 `g` 与车辆残差 `z`，并更新持久交互状态。
- `AdaptiveJointPlanContinuationDecoder` 在一次 agent-time 联合解码中生成 START 的完整计划，或修正 ROLL 的剩余 20 帧并生成新的 5 帧尾部。没有动作锚点、固定 carry mix 或重复 refiner。

## 运行

```bash
python world_model/scripts/prepare_hiqr_sequence.py
python world_model/scripts/train_hiqr_world_model.py
python world_model/scripts/test_hiqr_world_model.py
python world_model/scripts/evaluate_hiqr_long_tail.py
```

HiQR 的随机世界须传入 `HiQRWorldRandomness`。在已完成 START 的响应边界，`level="scene"` 同时改变 g/z，`level="residual"` 固定来自父世界的 g 并仅改变 z；两者都保留相同的已发生前缀。

离线评测报告轨迹误差及跟驰 gap、TTC、DRAC 和生成碰撞率；Flow×HiQR 评测还比较闭环回放的风险变量分布、跟驰误差与碰撞率。
