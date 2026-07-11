# 世界模型结果说明

当前活动模型为：

```text
catk_topk
```

checkpoint：

```text
catk_topk/checkpoints/best_world_model.pt
```

该目录由历史训练产物更名而来；本次仅调整模型组织、环境接口和路径，不重新训练或执行全量评测。

历史 highD logged-ego 重建指标：

| 指标 | 数值 |
|---|---:|
| EVT-tail ADE | 0.028524 m |
| EVT-tail FDE | 0.041313 m |
| EVT-tail gap MAE | 0.026337 m |
| logged-ego START->ROLL ADE | 0.055516 m |

这些指标验证背景车动作与轨迹重建。当前 START->ROLL replay 会使用真实 highD ego future 构造 ROLL history，因此不是 ADS 测试结果。

当前测试空间的环境随机变量是：

```text
E x Z_flow x Xi_world
```

其中 `Xi_world` 是 CAT-K 每个背景交通 chunk 实际选择的候选分支索引。结果目录只保留 checkpoint、配置、schema、训练摘要、训练历史和评测摘要；可再生成的 TensorBoard、可视化、逐事件 CSV 与已淘汰候选结果均已清理。

对应实现配置为 `world_model/scripts/configs/highd_world_model.yaml`，训练缓存为 `shared_dense_start_roll/`。该缓存由 161,314 个 highD 自然驾驶片段展开为 161,314 个 START 样本和 3,226,280 个 ROLL 样本；它仅用于训练与历史重建，不是未来 ADS 测试直接加载的场景集合。
