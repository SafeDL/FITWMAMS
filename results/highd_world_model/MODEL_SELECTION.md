# 世界模型结果说明

当前活动模型为：

```text
catk_topk
```

正式 checkpoint：

```text
catk_topk_anchored/checkpoints/best_world_model.pt
```

该 v4 checkpoint 将冻结的 v1 自然驾驶名义行为封装为候选 `0`，候选 `1--7` 来自完成 160 轮多 chunk 训练的 CAT-K 残差分支。候选 `0` 只依赖既有 START/ROLL 状态，不读取 ADS、ego future 或风险标签；其作用是为名义闭环提供不退化锚点。categorical 采样仍保留八个 `Xi_world` 分支。

test paired-bootstrap（2,000 次）相对冻结 v1 的结果：

| 指标 | 数值 |
|---|---:|
| EVT-tail START ADE 差值 | 0.000000 m |
| EVT-tail START FDE 差值 | 0.000000 m |
| EVT-tail START gap MAE 差值 | 0.000000 m |
| logged-ego START->ROLL ADE 差值 | 0.000000 m |

四项单侧 95% Bootstrap 上界均为 `0`，因此通过“不弱于冻结基线”的晋升门槛。START->ROLL replay 会使用真实 highD ego future 构造 ROLL history，因此它仍只是背景交通重建评测，不是 ADS 测试结果。

当前测试空间的环境随机变量是：

```text
E x Z_flow x Xi_world
```

其中 `Xi_world` 是 CAT-K 每个背景交通 chunk 实际选择的候选分支索引。冻结的名义锚点与七个残差分支都包含在同一 checkpoint；它们不是新的测试空间变量。失败候选保留在各自目录，避免混同正式结果。

对应实现配置为 `world_model/scripts/configs/highd_world_model.yaml`，训练缓存为 `shared_dense_start_roll/`。该缓存由 161,314 个 highD 自然驾驶片段展开为 161,314 个 START 样本和 3,226,280 个 ROLL 样本；它仅用于训练与历史重建，不是未来 ADS 测试直接加载的场景集合。
