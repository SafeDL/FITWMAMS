# 世界模型结果说明

当前活动模型为：

```text
catk_topk
```

正式 checkpoint：

```text
catk_topk/checkpoints/best_world_model.pt
```

该 checkpoint 是历史锚定训练的性能证据：候选 `0` 与冻结基线的名义行为一致，候选 `1--7` 来自多 chunk 训练的 CAT-K 残差分支。当前从零复现代码已将名义解码器内化为可训练模块：其内部 MAP 动作构成外层候选 `0`，并由 `nominal_logit_margin` 保证外层 argmax 选择该候选。代码不依赖这份 checkpoint 或外部基线权重；categorical 采样仍保留八个 `Xi_world` 分支。

test paired-bootstrap（2,000 次）相对冻结基线的结果：

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

其中 `Xi_world` 是 CAT-K 每个背景交通 chunk 实际选择的候选分支索引。名义分支与七个残差分支都包含在同一 checkpoint；它们不是新的测试空间变量。中间候选、训练日志和重复配置已清理，避免被误认为正式结果。

对应实现配置为 `world_model/scripts/configs/highd_world_model.yaml`，训练缓存为 `shared_dense_start_roll/`。该缓存由 161,314 个 highD 自然驾驶片段展开为 161,314 个 START 样本和 3,226,280 个 ROLL 样本；它仅用于训练与历史重建，不是未来 ADS 测试直接加载的场景集合。
