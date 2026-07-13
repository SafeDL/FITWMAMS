# CAT-K 正式结果

本目录保存历史锚定训练产生的 `catk_topk` checkpoint。它由两个已验证部分组成：

- 候选 `0`：与冻结基线一致的自然驾驶名义行为；
- 候选 `1--7`：MAP 校准、多 chunk 训练后的 CAT-K 联合残差候选。

锚定分支与残差全部封装在 `checkpoints/best_world_model.pt`，运行时只接收既有 START/ROLL 状态张量。测试空间仍为 `E x Z_flow x Xi_world`；它们不是额外 latent，`Xi_world` 仍是每秒的八分类索引。当前从零复现代码将名义解码器内化为可训练模块，内部 MAP 动作构成外层候选 `0`，不再依赖该锚定权重。

## 晋升证据

在 test split、固定 25 Hz 积分器、候选温度 `1.0`、2,000 次 paired bootstrap 下，候选相对冻结基线的以下差值均为 `0`，单侧 95% 上界也均为 `0`：

- EVT-tail START ADE；
- EVT-tail START FDE；
- EVT-tail START gap MAE；
- logged-ego START->ROLL ADE。

完整结果见 `paired_bootstrap_comparison.json`，全量重建指标见 `evaluation_summary.json`。logged-ego 指标只用于背景交通重建，不是 ADS 闭环成绩。
