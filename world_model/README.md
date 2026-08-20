# highD 世界模型与对比基线

本目录保留仍有独立实现和回归测试的世界模型组件。当前论文主模型位于
[`hierarchical_traffic_world_model/`](../hierarchical_traffic_world_model/)；这里不复制其训练、
评测或 checkpoint。

## 维护范围

- `src/hiqr/`：主模型复用的 HiQR 关系编码器、观测滤波器和配置；
- `src/cat_topk/`：CAT-TopK 对比模型；
- `src/firm/`：FIRM 对比模型；
- `src/ramp/`：RAMP 对比模型；
- `src/semi_markov/`：Semi-Markov 对比模型；
- `src/core/`、`src/traffic_graph/`、`src/relations/`：共享数据、动力学、指标和交通图实现。

旧 QR、旧 HiQR 和 HiQR-v2 不再作为并行版本维护。当前 HiQR 名称只指
`src/hiqr/` 中由分层模型实际复用的实现。其他四种模型是独立对比方法，不属于该版本链，
因此继续保留。

## 数据

规范 highD 序列统一保存在：

```text
results/highd_shared_training_data/highd_sequence_cache/sequence_cache/
```

它包含 96,055 条清洗后序列，recording-level train/validation/test 为
72,771/13,133/10,151。每条序列有 150 个状态点和 149 个真实转移；C0 有效的背景槽位在
完整窗口内持续有效。重建命令为：

```bash
conda run -n tread python process_highD/scripts/prepare_highd_sequences.py --rebuild
```

该入口只构建共享数据，不训练任何模型。默认配置是
`process_highD/scripts/configs/highd_sequences.yaml`。

保留的对比模型使用各自配置声明的数据协议；不得把旧的 125-transition 基线缓存解释成
当前 149-transition 规范序列。

## 对比模型入口

| 模型 | 训练 | 评测 | 结果目录 |
| --- | --- | --- | --- |
| CAT-TopK | `train_cat_topk.py` | `test_cat_topk_world_model.py` | `cat_topk_world_model/` |
| FIRM | `train_firm_world_model.py` | `test_firm_world_model.py` | `firm_world_model/` |
| RAMP | `train_ramp_world_model.py` | `test_ramp_world_model.py` | `ramp_world_model/` |
| Semi-Markov | `train_semi_markov_world_model.py` | `test_semi_markov_world_model.py` | `semi_markov_world_model/` |

脚本位于 `world_model/scripts/`，配置位于 `world_model/scripts/configs/`，对应的最后一次
完整结果摘要位于 `results/highd_world_model/`。这些结果只用于方法对比，不替代当前分层模型
的三目标评价。

## 验证

```bash
conda run -n tread python -m pytest world_model/tests -q
```

测试覆盖共享批处理、动力学、指标，以及保留对比模型的形状、因果历史、随机性、候选分支
和 snapshot/restore 契约。
