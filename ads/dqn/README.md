# DQN 自车

提供 49 维因果观测、五个语义动作、5 Hz 决策和 25 Hz 世界执行。
ADS 仅观察当前交通状态，不读取未来计划、驾驶员潜变量或事件风险。

| 入口 | 用途 |
| --- | --- |
| `train.py` | 训练及配对 replay-buffer 续训。 |
| `evaluate.py` | 固定事件的碰撞、越界、停车和行驶进度评测。 |
| `repair.py` | 混合正常、新失效和历史事件进行定向续训。 |
| `compare.py` | 同事件、同随机实现的续训前后对照。 |
| `cohort_report.py`、`multiseed_report.py` | 按事件群和训练种子汇总。 |
| `runtime.py`、`env.py` | 世界模型接入与观测、动作接口。 |

从仓库根目录运行：

```bash
conda activate tread
python -m ads.dqn.train \
  --events results/evt/manifests/mixed_normal_evt_train.csv \
  --sequence-cache results/highd_shared_training_data/highd_sequence_cache/sequence_cache \
  --world-config hierarchical_world_model/config.yaml \
  --output-dir ads/dqn/artifacts/dqn_run --steps 10000 --seed 20260926
```

`FrozenWorldFactory` 加载冻结的跟踪器、计划适用性桥接和 highD NPC 参数，
自然与生成场景使用同一接口。每辆 NPC 最多接管一次，之后自主运行至回合结束。
具体方法与验证范围见[世界模型说明](../../hierarchical_world_model/README.md)。

保留的基线为三种子 DQN、每种子训练 10,000 步：
`artifacts/policies/seed*/policy.zip` 是策略，同行的 `replay_buffer.pkl` 是对应续训状态。
[历史选策结果](../../archive/results/paper_benchmark/selection.json)没有在当前世界模型上重跑，
不能作为当前 ADS 成绩。新权重写入 `artifacts/`，评测 CSV 和 JSON 写入 `results/`；
输入文件的路径、大小及 SHA256 随结果记录。
