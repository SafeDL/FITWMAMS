# 背景交通世界模型

本目录保留四个从零训练或冻结复现的背景交通世界模型：RAMP-WM、FIRM-WM、Semi-Markov WM 与 CAT-TopK。每种方法只保留一个最佳 checkpoint、训练记录和正式测试结果。

## 统一入口

训练入口只负责训练；标准测试入口只评测自身 checkpoint：

```bash
python world_model/scripts/train_ramp_world_model.py
python world_model/scripts/train_firm_world_model.py
python world_model/scripts/train_semi_markov_world_model.py
python world_model/scripts/train_cat_topk.py

python world_model/scripts/test_ramp_world_model.py
python world_model/scripts/test_firm_world_model.py
python world_model/scripts/test_semi_markov_world_model.py
python world_model/scripts/test_cat_topk_world_model.py
```

RAMP/FIRM 的外层 EVT Flow × 内层世界模型组合测试是各自标准测试入口的显式模式：

```bash
python world_model/scripts/test_ramp_world_model.py --flow-composition
python world_model/scripts/test_firm_world_model.py --flow-composition
```

该模式固定为每个 held-out 回放条件 8 个 Flow 起点 × 每起点 4 条世界未来；它评估生成分布，不是逐 donor 轨迹重建，因而不报告 ADE/FDE。

## 正式长尾重建评测

```bash
python world_model/scripts/evaluate_long_tail_reproduction.py
```

该命令使用四个最终 checkpoint，在同一 held-out EVT-tail highD 事件上固定历史、B0、道路条件和 ego 回放；每个条件生成 1 条确定性轨迹及 32 条随机闭环未来。正式结果写入：

```text
results/highd_world_model/long_tail_reproduction/
  study_manifest.json
  selected_events.json
  overview/
  ramp_world_model/
  firm_world_model/
  semi_markov_world_model/
  cat_topk_world_model/
```

每个模型目录包含独立 `metrics.json`、六张 300 dpi 图和三段固定长尾事件 GIF。评测覆盖：

- 1--5 s ADE/FDE、`minADE@32`、`minFDE@32`；
- 加速度、jerk、曲率误差和速度 KL；
- 全车辆对碰撞、同车道 TTC/DRAC/gap/相对速度、制动响应；
- Wasserstein-1、KS、RBF-MMD 与交通行为特征 Fréchet 距离。

CAT-TopK 的 START 使用归档的未来首秒动作摘要，结果始终带有信息条件不对称标记，只作复现实验参考。

### 当前固定运行（2026-07-29）

本仓库的正式结果由固定种子 `20260729`、328 个 held-out EVT-tail 条件和每条件 32 个随机未来生成；
完整数值、哈希和图像以各模型的 `metrics.json` 为准。下表仅作便于阅读的摘要（距离单位为 m，越低越好）：

| 模型 | 确定性 FDE | minFDE@32 | minADE@32 | 特征 Fréchet | 碰撞 episode rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| RAMP-WM | 0.752 | 0.746 | 0.188 | 15.145 | 0.0000 |
| FIRM-WM | **0.710** | **0.473** | **0.166** | 4.159 | 0.0007 |
| Semi-Markov WM | 0.928 | 0.881 | 0.223 | **0.720** | 0.0000 |
| CAT-TopK（信息不对称，仅参考） | 0.779 | 0.570 | 0.152 | 4.050 | 0.0041 |

在同一结果中，FIRM 是当前 RAMP 架构改进中轨迹重建和多分支覆盖最好的版本；Semi-Markov 的聚合交通
特征分布最接近真实，但轨迹重建较弱。RAMP 的随机分支几乎没有降低其 FDE，说明其现有随机性尚未形成有效的
条件多模态覆盖。更关键的是，独立的 8×4 Flow 组合审核得到 RAMP/FIRM 的碰撞 episode rate 分别为
0.348/0.374；它们尚不具备可直接宣称的物理可靠长尾动态测试环境能力。该审核是非配对分布测试，不能与上述
重建 ADE/FDE 混用。

## 边界

- RAMP/Semi-Markov 在 START 使用冻结 B0 行为锚定；之后只读取已发生 ego 状态和已生成背景历史。
- FIRM 在 START 使用 C0、B0 和 Flow 条件；之后只读取已发生状态。
- 当前 highD 评测为固定车辆槽位、5 秒闭环背景重建；不包含车辆新增、消失或跨数据集泛化。
- 当前数据缺少可靠的路口让行/插入标注；社会行为指标限定为可验证的跟驰、制动响应和安全间距一致性。
