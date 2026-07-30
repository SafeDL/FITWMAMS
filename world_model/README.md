# 背景交通世界模型

本目录保留五个从零训练或冻结复现的背景交通世界模型：RAMP-WM、FIRM-WM、Semi-Markov WM、CAT-TopK 与 QR-WM。每种方法只保留一个最佳 checkpoint、训练记录和正式测试结果。

代码按职责组织：`src/core/` 放置缓存、动力学、指标与批处理等共享能力；`src/<方法名>/` 只包含该方法的模型、训练和评测；`scripts/` 仅保留稳定的命令行入口及配置；`tests/` 对共享能力和模型行为分别回归验证。

## 统一入口

训练入口只负责训练；标准测试入口只评测自身 checkpoint：

```bash
python world_model/scripts/train_ramp_world_model.py
python world_model/scripts/train_firm_world_model.py
python world_model/scripts/train_semi_markov_world_model.py
python world_model/scripts/train_cat_topk.py
python world_model/scripts/train_qr_world_model.py

python world_model/scripts/test_ramp_world_model.py
python world_model/scripts/test_firm_world_model.py
python world_model/scripts/test_semi_markov_world_model.py
python world_model/scripts/test_cat_topk_world_model.py
python world_model/scripts/test_qr_world_model.py
```

QR-WM 是独立的联合多智能体实现：relation-aware 多头 scene encoder 产生 agent/map context，单一 persistent scene memory 保存历史交互、已执行计划和 ego 响应；联合 agent-time refiner 在 `[time, background-agent, a/yaw_rate]` 控制 buffer 上进行时间注意力、车辆注意力及 scene/map 交叉注意力。QR-WM 不使用 traffic-light 特征，也不加载 RAMP/FIRM checkpoint。

Flow 组合接口固定为 76-D `C0+B0`，并统一由共享 START adapter 将背景相对速度还原为绝对速度。`B0[6,6]` 只在 START 使用：初始化行为 latent、scene memory 和第一个 25-frame 背景控制 buffer；首段以时间衰减的凸组合融合 B0 控制先验，后续 ROLL 仅维护移位/追加的 buffer 与 scene memory。`rollout_from_flow(..., ego_future_controls=[B,T,2])` 及 ADS rollout 由动力学推进 ego，绝不以 highD 未来状态覆盖；`rollout_reconstruction` 是唯一的高D replay 重建路径。

在线 ADS 可使用 `QRWorldModelEnvironment.reset_from_flow(C0, B0, metadata)`、`observe()`、`step(ego_action)`；每次只提供下一个 0.2 秒 `[5,2]` 控制块。Flow 组合结果同时写出逐样本 audit，保留 slot mask、primary risk slot、事件结构、条件密度和联合 `log_prob`，用于概率审计或重要性加权。

所有活跃世界模型的正式训练预算统一为 40 epoch。QR-WM 会在运行目录的 `tensorboard/` 写入 batch 训练损失、epoch 训练/验证损失和验证 FDE；训练后可运行 `tensorboard --logdir results/highd_world_model/qr_world_model/tensorboard` 查看曲线。

RAMP/FIRM 的外层 EVT Flow × 内层世界模型组合测试是各自标准测试入口的显式模式：

```bash
python world_model/scripts/test_ramp_world_model.py --flow-composition
python world_model/scripts/test_firm_world_model.py --flow-composition
python world_model/scripts/test_qr_world_model.py --flow-composition
```

该模式固定为每个 held-out 回放条件 8 个 Flow 起点 × 每起点 4 条世界未来；它评估生成分布，不是逐 donor 轨迹重建，因而不报告 ADE/FDE。当前 QR-WM 没有可加载训练 artifact，完成重训与测试前不能参与比较。

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

> 该固定运行是历史复现记录：其中 RAMP/FIRM/Semi-Markov/CAT-TopK 分别训练了 40/60/30/30 epoch，且尚未包含 QR-WM。因此它只用于复现这些冻结 checkpoint，不能作为统一 40-epoch 预算下 QR-WM 与基线的公平比较；完成统一预算重训并在相同 328 个条件上加入 QR-WM 后，才可作正式横向结论。

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
