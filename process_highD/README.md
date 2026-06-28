# highD Natural Segments + SEI EVT

`process_highD/` 当前只保留 highD 自然驾驶等长片段处理流程。它不再把 following、cut-in、lane-changing 作为入口场景，而是从 highD 全量自然驾驶轨迹中抽取固定 6 s 局部交通片段，计算安全包络侵入风险 `R_SEI`，再用单一 POT-GPD EVT 标定高风险尾部阈值。

设计说明见：

```text
doc/highd_goal.md
```

## 当前文件结构

主要脚本：

```text
process_highD/scripts/extract_highd_natural_segments.py
process_highD/scripts/build_natural_evt.py
process_highD/scripts/play_highd_natural_tail_events.py
normalizing/scripts/prepare_highd_tail_flow_dataset.py
```

核心实现：

```text
process_highD/src/safety_envelope_risk.py
process_highD/src/natural_segments.py
process_highD/src/natural_evt_pipeline.py
process_highD/src/natural_event_playback.py
process_highD/src/evt_diagnostics.py
process_highD/src/loader.py
process_highD/src/preprocess.py
process_highD/src/lane_utils.py
process_highD/src/io_utils.py
```

共享的 highway-env IDM ego helper 已移到 `tools/idm_ego.py`；`process_highD/` 内不再保留生成场景或分场景旧流程代码。

配置文件：

```text
process_highD/scripts/configs/highd_natural_evt.yaml
```

## 运行入口

全量抽取 6 s 自然片段并拟合 EVT：

```bash
python process_highD/scripts/extract_highd_natural_segments.py --recordings all --with-evt
```

只用已有 `natural_segments.csv` 重拟合 EVT：

```bash
python process_highD/scripts/build_natural_evt.py
```

整理 normalizing flow 长尾数据集：

```bash
python normalizing/scripts/prepare_highd_tail_flow_dataset.py
```

播放最高风险自然片段：

```bash
python process_highD/scripts/play_highd_natural_tail_events.py --top-k 5
```

`prepare_highd_tail_flow_dataset.py` 会复用已有 `natural_tail_contexts.csv`；如果该文件不存在或显式要求重建，则由上游 EVT 结果选择所有 POT exceedance 长尾片段，即 `event_risk > u`。`play_highd_natural_tail_events.py --top-k` 只限制 GIF 渲染数量，不影响 EVT 标定或 tail context 全量文件。

## 片段定义

- 采样频率：25 Hz。
- 自然驾驶窗口：150 帧，即 6.0 s。
- 风险评估窗口：完整 150 帧；自然驾驶筛选不区分 history/future。
- anchor stride：150 帧，即同一 ego 默认不重叠。
- ego 必须是 passenger car，完整窗口存在，anchor 速度 `>= 5 m/s`，窗口内 laneId 有效，且无异常标记。

每个 anchor 在窗口起始帧固定分配最多 6 个邻车 slot：

```text
same_front, same_rear,
left_front, left_rear,
right_front, right_rear
```

slot 在完整 6 s 窗口内固定跟踪同一批车辆；缺失 slot 不补车，用 mask 表示。窗口中途出现但未被 anchor slot 捕捉的车辆进入 `untracked_*` 审查字段，但不进入主 EVT 响应变量。

## 风险定义

默认风险变量为安全包络侵入风险 `R_SEI(tau)`，即 Safety-Envelope Intrusion Risk：

```text
R_SEI(tau) = prefix_max_t Phi_raw(t)
             + exposure_weight * dt * sum_t Phi_raw(t)
```

其中 `Phi_raw(t)` 是 ego 与 fixed slot 邻车在短时预测 horizon 内的原始正安全椭圆侵入 margin。该分数不做 sigmoid、softplus、log1p 或 `[0,1]` 映射；是否超过 1 由实际侵入强度和暴露时间决定。

每个窗口帧上，ego 与 slot 邻车做短时恒速度预测，默认 horizon 为 1.5 s、step 为 0.2 s。纵向边界采用 RSS 启发的 calibrated headway：

```text
d_long_safe = d0_x + T_gap * v_follower + closing^2 / (2 * b_x)
```

默认参数：

```text
T_gap = 0.7 s
d0_x = 1.0 m
b_x = 4.0 m/s^2
d0_y = 0.2 m
rho_y = 1.0 s
b_y = 0.8 m/s^2
exposure_weight = 0.15
```

该风险兼容多种驾驶行为。跟驰风险主要来自纵向安全边界侵入；换道、切入和并行接近风险通过相邻车道 slot、横向接近速度和纵横向联合椭圆侵入体现。following、cut-in、lane-change 只应作为后验解释标签，不再作为筛选入口。

`natural_risk_traces.npz` 中的 `risk_trace` 是非递减 prefix trajectory score，因此：

```text
event_risk = max(risk_trace)
```

旧 TTC/THW/DRAC/lateral/bbox overlap 分量只作为诊断列输出，不再作为 EVT 响应变量。

## 输出文件

全量输出目录：

```text
results/highd_natural_evt/
```

关键文件：

```text
natural_segments.csv
natural_risk_traces.npz
natural_segments_summary.json
natural_tail_contexts.csv
natural_tail_contexts_summary.json
evt/natural_evt_model.json
evt/natural_evt_summary.json
evt/natural_evt_threshold_sensitivity.csv
evt/figures/*.png
playbacks/*.gif                 # 可再生成
```

`natural_segments.csv` 当前为 161314 行、43 列，包含 segment id、recording、ego、window frame、slot id、slot 窗口存在比例、`max_sei_*` 诊断列、`event_risk`、`risk_trace_row`、瞬时峰值责任字段和 late-intruder 审查字段。输出不再包含旧 TTC/THW/DRAC 诊断列、slot 布尔列、`history_*`、`future_*` 或 `uir_*` 字段。

新增解释与审查字段：

```text
peak_slot_name
peak_neighbor_id
peak_pair_risk
peak_instant_risk
peak_instant_frame
peak_instant_offset
num_untracked_candidates
max_untracked_pair_risk
max_untracked_neighbor_id
max_untracked_risk_frame
untracked_risk_exceeds_tracked_peak
```

`natural_risk_traces.npz` 包含：

```text
risk_trace                  # shape = [N, 150]
slot_time_mask_packed       # packbits 后的 [N, 150, 6] slot presence
slot_time_mask_shape
slot_names
```

## 当前全量结果

当前结果覆盖 60 个 highD recording：

```text
num_segments: 161314
natural_segments.csv shape: (161314, 43)
risk_trace shape: (161314, 150)
slot_time_mask shape: (161314, 150, 6)
window_end_frame - window_start_frame: 149
max |max(risk_trace) - event_risk|: 2.220e-16
```

`R_SEI` 分位数：

```text
q50:   0.000000
q75:   0.000000
q90:   0.179211
q95:   0.410393
q99:   0.777643
q99.5: 0.882437
q99.9: 1.074055
max:   1.439776
```

约 81.1% 片段为零风险。这对上尾 EVT 可接受，但说明当前 `R_SEI` 更适合安全关键交互激活，不适合直接作为全分布驾驶舒适性评分。

## EVT 标定

当前只对筛选后的 161314 个 fixed-window 片段做单一 raw POT-GPD EVT。若需要去重，应在数据集筛选阶段通过 anchor/窗口规则处理，不在 EVT 阶段另造 declustered 样本。

当前 POT-GPD 标定：

```text
POT threshold u:              0.7196520567
num exceedances:              2209
exceedance rate:              0.0136937898
GPD xi:                      -0.2314678736
GPD beta:                     0.1821781993
KS:                           0.015805
Cramer-von Mises:             0.112952
Anderson-Darling:             0.657273
```

当前自然驾驶长尾事件库只使用 POT exceedance 阈值：

```text
u: POT exceedance threshold; natural_tail_contexts.csv uses event_risk > u
```

因此当前 `natural_tail_contexts.csv` 默认包含 `event_risk > u` 的 2209 个自然长尾片段。

阈值稳定性应检查 `xi`、`beta - xi*u` 和 `z1000`，不要只看 `beta`。阈值敏感性表位于：

```text
results/highd_natural_evt/evt/natural_evt_threshold_sensitivity.csv
```

字段：

```text
u, k, exceedance_rate, xi, beta, modified_scale, endpoint, z1000
```

## Late-Intruder 审查

固定 anchor slot 对当前 POT exceedance tail 的解释基本可用，但有少量片段需要关注 untracked 车辆。当前 `event_risk > u` 的 2209 个自然长尾片段分布在 60 个 recording、1899 个 ego；peak slot 分布为：

```text
same_front:  1581
same_rear:    443
left_front:    86
right_front:   63
right_rear:    24
left_rear:     12
```

在这 2209 个 tail contexts 中：

```text
max_untracked_pair_risk > peak_pair_risk: 28 segments
max_untracked_pair_risk > raw u:           3 segments
```

全量审查：

```text
num_untracked_candidates > 0: 98.79% segments
max_untracked_pair_risk > peak_pair_risk: 10561 segments
max_untracked_pair_risk > raw u: 99 segments
```

因此当前结果适合“固定 6-slot 输入下的自然驾驶 POT 长尾筛选”。如果后续目标变为“完整窗口内所有交互的长尾筛选”，应新增 `event_risk_all_candidates` 或 dynamic-slot 风险并重新标定 EVT。

## 长尾播放

`play_highd_natural_tail_events.py` 从 `natural_tail_contexts.csv` 读取最高风险真实自然片段，加载对应 highD recording，按同一预处理流程做方向统一和异常标记，然后导出 GIF。画面会高亮：

- ego vehicle；
- anchor 时刻固定分配的 slot 邻车；
- 同窗口内的周围背景车；
- 下方同步的 `R_SEI` prefix risk trace。

默认输出：

```text
results/highd_natural_evt/playbacks/
```

## 已删除的旧流程

旧 following/cut-in 分场景抽取、曝光估计、tail context 生成和生成场景播放入口已经从 `process_highD` 删除，包括旧的 `highd_default.yaml`、`estimate_*_exposure.py`、`select_*_tail_contexts.py`、`play_*_tail_events.py`、`event_extraction.py`、`event_playback.py`、`following_tail_generation.py` 和 `cutin_tail_generation.py`。共享 IDM helper 已移到 `tools/idm_ego.py`。

如果以后需要 following、cut-in、lane-change 标签，它们应作为 `R_SEI` 结果的后验解释标签，而不是重新成为筛选入口或 EVT 响应变量。

## 审查重点

1. 固定 6 s 窗口：`window_end_frame = window_start_frame + 149`。
2. `risk_trace_row` 必须和 `natural_risk_traces.npz` 行号一致。
3. `max(risk_trace)` 必须等于 `event_risk`。
4. 修改 SEI 参数、窗口长度、anchor stride 或 ego 筛选条件后，必须重新生成片段并重新标定 POT/GPD。
5. EVT 只使用筛选后的 fixed-window 样本；不在 EVT 阶段再做去簇后处理。
6. `natural_tail_contexts.csv` 应使用当前 POT threshold `u` 重新生成，playback 应能播放真实 highD tail 片段。
