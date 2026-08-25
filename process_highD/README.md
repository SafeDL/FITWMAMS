# highD 自然驾驶片段与 SEI EVT

`process_highD/` 当前只保留 highD 自然驾驶等长片段处理流程。它不再把 following、cut-in、lane-changing 作为入口场景，而是从 highD 全量自然驾驶轨迹中抽取固定 6 s 局部交通片段，计算安全包络侵入风险 `R_SEI`，再用单一 POT-GPD EVT 标定高风险尾部阈值。

设计说明见本文件以及 [`doc/world_model_goal.md`](../doc/world_model_goal.md)。

## 当前文件结构

主要脚本：

```text
process_highD/scripts/extract_highd_natural_segments.py
process_highD/scripts/build_natural_evt.py
process_highD/scripts/play_highd_natural_events.py
process_highD/scripts/prepare_highd_sequences.py
normalizing_flow/scripts/prepare_highd_natural_driving_flow_dataset.py
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

## 共享预处理

`process_highD/src/preprocess.py` 的 `prepare_recording(raw_dir, recording_id, config)` 是当前唯一的 highD 读取入口。它按固定顺序执行：读取 recording、统一行驶方向、标记异常轨迹、重采样到配置的目标帧率。

自然片段抽取、真实片段回放、`normalizing_flow/` 数据集构造和 `world_model/` 数据集构造均调用该函数。因此三处模块使用同一套原始轨迹坐标、异常标记和 25 Hz 采样规则。

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

整理全量自然驾驶 normalizing-flow 数据集：

```bash
python normalizing_flow/scripts/prepare_highd_natural_driving_flow_dataset.py
```

构建所有生成模型共用的 150 状态点（149 个真实转移）规范序列缓存：

```bash
python process_highD/scripts/prepare_highd_sequences.py --rebuild
```

该入口只从当前清洗后的自然驾驶数据构建共享缓存，不训练世界模型；序列表示、交通图和
动力学的共享实现仍位于 `world_model/src/core/`。

播放最高风险自然片段：

```bash
python process_highD/scripts/play_highd_natural_events.py --top-k 5
```

`prepare_highd_natural_driving_flow_dataset.py` 直接读取唯一的 `natural_segments.csv`，并验证
`lateral_event_complete=1`；它不读取也不构建 EVT-tail context 子集。EVT 的 `u` 只被写成
`is_evt_tail` 风险标定标签，用于 Flow 评估的尾部/非尾部分层。`play_highd_natural_events.py
--top-k` 只限制 GIF 渲染数量，不影响 EVT 标定或 Flow 数据集。

## 片段定义

- 采样频率：25 Hz。
- 自然驾驶窗口：150 帧，即 6.0 s。
- 风险评估窗口：完整 150 帧；自然驾驶筛选不区分 history/future。
- anchor stride：150 帧，即同一 ego 默认不重叠。
- ego 必须是 passenger car，完整窗口存在，anchor 速度 `>= 5 m/s`，窗口内 laneId 有效，且无异常标记。
- 若 ego、固定 slot 或窗口内相关车辆发生 laneId crossing，则必须是相邻车道
  变化，并在窗口内至少保留 1 秒 pre-cross、2 秒 post-cross 和 crossing 前后
  各 5 帧稳定车道；被窗口边界截断的横向事件不进入主数据。

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

该风险兼容多种驾驶行为。跟驰风险主要来自纵向安全边界侵入；换道、切入和并行接近风险通过相邻车道 slot、横向接近速度和纵横向联合椭圆侵入体现。

横向事件完整性属于上游数据质量契约，不是 cut-in 类别采样：没有换道的合法
跟驰片段仍然保留；窗口内发生换道但缺少完整前后语义的片段被清除；通过前方、
70% post-cross 同车道和无中间前车检查的事件另标为 strict cut-in。EVT、Flow
和世界模型均验证 `lateral_event_complete=1`，旧缓存不能静默复用。

`natural_risk_traces.npz` 中的 `risk_trace` 是非递减 prefix trajectory score，因此：

```text
event_risk = max(risk_trace)
```

旧 TTC/THW/DRAC/lateral/bbox overlap 分量只作为诊断列输出，不再作为 EVT 响应变量。

## 输出文件

全量输出目录：

```text
results/highd_natural_driving_evt/
```

关键文件：

```text
natural_segments.csv
natural_risk_traces.npz
natural_segments_summary.json
natural_tail_contexts.csv         # 按需从当前 EVT 模型生成，不长期保留
natural_tail_contexts_summary.json # 与上项同步生成
evt/natural_evt_model.json
evt/natural_evt_summary.json
evt/natural_evt_threshold_sensitivity.csv
evt/figures/*.png
playbacks/*.gif                 # 可再生成
```

新 `natural_segments.csv` 额外记录横向事件完整性、事件数量和 primary strict
cut-in 元数据。当前全量结果已按该契约重建。横向事件完整性是不可关闭的
清洗规则；旧口径数据及其下游缓存、模型和 EVT 阈值不作为历史副本保留。

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
lateral_event_complete
lateral_event_reject_reason
num_lane_changes
num_strict_cutins
primary_cutin_target_id
primary_cutin_cross_frame
```

`natural_risk_traces.npz` 包含：

```text
risk_trace                  # shape = [N, 150]
slot_time_mask_packed       # packbits 后的 [N, 150, 6] slot presence
slot_time_mask_shape
slot_names
```

## 当前全量结果与重建边界

60 个 recording 的完整性筛选结果：

```text
num_segments: 96055
lateral_event_complete: 100%
C0-active background slots stable through the full window: 100%
removed for background_slot_exits_window: 39850
```

其中移除的横向不完整窗口主要是 `insufficient_post_cross_context=22820`、
`insufficient_pre_cross_context=10281` 和 `unstable_lane_transition=510`。
150 个 25 Hz 状态点覆盖 6 秒采样窗口，首末状态时间差为 5.96 秒。

当前 `R_SEI` 的 q95/q99 为 0.393878/0.784674；重新拟合的 EVT 阈值
`u=0.534674`，tail 样本数为 2964，`z1000=1.100401`。

当前数据中约 82.89% 的片段为零风险。这对上尾 EVT 可接受，但说明当前
`R_SEI` 更适合安全关键交互激活，不适合直接作为全分布驾驶舒适性评分。

## EVT 标定

当前只对筛选后的 96055 个 fixed-window 片段做单一 raw POT-GPD EVT。若需要
去重，应在数据集筛选阶段通过 anchor/窗口规则处理，不在 EVT 阶段另造
declustered 样本。

当前 POT-GPD 标定：

```text
POT threshold u:              0.5346742868
num exceedances:              2964
exceedance rate:              0.0308573213
GPD xi:                      -0.2769226691
GPD beta:                     0.2555120557
KS:                           0.009178
Cramer-von Mises:             0.029464
Anderson-Darling:             0.236391
```

当前自然驾驶长尾事件库只使用 POT exceedance 阈值：

```text
u: POT exceedance threshold; natural_tail_contexts.csv uses event_risk > u
```

按需运行播放脚本时，`natural_tail_contexts.csv` 会由当前模型重新生成，并包含
`event_risk > u` 的 2964 个自然长尾片段；该派生文件不作为长期维护的核心产物。

阈值稳定性应检查 `xi`、`beta - xi*u` 和 `z1000`，不要只看 `beta`。阈值敏感性表位于：

```text
results/highd_natural_driving_evt/evt/natural_evt_threshold_sensitivity.csv
```

字段：

```text
u, k, exceedance_rate, xi, beta, modified_scale, endpoint, z1000
```

## Late-Intruder 审查

固定 anchor slot 对当前 POT exceedance tail 的解释基本可用，但有少量片段需要
关注 untracked 车辆。当前 2964 条 `event_risk > u` 片段覆盖 60 个 recording 和
2617 个 recording-ego group，peak slot 分布为：

```text
same_front:  2349
same_rear:    421
left_front:    99
right_front:   71
right_rear:    17
left_rear:      7
```

当前 tail contexts 中：

```text
max_untracked_pair_risk > peak_pair_risk: 62 segments
max_untracked_pair_risk > raw u:          51 segments
```

全量审查：

```text
num_untracked_candidates > 0: 98.62% segments
max_untracked_pair_risk > peak_pair_risk: 7718 segments
max_untracked_pair_risk > raw u: 125 segments
```

因此当前结果适合“固定 6-slot 输入下的自然驾驶 POT 长尾筛选”。如果后续目标变为“完整窗口内所有交互的长尾筛选”，应新增 `event_risk_all_candidates` 或 dynamic-slot 风险并重新标定 EVT。

## 长尾播放

`play_highd_natural_events.py` 可从 `natural_tail_contexts.csv` 读取最高风险
片段，也可从完整母数据随机抽取 `all`、`lane-change`、`strict-cutin` 或
`no-lane-change` cohort。它加载对应 highD recording，按同一预处理流程做方向
统一和异常标记，然后导出 GIF。画面会高亮：

- ego vehicle；
- anchor 时刻固定分配的 slot 邻车；
- strict cut-in 的真实 target 和 crossing 时刻；
- 同窗口内的周围背景车；
- 标题中的当前/整段 `R_SEI` prefix risk 值。

例如固定 seed 随机审查两个 strict cut-in：

```bash
python process_highD/scripts/play_highd_natural_events.py \
  --random-count 2 --random-seed 20260813 \
  --cohort strict-cutin --frame-stride 2
```

默认输出：

```text
results/highd_natural_driving_evt/playbacks/
```

## 已删除的旧流程

旧 following/cut-in 分场景抽取、曝光估计、tail context 生成和生成场景播放入口已经从 `process_highD` 删除，包括旧的 `highd_default.yaml`、`estimate_*_exposure.py`、`select_*_tail_contexts.py`、`play_*_tail_events.py`、`event_extraction.py`、`event_playback.py`、`following_tail_generation.py` 和 `cutin_tail_generation.py`。共享 IDM helper 已移到 `tools/idm_ego.py`。

following、cut-in、lane-change 仍不是 EVT 响应变量；其中 lane-change 完整性是
数据质量门槛，strict cut-in 是通过上游语义检查得到的解释标签。

## 审查重点

1. 固定 6 s 窗口：`window_end_frame = window_start_frame + 149`。
2. `risk_trace_row` 必须和 `natural_risk_traces.npz` 行号一致。
3. `max(risk_trace)` 必须等于 `event_risk`。
4. 修改 SEI 参数、窗口长度、anchor stride 或 ego 筛选条件后，必须重新生成片段并重新标定 POT/GPD。
5. EVT 只使用筛选后的 fixed-window 样本；不在 EVT 阶段再做去簇后处理。
6. `natural_tail_contexts.csv` 应使用当前 POT threshold `u` 重新生成，playback 应能播放真实 highD tail 片段。
