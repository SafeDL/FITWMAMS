# highD 自然驾驶等长片段与 EVT 当前目标

本文档记录 `process_highD/` 的当前执行状态、验收结果和后续必要改进；运行说明见 `process_highD/README.md`。

## 1. 当前结论

当前自然驾驶数据筛选和处理结果满足主流程要求：

- 覆盖 60 个 highD recording；
- 生成 161314 个自然驾驶片段；
- 每个片段为 6.0 s，即 150 帧；
- `window_end_frame - window_start_frame = 149`；
- `risk_trace` 形状为 `(161314, 150)`；
- `slot_time_mask_shape = [161314, 150, 6]`；
- `max(risk_trace) = event_risk`，最大数值误差为 `2.220e-16`；
- `natural_segments.csv` 形状为 `(161314, 43)`，已包含峰值责任对象和 late-intruder 审查字段；
- 新输出不再包含旧 TTC/THW/DRAC 诊断列、slot 布尔列、`history_*`、`future_*` 或 `uir_*` 字段。

当前风险分数不再压缩到 `[0,1]`。`R_SEI` 使用原始正安全椭圆侵入 margin 和线性暴露累计：

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

当前只对筛选后的 fixed-window 样本做单一 raw EVT，不再输出 declustered EVT。若需要去重，应在数据集筛选规则中处理，例如调整 anchor stride、窗口排斥规则或事件级采样规则，而不是在 EVT 阶段对筛选结果二次后处理。

当前 EVT 标定结果：

```text
POT threshold u:      0.7196520567
exceedances:          2209
exceedance rate:      0.0136937898
xi:                  -0.2314678736
beta:                 0.1821781993
KS:                   0.015805
CvM:                  0.112952
AD:                   0.657273
```

当前自然驾驶长尾事件库只使用 POT exceedance 阈值：

```text
u: POT exceedance threshold; natural_tail_contexts.csv uses event_risk > u
```

`natural_tail_contexts.csv` 默认应保存 `event_risk > u` 的全部 POT exceedances。当前对应 2209 个自然长尾片段。

约 81.1% 片段为零风险。这对上尾 EVT 可接受，但说明当前 `R_SEI` 更像安全关键交互激活分数，不适合作为全分布驾驶舒适性评分。

## 2. 必须保留的执行约定

1. 自然驾驶样本是单一等长片段，不再拆分 history/future。
2. 当前固定窗口为 6.0 s / 150 帧。
3. 风险评估窗口必须覆盖完整 6.0 s 片段。
4. 同一 ego 默认 anchor stride 为 6.0 s，避免相邻样本强重叠。
5. 安全包络侵入风险 `R_SEI` 是 EVT 主响应变量。
6. `R_SEI` 使用原始连续正侵入得分，不做 `[0,1]` 压缩。
7. `max_sei_*` 是保留的风险解释诊断列；旧 TTC、THW、DRAC、lateral、bbox overlap 分量不再输出。
8. following、cut-in、lane-change 不作为筛选入口，只可作为后验解释标签。
9. `process_highD/` 不保留废弃 wrapper、分场景旧脚本或只转发入口。

## 3. 已完成的关键修改

1. **6 秒完整窗口。** 自然驾驶筛选不再使用 1 s history + 5 s future 的解释；风险覆盖完整 150 帧。
2. **单一 raw EVT。** EVT 只使用 `natural_segments.csv` 中筛选后的 fixed-window 样本，不再做 declustered 后处理。
3. **阈值敏感性表。** EVT summary/CSV 包含：

   ```text
   u, k, exceedance_rate, xi, beta, modified_scale, endpoint, z1000
   ```

   其中 `modified_scale = beta - xi * u`。

4. **峰值责任对象。** `natural_segments.csv` 已增加：

   ```text
   peak_slot_name
   peak_neighbor_id
   peak_pair_risk
   peak_instant_risk
   peak_instant_frame
   peak_instant_offset
   ```

5. **late-intruder 审查。** `natural_segments.csv` 已增加：

   ```text
   num_untracked_candidates
   max_untracked_pair_risk
   max_untracked_neighbor_id
   max_untracked_risk_frame
   untracked_risk_exceeds_tracked_peak
   ```

6. **真实数据 playback。** `play_highd_natural_tail_events.py` 已能播放 `natural_tail_contexts.csv` 中真实 highD 长尾片段。
7. **字段与命名清理。** 用户可见风险名改为 `R_SEI`，实现文件为 `safety_envelope_risk.py`；新结果不再输出 `history_*`、`future_*` 或 `uir_*` 字段。

## 4. 当前重要发现

当前 POT exceedance tail 即 `event_risk > u` 有 2209 个片段，分布在 60 个 recording、1899 个 ego。peak slot 分布为：

- `same_front`: 1581；
- `same_rear`: 443；
- `left_front`: 86；
- `right_front`: 63；
- `right_rear`: 24；
- `left_rear`: 12。

late-intruder 审查：

```text
POT tail 中 max_untracked_pair_risk > peak_pair_risk: 28 segments
POT tail 中 max_untracked_pair_risk > raw u:           3 segments

num_untracked_candidates > 0: 98.79% segments
max_untracked_pair_risk > peak_pair_risk: 10561 segments
max_untracked_pair_risk > raw u: 99 segments
```

这说明固定 anchor slot 对当前 POT 长尾筛选基本可用，但 tail 中存在少量 untracked 风险高于 tracked peak 的片段。若后续目标是“完整窗口内所有交互风险筛选”，应新增 all-candidate 或 dynamic-slot 风险并重新标定 EVT。

## 5. 已剔除的不合适需求

以下需求不再进入当前 highD 处理目标：

1. **分场景建模入口。** 不再分别抽取 following、cut-in、lane-changing 并分别拟合 EVT。
2. **1 s history + 5 s future 切片。** 自然驾驶筛选阶段不区分历史段和预测段。
3. **旧 max-risk 响应变量。** 不再使用 `max(TTC, THW, DRAC, lateral, bbox overlap, hard brake, ...)` 作为 EVT 主变量。
4. **旧 cut-in/following exposure 流程。** 不再维护分场景抽取、曝光估计和播放链路。
5. **只转发的兼容入口。** 不再保留 `extract_highd_events.py` 或一键 wrapper。
6. **EVT 阶段去簇后处理。** 去重若有必要，应放入数据筛选规则，而不是在 EVT 拟合前另造低样本量数据。
7. **把 GPD beta 漂移直接判为拟合失败。** EVT 稳定性应看 `xi`、`beta - xi*u`、return level 和 QQ/PP/survival，而不是单独看阈值相关的 `beta`。
8. **把播放对象设为生成场景。** 当前 playback 目标是真实 highD 自然长尾片段。

## 6. 剩余改进需求

1. **如需去重，重设筛选规则。** 可以考虑更大的 anchor stride、ego-time 排斥规则或事件级 anchor 采样，但必须重新生成 `natural_segments.csv`，不能只在 EVT 阶段后处理。
2. **决定是否引入 all-candidate EVT 响应。** 当前 fixed-slot POT tail 中有 3 个片段的 untracked 风险超过 `u`；若论文目标强调完整自然交互风险，可新增 `event_risk_all_candidates` 或 dynamic-slot 版本。
3. **优化 untracked 审查性能。** 全量抽取含审查字段和 EVT 拟合约 20 分钟。当前可接受，但若频繁扫参数，需要缓存候选或降低重复 pairwise 计算。
4. **补充低风险背景评分。** 如果后续需要全量驾驶质量评估，应在 EVT 主风险之外增加低风险背景项，不要直接混入当前 EVT 响应变量。
5. **保持代码简洁。** 遵循 `doc/style.md`，不新增废弃 wrapper、无意义 fallback、未使用配置和旧分场景兼容分支。

## 7. 当前验收标准

任何后续修改完成后，至少需要通过以下检查：

```text
num_segments == 161314                 # 若未改变筛选规则
risk_trace.shape == (num_segments, 150)
slot_time_mask_shape == [num_segments, 150, 6]
window_end_frame - window_start_frame == 149
max(risk_trace) == event_risk
natural_segments.csv 包含 peak_* 与 untracked_* 字段
natural_segments.csv 不包含旧 TTC/THW/DRAC 诊断列、slot 布尔列、history_*、future_* 或 uir_* 字段
EVT summary/model/figures/threshold table 存在
结果目录不包含 declustered EVT 产物
natural_tail_contexts.csv 使用当前 POT threshold u 重新生成
play_highd_natural_tail_events.py 能播放真实 highD tail 片段
```

如果修改了筛选规则或风险函数，`num_segments` 和 `u` 可以变化，但必须在 README 和相关 summary 中明确记录变化原因。
