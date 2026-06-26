# process_highD：highD 事件、EVT 与长尾场景

`process_highD/` 负责把 highD 原始 CSV 转成 TREAD 使用的自然驾驶事件、风险缓存、
EVT 暴露量估计、长尾 scenario condition 和 diffusion 生成场景。它只保留 highD 读取、
预处理、事件抽取、EVT 拟合、tail context selection 和回放入口；跨模块复用的风险评分、
EVT 工具、暴露量汇总、绘图样式和 IDM ego 参数放在根目录 `tools/`。

默认配置在 `process_highD/scripts/configs/highd_default.yaml`。其中 highD 原始数据目录为
`../../../highD_dataset/Matlab/data`，输出目录为 `../../../results/highd_events`，
默认处理全部 60 个 recording，目标帧率为 25 Hz。

## 运行流程

### Car-Following

```bash
conda run -n tread python process_highD/scripts/extract_highd_events.py
conda run -n tread python process_highD/scripts/build_natural_dataset.py \
  --config diffusion/scripts/configs/natural_following.yaml
conda run -n tread python diffusion/scripts/train_following_diffusion.py
conda run -n tread python diffusion/scripts/evaluate_following_prior.py
conda run -n tread python process_highD/scripts/estimate_following_exposure.py
conda run -n tread python process_highD/scripts/select_following_tail_contexts.py
```

每一步的作用：

1. `extract_highd_events.py`：加载 highD CSV，校验原始目录，做坐标中心化、行驶方向归一化、
   异常帧标记和目标帧率抽样。随后抽取 following 与 cut-in 事件，并在同一遍 recording 遍历中
   写出 `events.csv`、`exposure_per_recording.csv`、following 风险/context cache 和 cut-in
   风险/context cache。
2. `build_natural_dataset.py --config natural_following.yaml`：必须从
   `following_event_segments.npz` 读取完整 following 片段，按 stride 切出 125 步训练窗口；
   若该 cache 缺失、字段不全或帧率不一致，脚本会直接报错。输出包括
   `scenario_conditions`、anchor-frame `initial_states`、`future_states` 和 lead 车 jerk `actions`。
3. `train_following_diffusion.py`：训练 following 自然驾驶动作 prior。denoiser 的条件输入是
   `scenario_conditions`，动作目标是 lead 车纵向 jerk。
4. `evaluate_following_prior.py`：在配置指定 split 上评估动作分布、rollout 重建误差和自然性图。
5. `estimate_following_exposure.py`：对固定 125 步 context 上的 `Y_long` 做 5 秒 run-length
   declustering，拟合 POT/GPD peak EVT，并用 highD 全车辆 km/时长估计 tail peak rate 和
   safety-critical intensity。
6. `select_following_tail_contexts.py`：读取 independent tail peaks 和 EVT 模型，保留经验 tail
   contexts，保存 tail scenario-condition Gaussian-copula 联合分布，并采样 5000 个 synthetic
   following tail conditions；随后调用 following diffusion prior 解码 lead 车轨迹，写入
   `results/highd_following_tail/contexts/` 和 `results/highd_following_tail/generated/`。

### Cut-In

```bash
conda run -n tread python process_highD/scripts/extract_highd_events.py
conda run -n tread python process_highD/scripts/build_natural_dataset.py \
  --config diffusion/scripts/configs/natural_cutin.yaml
conda run -n tread python diffusion/scripts/train_cutin_diffusion.py
conda run -n tread python diffusion/scripts/evaluate_cutin_prior.py
conda run -n tread python process_highD/scripts/estimate_cutin_exposure.py
conda run -n tread python process_highD/scripts/select_cutin_tail_contexts.py
```

`build_natural_dataset.py` 不带参数时默认使用
`diffusion/scripts/configs/natural_cutin.yaml`；上面仍显式传入配置，避免和 following 流程混淆。

每一步的作用：

1. `extract_highd_events.py`：同一遍 highD 遍历中检测 cut-in 候选。目标车必须是有换道记录的
   car，换道必须发生在相邻车道之间；ego 优先由 `followingId` 匹配，否则在目标车道后方找最近
   非 truck 车辆。默认 `require_immediate_leader=true`，cross 后至少 2 秒内 target 必须是 ego
   的最近前车。随后构建固定 100 步 cut-in context：anchor 从
   `cross_frame - context_pre_cross_steps` 的候选 `[15,20,25,30,35,45,50]` 中按事件确定性选择，
   窗口必须覆盖 cross 后至少 2 秒。`Y_cutin` 的纵向安全风险从 `cross_frame` 开始计算到该
   fixed window 结束，并与 following 使用同一套纵向公式和权重；横向 LTG 只在切入后
   `ltg_window_steps=5` 帧内计算。只有 `is_cutin >= 0.5` 的语义事件会写入 `cutin_event_scores.csv` 和
   `cutin_event_contexts.npz`。
2. `build_natural_dataset.py --config natural_cutin.yaml`：只使用已打分且语义成立的 cut-in 事件，
   围绕 `cross_frame` 生成等长 100 步窗口。anchor 取
   `cross_frame - cutin_pre_cross_steps` 的可用窗口候选，并要求 cross 后至少 2 秒在窗口内；当前配置
   `cutin_require_completion_in_window=false`，不强制 `cutin_end_frame` 落入训练窗口。
3. `train_cutin_diffusion.py`：训练 cut-in 自然驾驶动作 prior。denoiser 条件输入是
   `scenario_conditions`；`initial_states` 用于动作积分、评价和后续重构，不作为模型条件。
4. `evaluate_cutin_prior.py`：评估 cut-in prior 的动作分布、横向偏移、速度分布和轨迹重建误差。
5. `estimate_cutin_exposure.py`：对语义 cut-in 的 `Y_cutin` 做 declustering 和 POT/GPD 拟合，
   使用 highD 全车辆里程/时长作为暴露量分母，输出 cut-in independent tail peaks 和人类驾驶
   safety-critical intensity。
6. `select_cutin_tail_contexts.py`：读取 cut-in independent tail peaks，对 tail
   `scenario_conditions` 拟合 Gaussian copula 并保存联合分布，采样 5000 个条件，再用 cut-in
   diffusion prior 生成 target 轨迹。生成结果经过切入语义、车道进入、横向接近、横向进展和进入后保持等
   hard mask；若数量不足，会从同一 copula 分布补采样并继续解码，直到达到目标数量或 refill
   上限。

可选回放：

```bash
conda run -n tread python process_highD/scripts/play_following_tail_events.py
conda run -n tread python process_highD/scripts/play_cutin_tail_events.py
```

两个回放脚本从 generated scenario 中抽样，反查 highD recording 作为背景，并用
`tools/idm_ego.yaml` 中的 highway-env IDM 参数生成闭环 ego 响应。`select_*_tail_contexts.py`
只生成 adversary 轨迹：following 是 lead 车，cut-in 是 target 车；ego 轨迹在 playback 或
IDM_subset closed-loop 阶段计算。

回放时 `base_event_id` 指向采样 scenario condition 最近邻 highD tail context 的原始事件。
背景交通流从该 recording/frame 复现，并排除原始 ego/target 以及合成 ego/target 占用车道上的
灰色背景车，避免与合成主车和对抗车重叠。

`select_*_tail_contexts.py` 中的随机 condition 采样、扩散积分和 generated scenario 对比用于验证
条件扩散模型在给定长尾 scenario condition 下能复现相似测试场景；安全关键概率估计由 `IDM_subset/`
在同一 scenario-condition 联合分布和 diffusion latent 空间上执行。

## 主要文件

```text
process_highD/scripts/configs/highd_default.yaml
process_highD/scripts/extract_highd_events.py
process_highD/scripts/build_natural_dataset.py
process_highD/scripts/estimate_following_exposure.py
process_highD/scripts/estimate_cutin_exposure.py
process_highD/scripts/select_following_tail_contexts.py
process_highD/scripts/select_cutin_tail_contexts.py
process_highD/scripts/play_following_tail_events.py
process_highD/scripts/play_cutin_tail_events.py
process_highD/src/event_extraction.py
process_highD/src/preprocess.py
process_highD/src/loader.py
process_highD/src/following_tail_generation.py
process_highD/src/cutin_tail_generation.py
process_highD/src/event_playback.py
process_highD/src/idm_ego.py
tools/highd_longitudinal.py
tools/highd_cutin.py
tools/highd_exposure.py
tools/evt.py
tools/io.py
tools/plot_style.py
tools/idm_ego.yaml
```

## 主要输出

```text
results/highd_events/events.csv
results/highd_events/exposure_per_recording.csv
results/highd_events/following_event_scores.csv
results/highd_events/following_event_contexts.npz
results/highd_events/following_event_segments.npz
results/highd_events/following_event_cache_summary.json
results/highd_events/cutin_event_scores.csv
results/highd_events/cutin_event_contexts.npz
results/highd_events/cutin_event_cache_summary.json

results/diffusion_natural/following/dataset.npz
results/diffusion_natural/following/dataset_normalized.npz
results/diffusion_natural/following/normalization_stats.json
results/diffusion_natural/following/train_val_test_split.json
results/diffusion_natural/following/feature_schema.json
results/diffusion_natural/cutin/dataset.npz
results/diffusion_natural/cutin/dataset_normalized.npz
results/diffusion_natural/cutin/normalization_stats.json
results/diffusion_natural/cutin/train_val_test_split.json
results/diffusion_natural/cutin/feature_schema.json

results/highd_following_tail/evt/longitudinal_peak_evt_model.json
results/highd_following_tail/evt/longitudinal_peak_evt_summary.json
results/highd_following_tail/exposure/highd_exposure_summary.json
results/highd_following_tail/exposure/highd_independent_tail_peaks.csv
results/highd_following_tail/exposure/figures/
results/highd_following_tail/contexts/scenario_condition_distribution.npz
results/highd_following_tail/contexts/tail_contexts.npz
results/highd_following_tail/contexts/tail_context_summary.json
results/highd_following_tail/generated/diffusion_generated_scenarios.npz
results/highd_following_tail/generated/diffusion_generated_scenarios_summary.json
results/highd_following_tail/generated/figures/
results/highd_following_tail/generated/figures/distribution_similarity_summary.json
results/highd_following_tail/generated/event_playbacks/

results/highd_cutin_tail/evt/cutin_peak_evt_model.json
results/highd_cutin_tail/evt/cutin_peak_evt_summary.json
results/highd_cutin_tail/exposure/highd_cutin_exposure_summary.json
results/highd_cutin_tail/exposure/highd_independent_tail_peaks.csv
results/highd_cutin_tail/contexts/scenario_condition_distribution.npz
results/highd_cutin_tail/contexts/scenario_condition_distribution_summary.json
results/highd_cutin_tail/contexts/tail_contexts.npz
results/highd_cutin_tail/generated/diffusion_generated_scenarios.npz
results/highd_cutin_tail/generated/diffusion_generated_scenarios_summary.json
results/highd_cutin_tail/generated/figures/
results/highd_cutin_tail/generated/figures/distribution_similarity_summary.json
results/highd_cutin_tail/generated/event_playbacks/
```

## 实现口径

预处理：

- highD 原始 `x,y` 是 bounding box 左上角；代码先加车辆尺寸的一半，统一成车辆中心点。
- `drivingDirection == 1` 的车辆翻转 `x`、`xVelocity`、`xAcceleration` 和
  `precedingXVelocity`，使纵向前进方向统一为 positive x；`y` 和 lane ID 不翻转。
- `filter_abnormal_tracks()` 标记 `_abnormal` 帧，不直接删除车辆。事件抽取阶段若事件窗口内
  ego 或 target 有 `_abnormal` 帧，则跳过该事件。
- 默认 25 Hz 时不重采样；若配置为其他帧率，`resample_recording()` 按 source/target 的步长
  抽帧并更新 recording metadata。

事件与风险：

- `extract_highd_events.py` 只用语义和运动学规则筛选自然事件，不用风险分数筛选候选事件。
- 原始 following/cut-in 事件保持不等长；风险缓存使用与 diffusion 对齐的定长 context。
- following 的风险变量 `Y_long` 在 `context_anchor_frame` 后 125 个 future steps 上计算；
  完整 following 片段另存为 `following_event_segments.npz`，供 diffusion 训练自由滑窗。
- cut-in 的风险变量 `Y_cutin` 在 `cross_frame - context_pre_cross_steps` 锚定的 100 步窗口上
  计算，纵向风险从 `cross_frame` 开始到窗口结束；横向 LTG 只在切入后短窗口内计算。只有
  `is_cutin >= 0.5` 的语义事件进入后续流程。
- following 的统一极值尺度是 `S_EVT(Y_long)`；cut-in 的统一极值尺度是 `S_EVT(Y_cutin)`。

暴露量：

- following 和 cut-in 暴露量分母统一为 highD 全车辆累计里程和时长；主里程单位为 km。
  following 额外报告 following ego 暴露量作对照。
- 两类场景都对 5 秒 run-length decluster 后的 independent peaks 拟合 POT/GPD。
- safety-critical threshold 不使用固定 raw risk。`human_safety_threshold` 启用时，
  exposure summary 将 highD 人类驾驶全车辆配置的 all-vehicle-km return level 写为
  `collision_critical_level`。
  当前 following 阈值按 150/200/300 km 候选审计后设为 300 km return level；
  cut-in 阈值设为 3000 km return level。
- EVT 审计产物包括 POT threshold stability、mean residual life、QQ/PP、survival overlay、
  parametric-bootstrap KS/CvM/AD GOF，以及 km return-level threshold、survival/intensity
  threshold 和 threshold sensitivity 图。

## Tail Contexts

following tail context 默认配置：

```text
context_source = independent_tail_peaks
context_generation_method = gaussian_copula
include_empirical_contexts = true
num_synthetic_contexts = 5000
num_diffusion_scenarios = 5000
```

following 输出两类 context：

```text
highd_independent_tail_peak    empirical highD independent tail peak context
highd_tail_gaussian_copula     Gaussian copula sampled tail scenario context
```

following 同时输出 `scenario_condition_distribution.npz`，供 `IDM_subset/` 直接读取已拟合的
tail scenario-condition 联合分布。

cut-in 使用专用流程：

```text
semantic fixed-horizon cut-in contexts
-> EVT declustered independent tail peaks
-> Gaussian copula over tail scenario_conditions
-> sampled cut-in scenario_conditions
-> pretrained cut-in diffusion prior
-> semantic hard masks and quality metrics
-> 5000 generated cut-in scenarios
```

cut-in 同时输出 `tail_contexts.npz`。其中 empirical independent tail peaks 用作 subset 最近邻
base contexts，Gaussian-copula sampled rows 用于记录生成场景的 condition 和重构状态；安全概率
估计仍以 `scenario_condition_distribution.npz` 中保存的联合分布为准。

低维 tail scenario condition 与 diffusion 的 `scenario_conditions` 对齐。following 使用：

```text
ego_vx_0, initial_gap, initial_delta_v, lead_ax_0,
lead_speed_change, lead_min_ax, lead_braking_duration
```

cut-in 使用：

```text
ego_vx_0, initial_gap, initial_lateral_offset, initial_delta_vx,
target_ax_0, target_vy_0, target_ay_0, final_lateral_offset,
time_to_cross, target_speed_change
```

Gaussian copula 拟合时对 gap 使用 `log_initial_gap` 特征，输出给 diffusion 时恢复为
`initial_gap`。生成轨迹积分需要的 `initial_states` 由采样条件和最近邻 highD tail 事件的几何/
未建模状态共同重构；它用于物理积分、评价和 playback，不作为 denoiser 条件输入。

following tail generation 的诊断图位于
`results/highd_following_tail/generated/figures/`。其中
`scenario_condition_tail_vs_copula_sampled.png` 逐维对比 empirical tail conditions 与
Gaussian-copula sampled conditions，包含 `lead_braking_duration`；论文图
`results/paper_experiments/following/following_tail_diffusion_generalization_panel.png`
复用同一 condition 口径，并把该变量作为子图 f。

## 工程维护口径

`process_highD/` 只保留 highD 读取、事件抽取、EVT 暴露估计、tail context 生成和 playback
入口。跨模块共享的 EVT、风险评分、exposure、IO、绘图样式和 IDM ego 参数不在本目录复制，
统一从 `tools/` 读取。各入口脚本仍是公开运行接口；没有仅做转发的旧 wrapper。

`__pycache__/`、`.pyc`、模型 checkpoint、生成的 NPZ 样本和 GIF/MP4 playback 都是可再生成产物，
不应提交。若新增实验输出，优先写入 `results/` 下的任务目录，并同步检查根目录 `.gitignore`。
