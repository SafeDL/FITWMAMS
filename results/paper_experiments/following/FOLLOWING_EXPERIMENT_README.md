# Following 论文图

本目录基于已有 highD following EVT、exposure、diffusion、Monte Carlo 和
subset-simulation 结果生成 car-following 论文图。
本次没有生成 following diffusion 训练、EVT 拟合、subset simulation 或表格结果。

## 输入

- `subset_samples`: `IDM_subset/results/following/latent_subset_samples.npz`
- `evt_model`: `results/highd_following_tail/evt/longitudinal_peak_evt_model.json`
- `exposure_summary`: `results/highd_following_tail/exposure/highd_exposure_summary.json`
- `tail_condition_distribution`: `results/highd_following_tail/contexts/scenario_condition_distribution.npz`
- `tail_contexts`: `results/highd_following_tail/contexts/tail_contexts.npz`
- `tail_generated_scenarios`: `results/highd_following_tail/generated/diffusion_generated_scenarios.npz`
- `following_segment_cache`: `results/highd_events/following_event_segments.npz`

## 生成产物

- `results/paper_experiments/following/following_gpd_diagnostic_panel.png`
- `results/paper_experiments/following/following_safety_threshold_inverse_calibration.png`
- `results/paper_experiments/following/following_tail_diffusion_generalization_panel.png`
- `results/paper_experiments/following/following_tail_diffusion_acceleration_profiles.png`
- `results/paper_experiments/following/following_subset_level_score_histograms.png`

## 复用的已有产物

- 复用已有产物：`following_gpd_diagnostic_panel: results/highd_following_tail/evt/longitudinal_peak_evt_model.json`
- 复用已有产物：`following_safety_threshold_inverse_calibration: results/highd_following_tail/evt/longitudinal_peak_evt_model.json`
- 复用已有产物：`following_safety_threshold_inverse_calibration: results/highd_following_tail/exposure/highd_exposure_summary.json`
- 复用已有产物：`following_tail_diffusion_generalization_panel: results/highd_following_tail/contexts/scenario_condition_distribution.npz`
- 复用已有产物：`following_tail_diffusion_generalization_panel: results/highd_following_tail/contexts/tail_contexts.npz`
- 复用已有产物：`following_tail_diffusion_generalization_panel: results/highd_following_tail/generated/diffusion_generated_scenarios.npz`
- 复用已有产物：`following_tail_diffusion_generalization_panel: results/highd_events/following_event_segments.npz`
- 复用已有产物：`following_tail_diffusion_acceleration_profiles: results/highd_following_tail/generated/diffusion_generated_scenarios.npz`
- 复用已有产物：`following_subset_level_score_histograms: IDM_subset/results/following/latent_subset_samples.npz`

## 跳过的产物

- 无

## 解读说明

- following 论文图直接生成在本目录下，不使用 `figures/` 子目录。
- 所有论文图都使用共享 TREAD 论文样式：300 dpi 导出、Times 兼容衬线字体，以及 STIX/LaTeX 风格数学渲染。
- 诊断面板展示拟合后的 POT/GPD tail 诊断，绘图范围限制在 `Y_long = 10`。
- inverse calibration 图标出 exposure summary 中选定的 300 km all-vehicle return-level threshold。
- tail diffusion generalization 面板比较经验 following EVT-tail contexts 与生成的 lead trajectories；面板 f 使用 `process_highD` 中的 `lead_braking_duration` scenario-condition distribution。
- acceleration-profile 图用 5-95% envelope 和代表性制动模式概括 diffusion 生成的长尾 lead-vehicle acceleration traces。
- subset level histogram 展示 subset simulation 如何把质量集中到校准后的 EVT risk threshold 附近。
